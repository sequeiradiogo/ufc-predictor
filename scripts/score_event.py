"""
score_event.py -- Score the most recent prediction against actual UFC results.

Scrapes fight results from UFCStats, optionally odds from BFO, then updates
the prediction markdown with actual results and P/L analysis.

Usage:
  python scripts/score_event.py                       # auto-detect most recent unscored prediction
  python scripts/score_event.py --json path/to/f.json
  python scripts/score_event.py --json path/to/f.json --rescore   # re-score an already-scored event
  python scripts/score_event.py --summary-only                    # only rebuild predictions/SUMMARY.md
"""

import argparse
import difflib
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scrapers.ufcstats import _browser_session, _col_ps_text, _normalize_method
from scrapers.bestfightodds import (
    _fetch_bfo_events, _fetch_bfo_archive_events, _fetch_fighter_events,
    _scrape_event_odds, _name_key, _parse_bfo_date,
)
from utils.logger import get_logger

log = get_logger("score_event")

PREDICTIONS_DIR   = ROOT / "predictions"
SCORED_MARKER     = "Actual Result"
LOOKBACK_DAYS     = 10
MIN_CONFIDENCE    = 0.55  # default threshold for the high-confidence breakout + P/L gating


# -- UFCStats result scraping -------------------------------------------------

def scrape_results(event_url: str) -> list[dict]:
    """
    Scrape fight results from a completed UFCStats event page.
    The event listing puts the winner first, so fighter_links[0] is always
    the winner.
    """
    results = []
    try:
        with _browser_session() as get:
            soup = get(event_url)
            for row in soup.select("tr.b-fight-details__table-row__hover"):
                fighter_links = row.select("a[href*='fighter-details']")
                if len(fighter_links) < 2:
                    continue
                winner_name = fighter_links[0].get_text(strip=True)
                loser_name  = fighter_links[1].get_text(strip=True)
                tds = row.find_all("td", class_="b-fight-details__table-col")
                method_text = _col_ps_text(tds, 7, 0) if len(tds) > 7 else ""
                round_text  = _col_ps_text(tds, 8, 0) if len(tds) > 8 else ""
                if not method_text.strip() or not round_text.strip():
                    # Upcoming events list fighters but have no method/round yet --
                    # fighter_links[0] is NOT a winner in that case, just listing order.
                    continue
                method = _normalize_method(method_text)
                # W/L column: "win" for a decided fight, "draw" / "nc" otherwise.
                # The method column can't tell a majority draw from a majority
                # decision ("M-DEC" either way), and fighter_links[0] is then
                # just listing order -- a draw used to be scored as a win for
                # whoever was listed first.
                outcome = _col_ps_text(tds, 0, 0).lower()
                if outcome == "draw":
                    method = "Draw"
                elif outcome == "nc":
                    method = "No Contest"
                try:
                    round_num = int(round_text)
                except ValueError:
                    round_num = 0
                results.append({
                    "winner": winner_name,
                    "loser":  loser_name,
                    "method": method,
                    "round":  round_num,
                })
    except Exception as exc:
        log.error("Failed to scrape results from %s: %s", event_url, exc)
    return results


# -- BFO odds -----------------------------------------------------------------

def _american_to_decimal(odds: int | None) -> float | None:
    if odds is None:
        return None
    return round(odds / 100 + 1, 3) if odds > 0 else round(100 / abs(odds) + 1, 3)


def scrape_bfo_odds(
    event_name: str,
    event_date: date,
    fighter_pairs: list[tuple[str, str]],
) -> dict[str, tuple[int | None, int | None]]:
    """
    Look up closing odds on BFO for each (red, blue) fighter pair.
    Returns {fight_key: (odds_red, odds_blue)}.  fight_key = 'Red vs Blue'.

    score_event.py always runs after the event has happened, so the event has
    usually already dropped off BFO's homepage (upcoming events only) -- the
    archive of completed events is checked as well. BFO event names are often
    location-based (e.g. "UFC Oklahoma") rather than fighter-based like
    UFCStats/our own naming, so events are matched by date and confirmed by
    their card actually containing our fighters. The archive only covers the
    ~20 most recent events (all orgs), so any fight still uncovered after
    that is looked up through the fighter's own BFO page (needed when
    re-scoring older events).
    """
    try:
        bfo_events = _fetch_bfo_events()
    except Exception as exc:
        log.warning("BFO event list fetch failed: %s", exc)
        bfo_events = []

    try:
        archive_events = _fetch_bfo_archive_events()
    except Exception as exc:
        log.warning("BFO archive fetch failed: %s", exc)
        archive_events = []

    def _on_date(events: list[dict]) -> list[dict]:
        out = []
        for e in events:
            parsed = _parse_bfo_date(e["date_text"], ref_year=event_date.year)
            if parsed and abs((parsed - event_date).days) <= 1:
                out.append(e)
        return out

    fighter_keys = {_name_key(n) for pair in fighter_pairs for n in pair}
    matched_events, matchups = [], []
    seen_urls: set[str] = set()

    def _try_event(ev: dict) -> None:
        # Scrape a candidate and keep it only if its card contains our
        # fighters -- BFO covers many orgs on the same date. BFO also splits
        # a single UFC card across several listings under different names
        # (e.g. "UFC 332" for the main card + a bare "UFC" page for the
        # prelims), so every matching candidate is kept, not just the first.
        if ev["url"] in seen_urls:
            return
        seen_urls.add(ev["url"])
        try:
            ev_matchups = _scrape_event_odds(ev["url"])
        except Exception as exc:
            log.warning("BFO odds scrape failed for '%s': %s", ev["name"], exc)
            return
        ev_keys = {_name_key(m["r_name"]) for m in ev_matchups} | \
                  {_name_key(m["b_name"]) for m in ev_matchups}
        if ev_keys & fighter_keys:
            matched_events.append(ev)
            matchups.extend(ev_matchups)

    for cand in _on_date(bfo_events + archive_events):
        _try_event(cand)

    # Fights still uncovered (event older than the archive window, or a card
    # listing the archive missed): find the event via the fighter's BFO page.
    for red, blue in fighter_pairs:
        if _fuzzy_odds_lookup(red, blue, matchups) is not None:
            continue
        for name in (red, blue):
            try:
                fighter_events = _fetch_fighter_events(name)
            except Exception as exc:
                log.warning("BFO fighter lookup failed for '%s': %s", name, exc)
                continue
            for ev in _on_date(fighter_events):
                _try_event(ev)
            if _fuzzy_odds_lookup(red, blue, matchups) is not None:
                break

    if not matched_events:
        log.warning("No BFO event matched '%s' (%s)", event_name, event_date)
        return {}

    log.info("BFO event(s) matched: %s", ", ".join(e["name"] for e in matched_events))

    # Build lookup keyed by (red_key, blue_key)
    bfo_lookup: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    for m in matchups:
        rk = _name_key(m["r_name"])
        bk = _name_key(m["b_name"])
        bfo_lookup[(rk, bk)] = (m["odds_red"], m["odds_blue"])
        bfo_lookup[(bk, rk)] = (m["odds_blue"], m["odds_red"])

    out: dict[str, tuple[int | None, int | None]] = {}
    for red, blue in fighter_pairs:
        rk   = _name_key(red)
        bk   = _name_key(blue)
        key  = f"{red} vs {blue}"
        odds = bfo_lookup.get((rk, bk))
        if odds is None:
            # Exact key match failed -- fuzzy-match names (hyphenation like
            # "Saint-Denis" vs "Saint Denis", suffixes like "Kai Kamaka III",
            # nicknames like "Zach" vs "Zachary" all break the exact lookup).
            odds = _fuzzy_odds_lookup(red, blue, matchups)
        out[key] = odds if odds is not None else (None, None)
    return out


# -- Name matching ------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _name_key(a), _name_key(b)).ratio()


_ODDS_SIM_FLOOR = 0.75  # per-fighter similarity floor for odds fuzzy fallback


def _fuzzy_odds_lookup(
    red: str, blue: str, matchups: list[dict],
) -> tuple[int | None, int | None] | None:
    """Fuzzy-match (red, blue) against BFO matchups when exact keys don't align."""
    best_score, best = 0.0, None
    for m in matchups:
        s1, s2 = _sim(red, m["r_name"]), _sim(blue, m["b_name"])
        s3, s4 = _sim(blue, m["r_name"]), _sim(red, m["b_name"])
        if s1 >= _ODDS_SIM_FLOOR and s2 >= _ODDS_SIM_FLOOR:
            score, odds = s1 + s2, (m["odds_red"], m["odds_blue"])
        elif s3 >= _ODDS_SIM_FLOOR and s4 >= _ODDS_SIM_FLOOR:
            score, odds = s3 + s4, (m["odds_blue"], m["odds_red"])
        else:
            continue
        if score > best_score:
            best_score, best = score, odds
    return best


_SIM_FLOOR = 0.6  # minimum per-fighter similarity; prevents false positives when one fighter matches exactly


def _match_result(red: str, blue: str, results: list[dict]) -> dict | None:
    """Fuzzy-match a predicted fight to an actual result entry."""
    best_score, best = 0.0, None
    for r in results:
        s1 = _sim(red, r["winner"]); s2 = _sim(blue, r["loser"])
        s3 = _sim(blue, r["winner"]); s4 = _sim(red, r["loser"])
        if s1 >= _SIM_FLOOR and s2 >= _SIM_FLOOR:
            score = s1 + s2
        elif s3 >= _SIM_FLOOR and s4 >= _SIM_FLOOR:
            score = s3 + s4
        else:
            continue
        if score > best_score:
            best_score, best = score, r
    return best if best_score > 1.0 else None


# -- Result string formatting -------------------------------------------------

_METHOD_SHORT = {
    "KO/TKO":               "KO",
    "Submission":           "Sub",
    "Technical Submission": "Tech Sub",
    "Decision":             "Dec",
    "Decision - Unanimous": "Dec",
    "Decision - Split":     "Dec (Split)",
    "Decision - Majority":  "Dec (Majority)",
    "No Contest":           "NC",
    "Draw":                 "Draw",
}


_VOID_METHODS = ("Draw", "No Contest")


def _is_void(r: dict) -> bool:
    """Draw or no contest: no winner, so neither right nor wrong, and bets push."""
    return r["method"] in _VOID_METHODS


def _result_str(r: dict) -> str:
    if _is_void(r):
        return r["method"]
    method = _METHOD_SHORT.get(r["method"], r["method"])
    if "Dec" in method or "NC" in method or "Draw" in method:
        return f"{r['winner']} ({method})"
    return f"{r['winner']} ({method} R{r['round']})"


# -- Markdown update ----------------------------------------------------------

def _strip_scoring(md: str) -> str:
    """
    Undo a previous scoring pass (header date, result lines, Post-Event
    Summary) so the event can be scored again. The predictions table needs no
    undoing -- score_markdown() rebuilds it from the JSON either way.
    """
    md = re.sub(r" \| Scored: \S+", "", md, count=1)
    md = re.sub(
        r"(Fighters making their UFC debut were excluded.*?\n)"
        r"(?:\n?(?:\*\*Result:|\*High-confidence|\*Bookmaker favorite).*\n)+",
        r"\1",
        md,
        count=1,
    )
    md = re.sub(
        r"## Post-Event Summary\n.*?(?=## Raw Model Output)",
        "",
        md,
        count=1,
        flags=re.DOTALL,
    )
    return md


def _favorite_baseline(fight_rows: list[dict]) -> dict:
    """
    How the bookmaker favorite did on the fights the model is scored on.
    Returns n, fav_ok, model_ok, against (model picked the underdog), against_ok.
    """
    out = {"n": 0, "fav_ok": 0, "model_ok": 0, "against": 0, "against_ok": 0}
    for s in fight_rows:
        r, b = s["odds_red"], s["odds_blue"]
        if s["correct"] not in ("YES", "NO") or r is None or b is None or r == b:
            continue
        favorite = s["red_name"] if r < b else s["blue_name"]
        model_ok = s["correct"] == "YES"
        out["n"] += 1
        out["fav_ok"] += _sim(favorite, s["actual"]["winner"]) > 0.6
        out["model_ok"] += model_ok
        if _sim(s["pred_winner"], favorite) <= 0.6:
            out["against"] += 1
            out["against_ok"] += model_ok
    return out


def _favorite_is_red(s: dict) -> bool | None:
    """True/False for the red/blue corner as bookmaker favorite; None without odds or a pick'em."""
    r, b = s["odds_red"], s["odds_blue"]
    if r is None or b is None or r == b:
        return None
    return r < b


def _flat_bet(s: dict, on_red: bool) -> tuple[float, float, str] | None:
    """
    EUR 1 on one corner: (decimal odds, P/L, "Win"/"Loss"/"Push"), or None when
    there's no price or no result. A draw / no contest voids the bet (stake
    returned), as on a 2-way moneyline.
    """
    dec = _american_to_decimal(s["odds_red"] if on_red else s["odds_blue"])
    if dec is None or s["actual"] is None:
        return None
    if _is_void(s["actual"]):
        return dec, 0.0, "Push"
    won = _sim(s["red_name"] if on_red else s["blue_name"], s["actual"]["winner"]) > 0.6
    return (dec, round(dec - 1, 2), "Win") if won else (dec, -1.0, "Loss")


def _eur(x: float) -> str:
    return f"+EUR {x:.2f}" if x >= 0 else f"-EUR {abs(x):.2f}"


def _pl_str(pl: float, staked: float) -> str:
    roi = pl / staked * 100 if staked else 0.0
    return f"{_eur(pl)} on EUR {staked:.0f} staked ({roi:+.1f}% ROI)"


def score_markdown(
    md_path: Path,
    predictions: list[dict],
    results: list[dict],
    odds_map: dict[str, tuple[int | None, int | None]],
    min_confidence: float = MIN_CONFIDENCE,
    rescore: bool = False,
) -> str:
    """
    Update *md_path* with actual results and P/L.
    With *rescore*, an already-scored file is stripped and scored again.
    Returns a short summary string for logging.
    """
    md = md_path.read_text(encoding="utf-8")

    if SCORED_MARKER in md:
        if not rescore:
            log.info("Already scored: %s", md_path)
            return "already scored"
        md = _strip_scoring(md)

    today = date.today().isoformat()

    # Add "Scored: <date>" to the header line
    md = re.sub(
        r"(Model:.*?Generated: \S+)",
        rf"\1 | Scored: {today}",
        md,
        count=1,
    )

    # -- Match each prediction to an actual result ----------------------------
    fight_rows: list[dict] = []
    for p in predictions:
        red  = p["red_name"]
        blue = p["blue_name"]
        # Confidence is max(red_prob, blue_prob); JSON stores percents as floats (e.g. 63.5)
        red_prob  = float(p.get("red_prob", 50.0))
        blue_prob = float(p.get("blue_prob", 50.0))
        conf      = max(red_prob, blue_prob) / 100.0
        is_fifty  = abs(conf - 0.50) < 0.001

        pred_winner = p.get("winner") or (red if red_prob >= blue_prob else blue)
        actual      = _match_result(red, blue, results)
        fight_key   = f"{red} vs {blue}"

        if actual is None:
            result_str = "?"
            correct    = "?"
        elif _is_void(actual):
            result_str = _result_str(actual)
            correct    = "PUSH"
        elif is_fifty:
            result_str = _result_str(actual)
            correct    = "--"
        else:
            result_str = _result_str(actual)
            correct    = "YES" if _sim(pred_winner, actual["winner"]) > 0.6 else "NO"

        odds_red, odds_blue = odds_map.get(fight_key, (None, None))
        fight_rows.append({
            "fight_key":   fight_key,
            "pred_winner": pred_winner,
            "conf":        f"{conf:.1%}",
            "conf_val":    conf,
            "is_fifty":    is_fifty,
            "is_high_conf": conf >= min_confidence,
            "result_str":  result_str,
            "correct":     correct,
            "actual":      actual,
            "odds_red":    odds_red,
            "odds_blue":   odds_blue,
            "red_name":    red,
            "blue_name":   blue,
        })

    # -- Build updated table --------------------------------------------------
    # fight_rows is parallel to predictions; finish_str is stored in the JSON
    # at generation time (predict_event.py) so no markdown re-parse is needed.
    has_odds = any(r["odds_red"] is not None for r in fight_rows)

    if has_odds:
        header = "| Fight | Predicted Winner | Confidence | Likely Method | Odds (Red / Blue) | Actual Result | Correct? |"
        sep    = "|---|---|---|---|---|---|---|"
    else:
        header = "| Fight | Predicted Winner | Confidence | Likely Method | Actual Result | Correct? |"
        sep    = "|---|---|---|---|---|---|"

    new_rows: list[str] = []
    for i, s in enumerate(fight_rows):
        meth_col = predictions[i].get("finish_str", "?")

        if has_odds:
            r_str = f"{s['odds_red']:+d}"  if s["odds_red"]  is not None else "N/A"
            b_str = f"{s['odds_blue']:+d}" if s["odds_blue"] is not None else "N/A"
            row = (f"| {s['fight_key']} | {s['pred_winner']} | {s['conf']} "
                   f"| {meth_col} | {r_str} / {b_str} | {s['result_str']} | {s['correct']} |")
        else:
            row = (f"| {s['fight_key']} | {s['pred_winner']} | {s['conf']} "
                   f"| {meth_col} | {s['result_str']} | {s['correct']} |")
        new_rows.append(row)

    new_table = header + "\n" + sep + "\n" + "\n".join(new_rows)

    md = re.sub(
        r"\| Fight \|.*?\n\|[-|]+\|\n.*?(?=\n\n|\n---)",
        new_table,
        md,
        count=1,
        flags=re.DOTALL,
    )

    if SCORED_MARKER not in md:
        log.error(
            "Table replacement regex did not match in %s -- markdown format may have changed. "
            "File left unchanged to prevent data loss.",
            md_path,
        )
        return "error: table not found"

    # -- Accuracy summary line ------------------------------------------------
    n_correct  = sum(1 for s in fight_rows if s["correct"] == "YES")
    n_scored   = sum(1 for s in fight_rows if s["correct"] in ("YES", "NO"))
    n_excluded = sum(1 for s in fight_rows if s["correct"] == "--")
    n_push     = sum(1 for s in fight_rows if s["correct"] == "PUSH")
    acc_str    = f"{n_correct}/{n_scored} ({n_correct/n_scored:.1%})" if n_scored else "N/A"

    result_line = f"**Result: {acc_str}**"
    if n_excluded:
        s = "s" if n_excluded > 1 else ""
        result_line += f" *({n_excluded} fight{s} excluded: 50/50 pick{s})*"
    if n_push:
        result_line += f" *({n_push} excluded: draw/NC)*"

    # High-confidence breakout (supplementary -- does not affect the overall accuracy line above)
    n_correct_hc = sum(1 for s in fight_rows if s["correct"] == "YES" and s["is_high_conf"])
    n_scored_hc  = sum(1 for s in fight_rows if s["correct"] in ("YES", "NO") and s["is_high_conf"])
    hc_line = ""
    if n_scored_hc:
        hc_acc_str = f"{n_correct_hc}/{n_scored_hc} ({n_correct_hc/n_scored_hc:.1%})"
        hc_line = f"*High-confidence (>={min_confidence:.0%}) picks: {hc_acc_str}*\n"

    # Bookmaker-favorite baseline on the same fights (result known, not a 50/50
    # pick, both odds present and different). The model only adds value if it
    # beats this; Jun-Oct 2026 it trailed by ~15pp and nothing showed it.
    fav = _favorite_baseline(fight_rows)
    fav_line = ""
    if fav["n"]:
        fav_line = (f"*Bookmaker favorite: {fav['fav_ok']}/{fav['n']} ({fav['fav_ok']/fav['n']:.1%}) "
                    f"vs model {fav['model_ok']}/{fav['n']} ({fav['model_ok']/fav['n']:.1%}) on the same fights*\n")

    md = re.sub(
        r"(Fighters making their UFC debut were excluded.*?\n)",
        rf"\1\n{result_line}\n{hc_line}{fav_line}",
        md,
        count=1,
    )

    # -- Post-Event Summary + P/L ---------------------------------------------
    # Model: EUR 1 on each high-confidence pick. Favorite: EUR 1 on the
    # bookmaker favorite in those same fights, so both strategies are compared
    # bet for bet. Pushes (draw/NC) return the stake and aren't counted as
    # staked; fights without a price can't be bet and are skipped.
    pl_rows: list[str] = []
    net_pl = staked = 0.0
    fav_pl = fav_staked = 0.0

    for s in fight_rows:
        if s["is_fifty"] or not s["is_high_conf"]:
            continue
        bet = _flat_bet(s, _sim(s["pred_winner"], s["red_name"]) > 0.6)
        if bet is None:
            continue
        dec_odds, pl, result_label = bet
        net_pl += pl
        staked += result_label != "Push"

        fav_red = _favorite_is_red(s)
        fav_bet = _flat_bet(s, fav_red) if fav_red is not None else None
        if fav_bet is None:
            fav_cells = "-- | --"
        else:
            fav_dec, f_pl, f_label = fav_bet
            fav_pl += f_pl
            fav_staked += f_label != "Push"
            fav_name = s["red_name"] if fav_red else s["blue_name"]
            fav_cells = f"{fav_name} ({fav_dec:.2f}) | {_eur(f_pl)}"

        pl_rows.append(
            f"| {s['fight_key']} | {s['pred_winner']} | {dec_odds:.2f} | {result_label} "
            f"| {_eur(pl)} | {fav_cells} |"
        )

    # Always backing the favorite, on every fight with odds and a result
    # (no confidence gate) -- the plain "follow the bookmaker" strategy.
    all_fav_pl = all_fav_staked = 0.0
    all_fav_n = 0
    for s in fight_rows:
        fav_red = _favorite_is_red(s)
        fav_bet = _flat_bet(s, fav_red) if fav_red is not None else None
        if fav_bet is None:
            continue
        all_fav_n += 1
        all_fav_pl += fav_bet[1]
        all_fav_staked += fav_bet[2] != "Push"

    summary: list[str] = [
        "---",
        "",
        "## Post-Event Summary",
        "",
        f"- Fights predicted: {len(fight_rows)}"
        + (f" ({n_excluded} excluded: 50/50 pick{'s' if n_excluded > 1 else ''})" if n_excluded else "")
        + (f" ({n_push} excluded: draw/NC)" if n_push else ""),
        f"- Correct: {acc_str}",
    ]
    if n_scored_hc:
        summary.append(f"- High-confidence (>={min_confidence:.0%}) picks: {n_correct_hc}/{n_scored_hc} ({n_correct_hc/n_scored_hc:.1%})")
    if fav["n"]:
        summary.append(f"- Bookmaker favorite on the same {fav['n']} fights: {fav['fav_ok']}/{fav['n']} "
                       f"({fav['fav_ok']/fav['n']:.1%}) -- model {fav['model_ok']}/{fav['n']} ({fav['model_ok']/fav['n']:.1%})")
        summary.append(f"- Model picked against the favorite in {fav['against']} fight(s), right in {fav['against_ok']}")
    if all_fav_n:
        summary.append(f"- Always backing the favorite (EUR 1 on all {all_fav_n} fights with odds): "
                       f"{_pl_str(all_fav_pl, all_fav_staked)}")

    if pl_rows:
        summary += [
            "",
            f"### P/L (EUR 1 flat on picks >={min_confidence:.0%} confidence)",
            "",
            "| Fight | Model Pick | Odds (dec) | Result | P/L | Favorite (dec) | Favorite P/L |",
            "|---|---|---|---|---|---|---|",
        ] + pl_rows + [
            "",
            f"**Net P/L: {_pl_str(net_pl, staked)}**",
            "",
            f"**Favorite on the same fights: {_pl_str(fav_pl, fav_staked)}**",
        ]

    summary_block = "\n".join(summary) + "\n\n"

    # Insert summary before "## Raw Model Output"
    md = re.sub(
        r"---\n\n## Raw Model Output",
        summary_block + "## Raw Model Output",
        md,
        count=1,
    )

    md_path.write_text(md, encoding="utf-8")
    return acc_str


# -- Prediction finder --------------------------------------------------------

def find_unscored_prediction() -> Path | None:
    """Return the JSON path for the most recent unscored prediction, or None."""
    cutoff = date.today() - timedelta(days=LOOKBACK_DAYS)
    candidates: list[tuple[str, Path]] = []

    for json_path in sorted(PREDICTIONS_DIR.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta       = json.loads(json_path.read_text(encoding="utf-8"))
            event_date = date.fromisoformat(meta.get("date", "1970-01-01"))
        except Exception:
            continue
        if event_date < cutoff or event_date > date.today():
            continue
        md_path = json_path.with_suffix(".md")
        if not md_path.exists():
            continue
        if SCORED_MARKER not in md_path.read_text(encoding="utf-8"):
            candidates.append((meta.get("date", ""), json_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# -- Main ---------------------------------------------------------------------

SUMMARY_PATH = PREDICTIONS_DIR / "SUMMARY.md"


def event_fights(md: str) -> list[dict]:
    """Every scored fight row of an event markdown (YES / NO / PUSH), in card order."""
    out = []
    for line in md.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 6 or " vs " not in c[0] or c[-1] not in ("YES", "NO", "PUSH"):
            continue
        red, blue = c[0].split(" vs ", 1)
        odds = re.match(r"([+-]\d+) / ([+-]\d+)", c[4]) if len(c) >= 7 else None
        out.append({"red_name": red, "blue_name": blue, "pick": c[1],
                    "confidence": round(float(c[2].rstrip("%")) / 100, 4),
                    "odds": [int(odds.group(1)), int(odds.group(2))] if odds else None,
                    "result": c[-2], "correct": {"YES": True, "NO": False}.get(c[-1])})
    return out


def _event_record(md: str, min_confidence: float) -> dict | None:
    """Re-count one scored event from its markdown table (model, high-conf, favorite, P/L)."""
    rec = {"model_ok": 0, "n": 0, "hc_ok": 0, "hc_n": 0, "fav_ok": 0, "fav_model_ok": 0,
           "fav_n": 0, "against": 0, "against_ok": 0, "pl": 0.0, "staked": 0.0,
           "fav_pl": 0.0, "fav_staked": 0.0, "all_fav_pl": 0.0, "all_fav_staked": 0.0}
    for line in md.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 6 or " vs " not in c[0] or c[-1] not in ("YES", "NO"):
            continue
        red, blue = c[0].split(" vs ", 1)
        pick, ok = c[1], c[-1] == "YES"
        conf = float(c[2].rstrip("%")) / 100
        rec["n"] += 1
        rec["model_ok"] += ok
        if conf >= min_confidence:
            rec["hc_n"] += 1
            rec["hc_ok"] += ok
        odds = re.match(r"([+-]\d+) / ([+-]\d+)", c[4]) if len(c) >= 7 else None
        if odds and odds.group(1) != odds.group(2):
            favorite = red if int(odds.group(1)) < int(odds.group(2)) else blue
            winner = re.sub(r"\s*\(.*$", "", c[-2])
            rec["fav_n"] += 1
            rec["fav_ok"] += _sim(favorite, winner) > 0.6
            rec["fav_model_ok"] += ok
            if _sim(pick, favorite) <= 0.6:
                rec["against"] += 1
                rec["against_ok"] += ok
    if not rec["n"]:
        return None
    for key, label in (("pl", "Net P/L"), ("fav_pl", "Favorite on the same fights"),
                       ("all_fav_pl", r"Always backing the favorite \(.*?\)")):
        m = re.search(label + r": ([+-])EUR ([\d.]+) on EUR ([\d.]+) staked", md)
        if m:
            rec[key] = float(m.group(2)) * (1 if m.group(1) == "+" else -1)
            rec[key.replace("pl", "staked")] = float(m.group(3))
    return rec


def _scored_events(min_confidence: float) -> list[tuple[Path, str, dict]]:
    """(markdown path, event title, _event_record) for every scored event, oldest first."""
    out = []
    for md_path in sorted(PREDICTIONS_DIR.glob("*/*.md")):
        md = md_path.read_text(encoding="utf-8")
        if SCORED_MARKER not in md:
            continue
        rec = _event_record(md, min_confidence)
        if rec is None:
            continue
        out.append((md_path, md.splitlines()[0].lstrip("# ").split(" -- ")[0], rec))
    return out


def summary_totals(min_confidence: float = MIN_CONFIDENCE, events=None) -> dict:
    """The _event_record counts summed over every scored event, plus `events` (their number)."""
    tot = {"events": 0}
    for _, _, rec in _scored_events(min_confidence) if events is None else events:
        tot["events"] += 1
        for k, v in rec.items():
            tot[k] = tot.get(k, 0) + v
    return tot


def write_summary(min_confidence: float = MIN_CONFIDENCE) -> Path:
    """
    Regenerate predictions/SUMMARY.md: one row per scored event plus totals --
    model vs bookmaker favorite on the same fights, high-confidence accuracy,
    and flat-stake P/L. Rebuilt from the scored markdown, so it never drifts.
    """
    pct = lambda a, b: f"{a}/{b} ({a / b:.1%})" if b else "--"
    rows = []
    events = _scored_events(min_confidence)
    tot = summary_totals(min_confidence, events)
    for md_path, title, rec in events:
        roi = f"{rec['pl'] / rec['staked']:+.1%}" if rec["staked"] else "--"
        fav_roi = f"{rec['fav_pl'] / rec['fav_staked']:+.1%}" if rec["fav_staked"] else "--"
        all_roi = f"{rec['all_fav_pl'] / rec['all_fav_staked']:+.1%}" if rec["all_fav_staked"] else "--"
        rows.append(f"| {md_path.parent.name[:10]} | {title} | {pct(rec['model_ok'], rec['n'])} "
                    f"| {pct(rec['hc_ok'], rec['hc_n'])} | {pct(rec['fav_model_ok'], rec['fav_n'])} "
                    f"| {pct(rec['fav_ok'], rec['fav_n'])} | {rec['against_ok']}/{rec['against']} "
                    f"| {rec['pl']:+.2f} | {roi} | {rec['fav_pl']:+.2f} | {fav_roi} "
                    f"| {rec['all_fav_pl']:+.2f} | {all_roi} |")
    if not rows:
        return SUMMARY_PATH
    t = tot
    roi = f"{t['pl'] / t['staked']:+.1%}" if t["staked"] else "--"
    fav_roi = f"{t['fav_pl'] / t['fav_staked']:+.1%}" if t["fav_staked"] else "--"
    all_roi = f"{t['all_fav_pl'] / t['all_fav_staked']:+.1%}" if t["all_fav_staked"] else "--"
    gap = (t["fav_model_ok"] - t["fav_ok"]) / t["fav_n"] if t["fav_n"] else 0.0
    text = "\n".join([
        "# Prediction track record",
        "",
        f"Regenerated by `scripts/score_event.py` after every scored event ({date.today().isoformat()}). "
        "Favorite = the sportsbook favorite on BestFightOdds; the model only adds value where it beats it. "
        f"P/L is EUR 1 flat: the model on its >={min_confidence:.0%} picks, the favorite on those same fights, "
        "and \"always favorite\" on every fight with odds. Draws / no contests are pushes (stake returned).",
        "",
        f"**{len(rows)} events: model {pct(t['model_ok'], t['n'])} | high-confidence (>={min_confidence:.0%}) "
        f"{pct(t['hc_ok'], t['hc_n'])} | on fights with odds, model {pct(t['fav_model_ok'], t['fav_n'])} vs "
        f"favorite {pct(t['fav_ok'], t['fav_n'])} ({gap:+.1%}) | against the favorite: right "
        f"{t['against_ok']}/{t['against']}**",
        "",
        f"**P/L: model {t['pl']:+.2f} EUR on {t['staked']:.0f} staked ({roi}) | favorite on the same fights "
        f"{t['fav_pl']:+.2f} EUR on {t['fav_staked']:.0f} ({fav_roi}) | always favorite "
        f"{t['all_fav_pl']:+.2f} EUR on {t['all_fav_staked']:.0f} ({all_roi})**",
        "",
        f"| Date | Event | Model | High-conf (>={min_confidence:.0%}) | Model (fights with odds) | Favorite "
        "| Against favorite (right/picks) | Model P/L (EUR) | Model ROI | Fav P/L, same fights | Fav ROI "
        "| Always-fav P/L | Always-fav ROI |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        f"| **Total** | | **{pct(t['model_ok'], t['n'])}** | **{pct(t['hc_ok'], t['hc_n'])}** "
        f"| **{pct(t['fav_model_ok'], t['fav_n'])}** | **{pct(t['fav_ok'], t['fav_n'])}** "
        f"| **{t['against_ok']}/{t['against']}** | **{t['pl']:+.2f}** | **{roi}** "
        f"| **{t['fav_pl']:+.2f}** | **{fav_roi}** | **{t['all_fav_pl']:+.2f}** | **{all_roi}** |",
        "",
    ])
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    return SUMMARY_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the most recent event prediction.")
    parser.add_argument("--json", default=None, help="Prediction JSON path (default: auto-detect)")
    parser.add_argument(
        "--min-confidence", type=float, default=MIN_CONFIDENCE,
        help=f"Confidence threshold (0-1) for the high-confidence breakout and P/L gating (default: {MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--rescore", action="store_true",
        help="Re-score an already-scored prediction (requires --json): re-fetches results + odds",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only regenerate predictions/SUMMARY.md from the already-scored events",
    )
    args = parser.parse_args()
    if args.rescore and not args.json:
        parser.error("--rescore requires --json")
    if args.summary_only:
        print(f"Wrote {write_summary(args.min_confidence)}")
        _write_feed(args.min_confidence)
        return

    json_path = Path(args.json) if args.json else find_unscored_prediction()
    if json_path is None:
        print("No recent unscored predictions found. Nothing to do.")
        sys.exit(0)

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    log.info("Scoring: %s  (%s)", meta["event"], meta["date"])
    print(f"\nEvent:     {meta['event']}  ({meta['date']})")

    event_date = date.fromisoformat(meta.get("date", "1970-01-01"))
    if event_date > date.today():
        print(f"[ERROR] Event date {event_date} is in the future -- nothing to score yet.")
        sys.exit(1)

    event_url = meta.get("event_url")
    if not event_url:
        print(f"[ERROR] No event_url in {json_path}. Re-generate predictions with an updated predict_event.py.")
        sys.exit(1)

    print("Scraping results from UFCStats...")
    results = scrape_results(event_url)
    if not results:
        print("[WARN] No results found -- event may not have happened yet or UFCStats is not updated.")
        sys.exit(0)
    print(f"Found {len(results)} fight result(s).")

    predictions  = meta["fights"]
    fighter_pairs = [(p["red_name"], p["blue_name"]) for p in predictions]

    print("Fetching odds from BFO...")
    odds_map = scrape_bfo_odds(meta["event"], event_date, fighter_pairs)

    md_path = json_path.with_suffix(".md")
    acc_str = score_markdown(md_path, predictions, results, odds_map,
                             min_confidence=args.min_confidence, rescore=args.rescore)

    print(f"\nScored:    {md_path}")
    print(f"Accuracy:  {acc_str}")
    print(f"Summary:   {write_summary(args.min_confidence)}")
    _write_feed(args.min_confidence)


def _write_feed(min_confidence: float) -> None:
    """Refresh predictions/feed.json's track record (read by match-analyzer); never fails the run."""
    try:
        from scripts.build_feed import write_feed
        print(f"Feed:      {write_feed(min_confidence)}")
    except Exception as e:  # noqa: BLE001 -- the scoring itself is already written
        log.warning("feed.json not updated: %s", e)


if __name__ == "__main__":
    main()
