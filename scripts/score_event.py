"""
score_event.py -- Score the most recent prediction against actual UFC results.

Scrapes fight results from UFCStats, optionally odds from BFO, then updates
the prediction markdown with actual results and P/L analysis.

Usage:
  python scripts/score_event.py                       # auto-detect most recent unscored prediction
  python scripts/score_event.py --json path/to/f.json
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
from scrapers.bestfightodds import _fetch_bfo_events, _scrape_event_odds, _name_key, _best_match
from utils.logger import get_logger

log = get_logger("score_event")

PREDICTIONS_DIR = ROOT / "predictions"
SCORED_MARKER   = "Actual Result"
LOOKBACK_DAYS   = 10


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
                tds    = row.find_all("td", class_="b-fight-details__table-col")
                method = _normalize_method(_col_ps_text(tds, 7, 0)) if len(tds) > 7 else "Decision"
                try:
                    round_num = int(_col_ps_text(tds, 8, 0) or "0")
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
    fighter_pairs: list[tuple[str, str]],
) -> dict[str, tuple[int | None, int | None]]:
    """
    Look up closing odds on BFO for each (red, blue) fighter pair.
    Returns {fight_key: (odds_red, odds_blue)}.  fight_key = 'Red vs Blue'.
    """
    try:
        bfo_events = _fetch_bfo_events()
    except Exception as exc:
        log.warning("BFO event list fetch failed: %s", exc)
        return {}

    bfo_names = [e["name"] for e in bfo_events]
    matched   = _best_match(event_name, bfo_names, cutoff=0.60)
    if not matched:
        log.warning("No BFO event matched '%s'", event_name)
        return {}

    bfo_event = bfo_events[bfo_names.index(matched)]
    log.info("BFO event matched: '%s'", bfo_event["name"])

    try:
        matchups = _scrape_event_odds(bfo_event["url"])
    except Exception as exc:
        log.warning("BFO odds scrape failed: %s", exc)
        return {}

    # Build lookup keyed by (red_key, blue_key)
    bfo_lookup: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    for m in matchups:
        rk = _name_key(m["r_name"])
        bk = _name_key(m["b_name"])
        bfo_lookup[(rk, bk)] = (m["odds_red"], m["odds_blue"])
        bfo_lookup[(bk, rk)] = (m["odds_blue"], m["odds_red"])

    out: dict[str, tuple[int | None, int | None]] = {}
    for red, blue in fighter_pairs:
        rk  = _name_key(red)
        bk  = _name_key(blue)
        key = f"{red} vs {blue}"
        out[key] = bfo_lookup.get((rk, bk), (None, None))
    return out


# -- Name matching ------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _name_key(a), _name_key(b)).ratio()


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


def _result_str(r: dict) -> str:
    method = _METHOD_SHORT.get(r["method"], r["method"])
    if "Dec" in method or "NC" in method or "Draw" in method:
        return f"{r['winner']} ({method})"
    return f"{r['winner']} ({method} R{r['round']})"


# -- Markdown update ----------------------------------------------------------

def score_markdown(
    md_path: Path,
    predictions: list[dict],
    results: list[dict],
    odds_map: dict[str, tuple[int | None, int | None]],
) -> str:
    """
    Update *md_path* with actual results and P/L.
    Returns a short summary string for logging.
    """
    md = md_path.read_text(encoding="utf-8")

    if SCORED_MARKER in md:
        log.info("Already scored: %s", md_path)
        return "already scored"

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
            "is_fifty":    is_fifty,
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
    acc_str    = f"{n_correct}/{n_scored} ({n_correct/n_scored:.1%})" if n_scored else "N/A"

    result_line = f"**Result: {acc_str}**"
    if n_excluded:
        s = "s" if n_excluded > 1 else ""
        result_line += f" *({n_excluded} fight{s} excluded: 50/50 pick{s})*"

    md = re.sub(
        r"(Fighters making their UFC debut were excluded.*?\n)",
        rf"\1\n{result_line}\n",
        md,
        count=1,
    )

    # -- Post-Event Summary + P/L ---------------------------------------------
    pl_rows: list[str] = []
    net_pl = 0.0
    staked = 0.0

    for s in fight_rows:
        if s["is_fifty"] or s["actual"] is None or s["correct"] == "?":
            continue
        staked += 1.0

        # Odds for the model's pick
        is_red_pick = _sim(s["pred_winner"], s["red_name"]) > 0.6
        raw_odds    = s["odds_red"] if is_red_pick else s["odds_blue"]
        dec_odds    = _american_to_decimal(raw_odds)
        odds_str    = f"{dec_odds:.2f}" if dec_odds else "--"

        if s["correct"] == "YES":
            pl     = round((dec_odds - 1) if dec_odds else 0.0, 2)
            pl_str = f"+EUR {pl:.2f}"
        else:
            pl     = -1.0
            pl_str = "-EUR 1.00"
        net_pl += pl

        result_label = "Win" if s["correct"] == "YES" else "Loss"
        pl_rows.append(
            f"| {s['fight_key']} | {s['pred_winner']} | {odds_str} | {result_label} | {pl_str} |"
        )

    summary: list[str] = [
        "---",
        "",
        "## Post-Event Summary",
        "",
        f"- Fights predicted: {len(fight_rows)}"
        + (f" ({n_excluded} excluded: 50/50 pick{'s' if n_excluded > 1 else ''})" if n_excluded else ""),
        f"- Correct: {acc_str}",
    ]

    if pl_rows:
        roi     = net_pl / staked * 100 if staked else 0.0
        net_str = f"+EUR {net_pl:.2f}" if net_pl >= 0 else f"-EUR {abs(net_pl):.2f}"
        summary += [
            "",
            "### P/L (EUR 1 flat on each predicted winner)",
            "",
            "| Fight | Model Pick | Odds (dec) | Result | P/L |",
            "|---|---|---|---|---|",
        ] + pl_rows + [
            "",
            f"**Net P/L: {net_str} on EUR {staked:.0f} staked ({roi:+.1f}% ROI)**",
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
        if event_date < cutoff:
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Score the most recent event prediction.")
    parser.add_argument("--json", default=None, help="Prediction JSON path (default: auto-detect)")
    args = parser.parse_args()

    json_path = Path(args.json) if args.json else find_unscored_prediction()
    if json_path is None:
        print("No recent unscored predictions found. Nothing to do.")
        sys.exit(0)

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    log.info("Scoring: %s  (%s)", meta["event"], meta["date"])
    print(f"\nEvent:     {meta['event']}  ({meta['date']})")

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
    odds_map = scrape_bfo_odds(meta["event"], fighter_pairs)

    md_path = json_path.with_suffix(".md")
    acc_str = score_markdown(md_path, predictions, results, odds_map)

    print(f"\nScored:    {md_path}")
    print(f"Accuracy:  {acc_str}")


if __name__ == "__main__":
    main()
