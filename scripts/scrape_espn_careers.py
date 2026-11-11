"""
scrape_espn_careers.py -- Every pro bout outside the UFC/DWCS of every mapped UFC fighter (ESPN).

Reads raw_data/espn_fighter_map.csv (scripts/scrape_espn.py) and writes
raw_data/espn_career_fights.csv: one row per (mapped fighter, non-UFC bout) --
date, league slug, event name, opponent ESPN id, result W/L/D/NC, method, round/time.
UFC and DWCS bouts (ESPN league 'ufc') are skipped: UFCStats / espn_dwcs_fights.csv
already have them. Bouts after a fighter's UFC run are kept (a fighter who comes back
needs them), and features must filter by date anyway.

Requests: one eventlog per fighter, then competition + status per bout and one per
event (event name), deduplicated. Every response is cached on disk
(raw_data/espn_cache/, gitignored), so an interrupted run resumes where it stopped.

--opponents (second hop) then fetches the bouts of those bouts' non-UFC opponents
(date + winner only, up to their last meeting with a mapped fighter) into
raw_data/espn_opponent_bouts.csv, so opponent strength can be rated as of each bout.

Incremental (the monthly workflow): fighters already in espn_career_fights.csv are not
re-fetched -- a UFC fighter's pre-UFC career doesn't change -- and only opponents whose
needed history reaches past what is stored are. --full re-fetches everyone (bouts a
fighter had outside the UFC after their first scrape, e.g. after a release, are only
picked up that way). A full run is ~1.5 h + ~5 h for --opponents at ~9 req/s.

Usage:
    python scripts/scrape_espn_careers.py                     # new fighters, 4 workers
    python scripts/scrape_espn_careers.py --opponents         # after it
    python scripts/scrape_espn_careers.py --full --workers 2 --limit 50
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scrapers.espn import CachedFetcher, athlete_eventlog, bout_outcome, career_bout  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

MAP_CSV = ROOT_DIR / "raw_data" / "espn_fighter_map.csv"
OUT_CSV = ROOT_DIR / "raw_data" / "espn_career_fights.csv"
CACHE_DIR = ROOT_DIR / "raw_data" / "espn_cache"
OPP_CSV = ROOT_DIR / "raw_data" / "espn_opponent_bouts.csv"
OPP_COLS = ["competition_id", "date", "league", "a_espn_id", "b_espn_id", "winner"]

COLS = ["espn_id", "fighter_id", "date", "league", "event_id", "event_name", "competition_id",
        "opp_espn_id", "result", "method", "finish_round", "match_time_sec", "sched_rounds"]


def _parallel(fn, items: list, workers: int, what: str) -> dict:
    """{item: fn(item)} with progress logging; a failing item is logged and left out."""
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, it): it for it in items}
        for i, fut in enumerate(as_completed(futures), 1):
            it = futures[fut]
            try:
                out[it] = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one bad bout must not kill a 6-hour run
                log.warning("%s %s failed: %s", what, it, exc)
            if i % 500 == 0 or i == len(items):
                rate = i / max(time.time() - t0, 1e-9)
                log.info("%s: %d/%d (%.1f/s, ~%.0f min left)", what, i, len(items), rate,
                         (len(items) - i) / max(rate, 1e-9) / 60)
    return out


def _result(athletes: list[tuple[str, bool]], me: str, method: str) -> str:
    mine = [w for a, w in athletes if a == me]
    if mine and mine[0]:
        return "W"
    if any(w for a, w in athletes if a != me):
        return "L"
    return "NC" if "no contest" in method.lower() else "D"


def scrape_opponents(fetch: CachedFetcher, mapped: set[str], workers: int, limit: int | None,
                     full: bool = False) -> None:
    """Second hop: every bout (any league) of the non-UFC opponents in OUT_CSV -> OPP_CSV.

    Gives each opponent a dated history, so their strength *at the time* of a bout with a
    future UFC fighter can be rated -- ESPN only exposes current records, which would leak.
    """
    career = pd.read_csv(OUT_CSV, dtype=str)
    # Only bouts up to the last time an opponent met one of our fighters matter; the eventlog
    # is newest-first (checked on 30/30), so walk it oldest-first and stop past that date.
    need = career.groupby("opp_espn_id")["date"].max().to_dict()
    old = (pd.read_csv(OPP_CSV, dtype=str) if OPP_CSV.exists() and not full
           else pd.DataFrame(columns=OPP_COLS))
    # an opponent is covered up to their latest stored bout (the walk stores every bout up
    # to the needed date, the meeting itself included)
    covered = pd.concat([old[["a_espn_id", "date"]].set_axis(["id", "date"], axis=1),
                         old[["b_espn_id", "date"]].set_axis(["id", "date"], axis=1)]
                        ).groupby("id")["date"].max().to_dict()
    opps = sorted(o for o in set(need) - mapped if need[o] > covered.get(o, ""))
    log.info("%d unmapped opponents, %d to fetch (new or needing a longer history)",
             len(set(need) - mapped), len(opps))
    if limit:
        opps = opps[:limit]

    def one(aid: str) -> list[tuple[dict, dict]]:
        out = []
        for it in reversed(athlete_eventlog(fetch, aid)):
            d = bout_outcome(fetch, it["competition_ref"])
            if d and d["date"] > need[aid]:
                break
            if d:
                out.append((it, d))
        return out

    per_opp = _parallel(one, opps, workers, "opponents")
    bouts, details = {}, {}
    for pairs in per_opp.values():
        for it, d in pairs:
            bouts.setdefault(it["competition_id"], it)
            details.setdefault(it["competition_id"], d)
    log.info("%d opponents, %d distinct bouts", len(per_opp), len(bouts))
    rows = []
    for cid, d in details.items():
        if not d:
            continue
        (a, aw), (b, bw) = d["athletes"]
        outcome = a if aw else (b if bw else ("NC" if "no contest" in d["method"].lower() else "D"))
        rows.append({"competition_id": cid, "date": d["date"], "league": bouts[cid]["league"],
                     "a_espn_id": a, "b_espn_id": b, "winner": outcome})
    df = pd.concat([old, pd.DataFrame(rows, columns=OPP_COLS)], ignore_index=True)
    df = df.drop_duplicates("competition_id", keep="last").sort_values(["date", "competition_id"])
    df.to_csv(OPP_CSV, index=False)
    log.info("Wrote %d bouts to %s", len(df), OPP_CSV.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only the first N fighters (testing)")
    parser.add_argument("--opponents", action="store_true",
                        help="second hop: the opponents' own careers -> espn_opponent_bouts.csv")
    parser.add_argument("--full", action="store_true", help="re-fetch everyone, not only what is new")
    args = parser.parse_args()

    if args.opponents:
        mapped = set(pd.read_csv(MAP_CSV, dtype={"espn_id": str})["espn_id"])
        scrape_opponents(CachedFetcher(CACHE_DIR), mapped, args.workers, args.limit, full=args.full)
        return

    fmap = pd.read_csv(MAP_CSV, dtype={"espn_id": str})
    to_ufcstats_all = dict(zip(fmap["espn_id"], fmap["fighter_id"]))
    old = (pd.read_csv(OUT_CSV, dtype=str) if OUT_CSV.exists() and not args.full
           else pd.DataFrame(columns=COLS))
    old = old[old["espn_id"].isin(to_ufcstats_all)].copy()
    old["fighter_id"] = old["espn_id"].map(to_ufcstats_all)   # follow map corrections
    todo = [a for a in to_ufcstats_all if a not in set(old["espn_id"])]
    if args.limit:
        todo = todo[:args.limit]
    log.info("%d mapped fighters, %d already stored, %d to fetch", len(to_ufcstats_all),
             old["espn_id"].nunique(), len(todo))
    to_ufcstats = {a: to_ufcstats_all[a] for a in todo}
    fetch = CachedFetcher(CACHE_DIR)

    logs = _parallel(lambda a: athlete_eventlog(fetch, a), list(to_ufcstats), args.workers, "eventlogs")
    bouts = {}     # competition_id -> eventlog item (non-UFC only)
    for items in logs.values():
        for it in items:
            if it["league"] != "ufc":
                bouts.setdefault(it["competition_id"], it)
    log.info("%d fighters, %d distinct non-UFC bouts", len(logs), len(bouts))

    details = _parallel(lambda cid: career_bout(fetch, bouts[cid]["competition_ref"]),
                        list(bouts), args.workers, "bouts")
    events = {it["event_id"]: it["event_ref"] for it in bouts.values() if it["event_ref"]}
    names = _parallel(lambda eid: fetch.get(events[eid]).get("name", ""), list(events), args.workers, "events")

    rows = []
    for aid, items in logs.items():
        for it in items:
            d = details.get(it["competition_id"]) if it["league"] != "ufc" else None
            if not d or aid not in {a for a, _ in d["athletes"]}:
                continue
            opp = [a for a, _ in d["athletes"] if a != aid]
            rows.append({
                "espn_id": aid, "fighter_id": to_ufcstats[aid], "date": d["date"], "league": it["league"],
                "event_id": it["event_id"], "event_name": names.get(it["event_id"], ""),
                "competition_id": it["competition_id"], "opp_espn_id": opp[0] if opp else "",
                "result": _result(d["athletes"], aid, d["method"]), "method": d["method"],
                "finish_round": d["finish_round"], "match_time_sec": d["match_time_sec"],
                "sched_rounds": d["sched_rounds"],
            })
    df = pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
    df = df.sort_values(["fighter_id", "date", "competition_id"])
    df.to_csv(OUT_CSV, index=False)
    log.info("Wrote %d rows (%d fighters) to %s", len(df), df["espn_id"].nunique(), OUT_CSV.name)


if __name__ == "__main__":
    main()
