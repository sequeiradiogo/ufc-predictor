"""
scrape_espn.py -- Refresh the DWCS fight table and the ESPN <-> UFCStats fighter map.

Writes (both tracked in git, small):
  raw_data/espn_dwcs_fights.csv   two rows per Contender Series bout (one per fighter):
                                  result, method, round/time + fight_stats columns
  raw_data/espn_fighter_map.csv   espn_id -> UFCStats fighter_id, from UFC fights in both sources

Incremental: DWCS bouts already in the CSV are not re-fetched. The map is rebuilt
from every UFC event (~430 requests, a few minutes) since new UFC fights keep
linking DWCS alumni to their UFCStats ids.

Usage:
    python scripts/scrape_espn.py
    python scripts/scrape_espn.py --dry-run
    python scripts/scrape_espn.py --skip-map        # DWCS only
"""

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH  # noqa: E402
from scrapers.espn import (  # noqa: E402
    STAT_COLS, build_fighter_map, calendar, event_fights, full_stats, is_dwcs, is_ufc_card, result_of,
)
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

DWCS_CSV = ROOT_DIR / "raw_data" / "espn_dwcs_fights.csv"
MAP_CSV = ROOT_DIR / "raw_data" / "espn_fighter_map.csv"
FIRST_YEAR = 2017   # DWCS season 1

DWCS_COLS = ["event_id", "event_name", "competition_id", "date", "espn_id", "name",
             "opp_espn_id", "opp_name", "result", "method", "method_detail",
             "finish_round", "match_time_sec", "sched_rounds"] + STAT_COLS


def _dwcs_rows(event: dict, done: set[str]) -> list[dict]:
    rows = []
    for f in event_fights(event["event_id"]):
        if not f["completed"] or f["competition_id"] in done:
            continue
        for side, other in (("r", "b"), ("b", "r")):
            stats = full_stats(event["event_id"], f["competition_id"], f[f"{side}_espn_id"])
            if not stats:   # no core stats: keep the summary (no position split / reversals)
                stats = f[f"{side}_stats"]
            rows.append({
                "event_id": event["event_id"], "event_name": event["name"],
                "competition_id": f["competition_id"], "date": f["date"],
                "espn_id": f[f"{side}_espn_id"], "name": f[f"{side}_name"],
                "opp_espn_id": f[f"{other}_espn_id"], "opp_name": f[f"{other}_name"],
                "result": result_of(f, side), "method": f["method"], "method_detail": f["method_detail"],
                "finish_round": f["finish_round"], "match_time_sec": f["match_time_sec"],
                "sched_rounds": f["sched_rounds"], **{c: stats.get(c) for c in STAT_COLS},
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    parser.add_argument("--skip-map", action="store_true", help="don't rebuild the fighter map")
    args = parser.parse_args()

    today = date.today().isoformat()
    events = [e for e in calendar(range(FIRST_YEAR, date.today().year + 1)) if e["start"][:10] <= today]
    dwcs = [e for e in events if is_dwcs(e)]
    log.info("ESPN calendar %d-%d: %d past events, %d DWCS", FIRST_YEAR, date.today().year, len(events), len(dwcs))

    old = pd.read_csv(DWCS_CSV, dtype={"event_id": str, "competition_id": str, "espn_id": str,
                                       "opp_espn_id": str}) if DWCS_CSV.exists() else pd.DataFrame(columns=DWCS_COLS)
    done = set(old["competition_id"])
    new_rows = []
    for ev in dwcs:
        new_rows += _dwcs_rows(ev, done)
    log.info("DWCS: %d bouts stored, %d new", len(done), len(new_rows) // 2)
    if new_rows:
        dwcs_df = pd.concat([old, pd.DataFrame(new_rows, columns=DWCS_COLS)], ignore_index=True)
        dwcs_df = dwcs_df.sort_values(["date", "competition_id", "espn_id"]).reset_index(drop=True)
        if args.dry_run:
            log.info("[dry-run] would write %d rows to %s", len(dwcs_df), DWCS_CSV.name)
        else:
            dwcs_df.to_csv(DWCS_CSV, index=False)
            log.info("Wrote %d rows to %s", len(dwcs_df), DWCS_CSV.name)

    if args.skip_map:
        return
    with sqlite3.connect(f"file:{DB_UFCSTATS_PATH}?mode=ro", uri=True) as conn:
        fmap = build_fighter_map([e for e in events if is_ufc_card(e)], conn)
        names = dict(conn.execute("SELECT fighter_id, name FROM fighters"))
    map_df = pd.DataFrame(sorted(fmap.items()), columns=["espn_id", "fighter_id"])
    map_df["name"] = map_df["fighter_id"].map(names)
    if args.dry_run:
        log.info("[dry-run] would write %d ids to %s", len(map_df), MAP_CSV.name)
    else:
        map_df.to_csv(MAP_CSV, index=False)
        log.info("Wrote %d ids to %s", len(map_df), MAP_CSV.name)


if __name__ == "__main__":
    main()
