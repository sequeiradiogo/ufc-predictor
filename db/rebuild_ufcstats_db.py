"""
rebuild_ufcstats_db.py -- Rebuild ufc_ufcstats.db from the seed CSVs in raw_data/.

Use this after a fresh clone instead of running the 8-10 hour scrape:

    python db/rebuild_ufcstats_db.py

The CSVs include rolling-computed columns so rolling.py does not need to be re-run.
If you want to recompute rolling stats from scratch (e.g. after modifying rolling.py):

    python db/rebuild_ufcstats_db.py --no-rolling-cols
    python -c "from db.rolling import main; from config import DB_UFCSTATS_PATH; main(db_path=DB_UFCSTATS_PATH)"
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH, RAW_DIR
from utils.logger import get_logger

log = get_logger(__name__)

# Raw scraped columns only (subset of fight_stats used when --no-rolling-cols)
_RAW_FIGHT_STATS_COLS = [
    "fight_id", "fighter_id", "corner",
    "kd",
    "sig_str_landed", "sig_str_atmpted",
    "total_str_landed", "total_str_atmpted",
    "td_landed", "td_atmpted",
    "sub_att", "ctrl",
    "head_landed", "head_atmpted",
    "body_landed", "body_atmpted",
    "leg_landed", "leg_atmpted",
    "dist_landed", "dist_atmpted",
    "clinch_landed", "clinch_atmpted",
    "ground_landed", "ground_atmpted",
    "total_fight_time",
    "height", "reach", "stance", "dob", "weight",
]


def main(no_rolling_cols: bool = False) -> None:
    fighters_csv   = RAW_DIR / "ufcstats_fighters.csv"
    fights_csv     = RAW_DIR / "ufcstats_fights.csv"
    fight_stats_csv = RAW_DIR / "ufcstats_fight_stats.csv"

    for path in (fighters_csv, fights_csv, fight_stats_csv):
        if not path.exists():
            log.error("Missing seed CSV: %s", path)
            log.error("Run  python scripts/export_db_to_csv.py  to regenerate it.")
            sys.exit(1)

    log.info("Reading seed CSVs from %s ...", RAW_DIR)
    fighters   = pd.read_csv(fighters_csv)
    fights     = pd.read_csv(fights_csv)
    fight_stats = pd.read_csv(fight_stats_csv)

    if no_rolling_cols:
        keep = [c for c in _RAW_FIGHT_STATS_COLS if c in fight_stats.columns]
        fight_stats = fight_stats[keep]
        log.info("--no-rolling-cols: keeping %d raw columns only.", len(keep))

    log.info("Writing to %s ...", DB_UFCSTATS_PATH)
    conn = sqlite3.connect(str(DB_UFCSTATS_PATH))

    fighters.to_sql("fighters",    conn, if_exists="replace", index=False)
    fights.to_sql("fights",        conn, if_exists="replace", index=False)
    fight_stats.to_sql("fight_stats", conn, if_exists="replace", index=False)

    conn.close()

    log.info("Done.")
    log.info("  fighters:    %d rows", len(fighters))
    log.info("  fights:      %d rows", len(fights))
    log.info("  fight_stats: %d rows", len(fight_stats))
    if no_rolling_cols:
        log.info("Rolling stats were NOT included. Run rolling.py next:")
        log.info('  python -c "from db.rolling import main; from config import DB_UFCSTATS_PATH; main(db_path=DB_UFCSTATS_PATH)"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild ufc_ufcstats.db from seed CSVs.")
    parser.add_argument(
        "--no-rolling-cols",
        action="store_true",
        help="Import raw scraped columns only; rolling.py must be run afterwards.",
    )
    args = parser.parse_args()
    main(no_rolling_cols=args.no_rolling_cols)
