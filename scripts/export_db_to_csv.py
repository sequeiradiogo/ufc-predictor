"""
export_db_to_csv.py -- Export ufc_ufcstats.db tables to seed CSVs in raw_data/.

Run this after scraping or any time the DB changes significantly:

    python scripts/export_db_to_csv.py

Outputs:
    raw_data/ufcstats_fighters.csv
    raw_data/ufcstats_fights.csv
    raw_data/ufcstats_fight_stats.csv   (includes rolling-computed columns)

Commit the resulting CSVs so the DB can be rebuilt without re-scraping.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH, RAW_DIR
from utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    if not DB_UFCSTATS_PATH.exists():
        log.error("DB not found: %s", DB_UFCSTATS_PATH)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_UFCSTATS_PATH))

    for table in ("fighters", "fights", "fight_stats"):
        df  = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        out = RAW_DIR / f"ufcstats_{table}.csv"
        df.to_csv(out, index=False)
        log.info("%s: %d rows -> %s  (%d KB)", table, len(df), out, out.stat().st_size // 1024)

    conn.close()
    log.info("Export complete. Commit raw_data/ufcstats_*.csv to preserve the DB.")


if __name__ == "__main__":
    main()
