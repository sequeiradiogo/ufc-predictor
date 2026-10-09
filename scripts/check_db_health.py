"""
check_db_health.py -- Fail loudly if the UFCStats DB is in a state that
silently corrupts live predictions.

Each of these went unnoticed for months (Jul-Oct 2026) because nothing
errors -- predict.py just computes splm/sapm from garbage:
  - fights without match_time_sec / finish_round (fight minutes = 0)
  - fight_stats rows without a date (rolling.py chronology scrambled)
  - NULL rolling columns (rolling.py never ran on new fights)
  - fights without per-round stats (round-1 EWMA features go stale)
  - total_fight_time never made cumulative (rolling.py stale / not run)

Usage:
    python scripts/check_db_health.py            # exits 1 if any check fails
"""

import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH

# One fight's maximum duration (5 rounds x 300s). A cumulative pre-fight
# total_fight_time must exceed this somewhere in the table.
_MAX_SINGLE_FIGHT_SECS = 1500

_CHECKS = [
    ("fights without a duration (match_time_sec / finish_round)",
     "SELECT COUNT(*) FROM fights WHERE method IS NOT NULL AND (match_time_sec IS NULL OR finish_round IS NULL)"),
    ("fight_stats rows without a date",
     "SELECT COUNT(*) FROM fight_stats WHERE date IS NULL"),
    ("fight_stats rows with NULL rolling stats (splm / sapm / head_acc)",
     "SELECT COUNT(*) FROM fight_stats WHERE splm IS NULL OR sapm IS NULL OR head_acc IS NULL"),
    # backfill_rounds.py writes a round-0 sentinel when UFCStats has no round
    # data, so every fight should have at least one row here.
    ("fights without per-round stats (run scripts/backfill_rounds.py)",
     "SELECT COUNT(*) FROM fights f WHERE NOT EXISTS "
     "(SELECT 1 FROM fight_stats_rounds r WHERE r.fight_id = f.fight_id)"),
]


def main() -> int:
    if not DB_UFCSTATS_PATH.exists():
        print(f"[FAIL] DB not found: {DB_UFCSTATS_PATH}")
        return 1
    conn = sqlite3.connect(DB_UFCSTATS_PATH)
    failed = False
    for label, sql in _CHECKS:
        n = conn.execute(sql).fetchone()[0]
        print(f"[{'FAIL' if n else ' OK '}] {label}: {n}")
        failed |= bool(n)

    max_tft = conn.execute("SELECT MAX(CAST(total_fight_time AS REAL)) FROM fight_stats").fetchone()[0] or 0
    stale = max_tft <= _MAX_SINGLE_FIGHT_SECS
    print(f"[{'FAIL' if stale else ' OK '}] total_fight_time is cumulative (MAX = {max_tft:.0f}s)")
    failed |= stale
    conn.close()

    if failed:
        print("\nDB is unhealthy. Missing durations/dates mean fights were inserted without them "
              "(re-scrape those events); NULL rolling stats or a non-cumulative total_fight_time "
              "mean db/rolling.py has to be re-run:\n"
              "  python -c \"from db.rolling import main; from config import DB_UFCSTATS_PATH; "
              "main(db_path=DB_UFCSTATS_PATH)\"")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
