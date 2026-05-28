"""
refresh_data.py — Refresh the UFC database with new fight data and retrain models.
===================================================================================

This script handles the case where new UFC events have happened and you want to
update the database and models without rebuilding everything from scratch.

Two refresh modes
-----------------
  --csv   : You have a new/updated UFC.csv (e.g. downloaded from Kaggle).
            Rebuilds the full DB and retrains.

  --auto  : Attempts to scrape new fights from ufcstats.com since the last
            recorded event date.  Requires a scraper implementation — see
            the SCRAPER INTEGRATION section below.

Usage
-----
  # Refresh from an updated CSV file
  python refresh_data.py --csv path/to/updated_UFC.csv

  # Auto-scrape new events (requires scraper setup)
  python refresh_data.py --auto

  # Preview without making changes
  python refresh_data.py --csv path/to/UFC.csv --dry-run
"""

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH
from logger import get_logger
import run_pipeline

log = get_logger("refresh")


# ── Last event date ───────────────────────────────────────────────────────────

def get_last_event_date() -> date | None:
    """Return the date of the most recent fight in the database, or None."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    row  = cur.execute("SELECT MAX(date) FROM fights").fetchone()
    conn.close()
    if row and row[0]:
        return date.fromisoformat(row[0][:10])
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# To enable --auto mode, implement the function below.
# It should return a list of new fight dicts in the same format as UFC.csv.
#
# Recommended packages:
#   pip install requests beautifulsoup4 lxml
#
# Community scrapers to adapt:
#   https://github.com/WarriorMachines/ufcscraper
#   https://github.com/jasonrosas1/ufc-web-scraper
#
# The function receives the last known event date so it only fetches new data.
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_new_fights(since: date) -> list[dict]:
    """
    Scrape UFC events that occurred after *since* from ufcstats.com.

    ⚠️  NOT YET IMPLEMENTED — this is the integration point.

    Returns a list of fight dicts matching the UFC.csv column schema.
    Raise NotImplementedError until a scraper is wired in.
    """
    raise NotImplementedError(
        "Auto-scraping is not yet implemented.\n"
        "See the SCRAPER INTEGRATION section in refresh_data.py for guidance.\n"
        "Use  --csv path/to/updated_UFC.csv  for now."
    )


# ── Refresh logic ─────────────────────────────────────────────────────────────

def refresh_from_csv(csv_path: Path, dry_run: bool = False) -> None:
    """Rebuild DB from an updated CSV and retrain all models."""
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    last_date = get_last_event_date()
    if last_date:
        log.info("Current DB last event: %s", last_date)
    else:
        log.info("No existing DB found — will build from scratch.")

    log.info("Refreshing from CSV: %s", csv_path)

    if dry_run:
        log.info("[DRY RUN] Would run full pipeline with --csv %s", csv_path)
        run_pipeline.run_pipeline(list(run_pipeline.STEPS), dry_run=True, csv_path=csv_path)
        return

    run_pipeline.run_pipeline(
        steps=list(run_pipeline.STEPS),
        dry_run=False,
        csv_path=csv_path,
    )


def refresh_auto(dry_run: bool = False) -> None:
    """Scrape new fights and update the DB incrementally."""
    last_date = get_last_event_date()
    if last_date is None:
        log.error("No existing database found. Run the full pipeline first.")
        sys.exit(1)

    log.info("Last recorded event: %s. Checking for new fights…", last_date)

    try:
        new_fights = scrape_new_fights(since=last_date)
    except NotImplementedError as e:
        log.error(str(e))
        sys.exit(1)

    if not new_fights:
        log.info("No new fights found since %s. Database is up to date.", last_date)
        return

    log.info("Found %d new fights.", len(new_fights))

    if dry_run:
        log.info("[DRY RUN] Would insert %d new fights and retrain models.", len(new_fights))
        return

    # TODO: insert new_fights into DB, rerun rolling stats, regenerate ML data, retrain
    # For now, log and exit — implement once scraper is ready
    log.warning(
        "Incremental DB update not yet implemented. "
        "Use --csv with a full updated export for now."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the UFC database and retrain models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python refresh_data.py --csv path/to/updated_UFC.csv
  python refresh_data.py --csv path/to/updated_UFC.csv --dry-run
  python refresh_data.py --auto    # (requires scraper setup)
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",  type=Path, help="Path to updated UFC.csv")
    group.add_argument("--auto", action="store_true", help="Auto-scrape new events")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")

    args = parser.parse_args()

    last = get_last_event_date()
    log.info("UFC Predictor — Data Refresh")
    log.info("Last DB event : %s", last or "N/A (no DB)")

    if args.csv:
        refresh_from_csv(args.csv, dry_run=args.dry_run)
    else:
        refresh_auto(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
