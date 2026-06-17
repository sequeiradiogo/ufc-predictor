"""
scrape_history.py -- One-time historical scrape from UFCStats and full DB rebuild.

Scrapes all UFC events since a given date, ingests per-fight granular stats into
db/ufc_ufcstats.db, then runs rolling.py to compute pre-fight rolling windows.

Checkpoints to DB every CHECKPOINT_EVERY events so a crash loses at most a few
minutes of progress. Re-running after a crash automatically resumes from where
it left off (already-ingested events are skipped).

The original mdabbert DB (db/ufc_v2.db) is NOT touched.

Estimated runtime: 13-22 hours for a full scrape from 1993 (11,000+ pages).
Leave running overnight.

Usage:
    # Full historical rebuild (default: since UFC 1, Nov 1993)
    python scripts/scrape_history.py

    # Incremental catch-up from a specific date
    python scripts/scrape_history.py --since 2024-01-01

    # Scrape only, skip rolling stats computation
    python scripts/scrape_history.py --no-rolling

First-time setup (Playwright headless Chromium):
    python -m playwright install chromium
"""

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH
from db import ingest_ufcstats, rolling
from scrapers.ufcstats import scrape_events_iter
from utils.logger import get_logger

log = get_logger(__name__)

_UFC_BIRTH       = date(1993, 11, 11)
CHECKPOINT_EVERY = 10   # flush to DB after this many events


def _load_existing(db_path: Path) -> tuple[set[str], set[str]]:
    """Return (existing_fighter_ids, existing_event_ids) from the DB if it exists."""
    if not db_path.exists():
        return set(), set()
    conn = sqlite3.connect(str(db_path))
    fighter_ids = {row[0] for row in conn.execute("SELECT fighter_id FROM fighters").fetchall()}
    event_ids   = {row[0] for row in conn.execute(
        "SELECT DISTINCT event_id FROM fights WHERE event_id IS NOT NULL"
    ).fetchall()}
    conn.close()
    return fighter_ids, event_ids


def _flush(buffer: dict, db_path: Path) -> None:
    """Ingest the buffered events and reset the buffer in-place."""
    if not any(buffer.values()):
        return
    ingest_ufcstats.ingest(buffer, db_path=db_path)
    buffer["fighters"].clear()
    buffer["fights"].clear()
    buffer["fight_stats"].clear()
    buffer["round_stats"].clear()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape UFCStats history and build the per-fight rolling DB."
    )
    parser.add_argument(
        "--since", type=date.fromisoformat, default=_UFC_BIRTH,
        metavar="YYYY-MM-DD",
        help="Scrape events after this date (default: 1993-11-11, i.e. all history).",
    )
    parser.add_argument(
        "--no-rolling", action="store_true",
        help="Skip running rolling.py after ingest (useful for debugging ingest).",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("UFC per-fight DB builder")
    log.info("  Target DB        : %s", DB_UFCSTATS_PATH)
    log.info("  Since            : %s", args.since)
    log.info("  Checkpoint every : %d events", CHECKPOINT_EVERY)
    log.info("=" * 60)

    existing_fighter_ids, existing_event_ids = _load_existing(DB_UFCSTATS_PATH)
    if existing_event_ids:
        log.info(
            "Resuming -- %d events already ingested, %d fighters in DB.",
            len(existing_event_ids), len(existing_fighter_ids),
        )

    buffer: dict = {"fighters": [], "fights": [], "fight_stats": [], "round_stats": []}
    events_scraped = 0
    fights_total   = 0

    for _event, data in scrape_events_iter(
        since=args.since,
        existing_fighter_ids=existing_fighter_ids,
        skip_event_ids=existing_event_ids,
    ):
        buffer["fighters"].extend(data["fighters"])
        buffer["fights"].extend(data["fights"])
        buffer["fight_stats"].extend(data["fight_stats"])
        buffer["round_stats"].extend(data.get("round_stats", []))
        events_scraped += 1
        fights_total   += len(data["fights"])

        if events_scraped % CHECKPOINT_EVERY == 0:
            log.info(
                "Checkpoint: flushing %d events (%d fights so far)…",
                CHECKPOINT_EVERY, fights_total,
            )
            _flush(buffer, DB_UFCSTATS_PATH)
            # Update known fighter IDs so subsequent bio pages are skipped
            existing_fighter_ids, _ = _load_existing(DB_UFCSTATS_PATH)

    # Final flush for the last partial batch
    if any(buffer.values()):
        log.info("Final flush (%d remaining events)…", events_scraped % CHECKPOINT_EVERY or CHECKPOINT_EVERY)
        _flush(buffer, DB_UFCSTATS_PATH)

    if events_scraped == 0:
        log.info("No new events found. DB is already up to date.")
        return

    log.info("Scrape complete: %d events, %d fights.", events_scraped, fights_total)

    if not args.no_rolling:
        log.info("Computing rolling stats via rolling.py…")
        rolling.main(db_path=DB_UFCSTATS_PATH)
        log.info("Rolling stats complete.")

    log.info("Done. Run the ML pipeline next:")
    log.info("  python run_pipeline.py  (steps 4-9)")


if __name__ == "__main__":
    main()
