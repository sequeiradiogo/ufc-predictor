"""
refresh_data.py -- Refresh the UFC database with new fight data and retrain models.
====================================================================================

Two refresh modes
-----------------
  --csv   : You have a new/updated UFC.csv (e.g. downloaded from Kaggle).
            Rebuilds the full DB and retrains.

  --auto  : Scrapes new fights from three sources since the last recorded event:
              - ufcstats.com    (fight results and per-fight stats)
              - bestfightodds.com  (American moneyline odds)
              - kaggle martj42/ufc-rankings  (UFC rankings snapshots)
            Converts scraped data into ufc-master.csv format, appends to the CSV,
            rebuilds the DB from scratch via ingest_mdabbert, then reruns the full
            feature-engineering and training pipeline so ELO / diff features are
            recomputed on up-to-date data.

Usage
-----
  python refresh_data.py --csv path/to/updated_UFC.csv
  python refresh_data.py --auto
  python refresh_data.py --auto --dry-run   # preview without changes
"""

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from config import DB_PATH, RAW_DIR
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


def _existing_fighter_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT fighter_id FROM fighters").fetchall()
    return {r[0] for r in rows}


# ── Scraping ──────────────────────────────────────────────────────────────────

def scrape_new_fights(since: date) -> dict:
    """
    Scrape new fights since *since* from ufcstats.com and enrich with odds
    from bestfightodds.com.

    Returns {"fighters": [...], "fights": [...], "fight_stats": [...]}.
    """
    from scrapers.ufcstats import scrape_new_data
    from scrapers.bestfightodds import scrape_odds

    # Pass existing fighter IDs so we skip bio scraping for known fighters
    conn = sqlite3.connect(str(DB_PATH))
    known_fids = _existing_fighter_ids(conn)
    conn.close()

    data = scrape_new_data(since, existing_fighter_ids=known_fids)

    if not data["fights"]:
        return data

    # Enrich fight records with names for BFO matching
    fid_to_name: dict[str, str] = {f["fighter_id"]: f["name"] for f in data["fighters"]}
    # Also pull names for fighters already in DB
    conn = sqlite3.connect(str(DB_PATH))
    for fid, name in conn.execute("SELECT fighter_id, name FROM fighters").fetchall():
        fid_to_name.setdefault(fid, name)
    conn.close()

    for fight in data["fights"]:
        fight["r_name"] = fid_to_name.get(fight["r_fighter_id"], "")
        fight["b_name"] = fid_to_name.get(fight["b_fighter_id"], "")

    log.info("Fetching odds from BestFightOdds...")
    try:
        odds_map = scrape_odds(data["fights"])
        for fight in data["fights"]:
            odds_r, odds_b = odds_map.get(fight["fight_id"], (None, None))
            fight["odds_red"]  = odds_r
            fight["odds_blue"] = odds_b
    except Exception as exc:
        log.warning("BFO odds scrape failed (%s) -- continuing without odds", exc)

    # Strip the helper keys before returning
    for fight in data["fights"]:
        fight.pop("r_name", None)
        fight.pop("b_name", None)

    return data


# ── Incremental DB insert ─────────────────────────────────────────────────────

_FIGHTER_COLS   = ("fighter_id", "name", "height", "reach", "stance", "dob")
_FIGHT_COLS     = (
    "fight_id", "event_id", "date", "division",
    "r_fighter_id", "b_fighter_id", "winner_id",
    "method", "title_fight", "odds_red", "odds_blue",
)
_STAT_COLS = (
    "fight_id", "fighter_id", "corner",
    "kd",
    "sig_str_landed",  "sig_str_atmpted",
    "total_str_landed","total_str_atmpted",
    "td_landed",       "td_atmpted",
    "sub_att", "ctrl",
    "head_landed",   "head_atmpted",
    "body_landed",   "body_atmpted",
    "leg_landed",    "leg_atmpted",
    "dist_landed",   "dist_atmpted",
    "clinch_landed", "clinch_atmpted",
    "ground_landed", "ground_atmpted",
    "total_fight_time",
)


def _row(d: dict, cols: tuple) -> tuple:
    return tuple(d.get(c) for c in cols)


def _placeholders(cols: tuple) -> str:
    return ",".join("?" * len(cols))


def _db_cols(cur, table: str) -> set[str]:
    """Return the set of column names that actually exist in *table*."""
    return {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}


def _insert_new_data(data: dict, conn: sqlite3.Connection) -> set[str]:
    """
    Insert scraped rows into all three tables using INSERT OR IGNORE semantics.
    Adapts to the actual DB schema so it works with both UFCStats-format and
    mdabbert-format databases.

    Returns the set of affected fighter_ids (for incremental rolling stats).
    """
    cur = conn.cursor()

    # Auto-add odds columns to fights if they are missing (one-time migration)
    fight_db_cols = _db_cols(cur, "fights")
    for col in ("odds_red", "odds_blue"):
        if col not in fight_db_cols:
            cur.execute(f"ALTER TABLE fights ADD COLUMN {col} REAL")
            log.info("Migrated: added column %s to fights", col)
            fight_db_cols.add(col)

    # Restrict each insert to columns that actually exist in the DB
    fighter_cols = tuple(c for c in _FIGHTER_COLS if c in _db_cols(cur, "fighters"))
    fight_cols   = tuple(c for c in _FIGHT_COLS   if c in fight_db_cols)
    stat_cols    = tuple(c for c in _STAT_COLS     if c in _db_cols(cur, "fight_stats"))

    if len(stat_cols) < len(_STAT_COLS):
        missing = set(_STAT_COLS) - set(stat_cols)
        log.warning(
            "fight_stats is missing %d UFCStats columns (%s...). "
            "This DB was built from mdabbert format -- rolling stats will not be "
            "recomputed for new fights. Rebuild the DB from UFCStats CSV for full functionality.",
            len(missing), ", ".join(sorted(missing)[:3]),
        )

    # fighters
    if data["fighters"] and fighter_cols:
        cur.executemany(
            f"INSERT OR IGNORE INTO fighters ({','.join(fighter_cols)}) "
            f"VALUES ({_placeholders(fighter_cols)})",
            [_row(f, fighter_cols) for f in data["fighters"]],
        )
        log.info("Inserted/skipped %d fighters", len(data["fighters"]))

    # fights
    if data["fights"] and fight_cols:
        cur.executemany(
            f"INSERT OR IGNORE INTO fights ({','.join(fight_cols)}) "
            f"VALUES ({_placeholders(fight_cols)})",
            [_row(f, fight_cols) for f in data["fights"]],
        )
        log.info("Inserted/skipped %d fights", len(data["fights"]))

    # fight_stats
    if data["fight_stats"] and stat_cols:
        cur.executemany(
            f"INSERT OR IGNORE INTO fight_stats ({','.join(stat_cols)}) "
            f"VALUES ({_placeholders(stat_cols)})",
            [_row(s, stat_cols) for s in data["fight_stats"]],
        )
        log.info("Inserted/skipped %d fight_stats rows", len(data["fight_stats"]))

    affected_fids = {s["fighter_id"] for s in data["fight_stats"]}
    return affected_fids


# ── Refresh modes ─────────────────────────────────────────────────────────────

def refresh_from_csv(csv_path: Path, dry_run: bool = False) -> None:
    """Rebuild DB from an updated CSV and retrain all models."""
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    last_date = get_last_event_date()
    if last_date:
        log.info("Current DB last event: %s", last_date)
    else:
        log.info("No existing DB found -- will build from scratch.")

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


_MASTER_CSV = RAW_DIR / "ufc-master.csv"


def refresh_auto(dry_run: bool = False) -> None:
    """
    Scrape new fights, append to ufc-master.csv, rebuild DB, and retrain.

    Flow:
      1. Scrape ufcstats.com + BestFightOdds for new events
      2. Convert per-fight stats to pre-fight career snapshots (csv_builder)
      3. Append new rows to raw_data/ufc-master.csv
      4. Rebuild DB from updated CSV via ingest_mdabbert
      5. Refresh rankings from Kaggle (non-blocking)
      6. Rerun feature engineering + model training (steps 4-7)
         -- ELO and all diff features are recomputed from scratch on the full updated DB
    """
    last_date = get_last_event_date()
    if last_date is None:
        log.error("No existing database found. Run the full pipeline first.")
        sys.exit(1)

    if not _MASTER_CSV.exists():
        log.error("Source CSV not found: %s", _MASTER_CSV)
        log.error("Expected raw_data/ufc-master.csv to be the source-of-truth CSV.")
        sys.exit(1)

    log.info("Last recorded event: %s. Checking for new fights...", last_date)

    data = scrape_new_fights(since=last_date)

    if not data["fights"]:
        log.info("No new fights found since %s. Database is up to date.", last_date)
        return

    log.info(
        "Found %d new fights (%d fighters, %d stat rows).",
        len(data["fights"]),
        len(data["fighters"]),
        len(data["fight_stats"]),
    )

    # ---- Build career-snapshot CSV rows ----
    log.info("Converting scraped stats to career-snapshot format for CSV...")
    from scrapers.csv_builder import build_csv_rows
    new_rows_df = build_csv_rows(data, DB_PATH)

    if new_rows_df.empty:
        log.warning("No CSV rows could be built from scraped data. Aborting.")
        return

    if dry_run:
        log.info(
            "[DRY RUN] Would append %d rows to %s and retrain.",
            len(new_rows_df), _MASTER_CSV,
        )
        return

    # ---- Append to ufc-master.csv ----
    existing_df = pd.read_csv(_MASTER_CSV, low_memory=False)
    # Align columns: new_rows_df may be missing some CSV columns -- fill with NaN
    updated_df = pd.concat([existing_df, new_rows_df], ignore_index=True, sort=False)
    updated_df.to_csv(_MASTER_CSV, index=False)
    log.info(
        "Updated %s: %d total rows (+%d new).",
        _MASTER_CSV, len(updated_df), len(new_rows_df),
    )

    # ---- Rebuild DB from updated CSV ----
    log.info("Rebuilding DB from updated CSV via ingest_mdabbert...")
    from db.ingest_mdabbert import ingest
    ingest(_MASTER_CSV, DB_PATH)

    # ---- Rankings snapshot (optional -- non-blocking) ----
    log.info("Refreshing UFC rankings from Kaggle...")
    try:
        from scrapers.ufc_rankings import refresh_rankings
        refresh_rankings()
    except Exception as exc:
        log.warning("Rankings refresh skipped: %s", exc)

    # ---- Feature engineering + training (ELO recomputed from scratch) ----
    log.info("Running feature engineering and retraining models (steps 4-7)...")
    run_pipeline.run_pipeline(steps=[4, 5, 6, 7])

    log.info("Refresh complete.")


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
  python refresh_data.py --auto
  python refresh_data.py --auto --dry-run
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",  type=Path, help="Path to updated UFC.csv")
    group.add_argument("--auto", action="store_true", help="Auto-scrape new events")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")

    args = parser.parse_args()

    last = get_last_event_date()
    log.info("UFC Predictor -- Data Refresh")
    log.info("Last DB event : %s", last or "N/A (no DB)")

    if args.csv:
        refresh_from_csv(args.csv, dry_run=args.dry_run)
    else:
        refresh_auto(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
