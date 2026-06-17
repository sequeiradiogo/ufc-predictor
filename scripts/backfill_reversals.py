"""
backfill_reversals.py -- Backfill the `reversals` column for all historical fights.

The reversals column was added to fight_stats after the initial scrape. This script
visits every fight detail page already in the DB and updates the reversals count
for both fighters.

Resumable: fights where reversals IS NOT NULL are skipped automatically.

Estimated runtime: ~2 hours for a full DB (~7k fights at 1 req/sec).

Usage:
    python scripts/backfill_reversals.py
    python scripts/backfill_reversals.py --limit 100    # test first 100 fights
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH
from scrapers.ufcstats import _browser_session, _p_int, _id_from_url
from utils.logger import get_logger

log = get_logger(__name__)

BASE = "http://www.ufcstats.com"
_DELAY = 1.0


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add reversals column to fight_stats if not already present."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(fight_stats)").fetchall()}
    if "reversals" not in existing:
        log.info("Adding reversals column to fight_stats...")
        conn.execute("ALTER TABLE fight_stats ADD COLUMN reversals INTEGER")
        conn.commit()
        log.info("Column added.")
    else:
        log.info("reversals column already exists.")


def _get_pending_fights(conn: sqlite3.Connection, limit: int | None) -> list[dict]:
    """Return fights where at least one fighter has reversals IS NULL."""
    q = """
        SELECT DISTINCT f.fight_id, f.r_fighter_id, f.b_fighter_id
        FROM fights f
        JOIN fight_stats fs ON f.fight_id = fs.fight_id
        WHERE fs.reversals IS NULL
        ORDER BY f.date ASC
    """
    if limit:
        q += f" LIMIT {limit}"
    rows = conn.execute(q).fetchall()
    return [{"fight_id": r[0], "r_fid": r[1], "b_fid": r[2]} for r in rows]


def _parse_reversals(soup: BeautifulSoup, r_fid: str, b_fid: str) -> dict[str, int]:
    """Extract reversals from the overall totals section of a fight detail page."""
    sections = soup.select("section.b-fight-details__section.js-fight-section")
    result = {r_fid: 0, b_fid: 0}

    if len(sections) <= 1:
        return result

    tbody = sections[1].find("tbody", class_="b-fight-details__table-body")
    if not tbody:
        return result
    row = tbody.find("tr")
    if not row:
        return result
    tds = row.find_all("td")
    if not tds:
        return result

    # Determine which fighter is listed first in the table (p[0] vs p[1])
    td0_ps = tds[0].find_all("p") if tds else []
    td0_fid0 = ""
    if td0_ps:
        link0 = td0_ps[0].find("a")
        if link0 and link0.get("href"):
            td0_fid0 = _id_from_url(link0["href"])

    r_idx = 0 if (td0_fid0 == r_fid or td0_fid0 == "") else 1
    b_idx = 1 - r_idx

    result[r_fid] = _p_int(tds, 8, r_idx)
    result[b_fid] = _p_int(tds, 8, b_idx)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill reversals for historical fights in the UFCStats DB."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of fights to process (default: all pending).",
    )
    args = parser.parse_args()

    if not DB_UFCSTATS_PATH.exists():
        log.error("UFCStats DB not found at %s -- run scrape_history.py first.", DB_UFCSTATS_PATH)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_UFCSTATS_PATH))
    _migrate_schema(conn)

    pending = _get_pending_fights(conn, args.limit)
    log.info("%d fights need reversals backfill.", len(pending))

    if not pending:
        log.info("Nothing to do.")
        conn.close()
        return

    updated = 0
    errors  = 0

    with _browser_session() as get:
        for i, fight in enumerate(pending, 1):
            fight_id = fight["fight_id"]
            url = f"{BASE}/fight-details/{fight_id}"
            try:
                soup = get(url)
                rev = _parse_reversals(soup, fight["r_fid"], fight["b_fid"])
                for fid, val in rev.items():
                    conn.execute(
                        "UPDATE fight_stats SET reversals = ? WHERE fight_id = ? AND fighter_id = ?",
                        (val, fight_id, fid),
                    )
                conn.commit()
                updated += 1
                if i % 100 == 0:
                    log.info("Progress: %d/%d fights updated (%d errors)", i, len(pending), errors)
            except Exception as exc:
                log.warning("Failed fight %s: %s", fight_id, exc)
                errors += 1

    log.info("Backfill complete: %d updated, %d errors.", updated, errors)
    conn.close()


if __name__ == "__main__":
    main()
