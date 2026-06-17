"""
backfill_rounds.py -- Backfill fight_stats_rounds for all historical fights.

The fight_stats_rounds table was added after the initial scrape. This script
visits every fight detail page already in the DB and populates per-round stats
for all rounds.

Resumable: fights already present in fight_stats_rounds are skipped.

Estimated runtime: ~3 hours for a full DB (~7k fights at 1 req/sec).

Usage:
    python scripts/backfill_rounds.py
    python scripts/backfill_rounds.py --limit 50    # test first 50 fights
"""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH
from db.ingest_ufcstats import _create_schema
from scrapers.ufcstats import _browser_session, _parse_round_stats, _id_from_url
from utils.logger import get_logger

log = get_logger(__name__)

BASE = "http://www.ufcstats.com"


def _get_pending_fights(conn: sqlite3.Connection, limit: int | None) -> list[dict]:
    """Return fights not yet present in fight_stats_rounds."""
    q = """
        SELECT f.fight_id, f.r_fighter_id, f.b_fighter_id
        FROM fights f
        WHERE NOT EXISTS (
            SELECT 1 FROM fight_stats_rounds r WHERE r.fight_id = f.fight_id
        )
        ORDER BY f.date ASC
    """
    if limit:
        q += f" LIMIT {limit}"
    rows = conn.execute(q).fetchall()
    return [{"fight_id": r[0], "r_fid": r[1], "b_fid": r[2]} for r in rows]


def _determine_r_idx(soup, r_fid: str) -> int:
    """Determine which table column index (0 or 1) corresponds to the red corner."""
    from bs4 import BeautifulSoup
    sections = soup.select("section.b-fight-details__section.js-fight-section")
    if len(sections) <= 1:
        return 0
    tbody = sections[1].find("tbody", class_="b-fight-details__table-body")
    if not tbody:
        return 0
    row = tbody.find("tr")
    if not row:
        return 0
    tds = row.find_all("td")
    if not tds:
        return 0
    td0_ps = tds[0].find_all("p")
    if not td0_ps:
        return 0
    link0 = td0_ps[0].find("a")
    if not link0 or not link0.get("href"):
        return 0
    td0_fid0 = _id_from_url(link0["href"])
    return 0 if (td0_fid0 == r_fid or td0_fid0 == "") else 1


def _upsert_rounds(conn: sqlite3.Connection, round_stats: list[dict]) -> None:
    if not round_stats:
        return
    conn.executemany(
        """
        INSERT INTO fight_stats_rounds (
            fight_id, fighter_id, round,
            kd, sig_str_landed, sig_str_atmpted,
            td_landed, td_atmpted, sub_att, reversals, ctrl,
            head_landed, head_atmpted, body_landed, body_atmpted,
            leg_landed, leg_atmpted, dist_landed, dist_atmpted,
            clinch_landed, clinch_atmpted, ground_landed, ground_atmpted
        ) VALUES (
            :fight_id, :fighter_id, :round,
            :kd, :sig_str_landed, :sig_str_atmpted,
            :td_landed, :td_atmpted, :sub_att, :reversals, :ctrl,
            :head_landed, :head_atmpted, :body_landed, :body_atmpted,
            :leg_landed, :leg_atmpted, :dist_landed, :dist_atmpted,
            :clinch_landed, :clinch_atmpted, :ground_landed, :ground_atmpted
        )
        ON CONFLICT(fight_id, fighter_id, round) DO UPDATE SET
            kd                = excluded.kd,
            sig_str_landed    = excluded.sig_str_landed,
            sig_str_atmpted   = excluded.sig_str_atmpted,
            td_landed         = excluded.td_landed,
            td_atmpted        = excluded.td_atmpted,
            sub_att           = excluded.sub_att,
            reversals         = excluded.reversals,
            ctrl              = excluded.ctrl,
            head_landed       = excluded.head_landed,
            head_atmpted      = excluded.head_atmpted,
            body_landed       = excluded.body_landed,
            body_atmpted      = excluded.body_atmpted,
            leg_landed        = excluded.leg_landed,
            leg_atmpted       = excluded.leg_atmpted,
            dist_landed       = excluded.dist_landed,
            dist_atmpted      = excluded.dist_atmpted,
            clinch_landed     = excluded.clinch_landed,
            clinch_atmpted    = excluded.clinch_atmpted,
            ground_landed     = excluded.ground_landed,
            ground_atmpted    = excluded.ground_atmpted
        """,
        round_stats,
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill per-round stats for historical fights in the UFCStats DB."
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
    _create_schema(conn)  # ensures fight_stats_rounds table exists

    pending = _get_pending_fights(conn, args.limit)
    log.info("%d fights need round-stats backfill.", len(pending))

    if not pending:
        log.info("Nothing to do.")
        conn.close()
        return

    updated = 0
    no_rounds = 0
    errors  = 0

    with _browser_session() as get:
        for i, fight in enumerate(pending, 1):
            fight_id = fight["fight_id"]
            r_fid    = fight["r_fid"]
            b_fid    = fight["b_fid"]
            url = f"{BASE}/fight-details/{fight_id}"
            try:
                soup   = get(url)
                r_idx  = _determine_r_idx(soup, r_fid)
                rounds = _parse_round_stats(soup, fight_id, r_fid, b_fid, r_idx)
                if rounds:
                    _upsert_rounds(conn, rounds)
                    updated += 1
                else:
                    # Insert a sentinel row so this fight is not retried
                    conn.execute(
                        "INSERT OR IGNORE INTO fight_stats_rounds "
                        "(fight_id, fighter_id, round) VALUES (?, ?, 0)",
                        (fight_id, r_fid),
                    )
                    conn.commit()
                    no_rounds += 1
                if i % 100 == 0:
                    log.info(
                        "Progress: %d/%d processed (%d with rounds, %d no rounds, %d errors)",
                        i, len(pending), updated, no_rounds, errors,
                    )
            except Exception as exc:
                log.warning("Failed fight %s: %s", fight_id, exc)
                errors += 1

    log.info(
        "Backfill complete: %d with rounds, %d no-round pages, %d errors.",
        updated, no_rounds, errors,
    )
    conn.close()


if __name__ == "__main__":
    main()
