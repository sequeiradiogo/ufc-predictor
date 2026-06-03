"""
scripts/populate_odds.py -- Populate odds_red / odds_blue in the UFCStats DB.

Two modes:

  --csv       Bulk backfill from raw_data/ufc-master.csv (dry-run by default).
              The CSV has American moneyline odds (R_odds / B_odds) for ~95%
              of fights from 2010 onwards. Run with --apply to write to DB.

  --fight     Add odds for a single fight by fighter names and date.
              Always writes immediately (no dry-run needed).

Usage
-----
    # Dry-run: see how many fights would be matched
    python scripts/populate_odds.py --csv

    # Write matched odds to DB
    python scripts/populate_odds.py --csv --apply

    # Overwrite fights that already have odds
    python scripts/populate_odds.py --csv --apply --force

    # Use a different CSV file
    python scripts/populate_odds.py --csv --csv-path path/to/other.csv --apply

    # Add odds for one fight (e.g. after a monthly stat scrape)
    python scripts/populate_odds.py --fight "Islam Makhachev" "Arman Tsarukyan" \\
        --date 2026-06-07 --odds-red -350 --odds-blue 280
"""

import argparse
import difflib
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH, CSV_MASTER
from predict import search_fighter, resolve_fighter
from utils.logger import get_logger

log = get_logger(__name__)

LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
UNMATCHED_LOG = LOGS_DIR / "odds_unmatched.txt"

_MATCH_THRESHOLD = 0.85   # minimum SequenceMatcher ratio to accept a fuzzy match


# ── Name helpers ──────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return name.lower().strip()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _names_match(csv_r: str, csv_b: str, db_r: str, db_b: str) -> bool:
    """True if both names match (exact or fuzzy >= threshold)."""
    r_ok = (_norm(csv_r) == _norm(db_r)) or (_similarity(csv_r, db_r) >= _MATCH_THRESHOLD)
    b_ok = (_norm(csv_b) == _norm(db_b)) or (_similarity(csv_b, db_b) >= _MATCH_THRESHOLD)
    return r_ok and b_ok


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_db_fights(conn: sqlite3.Connection) -> dict[str, list[tuple]]:
    """
    Return all fights keyed by date.
    Each value is a list of (fight_id, r_name, b_name, odds_red, odds_blue).
    """
    rows = conn.execute(
        """
        SELECT f.fight_id, f.date, fi_r.name, fi_b.name, f.odds_red, f.odds_blue
        FROM fights f
        JOIN fighters fi_r ON f.r_fighter_id = fi_r.fighter_id
        JOIN fighters fi_b ON f.b_fighter_id = fi_b.fighter_id
        """
    ).fetchall()
    by_date: dict[str, list] = {}
    for fight_id, date, r_name, b_name, odds_r, odds_b in rows:
        by_date.setdefault(date, []).append((fight_id, r_name, b_name, odds_r, odds_b))
    return by_date


def _upsert_odds(conn: sqlite3.Connection, fight_id: str, odds_red: float, odds_blue: float) -> None:
    conn.execute(
        "UPDATE fights SET odds_red = ?, odds_blue = ? WHERE fight_id = ?",
        (odds_red, odds_blue, fight_id),
    )


# ── Mode A: CSV bulk backfill ─────────────────────────────────────────────────

def run_csv(csv_path: Path, apply: bool, force: bool) -> None:
    if not csv_path.exists():
        print(f"[ERROR]  CSV not found: {csv_path}")
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"[ERROR]  DB not found: {DB_PATH}")
        sys.exit(1)

    df = pd.read_csv(csv_path, usecols=["date", "R_fighter", "B_fighter", "R_odds", "B_odds"])
    df = df.dropna(subset=["R_odds"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(DB_PATH))
    by_date = _load_db_fights(conn)

    matched   = []   # (fight_id, odds_red, odds_blue, already_had_odds)
    unmatched = []   # (date, csv_r, csv_b)

    for _, row in df.iterrows():
        date   = row["date"]
        csv_r  = str(row["R_fighter"])
        csv_b  = str(row["B_fighter"])
        odds_r = float(row["R_odds"])
        odds_b = float(row["B_odds"]) if pd.notna(row["B_odds"]) else None

        candidates = by_date.get(date, [])
        found = None

        for fight_id, db_r, db_b, ex_odds_r, ex_odds_b in candidates:
            if _names_match(csv_r, csv_b, db_r, db_b):
                found = (fight_id, odds_r, odds_b, ex_odds_r is not None)
                break
            # Try reversed corners
            if _names_match(csv_r, csv_b, db_b, db_r):
                # CSV has corners swapped relative to DB — swap odds too
                found = (fight_id, odds_b, odds_r, ex_odds_r is not None)
                break

        if found:
            matched.append(found)
        else:
            unmatched.append((date, csv_r, csv_b))

    already_filled = sum(1 for _, _, _, had in matched if had)
    to_write       = [(fid, r, b) for fid, r, b, had in matched if (not had or force) and b is not None]
    skipped        = len(matched) - len(to_write)

    print()
    print(f"  CSV rows with odds : {len(df):>6}")
    print(f"  Matched to DB fight: {len(matched):>6}")
    print(f"  Already had odds   : {already_filled:>6}  {'(will overwrite with --force)' if not force else '(overwriting)'}")
    print(f"  Will write         : {len(to_write):>6}{'  [DRY RUN -- add --apply to write]' if not apply else ''}")
    print(f"  Unmatched          : {len(unmatched):>6}  (see {UNMATCHED_LOG.name})")

    total_db = conn.execute("SELECT COUNT(*) FROM fights").fetchone()[0]
    projected = len(to_write) + already_filled
    print(f"\n  DB coverage (projected): {projected} / {total_db} fights ({projected/total_db:.1%})")
    print()

    # Write unmatched log
    with open(UNMATCHED_LOG, "w", encoding="utf-8") as f:
        f.write(f"Unmatched CSV rows ({len(unmatched)} fights)\n")
        f.write("=" * 60 + "\n")
        for date, r, b in unmatched:
            f.write(f"{date} | {r} | {b}\n")

    if not apply:
        print("  Dry run complete. Re-run with --apply to write to DB.")
        conn.close()
        return

    for fight_id, odds_r, odds_b in to_write:
        _upsert_odds(conn, fight_id, odds_r, odds_b)
    conn.commit()
    conn.close()
    print(f"  Written {len(to_write)} odds rows to DB.")
    print(f"  Unmatched log: {UNMATCHED_LOG}")


# ── Mode B: Single-fight entry ────────────────────────────────────────────────

def run_fight(red_name: str, blue_name: str, date: str, odds_red: float, odds_blue: float) -> None:
    if not DB_PATH.exists():
        print(f"[ERROR]  DB not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    r_id, r_name = resolve_fighter(conn, red_name)
    b_id, b_name = resolve_fighter(conn, blue_name)

    # Try both corner orderings
    row = conn.execute(
        """
        SELECT fight_id FROM fights
        WHERE date = ?
          AND ((r_fighter_id = ? AND b_fighter_id = ?)
            OR (r_fighter_id = ? AND b_fighter_id = ?))
        """,
        (date, r_id, b_id, b_id, r_id),
    ).fetchone()

    if not row:
        print(f"\n[ERROR]  No fight found for {r_name} vs {b_name} on {date}.")
        print("  Check that fight stats have been scraped for this event first.")
        conn.close()
        sys.exit(1)

    fight_id = row[0]

    # Check if corners are swapped in DB (affects which fighter gets which odds)
    db_row = conn.execute(
        "SELECT r_fighter_id FROM fights WHERE fight_id = ?", (fight_id,)
    ).fetchone()
    if db_row[0] == b_id:
        # In the DB, b_name is actually Red corner — swap the odds
        odds_red, odds_blue = odds_blue, odds_red
        print(f"[INFO]  Corner assignment swapped in DB ({b_name} is Red). Odds adjusted.")

    _upsert_odds(conn, fight_id, odds_red, odds_blue)
    conn.commit()
    conn.close()

    print(f"\n  Odds saved: {r_name} {odds_red:+.0f}  |  {b_name} {odds_blue:+.0f}")
    print(f"  Fight: {fight_id}  |  Date: {date}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate odds_red / odds_blue in the UFCStats DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Dry-run: see match quality before writing
  python scripts/populate_odds.py --csv

  # Write to DB
  python scripts/populate_odds.py --csv --apply

  # Add odds for a single fight (e.g. after monthly stat scrape)
  python scripts/populate_odds.py --fight "Islam Makhachev" "Arman Tsarukyan" \\
      --date 2026-06-07 --odds-red -350 --odds-blue 280
        """,
    )

    sub = parser.add_subparsers(dest="mode")

    # -- csv mode
    csv_p = sub.add_parser("--csv", help="Bulk backfill from CSV (dry-run by default)")
    csv_p.add_argument("--apply",    action="store_true", help="Write to DB (default: dry-run)")
    csv_p.add_argument("--force",    action="store_true", help="Overwrite fights that already have odds")
    csv_p.add_argument("--csv-path", default=str(CSV_MASTER), metavar="PATH", help="CSV file path")

    # -- fight mode
    fight_p = sub.add_parser("--fight", help="Add odds for a single fight")
    fight_p.add_argument("red_fighter",  help="Red corner fighter name (partial OK)")
    fight_p.add_argument("blue_fighter", help="Blue corner fighter name (partial OK)")
    fight_p.add_argument("--date",      required=True, help="Fight date (YYYY-MM-DD)")
    fight_p.add_argument("--odds-red",  required=True, type=float, help="American moneyline for Red (e.g. -350)")
    fight_p.add_argument("--odds-blue", required=True, type=float, help="American moneyline for Blue (e.g. 280)")

    # argparse doesn't handle --csv / --fight as subcommands natively,
    # so we parse manually to support the -- prefix style.
    if len(sys.argv) < 2 or sys.argv[1] not in ("--csv", "--fight"):
        parser.print_help()
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "--csv":
        args = csv_p.parse_args(sys.argv[2:])
        run_csv(Path(args.csv_path), apply=args.apply, force=args.force)
    else:
        args = fight_p.parse_args(sys.argv[2:])
        run_fight(
            args.red_fighter, args.blue_fighter,
            date=args.date,
            odds_red=args.odds_red,
            odds_blue=args.odds_blue,
        )


if __name__ == "__main__":
    main()
