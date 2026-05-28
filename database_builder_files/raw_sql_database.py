"""
raw_sql_database.py — Build the raw UFC SQLite database from the source CSV.

Supersedes: 'raw SQL database.py'  (space in name caused issues)

Reads : UFC.csv  (path set via --csv argument or IN_CSV constant below)
Writes: database_builder_files/ufc.db

Usage:
    python database_builder_files/raw_sql_database.py
    python database_builder_files/raw_sql_database.py --csv path/to/UFC.csv
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

log = get_logger(__name__)

# Default CSV path — override with --csv flag
DEFAULT_CSV = ROOT_DIR / "raw_data" / "UFC.csv"
DEFAULT_DB  = Path(__file__).resolve().parent / "ufc.db"


# ── Column detection helpers ──────────────────────────────────────────────────

def detect_id_cols(cols: list[str]) -> tuple[str | None, str | None]:
    r = next((c for c in cols if c in ("r_id", "r_fighter_id", "r_fid", "r_fighter")), None)
    b = next((c for c in cols if c in ("b_id", "b_fighter_id", "b_fid", "b_fighter")), None)
    if r is None:
        r = next((c for c in cols if c.startswith("r_") and c.endswith("id")), None)
    if b is None:
        b = next((c for c in cols if c.startswith("b_") and c.endswith("id")), None)
    return r, b


def detect_name_cols(cols: list[str]) -> tuple[str | None, str | None]:
    rname = next((c for c in cols if c in ("r_name", "r_fighter", "r_fighter_name")), None)
    bname = next((c for c in cols if c in ("b_name", "b_fighter", "b_fighter_name")), None)
    return rname, bname


# ── Main build logic ──────────────────────────────────────────────────────────

def build_database(csv_path: Path, db_path: Path) -> None:
    """Read *csv_path* and write a 3-table SQLite DB to *db_path*."""

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("Loading CSV from %s…", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log.info("CSV loaded: %d rows × %d columns.", *df.shape)

    cols       = df.columns.tolist()
    r_id_col, b_id_col     = detect_id_cols(cols)
    r_name_col, b_name_col = detect_name_cols(cols)

    if r_id_col is None or b_id_col is None:
        log.error("Could not find r_id / b_id columns. Please update detect_id_cols().")
        sys.exit(1)

    log.info("ID columns   : %s, %s", r_id_col, b_id_col)
    log.info("Name columns : %s, %s", r_name_col, b_name_col)

    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    # ── fighters ──────────────────────────────────────────────────────────────
    log.info("Building fighters table…")
    r_df = df[[r_id_col]].rename(columns={r_id_col: "fighter_id"}).copy()
    r_df["name"] = df[r_name_col] if r_name_col else None
    b_df = df[[b_id_col]].rename(columns={b_id_col: "fighter_id"}).copy()
    b_df["name"] = df[b_name_col] if b_name_col else None

    fighters = (
        pd.concat([r_df, b_df], ignore_index=True)
        .drop_duplicates(subset=["fighter_id"])
        .reset_index(drop=True)
    )
    fighters["fighter_id"] = fighters["fighter_id"].astype(str)
    fighters.to_sql("fighters", conn, if_exists="replace", index=False)
    log.info("Inserted %d fighters.", len(fighters))

    # ── fights ────────────────────────────────────────────────────────────────
    log.info("Building fights table…")
    meta_candidates = [
        "fight_id", "event_id", "event_name", "date", "location", "division",
        "title_fight", "method", "finish_round", "match_time_sec", "total_rounds", "referee",
    ]
    meta_cols = [c for c in meta_candidates if c in cols]
    fights = (
        df[meta_cols + [r_id_col, b_id_col, "winner_id"]]
        .drop_duplicates(subset=["fight_id"])
        .copy()
        .rename(columns={r_id_col: "r_fighter_id", b_id_col: "b_fighter_id"})
    )
    fights["r_fighter_id"] = fights["r_fighter_id"].astype(str)
    fights["b_fighter_id"] = fights["b_fighter_id"].astype(str)
    fights.to_sql("fights", conn, if_exists="replace", index=False)
    log.info("Inserted %d fights.", len(fights))

    # ── fight_stats ───────────────────────────────────────────────────────────
    log.info("Building fight_stats table…")
    excluded_suffixes = ("id", "name", "fighter", "fighter_name")

    def is_stat(col: str, prefix: str) -> bool:
        base = col[len(prefix):]
        return not any(base.endswith(s) for s in excluded_suffixes)

    r_stat_cols = [c for c in cols if c.startswith("r_") and is_stat(c, "r_")]
    b_stat_cols = [c for c in cols if c.startswith("b_") and is_stat(c, "b_")]

    r_side = df[["fight_id", r_id_col]].copy().rename(columns={r_id_col: "fighter_id"})
    b_side = df[["fight_id", b_id_col]].copy().rename(columns={b_id_col: "fighter_id"})
    r_side["corner"] = "r"
    b_side["corner"] = "b"

    for c in r_stat_cols:
        r_side[c[len("r_"):]] = df[c]
    for c in b_stat_cols:
        b_side[c[len("b_"):]] = df[c]

    all_stats  = sorted({c[len("r_"):] for c in r_stat_cols} | {c[len("b_"):] for c in b_stat_cols})
    cols_order = ["fight_id", "fighter_id", "corner"] + all_stats
    for s in all_stats:
        r_side.setdefault(s, None)
        b_side.setdefault(s, None)

    stats_df = pd.concat([r_side[cols_order], b_side[cols_order]], ignore_index=True)
    stats_df["fighter_id"] = stats_df["fighter_id"].astype(str)
    for s in all_stats:
        stats_df[s] = pd.to_numeric(stats_df[s], errors="ignore")

    stats_df.to_sql("fight_stats", conn, if_exists="replace", index=False)
    log.info("Inserted %d fight_stats rows.", len(stats_df))

    # ── indexes ───────────────────────────────────────────────────────────────
    log.info("Creating indexes…")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fights_date      ON fights(date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fights_rfighter  ON fights(r_fighter_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fightstats_fight ON fight_stats(fight_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fightstats_fid   ON fight_stats(fighter_id);")
    conn.commit()

    # ── summary ───────────────────────────────────────────────────────────────
    n_f  = cur.execute("SELECT COUNT(*) FROM fighters").fetchone()[0]
    n_fi = cur.execute("SELECT COUNT(*) FROM fights").fetchone()[0]
    n_s  = cur.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0]
    log.info("Database complete → fighters: %d  fights: %d  fight_stats: %d", n_f, n_fi, n_s)
    conn.close()
    log.info("Saved to: %s", db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the raw UFC SQLite database from source CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to UFC.csv")
    parser.add_argument("--db",  type=Path, default=DEFAULT_DB,  help="Output SQLite database path")
    args = parser.parse_args()
    build_database(args.csv, args.db)


if __name__ == "__main__":
    main()
