"""
rolling.py — Compute pre-fight rolling statistics and upsert into the database.

Reads from and writes to the main UFC SQLite database (ufc_v2.db).
Every stat is shifted so a fighter's row for fight N only contains data
from fights 1 … N-1 — no future leakage.

Run:
    python database_builder_files/rolling.py

Or import and call main() from run_pipeline.py.
"""

import datetime
import math
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH
from logger import get_logger

log = get_logger(__name__)

pd.set_option("display.max_columns", None)

# ── Column groups ─────────────────────────────────────────────────────────────
_STATS        = ["body", "clinch", "dist", "ground", "head", "leg", "sig_str", "td", "total_str"]
_LANDED_COLS  = [s + "_landed"  for s in _STATS]
_ATMPTED_COLS = [s + "_atmpted" for s in _STATS]
_ACC_COLS     = [s + "_acc"     for s in _STATS]
_SUM_STATS    = ["ctrl", "kd", "sub_att"]
_AVG_STATS    = [
    "landed_body_per", "landed_clinch_per", "landed_dist_per",
    "landed_ground_per", "landed_head_per", "landed_leg_per",
]
_ROLLING_STATS        = ["losses", "wins"]
_TIME_DEPENDENT_STATS = ["sub_avg", "td_avg", "splm"]
_RELATIVE_STATS       = ["sapm", "str_def", "td_def"]

_TARGET_TABLE = "fight_stats"
_MATCH_COLS   = ["fight_id", "fighter_id"]
_BATCH        = 1000
_BUSY_TIMEOUT = 30_000


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load fight_stats joined with fight metadata."""
    query = """
        SELECT fs.*, f.date, f.winner_id, f.match_time_sec,
               f.finish_round, f.r_fighter_id, f.b_fighter_id
        FROM fight_stats AS fs
        JOIN fights AS f ON fs.fight_id = f.fight_id
    """
    df = pd.read_sql_query(query, conn, parse_dates=["date"])

    numeric_cols = _LANDED_COLS + _ATMPTED_COLS + _SUM_STATS + _AVG_STATS + _ROLLING_STATS
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Sort order is CRITICAL for correct rolling calculation
    df = df.sort_values(
        ["fighter_id", "date", "fight_id", "r_fighter_id", "b_fighter_id"]
    ).reset_index(drop=True)

    log.info("Loaded %d fight-stat rows for %d unique fighters.", len(df), df["fighter_id"].nunique())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Rolling-stat computation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_win_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative wins/losses *before* each fight (shift-based)."""
    rolling = df[["fighter_id", "date", "winner_id", "corner"] + _ROLLING_STATS].copy()
    rolling["is_win"]  = (rolling["fighter_id"] == rolling["winner_id"]).astype(int)
    rolling["is_loss"] = (
        (rolling["fighter_id"] != rolling["winner_id"]) &
        (rolling["winner_id"] != "None")
    ).astype(int)

    grp = rolling.groupby("fighter_id")
    rolling["wins_shift"]   = grp["is_win"].shift(1).fillna(0).astype(int)
    rolling["losses_shift"] = grp["is_loss"].shift(1).fillna(0).astype(int)
    rolling["wins_prov"]    = grp["wins_shift"].cumsum().astype(int)
    rolling["losses_prov"]  = grp["losses_shift"].cumsum().astype(int)

    rolling["wins"]   = rolling["wins_prov"]   + (rolling["wins"]   - grp["is_win"].transform("sum"))
    rolling["losses"] = rolling["losses_prov"] + (rolling["losses"] - grp["is_loss"].transform("sum"))

    rolling["outcome"] = np.where(
        rolling["is_win"],
        rolling["corner"],
        np.where(
            rolling["is_loss"] & (rolling["corner"] == "Blue"), "Red",
            np.where(rolling["is_loss"] & (rolling["corner"] == "Red"), "Blue", "NA"),
        ),
    )
    drop_cols = [
        "is_win", "is_loss", "wins_prov", "losses_prov",
        "wins_shift", "losses_shift", "fighter_id", "date", "winner_id",
    ]
    return rolling.drop(columns=drop_cols)


def _compute_cumulative_stats(df: pd.DataFrame) -> tuple:
    """Compute cumulative landed/attempted/accuracy/other stats before each fight."""
    grp = df.groupby("fighter_id")

    cum_landed  = grp[_LANDED_COLS].cumsum()
    cum_atmpted = grp[_ATMPTED_COLS].cumsum()
    cum_others  = grp[_SUM_STATS].cumsum()
    cum_avg     = grp[_AVG_STATS].cumsum()

    prior_landed  = cum_landed.groupby(df["fighter_id"]).shift(1).fillna(0)
    prior_atmpted = cum_atmpted.groupby(df["fighter_id"]).shift(1).fillna(0)
    prior_others  = cum_others.groupby(df["fighter_id"]).shift(1).fillna(0)
    prior_avg     = cum_avg.groupby(df["fighter_id"]).shift(1).fillna(0)

    prior_counts = df.groupby("fighter_id").cumcount().replace(0, np.nan)
    prior_avg    = prior_avg.div(prior_counts, axis=0).fillna(0)

    raw_acc    = (prior_landed.values / prior_atmpted.values) * 100
    prior_acc  = pd.DataFrame(raw_acc, columns=_ACC_COLS).fillna(0)

    return prior_landed, prior_atmpted, prior_acc, prior_others, prior_avg


def _compute_time_dependent(df: pd.DataFrame, prior_landed: pd.DataFrame, prior_others: pd.DataFrame) -> pd.DataFrame:
    """Compute per-minute / per-15-min stats from cumulative fight time."""
    td = df[["fighter_id", "date", "match_time_sec", "finish_round"] + _TIME_DEPENDENT_STATS].copy()
    td["fight_time"]      = td["match_time_sec"] + (td["finish_round"] - 1) * 300
    td["total_fight_time"] = (
        td.groupby("fighter_id")["fight_time"].cumsum()
          .groupby(df["fighter_id"]).shift(1).fillna(0)
    )

    time_div = td["total_fight_time"].replace(0, np.nan)
    td["td_avg"]  = (prior_landed["td_landed"]    / time_div) * 15 * 60
    td["sub_avg"] = (prior_others["sub_att"]       / time_div) * 15 * 60
    td["splm"]    = (prior_landed["sig_str_landed"] / time_div) * 60
    td = td.fillna(0)

    td = td.drop(columns=["date", "fighter_id", "match_time_sec", "finish_round", "fight_time"])
    return td


def _compute_relative_stats(df: pd.DataFrame, total_fight_time: pd.Series) -> pd.DataFrame:
    """Compute strike/takedown defense and strikes absorbed per minute."""
    rel = df[[
        "fighter_id", "date", "fight_id",
        "r_fighter_id", "b_fighter_id",
        "td_atmpted", "td_landed",
        "sig_str_atmpted", "sig_str_landed",
    ] + _RELATIVE_STATS].copy()

    rel["opponent_id"] = np.where(
        rel["fighter_id"] == rel["r_fighter_id"],
        rel["b_fighter_id"],
        rel["r_fighter_id"],
    )

    opp = df[["fight_id", "fighter_id", "td_atmpted", "td_landed", "sig_str_atmpted", "sig_str_landed"]].rename(
        columns={
            "fighter_id":      "opp_id",
            "td_atmpted":      "td_atmpted_against",
            "td_landed":       "td_landed_against",
            "sig_str_atmpted": "sig_str_atmpted_against",
            "sig_str_landed":  "sig_str_landed_against",
        }
    )
    rel = rel.merge(opp, left_on=["fight_id", "opponent_id"], right_on=["fight_id", "opp_id"], how="left")
    rel = rel.drop(columns=["opp_id"])

    against_cols = ["td_atmpted_against", "td_landed_against", "sig_str_atmpted_against", "sig_str_landed_against"]
    rel[against_cols] = rel.groupby("fighter_id")[against_cols].transform(
        lambda g: g.cumsum().shift(1).fillna(0)
    )

    rel["str_def"] = (
        ((rel["sig_str_atmpted_against"] - rel["sig_str_landed_against"]) / rel["sig_str_atmpted_against"]) * 100
    ).fillna(0)
    rel["sapm"] = (rel["sig_str_landed_against"] / total_fight_time.replace(0, np.nan) * 60).fillna(0)
    rel["td_def"] = (
        ((rel["td_atmpted_against"] - rel["td_landed_against"]) / rel["td_atmpted_against"]) * 100
    ).fillna(0)

    drop = [
        "fighter_id", "date", "fight_id", "r_fighter_id", "b_fighter_id",
        "td_atmpted", "td_landed", "sig_str_atmpted", "sig_str_landed",
        "td_atmpted_against", "td_landed_against",
        "sig_str_atmpted_against", "sig_str_landed_against",
    ]
    return rel.drop(columns=drop)


# ══════════════════════════════════════════════════════════════════════════════
# Assemble final DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def build_prior_df(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all rolling-stat components into a single DataFrame."""
    log.info("Computing win/loss rolling stats…")
    rolling = _compute_win_loss(df)

    log.info("Computing cumulative landed/attempted/accuracy stats…")
    prior_landed, prior_atmpted, prior_acc, prior_others, prior_avg = _compute_cumulative_stats(df)

    log.info("Computing time-dependent stats (splm, td_avg, sub_avg)…")
    time_dep = _compute_time_dependent(df, prior_landed, prior_others)

    log.info("Computing relative stats (str_def, sapm, td_def)…")
    relative = _compute_relative_stats(df, time_dep["total_fight_time"])

    prior_df = pd.concat(
        [
            df[["fighter_id", "date", "fight_id", "corner",
                "dob", "height", "reach", "stance", "weight"]].reset_index(drop=True),
            prior_landed.reset_index(drop=True),
            prior_atmpted.reset_index(drop=True),
            prior_acc.reset_index(drop=True),
            prior_others.reset_index(drop=True),
            prior_avg.reset_index(drop=True),
            rolling.reset_index(drop=True),
            time_dep.reset_index(drop=True),
            relative.reset_index(drop=True),
        ],
        axis=1,
    )

    # Remove any duplicate columns (e.g. sapm, str_def, td_def from original + computed)
    prior_df = prior_df.loc[:, ~prior_df.columns.duplicated()]
    log.info("Prior DataFrame built. Shape: %s", prior_df.shape)
    return prior_df


# ══════════════════════════════════════════════════════════════════════════════
# DB upsert
# ══════════════════════════════════════════════════════════════════════════════

def upsert_to_db(prior_df: pd.DataFrame, db_path: Path) -> None:
    """Backup the DB and upsert the computed rolling stats into fight_stats."""
    # ── Backup ────────────────────────────────────────────────────────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_name(db_path.stem + f"_backup_{stamp}" + db_path.suffix)
    shutil.copy2(db_path, backup)
    log.info("DB backup created: %s", backup)

    # ── Prepare export df ────────────────────────────────────────────────────
    df_export = prior_df.copy()
    for col, dtype in df_export.dtypes.items():
        if pd.api.types.is_datetime64_any_dtype(dtype):
            df_export[col] = df_export[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT};")
    cur = conn.cursor()

    try:
        # Ensure table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (_TARGET_TABLE,))
        if cur.fetchone() is None:
            raise RuntimeError(f"Target table '{_TARGET_TABLE}' not found.")

        # Add new columns if missing
        def _add_col(col_name: str, col_type: str) -> None:
            cur.execute(f"PRAGMA table_info({_TARGET_TABLE})")
            existing = {row[1] for row in cur.fetchall()}
            if col_name not in existing:
                log.info("Adding column '%s' to '%s'…", col_name, _TARGET_TABLE)
                cur.execute(f"ALTER TABLE {_TARGET_TABLE} ADD COLUMN {col_name} {col_type}")

        _add_col("total_fight_time", "REAL")
        _add_col("opponent_id",      "TEXT")
        conn.commit()

        # Unique index for upsert
        idx = f"idx_{_TARGET_TABLE}_{'_'.join(_MATCH_COLS)}_unique"
        cur.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS "{idx}"
            ON "{_TARGET_TABLE}" ({",".join(_MATCH_COLS)});
        """)
        conn.commit()

        # Determine write columns (intersection of DF and DB)
        cur.execute(f'PRAGMA table_info("{_TARGET_TABLE}");')
        tgt_cols  = {r[1] for r in cur.fetchall()}
        write_cols = [c for c in df_export.columns if c in tgt_cols]
        if not write_cols:
            raise RuntimeError("No matching columns to write.")

        cols_sql    = ",".join([f'"{c}"' for c in write_cols])
        placeholders = ",".join(["?"] * len(write_cols))
        set_clause  = ", ".join(
            [f'"{c}" = excluded."{c}"' for c in write_cols if c not in _MATCH_COLS]
        )
        sql = (
            f'INSERT INTO "{_TARGET_TABLE}" ({cols_sql}) VALUES ({placeholders}) '
            f'ON CONFLICT({",".join(_MATCH_COLS)}) DO UPDATE SET {set_clause};'
        )

        rows  = df_export[write_cols].where(df_export[write_cols].notnull(), None).values.tolist()
        total = len(rows)
        log.info("Upserting %d rows (%d columns)…", total, len(write_cols))
        t0 = time.time()

        for i in range(0, total, _BATCH):
            cur.executemany(sql, rows[i : i + _BATCH])
            conn.commit()
            log.info(
                "  Batch %d/%d done.", i // _BATCH + 1, math.ceil(total / _BATCH)
            )

        log.info("Upsert complete in %.2fs.", time.time() - t0)

    finally:
        conn.close()
        log.info("DB connection closed.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not DB_PATH.exists():
        log.error("Database not found: %s", DB_PATH)
        log.error("Run step 1 (raw_sql_database.py) and step 2 (keys.py) first.")
        sys.exit(1)

    log.info("Connecting to database: %s", DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))
    df = load_raw_data(conn)
    conn.close()

    prior_df = build_prior_df(df)
    upsert_to_db(prior_df, DB_PATH)
    log.info("Rolling stats update complete.")


if __name__ == "__main__":
    main()
