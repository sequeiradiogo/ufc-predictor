"""
ML_data_preparation.py — Build the ML feature dataset from the SQLite database.

Reads  : db/ufc_v2.db
Writes : ml/ufc_ml_data_with_debuts_and_elo.csv

Run:
    python ml/ML_data_preparation.py

New features added (v2):
  - Recent form      : recent_win_rate, recent_finish_rate, win_streak (last 3 fights)
  - Age              : age_diff  (Red age − Blue age at fight date)
  - Style matchup    : grapple_ratio, strike_ratio + striker_vs_wrestler interactions
  - Division         : one-hot encoded (12 known weight classes)
  - Title fight      : binary flag (0 / 1) passed directly to the model
  - Debutant imputation (v3): debutant stats replaced with division-average prior
    instead of zeros — better Bayesian baseline for fighters with no history.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DB_PATH, CSV_WITH_ELO,
    EXCLUDE_STAT_KEYWORDS,
    DIVISIONS,
    FINISH_METHOD_MAP,
    RECENT_FORM_WINDOW,
    MIN_FIGHT_DATE,
)
from logger import get_logger
from ml.ELO_calculator import build_elo_features

log = get_logger(__name__)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

_EPS = 1e-6   # avoid division by zero in ratios


# ═══════════════════════════════════════════════════════════════════════════════
# Recent-form helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_finish(method: str) -> int:
    """Return 1 if the fight ended by KO/TKO or Submission, else 0."""
    if not isinstance(method, str):
        return 0
    return int(method in FINISH_METHOD_MAP and FINISH_METHOD_MAP[method] > 0)


def compute_recent_form(conn: sqlite3.Connection, window: int = RECENT_FORM_WINDOW) -> pd.DataFrame:
    """
    For every (fight_id, fighter_id) pair compute three *pre-fight* rolling stats:
      - recent_win_rate    : fraction of last `window` fights won (0–1)
      - recent_finish_rate : fraction of last `window` fights ended by KO/TKO or Sub (0–1)
      - win_streak         : consecutive wins immediately before this fight (integer)

    shift(1) is applied so the current fight is never included → no leakage.

    Returns a DataFrame with columns:
        fight_id | fighter_id | recent_win_rate | recent_finish_rate | win_streak
    """
    log.info("Computing recent form (window=%d)…", window)

    df = pd.read_sql_query(
        """
        SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id, method
        FROM fights
        ORDER BY date ASC, fight_id ASC
        """,
        conn,
    )

    # Build a long-format table: one row per (fighter, fight)
    long_rows = []
    for _, row in df.iterrows():
        won_r    = 1 if row["winner_id"] == row["r_fighter_id"] else 0
        won_b    = 1 if row["winner_id"] == row["b_fighter_id"] else 0
        finished = _is_finish(row["method"])
        long_rows.append({
            "fight_id":   row["fight_id"],
            "date":       row["date"],
            "fighter_id": row["r_fighter_id"],
            "won":        won_r,
            "finished":   finished,
        })
        long_rows.append({
            "fight_id":   row["fight_id"],
            "date":       row["date"],
            "fighter_id": row["b_fighter_id"],
            "won":        won_b,
            "finished":   finished,
        })

    long = pd.DataFrame(long_rows)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    # Rolling stats (min_periods=1 so fighters with <window fights still get a value)
    grp = long.groupby("fighter_id", sort=False)

    long["recent_win_rate"] = (
        grp["won"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    long["recent_finish_rate"] = (
        grp["finished"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )

    # Fill NaN (first fight of a career) with 0
    for col in ("recent_win_rate", "recent_finish_rate"):
        long[col] = long[col].fillna(0)

    # win_streak is NOT computed here — mdabbert provides career_win_streak in
    # fight_stats which is more accurate (covers pre-DB history).
    return long[["fight_id", "fighter_id", "recent_win_rate", "recent_finish_rate"]]


# ═══════════════════════════════════════════════════════════════════════════════
# Age features
# ═══════════════════════════════════════════════════════════════════════════════

def add_age_features(ml_data: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add age_diff = age_red - age_blue.

    Two sources are supported:
    - dob_red / dob_blue ('YYYY/MM/DD') — original UFCStats format; age computed from fight date.
    - age_red / age_blue (numeric years) — mdabbert format; used directly.
    """
    log.info("Adding age features…")

    dob_col_available = (
        "dob_red" in df_raw.columns
        and df_raw["dob_red"].notna().any()
    )

    if dob_col_available:
        fight_date = pd.to_datetime(df_raw["date"])
        dob_red    = pd.to_datetime(df_raw["dob_red"],  format="%Y/%m/%d", errors="coerce")
        dob_blue   = pd.to_datetime(df_raw["dob_blue"], format="%Y/%m/%d", errors="coerce")
        age_red    = (fight_date - dob_red).dt.days  / 365.25
        age_blue   = (fight_date - dob_blue).dt.days / 365.25
    else:
        age_red  = pd.to_numeric(df_raw.get("age_red"),  errors="coerce")
        age_blue = pd.to_numeric(df_raw.get("age_blue"), errors="coerce")

    ml_data["age_diff"] = (age_red - age_blue).values
    return ml_data


# ═══════════════════════════════════════════════════════════════════════════════
# Style features
# ═══════════════════════════════════════════════════════════════════════════════

def add_style_features(ml_data: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute style ratios and matchup interaction features.

    Ratios (0–1):
        grapple_ratio = td_avg / (td_avg + splm + eps)
        strike_ratio  = splm   / (td_avg + splm + eps)

    Diff features (Red − Blue):
        grapple_ratio_diff
        strike_ratio_diff

    Matchup interactions (product of opposing ratios — high when styles clash):
        striker_vs_wrestler  = strike_ratio_red  × grapple_ratio_blue
        wrestler_vs_striker  = grapple_ratio_red × strike_ratio_blue
    """
    log.info("Adding style features…")

    def _ratios(suffix: str):
        splm    = pd.to_numeric(df_raw[f"splm_{suffix}"],   errors="coerce").fillna(0)
        td_avg  = pd.to_numeric(df_raw[f"td_avg_{suffix}"], errors="coerce").fillna(0)
        denom   = splm + td_avg + _EPS
        grapple = td_avg / denom
        strike  = splm   / denom
        return grapple, strike

    grapple_r, strike_r = _ratios("red")
    grapple_b, strike_b = _ratios("blue")

    ml_data["grapple_ratio_diff"]  = (grapple_r - grapple_b).values
    # strike_ratio = 1 - grapple_ratio per fighter, so strike_ratio_diff = -grapple_ratio_diff
    # (r~-0.92 for non-debutants) — dropped to avoid near-duplicate feature.
    ml_data["striker_vs_wrestler"] = (strike_r  * grapple_b).values
    ml_data["wrestler_vs_striker"] = (grapple_r * strike_b).values

    return ml_data


# ═══════════════════════════════════════════════════════════════════════════════
# Division features
# ═══════════════════════════════════════════════════════════════════════════════

def add_division_features(ml_data: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the division column using the 12 known UFC weight classes.
    Each column is named `div_<sanitised_division_name>` and takes values 0 / 1.
    Unknown divisions (catch-weight, open-weight, etc.) are encoded as all-zeros.
    """
    log.info("Adding division one-hot features…")

    def _sanitise(name: str) -> str:
        return "div_" + name.replace(" ", "_").replace("'", "")

    for div in DIVISIONS:
        col = _sanitise(div)
        ml_data[col] = (ml_data["division"].str.lower() == div).astype(int)

    return ml_data


# ═══════════════════════════════════════════════════════════════════════════════
# Debutant imputation
# ═══════════════════════════════════════════════════════════════════════════════

def impute_debutant_stats(df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """
    Replace zero-stat rows (debutants) with division-average stats as a Bayesian prior.

    A fighter's debut row has total_fight_time == 0 and all computed rolling stats
    at 0.  Using 0 for every diff feature is misleading — an average opponent looks
    like a world-beater compared to zeroes.

    Strategy:
      1. Compute per-division means from ALL established fighters (total_fight_time > 0).
      2. For each _red / _blue suffix, if total_fight_time_{suffix} == 0, replace
         every stat column with the division mean for that fight's division.
      3. A global mean is used as fallback for unknown/catch-weight divisions.

    Only numeric stat columns are imputed (not dob, stance, weight, etc.).
    Vectorised — no row-by-row loops.
    """
    log.info("Imputing debutant stats with division-average prior…")

    numeric_stat_cols = [
        c for c in stat_cols
        if not any(kw in c for kw in EXCLUDE_STAT_KEYWORDS)
        and c not in ("fight_id", "fighter_id", "corner", "date")
    ]

    for suffix in ("red", "blue"):
        time_col = f"total_fight_time_{suffix}"
        is_debut = pd.to_numeric(df[time_col], errors="coerce").fillna(0) == 0
        n_debuts = int(is_debut.sum())
        if n_debuts == 0:
            continue

        log.info("  %d %s-corner debutants — imputing with division prior.", n_debuts, suffix)

        stat_suffixed = [f"{s}_{suffix}" for s in numeric_stat_cols if f"{s}_{suffix}" in df.columns]
        if not stat_suffixed:
            continue

        # Convert stat columns to float64 in-place (allows assigning float means)
        for col in stat_suffixed:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Per-division means from established fighters (total_fight_time > 0)
        established_mask = pd.to_numeric(df[time_col], errors="coerce").fillna(0) > 0
        div_means = (
            df.loc[established_mask, ["division"] + stat_suffixed]
            .groupby("division")[stat_suffixed]
            .mean()
        )
        global_mean = df.loc[established_mask, stat_suffixed].mean()

        # Build a prior lookup: division → Series of means (fallback to global)
        # Shape: one row per division
        prior_df = div_means.reindex(df["division"].unique())
        # Fill missing divisions (catch-weight etc.) with global mean
        prior_df = prior_df.fillna(global_mean)
        prior_df.index.name = "division"

        # Map each row's division to its prior values (vectorised merge)
        mapped = df[["division"]].join(prior_df, on="division")[stat_suffixed]
        mapped.index = df.index

        # Apply imputation only on debut rows; keep original values elsewhere
        debut_idx = df.index[is_debut]
        df.loc[debut_idx, stat_suffixed] = mapped.loc[debut_idx]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Main dataset builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_ml_dataset() -> pd.DataFrame:
    """
    Query the database, compute Red−Blue difference features, merge ELO,
    add recent form / age / style / division / title-fight features, and
    return the final ML-ready DataFrame.
    """
    log.info("Connecting to database: %s", DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))

    # ── Discover fight_stats columns dynamically ──────────────────────────────
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(fight_stats)")
    stat_cols   = [row[1] for row in cur.fetchall()]
    ignore_cols = ["fight_id", "fighter_id", "corner"]

    select_red  = ", ".join([f'rs."{c}" AS "{c}_red"'  for c in stat_cols if c not in ignore_cols])
    select_blue = ", ".join([f'bs."{c}" AS "{c}_blue"' for c in stat_cols if c not in ignore_cols])

    query = f"""
        SELECT
            f.fight_id,
            f.date,
            f.division,
            f.title_fight,
            f.method,
            CASE
                WHEN f.winner_id = f.r_fighter_id THEN 1
                WHEN f.winner_id = f.b_fighter_id THEN 0
                ELSE NULL
            END AS red_win,
            {select_red},
            {select_blue}
        FROM fights f
        JOIN fight_stats rs ON f.fight_id = rs.fight_id AND rs.corner = 'r'
        JOIN fight_stats bs ON f.fight_id = bs.fight_id AND bs.corner = 'b'
        WHERE red_win IS NOT NULL;
    """

    log.info("Querying fight data…")
    df = pd.read_sql_query(query, conn)
    log.info("Raw fight data shape: %s", df.shape)

    # ── ELO features ─────────────────────────────────────────────────────────
    log.info("Building ELO features…")
    elo_df = build_elo_features(conn)
    df = df.merge(elo_df, on="fight_id", how="left")

    # ── Recent form per fighter ───────────────────────────────────────────────
    form_df = compute_recent_form(conn)

    # We need r_fighter_id / b_fighter_id to join form; fetch from fights table
    fighter_ids = pd.read_sql_query(
        "SELECT fight_id, r_fighter_id, b_fighter_id FROM fights", conn
    )
    df = df.merge(fighter_ids, on="fight_id", how="left")

    form_red  = form_df.rename(columns={
        "recent_win_rate":    "recent_win_rate_red",
        "recent_finish_rate": "recent_finish_rate_red",
    })
    form_blue = form_df.rename(columns={
        "recent_win_rate":    "recent_win_rate_blue",
        "recent_finish_rate": "recent_finish_rate_blue",
    })

    df = df.merge(
        form_red[["fight_id", "fighter_id", "recent_win_rate_red", "recent_finish_rate_red"]],
        left_on=["fight_id", "r_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    df = df.merge(
        form_blue[["fight_id", "fighter_id", "recent_win_rate_blue", "recent_finish_rate_blue"]],
        left_on=["fight_id", "b_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    conn.close()

    # ── Difference features (rolling stats) ───────────────────────────────────
    base_stats    = [c.replace("_red", "") for c in df.columns if c.endswith("_red")]
    stats_to_diff = [s for s in base_stats if not any(kw in s for kw in EXCLUDE_STAT_KEYWORDS)]

    log.info("Computing %d difference features…", len(stats_to_diff))

    ml_data = pd.DataFrame()
    ml_data["fight_id"] = df["fight_id"]
    ml_data["date"]     = df["date"]
    ml_data["division"] = df["division"]
    ml_data["target"]   = df["red_win"]

    r_time = pd.to_numeric(df["total_fight_time_red"],  errors="coerce").fillna(0)
    b_time = pd.to_numeric(df["total_fight_time_blue"], errors="coerce").fillna(0)

    for stat in stats_to_diff:
        col_red  = pd.to_numeric(df[f"{stat}_red"],  errors="coerce").fillna(0)
        col_blue = pd.to_numeric(df[f"{stat}_blue"], errors="coerce").fillna(0)
        ml_data[f"{stat}_diff"] = col_red - col_blue

    # ── Recent form differences ───────────────────────────────────────────────
    for stat in ("recent_win_rate", "recent_finish_rate"):
        r_col = df[f"{stat}_red"].fillna(0)
        b_col = df[f"{stat}_blue"].fillna(0)
        ml_data[f"{stat}_diff"] = (r_col - b_col).values

    # ── Age features ──────────────────────────────────────────────────────────
    ml_data = add_age_features(ml_data, df)

    # ── Style features ────────────────────────────────────────────────────────
    ml_data = add_style_features(ml_data, df)

    # ── Division one-hot encoding ─────────────────────────────────────────────
    ml_data = add_division_features(ml_data)

    # ── Title fight flag ──────────────────────────────────────────────────────
    ml_data["title_fight"] = pd.to_numeric(df["title_fight"], errors="coerce").fillna(0).astype(int).values

    # ── Exclusion filters ─────────────────────────────────────────────────────
    n_before = len(ml_data)

    # Drop fights before MIN_FIGHT_DATE (early era has unreliable stats)
    ml_data = ml_data[pd.to_datetime(ml_data["date"]) >= MIN_FIGHT_DATE].copy()
    n_date = n_before - len(ml_data)

    # Drop fights where either fighter has no prior recorded fights (debutant)
    # total_fight_time = wins + losses in the mdabbert schema; 0 means debut.
    is_debut_r = r_time.values == 0
    is_debut_b = b_time.values == 0
    debut_mask = pd.Series(is_debut_r | is_debut_b, index=df.index)
    # Align index after the date filter
    ml_data = ml_data[~debut_mask.reindex(ml_data.index, fill_value=False)].copy()
    n_debut = (n_before - n_date) - len(ml_data)

    log.info(
        "Exclusions: %d pre-%s fights, %d debut fights. %d rows remaining.",
        n_date, MIN_FIGHT_DATE[:4], n_debut, len(ml_data),
    )
    log.info("Final ML dataset shape: %s", ml_data.shape)
    return ml_data


def main() -> None:
    ml_data = build_ml_dataset()

    log.info("Saving dataset to: %s", CSV_WITH_ELO)
    ml_data.to_csv(CSV_WITH_ELO, index=False)
    log.info("Done. %d rows, %d columns.", *ml_data.shape)

    # Quick summary
    new_cols = [c for c in ml_data.columns if any(
        kw in c for kw in ("recent_", "age_", "grapple_", "strike_", "wrestler_", "div_", "title_")
    )]
    log.info("New feature columns (%d): %s", len(new_cols), new_cols)


if __name__ == "__main__":
    main()
