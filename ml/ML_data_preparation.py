"""
ML_data_preparation.py — Build the ML feature dataset from the SQLite database.

Reads  : db/ufc_ufcstats.db (DB_PATH)
Writes : ml/ufc_ml_data_with_debuts_and_elo.csv

Run:
    python ml/ML_data_preparation.py

New features added (v2):
  - Recent form      : recent_win_rate, recent_finish_rate (last 3 fights)
  - Age              : age_diff  (Red age - Blue age at fight date)
  - Style matchup    : grapple_ratio, strike_ratio + striker_vs_wrestler interactions
  - Division         : one-hot encoded (12 known weight classes)
  - Title fight      : binary flag (0 / 1) passed directly to the model
  - Debutant imputation (v3): debutant stats replaced with division-average prior

New features added (v3):
  - Height / reach diff    : physical advantage from fighters table
  - Stance matchup         : southpaw_adv_diff (-1/0/+1), both_southpaw flag
  - Finish-method rates    : ko_rate_diff, sub_rate_diff, dec_rate_diff
  - Inactivity             : days_since_last_diff (days between consecutive fights)
  - Strength of schedule   : sos_diff (avg ELO of last 5 opponents)

New features added (v4):
  - Win / loss streaks     : win_streak_diff, loss_streak_diff
  - Performance slopes     : str_acc_slope_diff, td_acc_slope_diff, splm_slope_diff
                             (linear trend of per-fight metrics over last TRAJECTORY_WINDOW fights)
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
    SHRINKAGE_LAMBDA,
    STARTING_ELO,
    SOS_WINDOW,
    KO_VULN_WINDOW,
    EWMA_SPAN,
    TRAJECTORY_WINDOW,
)
from utils.logger import get_logger
from ml.ELO_calculator import build_elo_features, build_glicko_features

log = get_logger(__name__)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

_EPS = 1e-6   # avoid division by zero in ratios

# Rolling stats to shrink toward the division mean (derived rates/accuracies,
# not raw counts which are already excluded from the diff loop).
_SHRINKAGE_COLS = [
    "splm", "sapm", "str_def", "td_def", "td_avg", "sub_avg",
    "sig_str_acc", "total_str_acc", "td_acc",
    "head_acc", "body_acc", "leg_acc", "dist_acc", "clinch_acc", "ground_acc",
    "landed_head_per", "landed_body_per", "landed_leg_per",
    "landed_dist_per", "landed_clinch_per", "landed_ground_per",
]


def apply_shrinkage(df: pd.DataFrame, lam: float = SHRINKAGE_LAMBDA) -> pd.DataFrame:
    """
    Shrink each fighter's rolling stats toward their division mean.

    smoothed = (n * raw + lam * div_mean) / (n + lam)

    Fighters with few prior fights (small n) are pulled toward the division
    average; established fighters (large n) keep their own stats.
    """
    df = df.copy()
    for suffix in ("red", "blue"):
        wins   = pd.to_numeric(df.get(f"wins_{suffix}",   0), errors="coerce").fillna(0)
        losses = pd.to_numeric(df.get(f"losses_{suffix}", 0), errors="coerce").fillna(0)
        n = wins + losses

        for col in _SHRINKAGE_COLS:
            col_s = f"{col}_{suffix}"
            if col_s not in df.columns:
                continue
            raw      = pd.to_numeric(df[col_s], errors="coerce").fillna(0)
            div_mean = df.groupby("division")[col_s].transform(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            ).fillna(raw.mean())
            df[col_s] = (n * raw + lam * div_mean) / (n + lam)
    return df


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
    For every (fight_id, fighter_id) pair compute two *pre-fight* rolling stats:
      - recent_win_rate    : fraction of last `window` fights won (0-1)
      - recent_finish_rate : fraction of last `window` fights ended by KO/TKO or Sub (0-1)

    shift(1) is applied so the current fight is never included -- no leakage.

    Returns a DataFrame with columns:
        fight_id | fighter_id | recent_win_rate | recent_finish_rate
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

    return long[["fight_id", "fighter_id", "recent_win_rate", "recent_finish_rate"]]


# ═══════════════════════════════════════════════════════════════════════════════
# Finish-method rate features
# ═══════════════════════════════════════════════════════════════════════════════

def compute_finish_rates(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    For every (fight_id, fighter_id) pair compute three pre-fight cumulative rates:
      - ko_rate  : fraction of career wins by KO/TKO  (0-1)
      - sub_rate : fraction of career wins by Submission (0-1)
      - dec_rate : fraction of career wins by Decision (0-1)

    shift(1) ensures the current fight result is never included.
    Fighters with zero wins before a fight have all rates set to 0.
    """
    log.info("Computing finish-method rates...")

    df = pd.read_sql_query(
        """
        SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id, method
        FROM fights
        ORDER BY date ASC, fight_id ASC
        """,
        conn,
    )

    long_rows = []
    for _, row in df.iterrows():
        method_cls = FINISH_METHOD_MAP.get(row["method"], -1)
        for fighter_id, is_winner in [
            (row["r_fighter_id"], row["winner_id"] == row["r_fighter_id"]),
            (row["b_fighter_id"], row["winner_id"] == row["b_fighter_id"]),
        ]:
            long_rows.append({
                "fight_id":   row["fight_id"],
                "date":       row["date"],
                "fighter_id": fighter_id,
                "won":        int(is_winner),
                "ko_win":     int(is_winner and method_cls == 1),
                "sub_win":    int(is_winner and method_cls == 2),
                "dec_win":    int(is_winner and method_cls == 0),
            })

    long = pd.DataFrame(long_rows)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    grp = long.groupby("fighter_id", sort=False)
    for col in ("won", "ko_win", "sub_win", "dec_win"):
        long[f"cum_{col}"] = grp[col].transform(lambda s: s.shift(1).cumsum().fillna(0))

    wins = long["cum_won"]
    long["ko_rate"]  = long["cum_ko_win"]  / wins.clip(lower=1)
    long["sub_rate"] = long["cum_sub_win"] / wins.clip(lower=1)
    long["dec_rate"] = long["cum_dec_win"] / wins.clip(lower=1)
    no_wins_mask = wins == 0
    for col in ("ko_rate", "sub_rate", "dec_rate"):
        long.loc[no_wins_mask, col] = 0.0

    return long[["fight_id", "fighter_id", "ko_rate", "sub_rate", "dec_rate"]]


# ═══════════════════════════════════════════════════════════════════════════════
# Inactivity features
# ═══════════════════════════════════════════════════════════════════════════════

def compute_inactivity(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    For every (fight_id, fighter_id) pair compute days since the fighter's
    previous fight (pre-fight inactivity gap).

    First career appearance has no prior date — filled with the dataset median.
    """
    log.info("Computing inactivity features...")

    df = pd.read_sql_query(
        """
        SELECT fight_id, date, r_fighter_id, b_fighter_id
        FROM fights
        ORDER BY date ASC, fight_id ASC
        """,
        conn,
    )

    long_rows = []
    for _, row in df.iterrows():
        for fid in [row["r_fighter_id"], row["b_fighter_id"]]:
            long_rows.append({
                "fight_id":   row["fight_id"],
                "date":       row["date"],
                "fighter_id": fid,
            })

    long = pd.DataFrame(long_rows)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    grp = long.groupby("fighter_id", sort=False)
    long["prev_date"] = grp["date"].transform(lambda s: s.shift(1))
    long["days_since_last"] = (long["date"] - long["prev_date"]).dt.days

    median_days = long["days_since_last"].median()
    long["days_since_last"] = long["days_since_last"].fillna(median_days)

    return long[["fight_id", "fighter_id", "days_since_last"]]


# ═══════════════════════════════════════════════════════════════════════════════
# Strength-of-schedule (SOS) features
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sos_features(df_fights: pd.DataFrame, window: int = SOS_WINDOW) -> pd.DataFrame:
    """
    For every (fight_id, fighter_id) pair compute the rolling average ELO of
    the last `window` opponents before this fight (strength of schedule).

    df_fights must contain: fight_id, date, r_fighter_id, b_fighter_id,
                            elo_red, elo_blue.

    Red fighter's opponent ELO = elo_blue (blue fighter's pre-fight per-division ELO).
    Blue fighter's opponent ELO = elo_red.

    shift(1) ensures the current opponent's ELO is not counted.
    """
    log.info("Computing SOS features (window=%d)...", window)

    long_rows = []
    for _, row in df_fights.iterrows():
        long_rows.append({
            "fight_id":   row["fight_id"],
            "date":       row["date"],
            "fighter_id": row["r_fighter_id"],
            "opp_elo":    row["elo_blue"],
        })
        long_rows.append({
            "fight_id":   row["fight_id"],
            "date":       row["date"],
            "fighter_id": row["b_fighter_id"],
            "opp_elo":    row["elo_red"],
        })

    long = pd.DataFrame(long_rows)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    grp = long.groupby("fighter_id", sort=False)
    long["sos"] = grp["opp_elo"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long["sos"] = long["sos"].fillna(STARTING_ELO)

    return long[["fight_id", "fighter_id", "sos"]]


# ═══════════════════════════════════════════════════════════════════════════════
# KO vulnerability
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ko_vulnerability(
    conn: sqlite3.Connection, window: int = KO_VULN_WINDOW
) -> pd.DataFrame:
    """
    For every (fight_id, fighter_id) pair compute the number of times the
    fighter was stopped by KO/TKO in their last `window` fights (as the loser).

    Distinct from ko_rate (which measures KO wins) — this captures chin damage:
    a fighter who has been knocked out recently is more susceptible going forward.

    shift(1) ensures the current fight result is never included.
    """
    log.info("Computing KO vulnerability (window=%d)...", window)

    df = pd.read_sql_query(
        """
        SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id, method
        FROM fights
        ORDER BY date ASC, fight_id ASC
        """,
        conn,
    )

    long_rows = []
    for _, row in df.iterrows():
        method_cls = FINISH_METHOD_MAP.get(row["method"], -1)
        for fighter_id, is_winner in [
            (row["r_fighter_id"], row["winner_id"] == row["r_fighter_id"]),
            (row["b_fighter_id"], row["winner_id"] == row["b_fighter_id"]),
        ]:
            # ko_stopped = fighter LOST by KO/TKO
            ko_stopped = int(not is_winner and method_cls == 1)
            long_rows.append({
                "fight_id":   row["fight_id"],
                "date":       row["date"],
                "fighter_id": fighter_id,
                "ko_stopped": ko_stopped,
            })

    long = pd.DataFrame(long_rows)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    grp = long.groupby("fighter_id", sort=False)
    long["ko_vuln"] = grp["ko_stopped"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    ).fillna(0)

    return long[["fight_id", "fighter_id", "ko_vuln"]]


# ═══════════════════════════════════════════════════════════════════════════════
# EWMA accuracy and variance
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ewma_stats(
    conn: sqlite3.Connection, span: int = EWMA_SPAN
) -> pd.DataFrame:
    """
    For every (fight_id, fighter_id) pair compute:
      - ewma_str_acc  : EWMA of per-fight sig-strike accuracy (landed/attempted)
      - ewma_td_acc   : EWMA of per-fight TD accuracy
      - str_acc_var   : rolling std of per-fight sig-strike accuracy (consistency)

    Per-fight accuracy uses raw counts (sig_str_landed / sig_str_atmpted for that
    specific fight), giving a recency-sensitive view vs the cumulative sig_str_acc.

    shift(1) ensures the current fight is never included.
    """
    log.info("Computing EWMA accuracy and variance (span=%d)...", span)

    fs = pd.read_sql_query(
        """
        SELECT fs.fight_id, fs.fighter_id,
               CAST(fs.sig_str_landed  AS REAL) AS str_land,
               CAST(fs.sig_str_atmpted AS REAL) AS str_att,
               CAST(fs.td_landed       AS REAL) AS td_land,
               CAST(fs.td_atmpted      AS REAL) AS td_att
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        ORDER BY f.date ASC, f.fight_id ASC
        """,
        conn,
    )
    dates = pd.read_sql_query(
        "SELECT fight_id, date FROM fights ORDER BY date ASC, fight_id ASC", conn
    )
    fs = fs.merge(dates, on="fight_id", how="left")
    fs["date"] = pd.to_datetime(fs["date"])

    _eps = 1e-6
    fs["pf_str_acc"] = fs["str_land"] / (fs["str_att"] + _eps)
    fs.loc[fs["str_att"] == 0, "pf_str_acc"] = 0.0
    fs["pf_td_acc"]  = fs["td_land"]  / (fs["td_att"]  + _eps)
    fs.loc[fs["td_att"]  == 0, "pf_td_acc"]  = 0.0

    fs = fs.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)
    grp = fs.groupby("fighter_id", sort=False)

    fs["ewma_str_acc"] = grp["pf_str_acc"].transform(
        lambda s: s.shift(1).ewm(span=span, min_periods=1).mean()
    ).fillna(0)
    fs["ewma_td_acc"] = grp["pf_td_acc"].transform(
        lambda s: s.shift(1).ewm(span=span, min_periods=1).mean()
    ).fillna(0)
    fs["str_acc_var"] = grp["pf_str_acc"].transform(
        lambda s: s.shift(1).rolling(span, min_periods=2).std()
    ).fillna(0)

    return fs[["fight_id", "fighter_id", "ewma_str_acc", "ewma_td_acc", "str_acc_var"]]


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
# Trajectory / momentum helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _running_win_streak(s: pd.Series) -> pd.Series:
    """Pre-fight consecutive win streak. shift(1) applied internally so the
    current fight result is excluded."""
    shifted = s.shift(1)
    out = np.zeros(len(shifted), dtype=int)
    streak = 0
    for i, v in enumerate(shifted):
        if pd.isna(v) or v == 0:
            streak = 0
        else:
            streak += 1
        out[i] = streak
    return pd.Series(out, index=s.index)


def _running_loss_streak(s: pd.Series) -> pd.Series:
    """Pre-fight consecutive loss streak. shift(1) applied internally."""
    shifted = s.shift(1)
    out = np.zeros(len(shifted), dtype=int)
    streak = 0
    for i, v in enumerate(shifted):
        if pd.isna(v) or v == 1:
            streak = 0
        else:
            streak += 1
        out[i] = streak
    return pd.Series(out, index=s.index)


def _rolling_slope(arr: np.ndarray) -> float:
    """Linear slope (via np.polyfit) of the values in arr. Used as a rolling apply fn."""
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def compute_trajectory_features(
    conn: sqlite3.Connection, window: int = TRAJECTORY_WINDOW
) -> pd.DataFrame:
    """
    For every (fight_id, fighter_id) pair compute:
      - win_streak    : consecutive wins immediately before this fight
      - loss_streak   : consecutive losses immediately before this fight
      - str_acc_slope : linear trend of per-fight sig-strike accuracy over last window fights
      - td_acc_slope  : linear trend of per-fight TD accuracy over last window fights
      - splm_slope    : linear trend of per-fight sig strikes per minute over last window fights

    All features use shift(1) -- the current fight is never included.
    Fighters with fewer than 2 prior fights get 0 for slope features.
    """
    log.info("Computing trajectory/momentum features (window=%d)...", window)

    # --- Win / loss streaks ---
    fights = pd.read_sql_query(
        """
        SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id
        FROM fights
        ORDER BY date ASC, fight_id ASC
        """,
        conn,
    )

    long_rows = []
    for _, row in fights.iterrows():
        won_r = int(row["winner_id"] == row["r_fighter_id"])
        for fighter_id, won in [
            (row["r_fighter_id"], won_r),
            (row["b_fighter_id"], 1 - won_r),
        ]:
            long_rows.append({
                "fight_id":   row["fight_id"],
                "date":       row["date"],
                "fighter_id": fighter_id,
                "won":        won,
            })

    long = pd.DataFrame(long_rows)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    grp = long.groupby("fighter_id", sort=False)
    long["win_streak"]  = grp["won"].transform(_running_win_streak)
    long["loss_streak"] = grp["won"].transform(_running_loss_streak)

    # --- Per-fight performance slopes ---
    # match_time_sec + (finish_round - 1) * 300 = per-fight duration in seconds
    fs = pd.read_sql_query(
        """
        SELECT fs.fight_id, fs.fighter_id,
               CAST(fs.sig_str_landed  AS REAL) AS str_land,
               CAST(fs.sig_str_atmpted AS REAL) AS str_att,
               CAST(fs.td_landed       AS REAL) AS td_land,
               CAST(fs.td_atmpted      AS REAL) AS td_att,
               CAST(f.match_time_sec   AS REAL) AS match_sec,
               CAST(f.finish_round     AS REAL) AS fin_round
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        ORDER BY f.date ASC, f.fight_id ASC
        """,
        conn,
    )
    dates = pd.read_sql_query(
        "SELECT fight_id, date FROM fights ORDER BY date ASC, fight_id ASC", conn
    )
    fs = fs.merge(dates, on="fight_id", how="left")
    fs["date"] = pd.to_datetime(fs["date"])

    _eps = 1e-6
    fs["pf_str_acc"] = fs["str_land"] / (fs["str_att"] + _eps)
    fs.loc[fs["str_att"] == 0, "pf_str_acc"] = 0.0
    fs["pf_td_acc"]  = fs["td_land"]  / (fs["td_att"]  + _eps)
    fs.loc[fs["td_att"]  == 0, "pf_td_acc"]  = 0.0

    fight_secs = fs["match_sec"].fillna(0) + (fs["fin_round"].fillna(1) - 1) * 300
    fight_min  = fight_secs / 60.0
    fs["pf_splm"] = fs["str_land"] / (fight_min + _eps)
    fs.loc[fight_min == 0, "pf_splm"] = 0.0

    fs = fs.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)
    fs_grp = fs.groupby("fighter_id", sort=False)

    for metric, col in [
        ("str_acc_slope", "pf_str_acc"),
        ("td_acc_slope",  "pf_td_acc"),
        ("splm_slope",    "pf_splm"),
    ]:
        fs[metric] = fs_grp[col].transform(
            lambda s, c=col: s.shift(1).rolling(window, min_periods=2).apply(
                _rolling_slope, raw=True
            )
        ).fillna(0)

    result = long[["fight_id", "fighter_id", "win_streak", "loss_streak"]].merge(
        fs[["fight_id", "fighter_id", "str_acc_slope", "td_acc_slope", "splm_slope"]],
        on=["fight_id", "fighter_id"],
        how="left",
    )
    for col in ("str_acc_slope", "td_acc_slope", "splm_slope"):
        result[col] = result[col].fillna(0)

    return result


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

    # ── Glicko-2 features ─────────────────────────────────────────────────────
    log.info("Building Glicko-2 features…")
    glicko_df = build_glicko_features(conn)
    df = df.merge(glicko_df, on="fight_id", how="left")

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

    # Note: height, reach, stance, dob, weight are already columns in fight_stats
    # (denormalized from the fighters table by rolling.py), so height_red/blue and
    # reach_red/blue are already in df and will be handled by the auto-diff loop.
    # stance_red/blue are present but excluded from the diff loop via EXCLUDE_STAT_KEYWORDS.

    # ── Finish-method rates ───────────────────────────────────────────────────
    finish_df = compute_finish_rates(conn)
    finish_red = finish_df.rename(columns={
        "ko_rate": "ko_rate_red", "sub_rate": "sub_rate_red", "dec_rate": "dec_rate_red",
    })
    finish_blue = finish_df.rename(columns={
        "ko_rate": "ko_rate_blue", "sub_rate": "sub_rate_blue", "dec_rate": "dec_rate_blue",
    })
    df = df.merge(
        finish_red[["fight_id", "fighter_id", "ko_rate_red", "sub_rate_red", "dec_rate_red"]],
        left_on=["fight_id", "r_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])
    df = df.merge(
        finish_blue[["fight_id", "fighter_id", "ko_rate_blue", "sub_rate_blue", "dec_rate_blue"]],
        left_on=["fight_id", "b_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    # ── Inactivity ────────────────────────────────────────────────────────────
    inact_df = compute_inactivity(conn)
    inact_red  = inact_df.rename(columns={"days_since_last": "days_since_last_red"})
    inact_blue = inact_df.rename(columns={"days_since_last": "days_since_last_blue"})
    df = df.merge(
        inact_red[["fight_id", "fighter_id", "days_since_last_red"]],
        left_on=["fight_id", "r_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])
    df = df.merge(
        inact_blue[["fight_id", "fighter_id", "days_since_last_blue"]],
        left_on=["fight_id", "b_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    # ── KO vulnerability ──────────────────────────────────────────────────────
    kovuln_df = compute_ko_vulnerability(conn)
    kovuln_red  = kovuln_df.rename(columns={"ko_vuln": "ko_vuln_red"})
    kovuln_blue = kovuln_df.rename(columns={"ko_vuln": "ko_vuln_blue"})
    df = df.merge(
        kovuln_red[["fight_id", "fighter_id", "ko_vuln_red"]],
        left_on=["fight_id", "r_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])
    df = df.merge(
        kovuln_blue[["fight_id", "fighter_id", "ko_vuln_blue"]],
        left_on=["fight_id", "b_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    # ── EWMA accuracy and variance ────────────────────────────────────────────
    ewma_df = compute_ewma_stats(conn)
    ewma_red  = ewma_df.rename(columns={
        "ewma_str_acc": "ewma_str_acc_red",
        "ewma_td_acc":  "ewma_td_acc_red",
        "str_acc_var":  "str_acc_var_red",
    })
    ewma_blue = ewma_df.rename(columns={
        "ewma_str_acc": "ewma_str_acc_blue",
        "ewma_td_acc":  "ewma_td_acc_blue",
        "str_acc_var":  "str_acc_var_blue",
    })
    df = df.merge(
        ewma_red[["fight_id", "fighter_id", "ewma_str_acc_red", "ewma_td_acc_red", "str_acc_var_red"]],
        left_on=["fight_id", "r_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])
    df = df.merge(
        ewma_blue[["fight_id", "fighter_id", "ewma_str_acc_blue", "ewma_td_acc_blue", "str_acc_var_blue"]],
        left_on=["fight_id", "b_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    # ── Trajectory / momentum ─────────────────────────────────────────────────
    traj_df   = compute_trajectory_features(conn)
    traj_cols = ["win_streak", "loss_streak", "str_acc_slope", "td_acc_slope", "splm_slope"]
    traj_red  = traj_df.rename(columns={c: f"{c}_red"  for c in traj_cols})
    traj_blue = traj_df.rename(columns={c: f"{c}_blue" for c in traj_cols})
    df = df.merge(
        traj_red[["fight_id", "fighter_id"] + [f"{c}_red" for c in traj_cols]],
        left_on=["fight_id", "r_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])
    df = df.merge(
        traj_blue[["fight_id", "fighter_id"] + [f"{c}_blue" for c in traj_cols]],
        left_on=["fight_id", "b_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    conn.close()

    # ── Strength of schedule (uses elo cols already in df) ────────────────────
    sos_df = compute_sos_features(
        df[["fight_id", "date", "r_fighter_id", "b_fighter_id", "elo_red", "elo_blue"]]
    )
    sos_red  = sos_df.rename(columns={"sos": "sos_red"})
    sos_blue = sos_df.rename(columns={"sos": "sos_blue"})
    df = df.merge(
        sos_red[["fight_id", "fighter_id", "sos_red"]],
        left_on=["fight_id", "r_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])
    df = df.merge(
        sos_blue[["fight_id", "fighter_id", "sos_blue"]],
        left_on=["fight_id", "b_fighter_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    # ── Shrinkage toward division mean ────────────────────────────────────────
    df = apply_shrinkage(df)

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

    # height_diff and reach_diff are computed automatically by the diff loop
    # above — fight_stats stores height and reach per row so height_red/blue
    # are already in df with no extra merge needed.

    # ── Stance matchup ────────────────────────────────────────────────────────
    # stance_red / stance_blue come from fight_stats (same denormalized source)
    stance_r = df["stance_red"].fillna("Orthodox").str.strip().str.lower()
    stance_b = df["stance_blue"].fillna("Orthodox").str.strip().str.lower()
    red_sp = stance_r == "southpaw"
    blue_sp = stance_b == "southpaw"
    red_or = stance_r == "orthodox"
    blue_or = stance_b == "orthodox"
    # +1 if red has southpaw advantage, -1 if blue does, 0 otherwise
    ml_data["southpaw_adv_diff"] = (
        (red_sp & blue_or).astype(int) - (red_or & blue_sp).astype(int)
    ).values
    # Mirror-stance signal (both southpaw; symmetric so no _diff needed)
    ml_data["both_southpaw"] = (red_sp & blue_sp).astype(int).values

    # ── Finish-method rate differences ────────────────────────────────────────
    for stat in ("ko_rate", "sub_rate", "dec_rate"):
        r_col = pd.to_numeric(df[f"{stat}_red"],  errors="coerce").fillna(0)
        b_col = pd.to_numeric(df[f"{stat}_blue"], errors="coerce").fillna(0)
        ml_data[f"{stat}_diff"] = (r_col - b_col).values

    # ── Inactivity difference ─────────────────────────────────────────────────
    inact_r = pd.to_numeric(df["days_since_last_red"],  errors="coerce").fillna(365)
    inact_b = pd.to_numeric(df["days_since_last_blue"], errors="coerce").fillna(365)
    ml_data["days_since_last_diff"] = (inact_r - inact_b).values

    # ── Strength-of-schedule difference ──────────────────────────────────────
    sos_r = pd.to_numeric(df["sos_red"],  errors="coerce").fillna(STARTING_ELO)
    sos_b = pd.to_numeric(df["sos_blue"], errors="coerce").fillna(STARTING_ELO)
    ml_data["sos_diff"] = (sos_r - sos_b).values

    # ── KO vulnerability difference ───────────────────────────────────────────
    ko_vuln_r = pd.to_numeric(df["ko_vuln_red"],  errors="coerce").fillna(0)
    ko_vuln_b = pd.to_numeric(df["ko_vuln_blue"], errors="coerce").fillna(0)
    ml_data["ko_vuln_diff"] = (ko_vuln_r - ko_vuln_b).values

    # ── EWMA accuracy differences ─────────────────────────────────────────────
    for stat in ("ewma_str_acc", "ewma_td_acc", "str_acc_var"):
        r_col = pd.to_numeric(df[f"{stat}_red"],  errors="coerce").fillna(0)
        b_col = pd.to_numeric(df[f"{stat}_blue"], errors="coerce").fillna(0)
        ml_data[f"{stat}_diff"] = (r_col - b_col).values

    # ── Trajectory / momentum differences ────────────────────────────────────
    for stat in ("win_streak", "loss_streak", "str_acc_slope", "td_acc_slope", "splm_slope"):
        r_col = pd.to_numeric(df[f"{stat}_red"],  errors="coerce").fillna(0)
        b_col = pd.to_numeric(df[f"{stat}_blue"], errors="coerce").fillna(0)
        ml_data[f"{stat}_diff"] = (r_col - b_col).values

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
