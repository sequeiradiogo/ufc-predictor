"""
ML_data_preparation_v1.py -- Build the v1 (mdabbert) feature CSV.

Uses ufc_v2.db (career-aggregate fight_stats) and applies the same
feature improvements as v2 PRs 36, 46, 50, 52:
  PR 36: stance matchup, finish rates, inactivity, KO vulnerability, SOS,
         corrected symmetry augmentation for striker/wrestler interaction terms
  PR 46: calibrated ensemble (handled in train_v1_models.py)
  PR 46: slope features -- slope of career avg_sig_str_pct, splm, avg_td_pct
         over the last TRAJECTORY_WINDOW fights (slope of the running career
         average, not per-fight values -- still a valid trajectory signal)
  PR 50: age_diff removed (dead feature)
  PR 52: Glicko-2 (glicko_diff, glicko_rd_diff)

Features NOT ported (require per-fight raw counts -- unavailable in v1 DB):
  ewma_str_acc_diff, ewma_td_acc_diff, str_acc_var_diff

v1 already covers win/loss streaks via career_win_streak_diff / career_lose_streak_diff.

Output: ml/ufc_ml_data_v1.csv

Usage:
    python ml/ML_data_preparation_v1.py
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    CSV_V1_WITH_ELO,
    DB_PATH,
    DB_V1_PATH,
    DIVISIONS,
    FINISH_METHOD_MAP,
    META_COLS,
    MIN_FIGHT_DATE,
    RANDOM_STATE,
    RECENT_FORM_WINDOW,
    SAMPLE_WEIGHT_ALPHA,
    STARTING_ELO,
    TARGET_COL,
    TRAIN_TEST_SPLIT,
    TRAJECTORY_WINDOW,
)
from ml.ELO_calculator import build_elo_features, build_glicko_features
from ml.ML_data_preparation import (
    _rolling_slope,
    compute_finish_rates,
    compute_inactivity,
    compute_ko_vulnerability,
    compute_sample_weights,
    compute_sos_features,
)
from utils.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-6

# Career-aggregate columns from v1 fight_stats to diff
_STAT_COLS = [
    "wins", "losses",
    "career_win_streak", "career_lose_streak", "longest_win_streak",
    "total_rounds_fought", "total_title_bouts",
    "avg_sig_str_pct", "avg_sub_att", "avg_td_pct",
    "height", "reach",
    "splm", "td_avg",
    "win_by_ko", "win_by_sub", "win_by_dec_unanimous", "win_by_dec_split",
    # age_diff excluded per permutation importance (PR 50)
]


# ── Recent form ───────────────────────────────────────────────────────────────

def _compute_recent_form_v1(
    conn: sqlite3.Connection,
    window: int = RECENT_FORM_WINDOW,
) -> pd.DataFrame:
    """
    Compute pre-fight recent win rate and finish rate per (fight_id, fighter_id).

    Uses the same fights table as v2 -- leakage-free via shift(1) approach.
    """
    log.info("Computing recent form (window=%d)...", window)

    df = pd.read_sql_query(
        """
        SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id, method
        FROM fights
        ORDER BY date ASC, fight_id ASC
        """,
        conn,
    )

    finish_methods = {m for m, cls in FINISH_METHOD_MAP.items() if cls > 0}

    long_rows = []
    for _, row in df.iterrows():
        for fid, is_win in [
            (row["r_fighter_id"], row["winner_id"] == row["r_fighter_id"]),
            (row["b_fighter_id"], row["winner_id"] == row["b_fighter_id"]),
        ]:
            long_rows.append({
                "fight_id":   row["fight_id"],
                "date":       row["date"],
                "fighter_id": fid,
                "won":        int(is_win),
                "finished":   int(row["method"] in finish_methods),
            })

    long = pd.DataFrame(long_rows)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    grp = long.groupby("fighter_id", sort=False)
    long["recent_win_rate"] = grp["won"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).fillna(0)
    long["recent_finish_rate"] = grp["finished"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).fillna(0)

    return long[["fight_id", "fighter_id", "recent_win_rate", "recent_finish_rate"]]


# ── Slope features (PR 46 equivalent for v1) ─────────────────────────────────

def compute_slope_features_v1(
    conn: sqlite3.Connection, window: int = TRAJECTORY_WINDOW
) -> pd.DataFrame:
    """
    For every (fight_id, fighter_id) pair compute the linear slope of the
    fighter's career-average striking accuracy, TD accuracy, and SPLM over
    the last `window` fights.

    v1 fight_stats stores career averages as pre-fight snapshots, so these
    are already leakage-free. shift(1) ensures the current fight's snapshot
    is not included in the window used to compute the slope going INTO it.

    Fighters with fewer than 2 prior fights get slope=0.
    """
    log.info("Computing slope features (window=%d)...", window)

    fs = pd.read_sql_query("SELECT fight_id, fighter_id, avg_sig_str_pct, splm, avg_td_pct FROM fight_stats", conn)
    dates = pd.read_sql_query("SELECT fight_id, date FROM fights ORDER BY date ASC, fight_id ASC", conn)
    fs = fs.merge(dates, on="fight_id", how="left")
    fs["date"] = pd.to_datetime(fs["date"])
    for col in ("avg_sig_str_pct", "splm", "avg_td_pct"):
        fs[col] = pd.to_numeric(fs[col], errors="coerce").fillna(0)

    fs = fs.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)
    grp = fs.groupby("fighter_id", sort=False)

    for metric, col in [
        ("str_acc_slope", "avg_sig_str_pct"),
        ("splm_slope",    "splm"),
        ("td_acc_slope",  "avg_td_pct"),
    ]:
        fs[metric] = grp[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).apply(_rolling_slope, raw=True)
        ).fillna(0)

    return fs[["fight_id", "fighter_id", "str_acc_slope", "splm_slope", "td_acc_slope"]]


# ── v2 defensive metrics enrichment ──────────────────────────────────────────

def _asof_defensive(corner_df: pd.DataFrame, v2_fs: pd.DataFrame) -> pd.DataFrame:
    """
    For each (fight_id, date, v2_fighter_id) in corner_df return the most
    recent v2 sapm/str_def/td_def row with date <= fight date.
    """
    results = []
    v2_by_fighter = {fid: grp for fid, grp in v2_fs.groupby("v2_fighter_id")}

    for v2id, grp_corner in corner_df.groupby("v2_fighter_id"):
        grp_v2 = v2_by_fighter.get(v2id)
        if grp_v2 is None:
            continue
        grp_corner_s = grp_corner[["fight_id", "date"]].sort_values("date")
        grp_v2_s = grp_v2[["date", "sapm", "str_def", "td_def"]].sort_values("date")
        m = pd.merge_asof(grp_corner_s, grp_v2_s, on="date", direction="backward")
        results.append(m)

    if not results:
        return pd.DataFrame(columns=["fight_id", "sapm", "str_def", "td_def"])
    return pd.concat(results, ignore_index=True)[["fight_id", "sapm", "str_def", "td_def"]]


def enrich_from_v2(wide: pd.DataFrame, v1_conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Add pre-fight sapm, str_def, td_def from v2 (UFCStats) DB to wide DF,
    matched by fighter name.  Unmatched fighters get 0 for all three metrics.
    """
    if not DB_PATH.exists():
        log.warning("v2 DB not found -- skipping sapm/str_def/td_def enrichment.")
        for corner in ("red", "blue"):
            for col in ("sapm", "str_def", "td_def"):
                wide[f"{col}_{corner}"] = 0.0
        return wide

    v1_fighters = pd.read_sql("SELECT fighter_id, name FROM fighters", v1_conn)
    v1_id_to_name = dict(zip(v1_fighters["fighter_id"], v1_fighters["name"]))

    v2_conn = sqlite3.connect(str(DB_PATH))
    v2_fighters = pd.read_sql("SELECT fighter_id, name FROM fighters", v2_conn)
    name_to_v2id = dict(zip(v2_fighters["name"], v2_fighters["fighter_id"]))

    v1_to_v2 = {
        v1id: name_to_v2id[name]
        for v1id, name in v1_id_to_name.items()
        if name in name_to_v2id
    }
    log.info(
        "Name-matched %d/%d v1 fighters to v2 DB for sapm/str_def/td_def.",
        len(v1_to_v2), len(v1_id_to_name),
    )

    v2_fs = pd.read_sql(
        """
        SELECT fs.fighter_id AS v2_fighter_id,
               f.date,
               CAST(fs.sapm    AS REAL) AS sapm,
               CAST(fs.str_def AS REAL) AS str_def,
               CAST(fs.td_def  AS REAL) AS td_def
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        ORDER BY f.date ASC, f.fight_id ASC
        """,
        v2_conn,
    )
    v2_conn.close()

    v2_fs["date"] = pd.to_datetime(v2_fs["date"])
    for col in ("sapm", "str_def", "td_def"):
        v2_fs[col] = v2_fs[col].fillna(0.0)

    wide["date"] = pd.to_datetime(wide["date"])

    for corner, fid_col in [("red", "r_fighter_id"), ("blue", "b_fighter_id")]:
        corner_df = wide[["fight_id", "date", fid_col]].copy()
        corner_df["v2_fighter_id"] = corner_df[fid_col].map(v1_to_v2)
        matched = corner_df.dropna(subset=["v2_fighter_id"])

        lookup = _asof_defensive(matched, v2_fs)
        if not lookup.empty:
            lookup = lookup.set_index("fight_id")
        for col in ("sapm", "str_def", "td_def"):
            wide[f"{col}_{corner}"] = (
                wide["fight_id"].map(lookup[col] if not lookup.empty else pd.Series(dtype=float)).fillna(0.0)
            )

    return wide


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_v1_dataset(conn: sqlite3.Connection, min_date: str | None = None) -> pd.DataFrame:
    """
    Build the full v1 ML feature DataFrame (one row per fight, pre-augmentation).

    Returns a DataFrame with META_COLS + all feature columns.
    Does NOT apply symmetry augmentation -- that is done per-fold during training.
    """
    log.info("Loading fights and fight_stats from v1 DB...")

    fights = pd.read_sql_query(
        """
        SELECT fight_id, date, division,
               r_fighter_id, b_fighter_id, winner_id, method, title_fight
        FROM fights
        ORDER BY date ASC, fight_id ASC
        """,
        conn,
    )
    fights["date"] = pd.to_datetime(fights["date"])

    fs = pd.read_sql_query("SELECT * FROM fight_stats", conn)

    r_stats = (
        fs[fs["corner"] == "r"]
        .drop(columns=["corner", "fighter_id"])
        .rename(columns=lambda c: f"r_{c}" if c != "fight_id" else c)
    )
    b_stats = (
        fs[fs["corner"] == "b"]
        .drop(columns=["corner", "fighter_id"])
        .rename(columns=lambda c: f"b_{c}" if c != "fight_id" else c)
    )

    wide = (
        fights
        .merge(r_stats, on="fight_id", how="left")
        .merge(b_stats, on="fight_id", how="left")
    )

    # ── ELO ──────────────────────────────────────────────────────────────────
    log.info("Building ELO features...")
    elo_df = build_elo_features(conn)
    wide = wide.merge(elo_df, on="fight_id", how="left")

    # ── Glicko-2 (PR 52) ─────────────────────────────────────────────────────
    log.info("Building Glicko-2 features...")
    glicko_df = build_glicko_features(conn)
    wide = wide.merge(glicko_df, on="fight_id", how="left")

    # ── Recent form ───────────────────────────────────────────────────────────
    form_df = _compute_recent_form_v1(conn)

    form_red  = form_df.rename(columns={
        "recent_win_rate":    "recent_win_rate_red",
        "recent_finish_rate": "recent_finish_rate_red",
    })
    form_blue = form_df.rename(columns={
        "recent_win_rate":    "recent_win_rate_blue",
        "recent_finish_rate": "recent_finish_rate_blue",
    })
    wide = wide.merge(
        form_red[["fight_id", "fighter_id", "recent_win_rate_red", "recent_finish_rate_red"]],
        left_on=["fight_id", "r_fighter_id"], right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])
    wide = wide.merge(
        form_blue[["fight_id", "fighter_id", "recent_win_rate_blue", "recent_finish_rate_blue"]],
        left_on=["fight_id", "b_fighter_id"], right_on=["fight_id", "fighter_id"],
        how="left",
    ).drop(columns=["fighter_id"])

    # ── Finish rates (PR 36) ─────────────────────────────────────────────────
    fin_df = compute_finish_rates(conn)
    for corner, fid_col in [("red", "r_fighter_id"), ("blue", "b_fighter_id")]:
        suffix = f"_{corner}"
        fin_renamed = fin_df.rename(columns={
            "ko_rate":  f"ko_rate{suffix}",
            "sub_rate": f"sub_rate{suffix}",
            "dec_rate": f"dec_rate{suffix}",
        })
        wide = wide.merge(
            fin_renamed[["fight_id", "fighter_id", f"ko_rate{suffix}", f"sub_rate{suffix}", f"dec_rate{suffix}"]],
            left_on=["fight_id", fid_col], right_on=["fight_id", "fighter_id"],
            how="left",
        ).drop(columns=["fighter_id"])

    # ── Inactivity (PR 36) ────────────────────────────────────────────────────
    inact_df = compute_inactivity(conn)
    for corner, fid_col in [("red", "r_fighter_id"), ("blue", "b_fighter_id")]:
        suffix = f"_{corner}"
        inact_renamed = inact_df.rename(columns={"days_since_last": f"days_since_last{suffix}"})
        wide = wide.merge(
            inact_renamed[["fight_id", "fighter_id", f"days_since_last{suffix}"]],
            left_on=["fight_id", fid_col], right_on=["fight_id", "fighter_id"],
            how="left",
        ).drop(columns=["fighter_id"])

    # ── KO vulnerability (PR 36) ──────────────────────────────────────────────
    kovuln_df = compute_ko_vulnerability(conn)
    for corner, fid_col in [("red", "r_fighter_id"), ("blue", "b_fighter_id")]:
        suffix = f"_{corner}"
        kv_renamed = kovuln_df.rename(columns={"ko_vuln": f"ko_vuln{suffix}"})
        wide = wide.merge(
            kv_renamed[["fight_id", "fighter_id", f"ko_vuln{suffix}"]],
            left_on=["fight_id", fid_col], right_on=["fight_id", "fighter_id"],
            how="left",
        ).drop(columns=["fighter_id"])

    # ── Slope features (PR 46 equivalent) ────────────────────────────────────
    slope_df = compute_slope_features_v1(conn)
    slope_cols = ["str_acc_slope", "splm_slope", "td_acc_slope"]
    for corner, fid_col in [("red", "r_fighter_id"), ("blue", "b_fighter_id")]:
        suffix = f"_{corner}"
        sl_renamed = slope_df.rename(columns={c: f"{c}{suffix}" for c in slope_cols})
        wide = wide.merge(
            sl_renamed[["fight_id", "fighter_id"] + [f"{c}{suffix}" for c in slope_cols]],
            left_on=["fight_id", fid_col], right_on=["fight_id", "fighter_id"],
            how="left",
        ).drop(columns=["fighter_id"])

    # ── SOS (PR 36) -- uses elo_red/blue already in wide ─────────────────────
    sos_df = compute_sos_features(
        wide[["fight_id", "date", "r_fighter_id", "b_fighter_id", "elo_red", "elo_blue"]]
    )
    for corner, fid_col in [("red", "r_fighter_id"), ("blue", "b_fighter_id")]:
        suffix = f"_{corner}"
        sos_renamed = sos_df.rename(columns={"sos": f"sos{suffix}"})
        wide = wide.merge(
            sos_renamed[["fight_id", "fighter_id", f"sos{suffix}"]],
            left_on=["fight_id", fid_col], right_on=["fight_id", "fighter_id"],
            how="left",
        ).drop(columns=["fighter_id"])

    # ── v2 defensive metrics (sapm, str_def, td_def) via name matching ────────
    wide = enrich_from_v2(wide, conn)

    # ── Build output ──────────────────────────────────────────────────────────
    log.info("Building diff and derived features...")

    ml = pd.DataFrame()
    ml["fight_id"] = wide["fight_id"]
    ml["date"]     = wide["date"]
    ml["division"] = wide["division"]
    ml["target"]   = (wide["winner_id"] == wide["r_fighter_id"]).astype(int)

    # Career-aggregate diff features
    for col in _STAT_COLS:
        r_vals = pd.to_numeric(wide.get(f"r_{col}", 0), errors="coerce").fillna(0)
        b_vals = pd.to_numeric(wide.get(f"b_{col}", 0), errors="coerce").fillna(0)
        ml[f"{col}_diff"] = r_vals - b_vals

    # Rating diffs
    ml["elo_diff"]      = wide["elo_red"].fillna(STARTING_ELO)  - wide["elo_blue"].fillna(STARTING_ELO)
    ml["glicko_diff"]   = wide["glicko_red"].fillna(1500)        - wide["glicko_blue"].fillna(1500)
    ml["glicko_rd_diff"]= wide["glicko_rd_red"].fillna(350)      - wide["glicko_rd_blue"].fillna(350)

    # Recent form diffs
    for stat in ("recent_win_rate", "recent_finish_rate"):
        r_col = wide[f"{stat}_red"].fillna(0)
        b_col = wide[f"{stat}_blue"].fillna(0)
        ml[f"{stat}_diff"] = (r_col - b_col).values

    # Style ratios
    r_splm   = pd.to_numeric(wide.get("r_splm",   0), errors="coerce").fillna(0)
    r_td_avg = pd.to_numeric(wide.get("r_td_avg", 0), errors="coerce").fillna(0)
    b_splm   = pd.to_numeric(wide.get("b_splm",   0), errors="coerce").fillna(0)
    b_td_avg = pd.to_numeric(wide.get("b_td_avg", 0), errors="coerce").fillna(0)

    r_denom = r_splm + r_td_avg + _EPS
    b_denom = b_splm + b_td_avg + _EPS
    r_grapple = r_td_avg / r_denom
    r_strike  = r_splm   / r_denom
    b_grapple = b_td_avg / b_denom
    b_strike  = b_splm   / b_denom

    ml["grapple_ratio_diff"]  = r_grapple - b_grapple
    ml["striker_vs_wrestler"] = r_strike  * b_grapple
    ml["wrestler_vs_striker"] = r_grapple * b_strike

    # Stance matchup (PR 36) -- fight_stats already has stance per row
    stance_r = wide["r_stance"].fillna("Orthodox").str.strip().str.lower()
    stance_b = wide["b_stance"].fillna("Orthodox").str.strip().str.lower()
    red_sp  = stance_r == "southpaw"
    blue_sp = stance_b == "southpaw"
    red_or  = stance_r == "orthodox"
    blue_or = stance_b == "orthodox"
    ml["southpaw_adv_diff"] = ((red_sp & blue_or).astype(int) - (red_or & blue_sp).astype(int)).values
    ml["both_southpaw"]     = (red_sp & blue_sp).astype(int).values

    # Finish rate diffs (PR 36)
    for stat in ("ko_rate", "sub_rate", "dec_rate"):
        r_col = pd.to_numeric(wide[f"{stat}_red"],  errors="coerce").fillna(0)
        b_col = pd.to_numeric(wide[f"{stat}_blue"], errors="coerce").fillna(0)
        ml[f"{stat}_diff"] = (r_col - b_col).values

    # Inactivity (PR 36)
    inact_r = pd.to_numeric(wide["days_since_last_red"],  errors="coerce").fillna(365)
    inact_b = pd.to_numeric(wide["days_since_last_blue"], errors="coerce").fillna(365)
    ml["days_since_last_diff"] = (inact_r - inact_b).values

    # SOS (PR 36)
    sos_r = pd.to_numeric(wide["sos_red"],  errors="coerce").fillna(STARTING_ELO)
    sos_b = pd.to_numeric(wide["sos_blue"], errors="coerce").fillna(STARTING_ELO)
    ml["sos_diff"] = (sos_r - sos_b).values

    # KO vulnerability (PR 36)
    kv_r = pd.to_numeric(wide["ko_vuln_red"],  errors="coerce").fillna(0)
    kv_b = pd.to_numeric(wide["ko_vuln_blue"], errors="coerce").fillna(0)
    ml["ko_vuln_diff"] = (kv_r - kv_b).values

    # Slope features (PR 46 equivalent for v1)
    for stat in ("str_acc_slope", "splm_slope", "td_acc_slope"):
        r_col = pd.to_numeric(wide[f"{stat}_red"],  errors="coerce").fillna(0)
        b_col = pd.to_numeric(wide[f"{stat}_blue"], errors="coerce").fillna(0)
        ml[f"{stat}_diff"] = (r_col - b_col).values

    # v2 defensive metrics (sapm, str_def, td_def)
    for stat in ("sapm", "str_def", "td_def"):
        r_col = pd.to_numeric(wide[f"{stat}_red"],  errors="coerce").fillna(0)
        b_col = pd.to_numeric(wide[f"{stat}_blue"], errors="coerce").fillna(0)
        ml[f"{stat}_diff"] = (r_col - b_col).values

    # UFC ranking differential (unranked encoded as 16)
    _UNRANKED = 16.0
    r_rank = pd.to_numeric(wide.get("r_weightclass_rank", 0), errors="coerce").fillna(0)
    b_rank = pd.to_numeric(wide.get("b_weightclass_rank", 0), errors="coerce").fillna(0)
    r_rank_enc = r_rank.where(r_rank > 0, _UNRANKED)
    b_rank_enc = b_rank.where(b_rank > 0, _UNRANKED)
    ml["weightclass_rank_diff"] = (r_rank_enc - b_rank_enc).values

    # Division one-hot
    def _div_col(name: str) -> str:
        return "div_" + name.replace(" ", "_").replace("'", "")

    div_lower = wide["division"].str.lower().str.strip().fillna("")
    for div in DIVISIONS:
        ml[_div_col(div)] = (div_lower == div).astype(int).values

    ml["title_fight"] = pd.to_numeric(wide["title_fight"], errors="coerce").fillna(0).astype(int).values

    # ── Exclusion filters ─────────────────────────────────────────────────────
    n_before = len(ml)

    cutoff = min_date or MIN_FIGHT_DATE
    ml = ml[pd.to_datetime(ml["date"]) >= cutoff].copy()
    n_date = n_before - len(ml)

    # Debut filter: total_fight_time == 0 in v1 means no prior recorded fights
    r_time = pd.to_numeric(wide["r_total_fight_time"], errors="coerce").fillna(0)
    b_time = pd.to_numeric(wide["b_total_fight_time"], errors="coerce").fillna(0)
    debut_mask = pd.Series((r_time.values == 0) | (b_time.values == 0), index=wide.index)
    ml = ml[~debut_mask.reindex(ml.index, fill_value=False)].copy()
    n_debut = (n_before - n_date) - len(ml)

    log.info(
        "Filtered: %d pre-%s | %d debuts | %d rows remaining",
        n_date, cutoff, n_debut, len(ml),
    )

    return ml


# ── Symmetry augmentation (with PR 36 bug fix) ───────────────────────────────

def make_symmetric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip Red <-> Blue to double training data and remove corner bias.
    Applied per-fold during training, NOT in the saved CSV.

    Fixes PR 36 bug: striker_vs_wrestler and wrestler_vs_striker
    must be swapped (not just negated) when corners are flipped.
    """
    df_flip = df.copy()
    df_flip[TARGET_COL] = 1 - df[TARGET_COL]
    diff_cols = [c for c in df.columns if "_diff" in c]
    df_flip[diff_cols] = df_flip[diff_cols] * -1
    if "striker_vs_wrestler" in df.columns and "wrestler_vs_striker" in df.columns:
        df_flip["striker_vs_wrestler"] = df["wrestler_vs_striker"]
        df_flip["wrestler_vs_striker"] = df["striker_vs_wrestler"]
    return pd.concat([df, df_flip], ignore_index=True).sort_values("date").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v1 feature CSV.")
    parser.add_argument(
        "--min-date",
        default=None,
        metavar="YYYY-MM-DD",
        help=f"Earliest fight date to include (default: config.MIN_FIGHT_DATE={MIN_FIGHT_DATE}).",
    )
    args = parser.parse_args()

    if not DB_V1_PATH.exists():
        log.error("v1 DB not found at '%s'.", DB_V1_PATH)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_V1_PATH))
    df   = build_v1_dataset(conn, min_date=args.min_date)
    conn.close()

    feature_cols = [c for c in df.columns if c not in META_COLS]
    log.info("Feature set: %d features", len(feature_cols))
    log.info("Feature names: %s", feature_cols)

    CSV_V1_WITH_ELO.parent.mkdir(exist_ok=True)
    df.to_csv(CSV_V1_WITH_ELO, index=False)
    log.info("Saved %d rows to %s", len(df), CSV_V1_WITH_ELO)

    # Quick sanity check
    n_test = int(len(df) * (1 - TRAIN_TEST_SPLIT))
    log.info(
        "Train/test split: %d train | %d test (%.0f%% split)",
        len(df) - n_test, n_test, (1 - TRAIN_TEST_SPLIT) * 100,
    )


if __name__ == "__main__":
    main()
