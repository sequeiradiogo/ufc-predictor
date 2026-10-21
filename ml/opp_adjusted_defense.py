"""
opp_adjusted_defense.py -- Striking defence measured against what each opponent usually does.

For every prior fight j of fighter F against opponent O_j:

    str_def residual = strikes F absorbed per minute in fight j
                       - O_j's pre-fight sig. strikes landed per minute
    acc_def residual = O_j's accuracy against F in fight j
                       - O_j's pre-fight accuracy

F's feature before a fight is the minutes-weighted mean of those residuals over
every earlier fight, shrunk toward 0 with RESID_PSEUDO_MIN pseudo-minutes
(negative = F absorbs less than his opponents usually land = good defence).
Opponent rates are themselves shrunk toward the league average, so a debuting
opponent counts as league-average instead of as zero.

str_def / sapm / head_def already describe defence, but not relative to who the
fighter faced; the existing opp_adj_* features only adjust offence. Added
2026-10-21: on the 2025-02+ test split, 9 seeds, adding adj_str_def_diff +
adj_acc_def_diff improved ensemble hold-out log-loss by 0.002 (7/9 seeds) and
base-model test log-loss by 0.003 (9/9).

Training (scripts/add_computed_features_to_csv.py -> prefight_adj_defense) and
live inference (predict.py -> current_adj_defense) both go through
_fight_rows / _with_residuals below, so the two can only differ by which
fights are in the DB. Reads the UFCStats DB (fights / fight_stats).
"""

import sqlite3

import numpy as np
import pandas as pd

# League-wide rates over the UFCStats history (1994-2026), frozen so training
# and inference -- which sees a DB truncated at the event date -- shrink toward
# the same constant.
LEAGUE_SLPM = 3.51
LEAGUE_STR_ACC = 0.449

RATE_PSEUDO_MIN = 15.0    # opponent's strikes-per-minute prior, in minutes
ACC_PSEUDO_ATT = 30.0     # opponent's accuracy prior, in attempts
RESID_PSEUDO_MIN = 15.0   # fighter's mean-residual prior (toward 0), in minutes

FEATURES = ("adj_str_def", "adj_acc_def")


def _fight_rows(conn: sqlite3.Connection, before_date: str | None = None) -> pd.DataFrame:
    """One row per (fighter, fight) with the opponent's counts, fights before *before_date*."""
    where = "WHERE f.date < ?" if before_date else ""
    params = (str(before_date)[:10],) if before_date else ()
    df = pd.read_sql_query(
        f"""
        SELECT fs.fighter_id, f.fight_id, f.date,
               (CAST(f.match_time_sec AS REAL) + (CAST(f.finish_round AS REAL) - 1) * 300) / 60.0 AS dur,
               CAST(fs.sig_str_landed   AS REAL) AS sl,
               CAST(fs.sig_str_atmpted  AS REAL) AS sa,
               CAST(opp.sig_str_landed  AS REAL) AS o_sl,
               CAST(opp.sig_str_atmpted AS REAL) AS o_sa,
               opp.fighter_id AS opp_id
        FROM fight_stats fs
        JOIN fights f        ON f.fight_id = fs.fight_id
        JOIN fight_stats opp ON opp.fight_id = fs.fight_id AND opp.fighter_id != fs.fighter_id
        {where}
        """,
        conn,
        params=params,
    )
    num = ["dur", "sl", "sa", "o_sl", "o_sa"]
    df[num] = df[num].astype(float)          # an empty result would otherwise be object dtype
    df = df[df["dur"] > 0].copy()
    df[num[1:]] = df[num[1:]].fillna(0)
    return df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)


def _with_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Add each fight's defensive residuals (needs the opponent's pre-fight rates)."""
    g = df.groupby("fighter_id")
    prior = lambda col: g[col].cumsum() - df[col]          # strictly before this fight
    pre = pd.DataFrame({
        "fight_id": df["fight_id"],
        "opp_id":   df["fighter_id"],                     # this row's fighter, as someone's opponent
        "O_slpm": (prior("sl") + RATE_PSEUDO_MIN * LEAGUE_SLPM) / (prior("dur") + RATE_PSEUDO_MIN),
        "O_acc":  (prior("sl") + ACC_PSEUDO_ATT * LEAGUE_STR_ACC) / (prior("sa") + ACC_PSEUDO_ATT),
    })
    df = df.merge(pre, on=["fight_id", "opp_id"], how="left")
    df["r_str_def"] = df["o_sl"] / df["dur"] - df["O_slpm"]
    df["r_acc_def"] = np.where(df["o_sa"] > 0, df["o_sl"] / df["o_sa"].clip(lower=1), np.nan) - df["O_acc"]
    return df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)


def _weights(df: pd.DataFrame, resid: str) -> tuple[pd.Series, pd.Series]:
    w = df["dur"].where(df[resid].notna(), 0.0)
    return df[resid].fillna(0.0) * w, w


def prefight_adj_defense(conn: sqlite3.Connection) -> pd.DataFrame:
    """Pre-fight adj_str_def / adj_acc_def for every (fighter_id, fight_id) in the DB."""
    df = _with_residuals(_fight_rows(conn))
    out = df[["fighter_id", "fight_id", "date"]].copy()
    for feat in FEATURES:
        x, w = _weights(df, "r_" + feat[len("adj_"):])
        num = x.groupby(df["fighter_id"]).cumsum() - x
        den = w.groupby(df["fighter_id"]).cumsum() - w
        out[feat] = num / (den + RESID_PSEUDO_MIN)
    return out


def current_adj_defense(
    conn: sqlite3.Connection, fighter_ids: list[str], before_date: str | None = None,
) -> dict[str, dict[str, float]]:
    """adj_str_def / adj_acc_def going into a fight on *before_date* (0.0 for a debut)."""
    df = _with_residuals(_fight_rows(conn, before_date))
    result = {}
    for fid in fighter_ids:
        sub = df[df["fighter_id"] == fid]
        vals = {}
        for feat in FEATURES:
            x, w = _weights(sub, "r_" + feat[len("adj_"):])
            vals[feat] = float(x.sum() / (w.sum() + RESID_PSEUDO_MIN))
        result[fid] = vals
    return result
