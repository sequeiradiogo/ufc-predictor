"""
adj_performance.py -- Opponent-adjusted significant-strike landing ratio (issue #68).

The landing ratio of a fight is the fighter's share of the significant strikes
both fighters landed in it: sl / (sl + opponent sl). It is the per-fight strike
differential as a share, which none of the career rates (splm, sapm, accuracy)
expresses on its own.

The construction follows mma-ai.net's "Adjusted Performance" (a robust z-score
against what each opponent usually allows), for this one stat:

  1. Per prior fight j of fighter F against opponent O, O's baseline is what O's
     own earlier opponents achieved against O (fights strictly before j): a
     time-decayed mean and mean absolute deviation of their landing ratios
     (half-life HALF_LIFE_DAYS), shrunk toward the division prior with
     w = n_eff / (n_eff + K_OPP), n_eff = Kish effective sample size
     (sum w)^2 / sum w^2. An opponent with no history is the division prior.
  2. adjperf_j = clip((ratio_Fj - mu_O) / max(mad_O, MAD_FLOOR_FRAC * mad_div), +/-CLIP).
  3. F's feature before a fight = time-decayed mean of adjperf_j over every
     earlier fight, shrunk toward 0 by n_eff / (n_eff + S_FIGHTER).

Positive = out-landed his opponents by more than they usually get out-landed.
Tested 2026-11 (issue #68): over 7 forecast origins not used to choose it,
hold-out log-loss -0.0025 (6/7 origins, 16/21 seeds), better at 11 of 14
origins overall -- strong for hold-outs starting 2024-04 or later (-0.003 to
-0.013), about neutral for those starting 2023-02..2024-01.
Eleven other AdjPerf stats (accuracies, defences, round-1 strikes, ...) were
rejected: they duplicate the ewma_* features and destabilised RF / LightGBM.

Training (scripts/add_computed_features_to_csv.py -> prefight_adj_land_ratio)
and live inference (predict.py -> current_adj_land_ratio) both go through
_fight_rows / _adjperf / _fighter_value below. Reads the UFCStats DB.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

FEATURE = "adj_sig_land_ratio"

HALF_LIFE_DAYS = 548.0    # 1.5 years, for both the opponent baseline and the fighter mean
K_OPP = 4.0               # shrink of the opponent baseline toward the division prior
S_FIGHTER = 2.0           # shrink of the fighter's mean toward 0, in effective fights
CLIP = 7.0
MAD_FLOOR_FRAC = 0.25     # floor of the opponent's MAD, as a share of the division's

# Frozen division priors of the per-fight landing ratio, UFCStats fights before
# 2018-01-01: the mean is 0.5 by construction (both corners are counted), the
# mean absolute deviation is per division. Unknown divisions use MAD_GLOBAL.
PRIOR_MEAN = 0.5
DIV_MAD: dict[str, float] = {
    "bantamweight": 0.157373, "catch weight": 0.189092, "featherweight": 0.151087,
    "flyweight": 0.156379, "heavyweight": 0.222577, "light heavyweight": 0.210325,
    "lightweight": 0.169123, "middleweight": 0.192801, "open weight": 0.322111,
    "welterweight": 0.180991, "women's bantamweight": 0.156110,
    "women's featherweight": 0.145354, "women's flyweight": 0.142785,
    "women's strawweight": 0.138410,
}
MAD_GLOBAL = 0.184233


def _fight_rows(conn: sqlite3.Connection, before_date: str | None = None) -> pd.DataFrame:
    """One row per (fighter, fight) with the fight's landing ratio, fights before *before_date*."""
    where = "WHERE f.date < ?" if before_date else ""
    params = (str(before_date)[:10],) if before_date else ()
    df = pd.read_sql_query(
        f"""
        SELECT p.fighter_id, o.fighter_id AS opp_id, f.fight_id, f.date,
               LOWER(f.division) AS division,
               (CAST(f.match_time_sec AS REAL) + (CAST(f.finish_round AS REAL) - 1) * 300) AS secs,
               CAST(p.sig_str_landed AS REAL) AS sl,
               CAST(o.sig_str_landed AS REAL) AS o_sl
        FROM fights f
        JOIN fight_stats p ON p.fight_id = f.fight_id
        JOIN fight_stats o ON o.fight_id = f.fight_id AND o.fighter_id != p.fighter_id
        {where}
        """,
        conn,
        params=params,
    )
    df["secs"] = df["secs"].astype(float)
    df = df[df["secs"] > 0].copy()
    tot = df["sl"].astype(float) + df["o_sl"].astype(float)
    df["ratio"] = np.where(tot > 0, df["sl"].astype(float) / tot.where(tot > 0, 1.0), np.nan)
    df["day"] = pd.to_datetime(df["date"]).values.astype("datetime64[D]").astype(np.int64)
    return df.sort_values(["fighter_id", "day", "fight_id"]).reset_index(drop=True)


def _decay(dt_days: np.ndarray) -> np.ndarray:
    return 0.5 ** (dt_days / HALF_LIFE_DAYS)


def _adjperf(df: pd.DataFrame, rows: np.ndarray, by_fighter: dict) -> np.ndarray:
    """Per-fight adjusted performance for *rows* (positions in df)."""
    ratio, day = df["ratio"].values, df["day"].values
    opp, div = df["opp_id"].values, df["division"].values
    out = np.full(len(rows), np.nan)
    for n, i in enumerate(rows):
        if np.isnan(ratio[i]):
            continue
        mad_div = DIV_MAD.get(div[i], MAD_GLOBAL)
        mu, mad = PRIOR_MEAN, mad_div
        hist = by_fighter.get(opp[i])
        if hist is not None:
            hist = hist[day[hist] < day[i]]
            allowed = 1.0 - ratio[hist]            # what O's opponents achieved against O
            ok = ~np.isnan(allowed)
            if ok.any():
                w = _decay(day[i] - day[hist][ok])
                v = allowed[ok]
                n_eff = w.sum() ** 2 / (w ** 2).sum()
                m_o = (w * v).sum() / w.sum()
                mad_o = (w * np.abs(v - m_o)).sum() / w.sum()
                a = n_eff / (n_eff + K_OPP)
                mu = a * m_o + (1 - a) * PRIOR_MEAN
                mad = a * mad_o + (1 - a) * mad_div
        mad = max(mad, MAD_FLOOR_FRAC * mad_div, 1e-6)
        out[n] = np.clip((ratio[i] - mu) / mad, -CLIP, CLIP)
    return out


def _fighter_value(day_now: int, days: np.ndarray, adj: np.ndarray) -> float:
    """Decayed mean of a fighter's earlier per-fight values, shrunk toward 0 (0.0 with none)."""
    ok = ~np.isnan(adj) & (days < day_now)
    if not ok.any():
        return 0.0
    w = _decay(day_now - days[ok])
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return float((w * adj[ok]).sum() / w.sum() * n_eff / (n_eff + S_FIGHTER))


def _index(df: pd.DataFrame) -> dict:
    return {f: np.asarray(ix) for f, ix in df.groupby("fighter_id").indices.items()}


def prefight_adj_land_ratio(conn: sqlite3.Connection) -> pd.DataFrame:
    """Pre-fight adj_sig_land_ratio for every (fighter_id, fight_id) in the DB."""
    df = _fight_rows(conn)
    by_fighter = _index(df)
    adj = _adjperf(df, np.arange(len(df)), by_fighter)
    day = df["day"].values
    vals = np.zeros(len(df))
    for ix in by_fighter.values():
        for i in ix:
            vals[i] = _fighter_value(day[i], day[ix], adj[ix])
    out = df[["fighter_id", "fight_id", "date"]].copy()
    out[FEATURE] = vals
    return out


def current_adj_land_ratio(
    conn: sqlite3.Connection, fighter_ids: list[str], before_date: str | None = None,
) -> dict[str, dict[str, float]]:
    """adj_sig_land_ratio going into a fight on *before_date* (0.0 for a debut)."""
    df = _fight_rows(conn, before_date)
    by_fighter = _index(df)
    day_now = (np.datetime64(str(before_date)[:10], "D").astype(np.int64) if before_date
               else np.datetime64("today", "D").astype(np.int64))
    result = {}
    for fid in fighter_ids:
        ix = by_fighter.get(fid)
        if ix is None:
            result[fid] = {FEATURE: 0.0}
            continue
        adj = _adjperf(df, ix, by_fighter)
        result[fid] = {FEATURE: _fighter_value(day_now, df["day"].values[ix], adj)}
    return result
