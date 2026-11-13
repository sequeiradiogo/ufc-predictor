"""
shrinkage.py -- Pull noisy per-fighter rates toward the league average (issue #69).

Career rates (strikes per minute, takedowns per 15 min, accuracies, defences,
EWMA and opponent-adjusted stats, finish rates) come from UFC fights only, so
after one or two fights they are mostly noise -- and every feature is a
Red-minus-Blue diff, so the models can't tell how much evidence is behind one.
Each rate is shrunk by the fighter's pre-fight UFC rounds:

    shrunk = (rate * rounds + prior * k) / (rounds + k)

k is SHRINK_K (config.py; 0 = off): k pseudo-rounds of league average. With
k = 6, 3 rounds keep 1/3 of the fighter's own rate, 60 rounds keep 91%.

One code path for training and live inference: ML_data_preparation_v1.py
shrinks the per-corner columns of each fight (shrink_wide) before any diff or
derived feature is built, and predict.build_feature_vector shrinks red_stats /
extra_r and the Blue ones (shrink_corner) the same way. Both read the same
pre-fight total_rounds_fought, so check_live_parity.py covers it.

PRIORS are frozen, rounds-weighted league means over fights from 2018-01-01
to 2022-06-30 -- before the base-model training cut-off of every evaluation
origin used for #69 (the earliest, 18 months back, is 2022-10-22), so no
experiment evaluates on fights its priors were computed from. Regenerate with
compute_priors() only together with a retrain.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

PRIOR_START = "2018-01-01"
PRIOR_END   = "2022-06-30"

# compute_priors(ufc_v2.db), 2026-11-11. Same units as the stored columns
# (some accuracies are fractions, some percentages).
PRIORS: dict[str, float] = {
    # career
    "avg_sig_str_pct": 0.452094,
    "avg_sub_att": 0.403286,
    "splm": 3.8022,
    "td_avg": 1.5195,
    # finish rates (share of wins)
    "ko_rate": 0.317541,
    "sub_rate": 0.182354,
    "dec_rate": 0.47996,
    # defence / zone accuracy
    "sapm": 3.34479,
    "td_def": 65.0501,
    "head_acc": 37.3291,
    "body_acc": 68.3123,
    "leg_acc": 80.7972,
    "head_def": 65.7458,
    "body_def": 33.828,
    "dist_def": 62.4703,
    # EWMA
    "ewma_str_acc": 0.468584,
    "ewma_td_acc": 0.255378,
    "ewma_head_acc": 0.378393,
    "ewma_body_acc": 0.64162,
    "ewma_dist_acc": 0.407502,
    "ewma_splm": 4.29479,
    "ewma_td_avg": 1.4386,
    "ewma_sapm": 3.93565,
    "ewma_clinch_per": 0.490251,
    "ewma_sub_att": 0.678559,
    "ewma_kd_received": 0.0563748,
    # opponent-adjusted
    "opp_adj_splm": 3.98587,
    "opp_adj_td_avg": 1.73778,
    "opp_adj_head_acc": 39.7798,
    "opp_adj_body_acc": 71.1415,
    "opp_adj_dist_acc": 41.2106,
    # round 1 EWMA
    "ewma_ctrl_r1": 0.175587,
    "ewma_splm_r1": 3.91776,
}


def compute_priors(conn: sqlite3.Connection, stats=None,
                   start: str = PRIOR_START, end: str = PRIOR_END) -> dict[str, float]:
    """Rounds-weighted mean of each stat over per-corner pre-fight snapshots in [start, end].

    Weighting by total_rounds_fought gives debut rows (0 rounds, stats stored as 0) no weight.
    """
    stats = list(stats or PRIORS)
    fs = pd.read_sql_query(
        f"SELECT fs.total_rounds_fought, {', '.join('fs.' + s for s in stats)} "
        "FROM fight_stats fs JOIN fights f USING (fight_id) WHERE f.date BETWEEN ? AND ?",
        conn, params=(start, end),
    )
    w = pd.to_numeric(fs["total_rounds_fought"], errors="coerce").fillna(0).clip(lower=0)
    out = {}
    for s in stats:
        v = pd.to_numeric(fs[s], errors="coerce")
        m = v.notna() & (w > 0)
        out[s] = float(np.average(v[m], weights=w[m]))
    return out


def _k(k: float | None) -> float:
    if k is None:
        from config import SHRINK_K
        k = SHRINK_K
    return float(k)


def shrink(value, rounds, prior: float, k: float):
    """(value * rounds + prior * k) / (rounds + k); scalars or arrays. NaN values stay NaN."""
    rounds = np.clip(np.asarray(rounds, dtype=float), 0, None)
    rounds = np.where(np.isnan(rounds), 0.0, rounds)
    out = (np.asarray(value, dtype=float) * rounds + prior * k) / (rounds + k)
    return float(out) if np.ndim(out) == 0 else out


def shrink_corner(stats, rounds, k: float | None = None):
    """Copy of one fighter's stats (dict or Series) with every PRIORS key present shrunk.

    Missing keys stay missing, so callers' own defaults still apply.
    """
    k = _k(k)
    if k <= 0 or stats is None:
        return stats
    out = stats.copy() if isinstance(stats, pd.Series) else dict(stats)
    r = float(pd.to_numeric(rounds, errors="coerce") or 0.0)
    for s, prior in PRIORS.items():
        if s in out:
            v = pd.to_numeric(out[s], errors="coerce")
            if v is not None and not pd.isna(v):
                out[s] = shrink(float(v), r, prior, k)
    return out


def shrink_wide(wide: pd.DataFrame, k: float | None = None) -> pd.DataFrame:
    """Shrink r_<stat> / b_<stat> columns of a one-row-per-fight frame in place (returns it)."""
    k = _k(k)
    if k <= 0:
        return wide
    for corner in ("r", "b"):
        rounds = pd.to_numeric(wide.get(f"{corner}_total_rounds_fought"), errors="coerce").fillna(0).values
        for s, prior in PRIORS.items():
            col = f"{corner}_{s}"
            if col in wide.columns:
                wide[col] = shrink(pd.to_numeric(wide[col], errors="coerce").values, rounds, prior, k)
    return wide

