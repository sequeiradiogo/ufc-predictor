"""
estimate_shrink_k.py -- Estimate one shrinkage k per stat (ml/shrinkage.STAT_K, issue #69).

k = noise variance / real between-fighter variance, in pre-fight UFC rounds.
For each fighter and each UFC fight t with at least one prior UFC fight:

  xbar = career value before fight t (same pooled formulas as
         predict.compute_live_career_stats)
  r    = pre-fight UFC rounds (total_rounds_fought)
  y    = fight t's own value
  mu   = division prior of fight t (ml/shrinkage.priors_for(division, True))

The posterior mean gives E[y - mu | xbar] = r/(r+k) * (xbar - mu); noise in y
doesn't bias the slope. One k per stat minimises the squared error, weighted by
fight t's rounds. Only fights t dated PRIOR_START..PRIOR_END (the priors' window)
are used, so the estimate never touches an evaluation window. A 90% interval
comes from resampling fighters.

The result is printed, not written: STAT_K is frozen by hand in ml/shrinkage.py
(capped at STAT_K_MAX), and changing it needs a full retrain.

Usage:
    python scripts/estimate_shrink_k.py
    python scripts/estimate_shrink_k.py --stats splm td_def --bootstrap 0
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH  # noqa: E402
from ml.shrinkage import PRIOR_END, PRIOR_START, PRIORS, priors_for  # noqa: E402

_OWN = ["sig_str_landed", "sig_str_atmpted", "td_landed", "td_atmpted", "sub_att",
        "head_landed", "head_atmpted", "body_landed", "body_atmpted",
        "leg_landed", "leg_atmpted", "dist_landed", "dist_atmpted"]
_OPP = ["sig_str_landed", "td_landed", "td_atmpted", "head_landed", "head_atmpted",
        "body_landed", "body_atmpted", "dist_landed", "dist_atmpted"]

# stat: (numerator, denominator, scale, invert) -> value = scale * num / den,
# or scale * (den - num) / den for defences. Same units as the stored columns.
DEFS = {
    "splm":            ("p_sig_str_landed", "mins", 1.0, False),
    "sapm":            ("o_sig_str_landed", "mins", 1.0, False),
    "td_avg":          ("p_td_landed", "mins", 15.0, False),
    "avg_sub_att":     ("p_sub_att", "one", 1.0, False),
    "avg_sig_str_pct": ("p_sig_str_landed", "p_sig_str_atmpted", 1.0, False),
    "td_acc":          ("p_td_landed", "p_td_atmpted", 1.0, False),
    "head_acc":        ("p_head_landed", "p_head_atmpted", 100.0, False),
    "body_acc":        ("p_body_landed", "p_body_atmpted", 100.0, False),
    "leg_acc":         ("p_leg_landed", "p_leg_atmpted", 100.0, False),
    "dist_acc":        ("p_dist_landed", "p_dist_atmpted", 100.0, False),
    "td_def":          ("o_td_landed", "o_td_atmpted", 100.0, True),
    "head_def":        ("o_head_landed", "o_head_atmpted", 100.0, True),
    "body_def":        ("o_body_landed", "o_body_atmpted", 100.0, True),
    "dist_def":        ("o_dist_landed", "o_dist_atmpted", 100.0, True),
}
_BOUNDS = (np.log(0.1), np.log(2000.0))


def load_fights(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per (fight, fighter) with own and opponent counts, ordered by fighter and date."""
    q = f"""
    SELECT f.fight_id, f.date, f.division, p.fighter_id,
           CAST(f.finish_round AS REAL) AS rnd, CAST(f.match_time_sec AS REAL) AS mts,
           {', '.join(f'CAST(p.{c} AS REAL) AS p_{c}' for c in _OWN)},
           {', '.join(f'CAST(o.{c} AS REAL) AS o_{c}' for c in _OPP)}
    FROM fights f
    JOIN fight_stats p ON p.fight_id = f.fight_id
    JOIN fight_stats o ON o.fight_id = f.fight_id AND o.fighter_id != p.fighter_id
    ORDER BY p.fighter_id, f.date, f.fight_id
    """
    df = pd.read_sql_query(q, conn)
    df["mins"] = (df["mts"] + (df["rnd"] - 1) * 300) / 60.0
    df["one"] = 1.0
    return df


def _fit_k(r, dx, dy, w) -> float:
    def loss(logk):
        return np.sum(w * (dy - r / (r + np.exp(logk)) * dx) ** 2)
    return float(np.exp(minimize_scalar(loss, bounds=_BOUNDS, method="bounded").x))


def estimate(df: pd.DataFrame, stat: str, n_boot: int = 100, seed: int = 0) -> dict:
    num, den, scale, inv = DEFS[stat]
    g = df.groupby("fighter_id", sort=False)
    r_prev = g["rnd"].cumsum() - df["rnd"]
    n_prev = g.cumcount()
    cn = g[num].cumsum() - df[num]
    cd = g[den].cumsum() - df[den]
    if inv:
        xbar, y = scale * (cd - cn) / cd, scale * (df[den] - df[num]) / df[den]
    else:
        xbar, y = scale * cn / cd, scale * df[num] / df[den]
    m = ((df["date"] >= PRIOR_START) & (df["date"] <= PRIOR_END) & (n_prev >= 1) & (df["mins"] > 0)
         & (cd > 0) & (df[den] > 0) & np.isfinite(xbar) & np.isfinite(y))
    d = df.loc[m]
    if stat in PRIORS:
        mu = np.array([priors_for(v, True)[stat] for v in d["division"]])
    else:  # no frozen prior (td_acc, dist_acc): division mean of y in the window
        mu = y[m].groupby(d["division"]).transform("mean").values
    r, dx, dy, w = r_prev[m].values, xbar[m].values - mu, y[m].values - mu, d["rnd"].values

    out = {"stat": stat, "n": int(m.sum()), "k": _fit_k(r, dx, dy, w),
           "k_unweighted": _fit_k(r, dx, dy, np.ones_like(w))}
    if n_boot:
        fids = d["fighter_id"].values
        uf = np.unique(fids)
        idx = {f: np.where(fids == f)[0] for f in uf}
        rng = np.random.default_rng(seed)
        ks = []
        for _ in range(n_boot):
            pick = np.concatenate([idx[f] for f in rng.choice(uf, len(uf))])
            ks.append(_fit_k(r[pick], dx[pick], dy[pick], w[pick]))
        out["ci_lo"], out["ci_hi"] = (float(v) for v in np.percentile(ks, [5, 95]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stats", nargs="*", choices=list(DEFS), help="default: all")
    ap.add_argument("--bootstrap", type=int, default=100, help="resamples for the 90%% CI (0 = none)")
    args = ap.parse_args()

    with sqlite3.connect(DB_UFCSTATS_PATH) as conn:
        df = load_fights(conn)
    res = {}
    for stat in args.stats or DEFS:
        e = estimate(df, stat, args.bootstrap)
        ci = f"  90% CI [{e['ci_lo']:6.2f}, {e['ci_hi']:6.2f}]" if "ci_lo" in e else ""
        print(f"{stat:16s} n={e['n']:5d}  k={e['k']:7.2f}  (unweighted {e['k_unweighted']:7.2f}){ci}", flush=True)
        res[stat] = round(e["k"], 1)
    print("\nk per base stat:", res)


if __name__ == "__main__":
    main()
