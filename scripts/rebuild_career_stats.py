"""
rebuild_career_stats.py -- Recompute the career-stat columns of ufc-master.csv
from the UFCStats DB with predict.py's own live function, pre-fight only.

Why: the Kaggle-sourced rows (2010 - Mar 2026) leak the fight being predicted
into their "pre-fight" career averages -- avg_SIG_STR_pct / avg_TD_pct /
avg_SIG_STR_landed / avg_TD_landed match the post-fight value in ~60-65% of
rows, and wins / total_rounds_fought / total_title_bouts / longest_win_streak
partially. That inflated the backtest and taught the models to lean on signal
that doesn't exist at prediction time. Rows appended by csv_builder.py are
leak-free but only approximate the same stats.

Each row's value is compute_live_career_stats(name, before_date=<fight date>):
the exact function inference uses, restricted to fights strictly before the
fight -- so training and live features match by construction.
The trajectory slopes (str_acc / splm / td_acc) and the static bio columns
(height, reach, stance) come from the same call; recent form and KO/sub/dec
win rates from compute_recent_form() / compute_finish_rates_single() with the
same cutoff.

Rows whose fighter can't be resolved in the UFCStats DB keep their existing
values (reported). Debuts (resolved, no prior fights) get zeros.

Run after scripts/add_rankings_to_csv.py and before
scripts/add_computed_features_to_csv.py (the style features read splm/td_avg).

Usage:
    python scripts/rebuild_career_stats.py
    python scripts/rebuild_career_stats.py --dry-run     # report changes, don't write
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH, NAME_ALIASES, RAW_DIR
from predict import (
    _resolve_ufcstats_id, compute_finish_rates_single, compute_live_career_stats, compute_recent_form,
)

MASTER_CSV = RAW_DIR / "ufc-master.csv"

# ufc-master.csv column suffix -> compute_live_career_stats() key
CAREER_COLS = {
    "wins":                      "wins",
    "losses":                    "losses",
    "current_win_streak":        "career_win_streak",
    "current_lose_streak":       "career_lose_streak",
    "longest_win_streak":        "longest_win_streak",
    "total_rounds_fought":       "total_rounds_fought",
    "total_title_bouts":         "total_title_bouts",
    "win_by_KO/TKO":             "win_by_ko",
    "win_by_Submission":         "win_by_sub",
    "win_by_Decision_Unanimous": "win_by_dec_unanimous",
    "win_by_Decision_Split":     "win_by_dec_split",
    "avg_SIG_STR_pct":           "avg_sig_str_pct",
    "avg_TD_pct":                "avg_td_pct",
    "avg_SIG_STR_landed":        "splm",
    "avg_TD_landed":             "td_avg",
    "avg_SUB_ATT":               "avg_sub_att",
    "str_acc_slope":             "str_acc_slope",
    "splm_slope":                "splm_slope",
    "td_acc_slope":              "td_acc_slope",
}
# Form / finish-rate columns -> key of compute_recent_form() / compute_finish_rates_single()
FORM_COLS = {"recent_win_rate": "recent_win_rate", "recent_finish_rate": "recent_finish_rate",
             "ko_rate": "ko_rate", "sub_rate": "sub_rate", "dec_rate": "dec_rate"}
# Static bio columns, also read by inference from the UFCStats fighters table
BIO_COLS = {"Height_cms": "height", "Reach_cms": "reach", "Stance": "stance"}


def main(dry_run: bool = False) -> None:
    logging.getLogger("predict").setLevel(logging.WARNING)
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_UFCSTATS_PATH)
    has_dob = {r[0] for r in conn.execute("SELECT name FROM fighters WHERE dob IS NOT NULL AND dob != ''")}
    # compute_live_career_stats() resolves exact UFCStats names only; Kaggle
    # names differ by alias (config.NAME_ALIASES) or just letter case.
    canonical = {r[0].lower().strip(): r[0] for r in conn.execute("SELECT name FROM fighters")}

    def ufcstats_name(name: str) -> str:
        key = str(name).lower().strip()
        return NAME_ALIASES.get(key) or canonical.get(key, name)

    new = {f"{s}_{c}": df[f"{s}_{c}"].copy() for s in ("R", "B") for c in [*CAREER_COLS, *FORM_COLS, *BIO_COLS, "age"]}
    unresolved, inconsistent, debuts = set(), [], 0
    for i, d in enumerate(dates):
        for side in ("R", "B"):
            name = ufcstats_name(df.at[i, f"{side}_fighter"])
            try:
                stats = compute_live_career_stats(conn, name, before_date=d)
            except ValueError:
                inconsistent.append((d, name))
                continue
            if stats is None:
                if _resolve_ufcstats_id(conn, name) is None:
                    unresolved.add(name)
                    continue
                debuts += 1
                stats = {k: 0.0 for k in CAREER_COLS.values()}
                h, r, st = conn.execute("SELECT height, reach, stance FROM fighters WHERE name = ?", (name,)).fetchone()
                stats.update(height=float(h or 0), reach=float(r or 0), stance=(st or "Orthodox").strip())
            for col, key in CAREER_COLS.items():
                new[f"{side}_{col}"].iat[i] = round(float(stats[key]), 4)
            fid = _resolve_ufcstats_id(conn, name)
            form = {**compute_recent_form(conn, fid, before_date=d),
                    **compute_finish_rates_single(conn, fid, before_date=d)}
            for col, key in FORM_COLS.items():
                new[f"{side}_{col}"].iat[i] = round(float(form[key]), 4)
            for col, key in BIO_COLS.items():
                new[f"{side}_{col}"].iat[i] = stats[key] if key == "stance" else round(float(stats[key]), 2)
            if "age" in stats and name in has_dob:
                new[f"{side}_age"].iat[i] = round(float(stats["age"]), 2)
        if i % 1000 == 0:
            print(f"  {i}/{len(df)} rows", flush=True)

    print(f"\nRows: {len(df)} | debut corners zeroed: {debuts} | "
          f"unresolved fighters (kept as-is): {len(unresolved)} | inconsistent DB time (kept): {len(inconsistent)}")
    if unresolved:
        print("  unresolved sample:", sorted(unresolved)[:10])
    for col in [*CAREER_COLS, *FORM_COLS, *BIO_COLS, "age"]:
        if col == "Stance":
            print(f"  R_{col:28s} changed in {(df[f'R_{col}'].astype(str) != new[f'R_{col}'].astype(str)).mean():6.1%} of rows")
            continue
        old, upd = pd.to_numeric(df[f"R_{col}"], errors="coerce"), pd.to_numeric(new[f"R_{col}"], errors="coerce")
        changed = ~(np.isclose(old, upd, rtol=1e-3, atol=1e-3) | (old.isna() & upd.isna()))
        print(f"  R_{col:28s} changed in {changed.mean():6.1%} of rows")

    if dry_run:
        print("\nDry run -- no changes written.")
        return
    for col, values in new.items():
        df[col] = values
    df.to_csv(MASTER_CSV, index=False)
    print(f"\nWrote {MASTER_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args().dry_run)
