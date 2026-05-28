"""
backtest.py — Simulate predictions at each historical fight and report accuracy.

Loads the saved XGBoost (or LR) model and scores it against the full dataset,
reporting year-by-year accuracy, overall metrics, and model drift over time.

Usage
-----
    python backtest.py
    python backtest.py --model lr
    python backtest.py --from-year 2018
    python backtest.py --save-csv backtest_results.csv
    python backtest.py --odds                 # value-bet ROI simulation
    python backtest.py --odds --min-edge 0.05 # only bet when edge >= 5%
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    CSV_WITH_ELO, DB_PATH,
    MODEL_XGB_PATH, MODEL_XGB_FEATURES,
    MODEL_LR_PATH, MODEL_LR_SCALER, MODEL_LR_FEATURES,
    TARGET_COL, META_COLS,
)
from logger import get_logger
from odds import american_to_prob, remove_vig, kelly_fraction

log = get_logger(__name__)


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(model_type: str) -> tuple:
    """
    Load saved model artifacts.

    Returns (model_or_base, feature_names, scaler, platt, model_label)
    """
    if model_type == "xgb":
        if not MODEL_XGB_PATH.exists():
            print(f"[ERROR] XGBoost model not found at '{MODEL_XGB_PATH}'.")
            print("   Run: python ML_models/XGBoost.py")
            sys.exit(1)
        model         = joblib.load(MODEL_XGB_PATH)
        feature_names = joblib.load(MODEL_XGB_FEATURES)
        scaler        = None
        platt         = None
        label         = "XGBoost"
    else:
        if not MODEL_LR_PATH.exists():
            print(f"[ERROR] LR model not found at '{MODEL_LR_PATH}'.")
            print("   Run: python ML_models/logistic_regression.py")
            sys.exit(1)
        artifact      = joblib.load(MODEL_LR_PATH)
        feature_names = joblib.load(MODEL_LR_FEATURES)
        scaler        = joblib.load(MODEL_LR_SCALER)
        model         = artifact["base"]
        platt         = artifact["platt"]
        label         = "Logistic Regression"

    return model, feature_names, scaler, platt, label


# ── Score dataset ─────────────────────────────────────────────────────────────

def score_dataset(
    df: pd.DataFrame,
    model,
    feature_names: list[str],
    scaler,
    platt,
) -> pd.DataFrame:
    """
    Generate predictions for every row in *df* and return a results DataFrame.

    Columns: date | year | target | predicted | correct | prob_red | confidence
    """
    meta_to_drop = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=meta_to_drop).fillna(0)

    # Align features — only use columns the model was trained on
    common = [f for f in feature_names if f in X.columns]
    X_aligned = X[common]

    if scaler is not None:
        X_input = scaler.transform(X_aligned)
    else:
        X_input = X_aligned.values

    if platt is not None:
        raw_probs  = model.predict_proba(X_input)[:, 1]
        prob_red   = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    else:
        prob_red = model.predict_proba(X_input)[:, 1]

    predicted  = (prob_red >= 0.5).astype(int)
    confidence = np.where(prob_red >= 0.5, prob_red, 1 - prob_red)

    results = pd.DataFrame({
        "date":       df["date"].values,
        "year":       pd.to_datetime(df["date"]).dt.year.values,
        "division":   df["division"].values if "division" in df.columns else np.nan,
        "target":     df[TARGET_COL].values,
        "predicted":  predicted,
        "correct":    (df[TARGET_COL].values == predicted).astype(int),
        "prob_red":   prob_red,
        "confidence": confidence,
    })
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: pd.DataFrame, label: str, from_year: int | None = None) -> None:
    """Print overall + year-by-year accuracy report."""
    if from_year:
        results = results[results["year"] >= from_year].copy()

    print(f"\n{'=' * 60}")
    print(f"  Backtest Report — {label}")
    if from_year:
        print(f"  (from {from_year} onwards)")
    print(f"{'=' * 60}")

    total   = len(results)
    correct = results["correct"].sum()
    overall = correct / total if total else 0
    mean_conf = results["confidence"].mean()

    print(f"\n  Overall accuracy:  {overall:.2%}  ({correct}/{total} fights)")
    print(f"  Mean confidence:   {mean_conf:.1%}")
    print(f"  Date range:        {results['date'].min()} to {results['date'].max()}")

    # Brier score (lower = better calibrated)
    brier = np.mean((results["prob_red"] - results["target"]) ** 2)
    print(f"  Brier score:       {brier:.4f}")

    # Naive baseline: always predict red wins (~64% red bias in UFC)
    naive = results["target"].mean()
    naive_acc = max(naive, 1 - naive)
    print(f"  Naive baseline:    {naive_acc:.2%}  (always pick {'Red' if naive > 0.5 else 'Blue'})")
    print(f"  Model edge:        {overall - naive_acc:+.2%}")

    # ── Year-by-year table ────────────────────────────────────────────────────
    yearly = (
        results.groupby("year")
        .agg(
            fights=("correct", "count"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    yearly["accuracy_pct"]    = yearly["accuracy"].map(lambda x: f"{x:.1%}")
    yearly["avg_conf_pct"]    = yearly["avg_confidence"].map(lambda x: f"{x:.1%}")

    print(f"\n  {'Year':<6} {'Fights':>7} {'Correct':>8} {'Accuracy':>10} {'Avg Conf':>10}")
    print(f"  {'-'*46}")
    for _, row in yearly.iterrows():
        marker = " ←" if row["accuracy"] < 0.60 else ""
        print(f"  {int(row['year']):<6} {int(row['fights']):>7} {int(row['correct']):>8} "
              f"{row['accuracy_pct']:>10} {row['avg_conf_pct']:>10}{marker}")

    # ── Division breakdown ────────────────────────────────────────────────────
    if "division" in results.columns and results["division"].notna().any():
        print(f"\n  {'Division':<25} {'Fights':>7} {'Accuracy':>10}")
        print(f"  {'-'*45}")
        div_acc = (
            results.groupby("division")
            .agg(fights=("correct", "count"), accuracy=("correct", "mean"))
            .sort_values("accuracy", ascending=False)
        )
        for div, row in div_acc.iterrows():
            if row["fights"] >= 30:   # only report divisions with enough fights
                print(f"  {str(div):<25} {int(row['fights']):>7} {row['accuracy']:.1%}")

    print()


# ── Odds backtest ─────────────────────────────────────────────────────────────

def _load_odds(db_path: Path) -> pd.DataFrame:
    """Pull fight_id + odds_red + odds_blue from the DB fights table."""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT fight_id, odds_red, odds_blue FROM fights WHERE odds_red IS NOT NULL AND odds_blue IS NOT NULL",
        conn,
    )
    conn.close()
    return df


def print_odds_report(results: pd.DataFrame, min_edge: float, from_year: int | None) -> None:
    """
    Two simulations for fights where odds are available:

    1. Naive model picks: bet 1 unit on every fight the model predicts to win.
    2. Value bets only:   bet 1 unit (+ quarter-Kelly) only when edge >= min_edge.
    """
    df = results.dropna(subset=["odds_red", "odds_blue"]).copy()
    if from_year:
        df = df[df["year"] >= from_year].copy()

    coverage = len(df)
    total    = len(results[results["year"] >= from_year] if from_year else results)

    print(f"\n{'=' * 60}")
    print(f"  Odds Backtest")
    if from_year:
        print(f"  (from {from_year} onwards)")
    print(f"{'=' * 60}")
    print(f"\n  Fights with odds: {coverage} / {total}  ({coverage/total:.0%} coverage)")

    if coverage == 0:
        print("  No odds data available — run the pipeline with a dataset that includes odds.")
        return

    # ── Section 1: Naive model picks (1 unit on every prediction) ─────────────
    raw_p_red_all  = df["odds_red"].apply(american_to_prob)
    raw_p_blue_all = df["odds_blue"].apply(american_to_prob)
    df["dec_red"]  = 1 / raw_p_red_all
    df["dec_blue"] = 1 / raw_p_blue_all

    # Model picks red when prob_red >= 0.5
    df["pick_red"]  = df["prob_red"] >= 0.5
    df["pick_won"]  = df.apply(
        lambda r: r["target"] == 1 if r["pick_red"] else r["target"] == 0, axis=1
    )
    df["pick_dec_odds"] = df.apply(
        lambda r: r["dec_red"] if r["pick_red"] else r["dec_blue"], axis=1
    )
    df["naive_pnl"] = df.apply(
        lambda r: (r["pick_dec_odds"] - 1) if r["pick_won"] else -1.0, axis=1
    )

    naive_total  = len(df)
    naive_won    = df["pick_won"].sum()
    naive_pnl    = df["naive_pnl"].sum()
    naive_roi    = naive_pnl / naive_total * 100

    print(f"\n  -- Section 1: Bet 1 unit on every model pick --")
    print(f"  Fights:    {naive_total}")
    print(f"  Wins:      {naive_won}  ({naive_won/naive_total:.1%})")
    outcome = "PROFIT" if naive_pnl >= 0 else "LOSS"
    print(f"  Total P&L: {naive_pnl:+.2f} units  [{outcome}]")
    print(f"  ROI:       {naive_roi:+.1f}%  (per unit staked)")

    # ── Section 2: Value bets only (edge >= min_edge) ─────────────────────────
    print(f"\n  -- Section 2: Value bets only (edge >= {min_edge:.0%}) --")

    raw_p_red  = df["odds_red"].apply(american_to_prob)
    raw_p_blue = df["odds_blue"].apply(american_to_prob)
    fair_probs = [remove_vig(r, b) for r, b in zip(raw_p_red, raw_p_blue)]
    df["fair_p_red"]  = [fp[0] for fp in fair_probs]
    df["fair_p_blue"] = [fp[1] for fp in fair_probs]
    df["edge_red"]    = df["prob_red"]       - df["fair_p_red"]
    df["edge_blue"]   = (1 - df["prob_red"]) - df["fair_p_blue"]

    # Determine best side to bet on each fight
    df["bet_red"]  = df["edge_red"]  >= min_edge
    df["bet_blue"] = df["edge_blue"] >= min_edge
    bets = df[df["bet_red"] | df["bet_blue"]].copy()

    if bets.empty:
        print(f"\n  No value bets found with edge >= {min_edge:.0%}.")
        print(f"  Tip: lower --min-edge (current: {min_edge:.0%}) or retrain the model.")
        return

    # For fights where both sides qualify, pick the higher-edge side
    def _resolve(row):
        if row["bet_red"] and row["bet_blue"]:
            return "red" if row["edge_red"] >= row["edge_blue"] else "blue"
        return "red" if row["bet_red"] else "blue"

    bets["bet_side"]   = bets.apply(_resolve, axis=1)
    bets["bet_edge"]   = bets.apply(lambda r: r["edge_red"]  if r["bet_side"] == "red" else r["edge_blue"],  axis=1)
    bets["bet_dec"]    = bets.apply(lambda r: r["dec_red"]   if r["bet_side"] == "red" else r["dec_blue"],   axis=1)
    bets["bet_won"]    = bets.apply(lambda r: r["target"] == 1 if r["bet_side"] == "red" else r["target"] == 0, axis=1)

    # Flat stake P&L: +profit on win, -1 on loss
    bets["flat_pnl"]   = bets.apply(
        lambda r: (r["bet_dec"] - 1) if r["bet_won"] else -1.0, axis=1
    )

    # Quarter-Kelly stake P&L (bankroll = 1000 units)
    BANKROLL = 1000.0
    bets["kelly_stake"] = bets.apply(
        lambda r: kelly_fraction(r["bet_edge"], r["bet_dec"]) * BANKROLL, axis=1
    )
    bets["kelly_pnl"] = bets.apply(
        lambda r: r["kelly_stake"] * (r["bet_dec"] - 1) if r["bet_won"] else -r["kelly_stake"], axis=1
    )

    n_bets   = len(bets)
    n_won    = bets["bet_won"].sum()
    win_rate = n_won / n_bets
    flat_roi = bets["flat_pnl"].sum() / n_bets * 100
    kelly_roi_pct = bets["kelly_pnl"].sum() / BANKROLL * 100

    print(f"\n  Bets placed:   {n_bets}  ({n_bets/coverage:.0%} of odds-covered fights)")
    print(f"  Win rate:      {win_rate:.1%}  ({n_won}/{n_bets})")
    print(f"  Flat ROI:      {flat_roi:+.1f}%  (per unit staked)")
    print(f"  Kelly ROI:     {kelly_roi_pct:+.1f}%  (on {BANKROLL:.0f}-unit bankroll, quarter-Kelly)")

    # Avg edge on bets placed
    print(f"  Avg edge:      {bets['bet_edge'].mean():+.1%}")
    print(f"  Avg odds (dec):{bets['bet_dec'].mean():.2f}x")

    # Year-by-year breakdown
    yearly = (
        bets.groupby("year")
        .agg(
            bets=("bet_won", "count"),
            wins=("bet_won", "sum"),
            flat_pnl=("flat_pnl", "sum"),
        )
        .reset_index()
    )
    yearly["win_rate"] = yearly["wins"] / yearly["bets"]
    yearly["flat_roi"] = yearly["flat_pnl"] / yearly["bets"] * 100

    print(f"\n  {'Year':<6} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'Flat ROI':>10}")
    print(f"  {'-'*38}")
    for _, row in yearly.iterrows():
        print(f"  {int(row['year']):<6} {int(row['bets']):>5} {int(row['wins']):>5} "
              f"{row['win_rate']:>7.1%} {row['flat_roi']:>+9.1f}%")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    model_type: str = "xgb",
    from_year:  int | None = None,
    save_csv:   Path | None = None,
    run_odds:   bool = False,
    min_edge:   float = 0.03,
) -> None:
    if not CSV_WITH_ELO.exists():
        print(f"[ERROR] ML dataset not found at '{CSV_WITH_ELO}'.")
        print("   Run: python ML_models/ML_data_preparation.py")
        sys.exit(1)

    log.info("Loading ML dataset from %s…", CSV_WITH_ELO)
    df = pd.read_csv(CSV_WITH_ELO)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("Loaded %d rows.", len(df))

    model, feature_names, scaler, platt, label = load_model(model_type)
    log.info("Scoring full dataset…")

    results = score_dataset(df, model, feature_names, scaler, platt)
    print_report(results, label, from_year)

    if run_odds:
        if not DB_PATH.exists():
            print(f"[ERROR] Database not found at '{DB_PATH}'. Run the pipeline first.")
        else:
            odds_df = _load_odds(DB_PATH)
            if odds_df.empty:
                print("\n  [ODDS] No odds data in DB — ingest a dataset that includes R_odds/B_odds.")
            else:
                # fight_id is a meta col in the CSV — re-attach it from df
                results["fight_id"] = df["fight_id"].values if "fight_id" in df.columns else np.nan
                results = results.merge(odds_df, on="fight_id", how="left")
                log.info("Odds joined: %d / %d fights have odds.", results["odds_red"].notna().sum(), len(results))
                print_odds_report(results, min_edge, from_year)

    if save_csv:
        results.to_csv(save_csv, index=False)
        log.info("Results saved to %s", save_csv)
        print(f"  Results saved to: {save_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest the saved model against historical fights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python backtest.py                          # XGBoost, all years
  python backtest.py --model lr               # Logistic Regression
  python backtest.py --from-year 2020         # only fights from 2020 onward
  python backtest.py --save-csv results.csv   # save fight-level results
  python backtest.py --odds                   # value-bet ROI simulation
  python backtest.py --odds --min-edge 0.05   # only bet when edge >= 5%
        """,
    )
    parser.add_argument("--model",     choices=["xgb", "lr"], default="xgb")
    parser.add_argument("--from-year", type=int, default=None,
                        help="Only report from this year onwards.")
    parser.add_argument("--save-csv",  type=Path, default=None,
                        help="Save fight-level results to a CSV file.")
    parser.add_argument("--odds",      action="store_true",
                        help="Run value-bet ROI simulation using DB odds data.")
    parser.add_argument("--min-edge",  type=float, default=0.03,
                        help="Minimum model edge (vs vig-stripped market) to place a bet (default: 0.03).")
    args = parser.parse_args()
    main(args.model, args.from_year, args.save_csv, args.odds, args.min_edge)
