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
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    CSV_WITH_ELO,
    MODEL_XGB_PATH, MODEL_XGB_FEATURES,
    MODEL_LR_PATH, MODEL_LR_SCALER, MODEL_LR_FEATURES,
    TARGET_COL, META_COLS,
)
from logger import get_logger

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    model_type: str = "xgb",
    from_year:  int | None = None,
    save_csv:   Path | None = None,
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
        """,
    )
    parser.add_argument("--model",     choices=["xgb", "lr"], default="xgb")
    parser.add_argument("--from-year", type=int, default=None,
                        help="Only report from this year onwards.")
    parser.add_argument("--save-csv",  type=Path, default=None,
                        help="Save fight-level results to a CSV file.")
    args = parser.parse_args()
    main(args.model, args.from_year, args.save_csv)
