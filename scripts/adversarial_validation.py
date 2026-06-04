"""
adversarial_validation.py -- Detect distribution shift between training era and recent fights.

Trains a LightGBM classifier to distinguish old fights (label=0) from recent
fights (label=1) using the same feature set as the outcome models. AUC near
0.5 means no detectable shift; AUC > 0.65 indicates meaningful shift that
may warrant time-based sample weighting or trimming old training data.

Usage
-----
    python scripts/adversarial_validation.py
    python scripts/adversarial_validation.py --cutoff 2018-01-01
    python scripts/adversarial_validation.py --cutoff 2022-01-01 --save-csv shift_report.csv
    python scripts/adversarial_validation.py --top-n 30
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import CSV_WITH_ELO, META_COLS
from utils.logger import get_logger

log = get_logger(__name__)

MIN_CLASS_SIZE = 50


# -- Data loading --------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    log.info("Loading ML dataset from %s...", csv_path)
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("Loaded %d fights spanning %s to %s.",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df


# -- Labeling and splitting ----------------------------------------------------

def label_and_split(
    df: pd.DataFrame,
    cutoff: str,
    random_state: int = 42,
) -> tuple:
    try:
        cutoff_dt = pd.to_datetime(cutoff)
    except ValueError:
        print(f"[ERROR] Cannot parse cutoff date '{cutoff}'. Expected format: YYYY-MM-DD.")
        sys.exit(1)

    adv_label = (df["date"] >= cutoff_dt).astype(int)
    n_old = int((adv_label == 0).sum())
    n_recent = int((adv_label == 1).sum())

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()

    if n_old == 0:
        print(
            f"[ERROR] No fights before cutoff '{cutoff}'. "
            f"Dataset runs from {date_min} to {date_max}. "
            "Try an earlier --cutoff."
        )
        sys.exit(1)
    if n_recent == 0:
        print(
            f"[ERROR] No fights from cutoff '{cutoff}' onward. "
            f"Dataset runs from {date_min} to {date_max}. "
            "Try a later --cutoff."
        )
        sys.exit(1)

    if n_old < MIN_CLASS_SIZE:
        print(
            f"[WARNING] Only {n_old} fights before cutoff '{cutoff}'. "
            "AUC may be unreliable. Try an earlier --cutoff."
        )
    if n_recent < MIN_CLASS_SIZE:
        print(
            f"[WARNING] Only {n_recent} fights from cutoff '{cutoff}' onward. "
            "AUC may be unreliable. Try a later --cutoff."
        )

    drop_cols = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop_cols).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, adv_label, test_size=0.5, random_state=random_state, stratify=adv_label
    )
    log.info(
        "Split: %d train / %d test (50/50 random, stratified).",
        len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test, n_old, n_recent


# -- Model training ------------------------------------------------------------

def train_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        importance_type="split",
        verbosity=-1,
        random_state=42,
    )
    log.info("Training adversarial classifier on %d samples...", len(X_train))
    model.fit(X_train, y_train)
    log.info("Training complete.")
    return model


# -- Evaluation ----------------------------------------------------------------

def evaluate(model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    probs = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))
    log.info("Adversarial AUC: %.4f", auc)
    return auc


# -- Feature importances -------------------------------------------------------

def get_feature_importances(
    model: LGBMClassifier,
    feature_names: list,
    top_n: int = 20,
) -> pd.DataFrame:
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df.head(top_n)


# -- Interpretation ------------------------------------------------------------

def interpret_auc(auc: float) -> tuple:
    if auc < 0.55:
        label = "No detectable distribution shift"
        action = "No action needed. Full history training is appropriate."
    elif auc < 0.65:
        label = "Mild distribution shift detected"
        action = (
            "Investigate top shifting features. "
            "Consider removing pre-2010 fights from training."
        )
    else:
        label = "Significant distribution shift detected"
        action = "Add time-based sample weights to training (recent fights weighted higher)."
    return label, action


# -- Report printing -----------------------------------------------------------

def print_report(
    auc: float,
    cutoff: str,
    n_old: int,
    n_recent: int,
    importances: pd.DataFrame,
) -> None:
    label, action = interpret_auc(auc)
    total = n_old + n_recent

    sep = "=" * 62
    thin = "-" * 62

    print()
    print(sep)
    print("  Adversarial Validation Report")
    print(f"  Cutoff: {cutoff}")
    print(sep)
    print()
    print("  Dataset split")
    print(f"  Old fights    (before {cutoff}, label=0): {n_old:>7}")
    print(f"  Recent fights (from   {cutoff}, label=1): {n_recent:>7}")
    print(f"  Total:                                    {total:>7}")
    print()
    print(f"  Adversarial AUC (test split, 50/50 random):  {auc:.4f}")
    print(f"  Interpretation:  {label}")
    print()
    print("  Recommended action:")
    print(f"    {action}")
    print()
    print(f"  Top {len(importances)} most-shifted features (LightGBM split importance)")
    print(thin)
    print(f"  {'Rank':>4}  {'Feature':<38}  {'Importance':>10}")
    print(f"  {'----':>4}  {'-------':<38}  {'----------':>10}")
    for i, row in importances.iterrows():
        rank = i + 1
        print(f"  {rank:>4}  {row['feature']:<38}  {int(row['importance']):>10}")
    print(thin)
    print()


# -- Main ----------------------------------------------------------------------

def main(
    cutoff: str = "2020-01-01",
    save_csv: Path = None,
    top_n: int = 20,
) -> None:
    if not CSV_WITH_ELO.exists():
        print(f"[ERROR] ML dataset not found at '{CSV_WITH_ELO}'.")
        print("   Run: python run_pipeline.py --steps 4")
        sys.exit(1)

    df = load_data(CSV_WITH_ELO)
    X_train, X_test, y_train, y_test, n_old, n_recent = label_and_split(df, cutoff)
    model = train_classifier(X_train, y_train)
    auc = evaluate(model, X_test, y_test)
    importances = get_feature_importances(model, list(X_train.columns), top_n=top_n)

    print_report(auc, cutoff, n_old, n_recent, importances)

    if save_csv is not None:
        full_imp = get_feature_importances(model, list(X_train.columns), top_n=len(X_train.columns))
        full_imp.to_csv(save_csv, index=False)
        log.info("Feature importances saved to %s", save_csv)
        print(f"  Feature importances saved to: {save_csv}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adversarial validation: detect distribution shift between old and recent UFC fights.",
        epilog=(
            "Examples:\n"
            "  python scripts/adversarial_validation.py\n"
            "  python scripts/adversarial_validation.py --cutoff 2018-01-01\n"
            "  python scripts/adversarial_validation.py --cutoff 2022-01-01 --save-csv shift.csv\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2020-01-01",
        help="Date boundary (YYYY-MM-DD). Fights before = old (0), from = recent (1). Default: 2020-01-01",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Export full feature importance ranking to a CSV file.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="Number of top features to display in the report. Default: 20",
    )

    args = parser.parse_args()
    main(cutoff=args.cutoff, save_csv=args.save_csv, top_n=args.top_n)
