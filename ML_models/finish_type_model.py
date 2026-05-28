"""
finish_type_model.py — Secondary model predicting HOW a fight ends.

Classes:
    0 = Decision   (unanimous, split, or majority)
    1 = KO / TKO
    2 = Submission

Uses the same fight-level diff features as the main win-prediction model.
Fights with unknown or rare finish methods (DQ, Overturned, etc.) are excluded.

Run:
    python ML_models/finish_type_model.py

Saves:
    models/finish_type.joblib
    models/finish_type_features.joblib
"""

import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ── Project imports ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DB_PATH,
    CSV_WITH_ELO,
    TARGET_COL, META_COLS,
    TRAIN_TEST_SPLIT, RANDOM_STATE,
    FINISH_METHOD_MAP, FINISH_CLASS_NAMES,
    MODEL_FINISH_PATH, MODEL_FINISH_FEATURES,
)
from logger import get_logger

log = get_logger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_finish_dataset() -> pd.DataFrame:
    """
    Load the prepared ML CSV, attach the finish-type label from the DB,
    and return a DataFrame ready for training.

    Rows with unmapped finish methods (DQ, Overturned, etc.) are dropped.
    """
    if not CSV_WITH_ELO.exists():
        raise FileNotFoundError(
            f"ML dataset not found at '{CSV_WITH_ELO}'.\n"
            "Run 'python ML_models/ML_data_preparation.py' first."
        )

    log.info("Loading ML dataset from %s…", CSV_WITH_ELO)
    ml = pd.read_csv(CSV_WITH_ELO)
    ml["date"] = pd.to_datetime(ml["date"])

    # Fetch method column from DB
    log.info("Fetching finish methods from database…")
    conn = sqlite3.connect(str(DB_PATH))
    methods = pd.read_sql_query("SELECT fight_id, method FROM fights", conn)
    conn.close()

    df = ml.merge(methods, on="fight_id", how="left")

    # Map to integer class; unmapped rows → -1 → drop
    df["finish_class"] = df["method"].map(FINISH_METHOD_MAP)
    before = len(df)
    df = df.dropna(subset=["finish_class"])
    df["finish_class"] = df["finish_class"].astype(int)
    dropped = before - len(df)
    if dropped:
        log.info("Dropped %d rows with unmapped finish method (DQ, Overturned, etc.)", dropped)

    log.info("Finish dataset: %d rows", len(df))
    log.info(
        "Class distribution: %s",
        dict(zip(FINISH_CLASS_NAMES, [
            (df["finish_class"] == i).sum() for i in range(len(FINISH_CLASS_NAMES))
        ])),
    )
    return df.sort_values("date").reset_index(drop=True)


# ── Train / test split ────────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame, split_ratio: float = TRAIN_TEST_SPLIT):
    idx = int(len(df) * split_ratio)
    return df.iloc[:idx], df.iloc[idx:]


# ── Preprocess ────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop metadata + finish_class + method; return (X, y)."""
    drop_cols = META_COLS + ["finish_class", "method", TARGET_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).fillna(0)
    y = df["finish_class"]
    return X, y


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(df: pd.DataFrame, n_splits: int = 5) -> list[float]:
    log.info("Running %d-fold TimeSeriesSplit CV for finish-type model…", n_splits)
    tss = TimeSeriesSplit(n_splits=n_splits)
    X_all, y_all = preprocess(df)
    scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(tss.split(X_all), 1):
        X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

        m = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
        )
        m.fit(X_tr, y_tr, verbose=False)
        score = accuracy_score(y_val, m.predict(X_val))
        scores.append(score)
        log.info("  Fold %d: %.2f%%", fold, score * 100)

    log.info("CV Mean: %.2f%%  ± %.2f%%", np.mean(scores) * 100, np.std(scores) * 100)
    return scores


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_confusion(y_test: pd.Series, predictions: np.ndarray) -> None:
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"Pred {n}" for n in FINISH_CLASS_NAMES],
        yticklabels=[f"True {n}" for n in FINISH_CLASS_NAMES],
    )
    plt.title("Finish Type — Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_finish_dataset()

    cv_scores = cross_validate(df)

    df_train, df_test = time_series_split(df)
    X_train, y_train  = preprocess(df_train)
    X_test,  y_test   = preprocess(df_test)
    feature_names     = list(X_train.columns)

    log.info("Training finish-type XGBoost on %d samples…", len(X_train))
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, verbose=False)

    predictions = model.predict(X_test)
    proba       = model.predict_proba(X_test)
    acc         = accuracy_score(y_test, predictions)

    log.info("Test Accuracy: %.2f%%", acc * 100)
    log.info("Mean CV Accuracy: %.2f%% ± %.2f%%", np.mean(cv_scores) * 100, np.std(cv_scores) * 100)

    print("\n=== FINISH TYPE MODEL RESULTS ===")
    print(f"Test Accuracy:    {acc:.2%}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")
    print(f"\nClass distribution in test set:")
    for i, name in enumerate(FINISH_CLASS_NAMES):
        n = (y_test == i).sum()
        print(f"  {name:12s}: {n:5d}  ({n/len(y_test):.1%})")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=FINISH_CLASS_NAMES))

    # Feature importance (top 15)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    print("\nTop 15 Features (XGBoost importance):")
    print(importances.nlargest(15).to_string())

    plot_confusion(y_test, predictions)

    # Save
    log.info("Saving finish-type model to %s…", MODEL_FINISH_PATH)
    joblib.dump(model,         MODEL_FINISH_PATH)
    joblib.dump(feature_names, MODEL_FINISH_FEATURES)
    log.info("Saved: %s, %s", MODEL_FINISH_PATH.name, MODEL_FINISH_FEATURES.name)


if __name__ == "__main__":
    main()
