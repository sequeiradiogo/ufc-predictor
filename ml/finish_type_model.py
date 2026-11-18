"""
finish_type_model.py -- Secondary model predicting HOW a fight ends.

Classes:
    0 = Decision   (unanimous, split, or majority)
    1 = KO / TKO
    2 = Submission

How a fight ends doesn't depend on who is in the red corner, so the model uses
corner-invariant features only (issue #30): for each stat in
config.FINISH_INV_STATS, the sum and the absolute difference of the two fighters'
pre-fight values, plus the fight context (division one-hots, title_fight,
five_rounds). Two KO artists (0.8 - 0.8) and two decision fighters (0.2 - 0.2)
look the same to a corner diff; their sum tells them apart. The columns are in
the v1 feature CSV (ML_data_preparation_v1.py) and built live by
predict.build_feature_vector.

Until 2026-11 the model was a 400-tree XGBoost on every corner-diff column; its
test log-loss was worse than division + title + rounds alone at 8 of 8 forecast
origins (and worse than the class frequencies at 7 of 8). The current one is a
small XGBoost (config.FINISH_XGB_PARAMS) on the invariant features: -0.028
log-loss vs that context-only model over 5 exploration origins (5/5) and -0.033
over 3 unseen confirmation origins (3/3), 5 seeds each.

Fights with unknown or rare finish methods (DQ, Overturned, etc.) are excluded.

Run:
    python ml/finish_type_model.py            # eval tier -> models_v1/
    python ml/finish_type_model.py --prod      # prod tier -> models_v1_prod/ (100% of data)

Saves:
    models_v1/finish_type.joblib, models_v1/finish_type_features.joblib
    (or models_v1_prod/... with --prod)
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# -- Project imports ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DB_V1_PATH,
    CSV_V1_WITH_ELO,
    TRAIN_TEST_SPLIT, RANDOM_STATE,
    FINISH_METHOD_MAP, FINISH_CLASS_NAMES,
    FINISH_ONLY_FEATURES, FINISH_XGB_PARAMS,
    MODEL_V1_FINISH_PATH, MODEL_V1_FINISH_FEATURES,
    MODEL_V1_PROD_FINISH_PATH, MODEL_V1_PROD_FINISH_FEATURES,
)
from ml.shrinkage import csv_settings_path, read_settings, stamp_models_dir
from utils.logger import get_logger

log = get_logger(__name__)

_CLASSES = list(range(len(FINISH_CLASS_NAMES)))


# -- Data loading ---------------------------------------------------------------

def load_finish_dataset() -> pd.DataFrame:
    """
    Load the v1 feature CSV, attach the finish-type label from the mdabbert DB,
    and return a DataFrame ready for training.

    Rows with unmapped finish methods (DQ, Overturned, etc.) are dropped.
    """
    if not CSV_V1_WITH_ELO.exists():
        raise FileNotFoundError(
            f"v1 ML dataset not found at '{CSV_V1_WITH_ELO}'.\n"
            "Run 'python ml/ML_data_preparation_v1.py' first."
        )

    log.info("Loading v1 ML dataset from %s...", CSV_V1_WITH_ELO)
    ml = pd.read_csv(CSV_V1_WITH_ELO)
    ml["date"] = pd.to_datetime(ml["date"])
    missing = [c for c in FINISH_ONLY_FEATURES if c not in ml.columns]
    if missing:
        raise ValueError(f"{CSV_V1_WITH_ELO.name} has no {missing[:3]}... -- "
                         "rebuild it with 'python ml/ML_data_preparation_v1.py'.")

    # Fetch method column from the mdabbert DB
    log.info("Fetching finish methods from database...")
    conn = sqlite3.connect(str(DB_V1_PATH))
    methods = pd.read_sql_query("SELECT fight_id, method FROM fights", conn)
    conn.close()

    df = ml.merge(methods, on="fight_id", how="left")

    # Map to integer class; unmapped rows -> NaN -> drop
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
            (df["finish_class"] == i).sum() for i in _CLASSES
        ])),
    )
    return df.sort_values("date").reset_index(drop=True)


# -- Features -------------------------------------------------------------------

def context_columns(df: pd.DataFrame) -> list[str]:
    """Fight context: division one-hots, title_fight, five_rounds."""
    return [c for c in df.columns if c.startswith("div_")] + ["title_fight", "five_rounds"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Corner-invariant stats + fight context -- no corner diffs."""
    inv = [c for c in FINISH_ONLY_FEATURES if c != "five_rounds"]
    return inv + context_columns(df)


def preprocess(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) on *cols* (default: feature_columns)."""
    cols = cols or feature_columns(df)
    return df[cols].fillna(0), df["finish_class"]


def make_model(seed: int = RANDOM_STATE) -> XGBClassifier:
    return XGBClassifier(**FINISH_XGB_PARAMS, objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", random_state=seed)


def context_baseline(df_train: pd.DataFrame) -> LogisticRegression:
    """Division + title + rounds only: the bar the model must beat (issue #30)."""
    X, y = preprocess(df_train, context_columns(df_train))
    return LogisticRegression(C=1.0, max_iter=2000).fit(X.values, y.values)


# -- Train / test split ---------------------------------------------------------

def time_series_split(df: pd.DataFrame, split_ratio: float = TRAIN_TEST_SPLIT):
    idx = int(len(df) * split_ratio)
    return df.iloc[:idx], df.iloc[idx:]


# -- Cross-validation -------------------------------------------------------------

def cross_validate(df: pd.DataFrame, n_splits: int = 5) -> list[float]:
    """Time-ordered CV; returns the log-loss of each fold."""
    log.info("Running %d-fold TimeSeriesSplit CV for finish-type model...", n_splits)
    X_all, y_all = preprocess(df)
    scores: list[float] = []
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X_all), 1):
        m = make_model().fit(X_all.iloc[tr].values, y_all.iloc[tr].values, verbose=False)
        score = log_loss(y_all.iloc[va], m.predict_proba(X_all.iloc[va].values), labels=_CLASSES)
        scores.append(score)
        log.info("  Fold %d: log-loss %.4f", fold, score)
    log.info("CV mean log-loss: %.4f +/- %.4f", np.mean(scores), np.std(scores))
    return scores


# -- Main -------------------------------------------------------------------------

def main(prod: bool = False) -> None:
    df = load_finish_dataset()
    # Same tier directory as the win models: must have been built with the same
    # rate-shrinkage settings (ml/shrinkage.py).
    stamp_models_dir(MODEL_V1_PROD_FINISH_PATH.parent if prod else MODEL_V1_FINISH_PATH.parent,
                     read_settings(csv_settings_path(CSV_V1_WITH_ELO)), partial=True)

    if prod:
        # Production tier: train on 100% of data, no held-out evaluation.
        X_train, y_train = preprocess(df)
        log.info("Training PRODUCTION finish-type XGBoost on %d samples (100%% of data)...", len(X_train))
        model = make_model().fit(X_train.values, y_train.values, verbose=False)
        finish_path, finish_feats_path = MODEL_V1_PROD_FINISH_PATH, MODEL_V1_PROD_FINISH_FEATURES
    else:
        cv_scores = cross_validate(df)

        df_train, df_test = time_series_split(df)
        X_train, y_train = preprocess(df_train)
        X_test, y_test = preprocess(df_test)

        log.info("Training finish-type XGBoost on %d samples...", len(X_train))
        model = make_model().fit(X_train.values, y_train.values, verbose=False)

        proba = model.predict_proba(X_test.values)
        predictions = proba.argmax(axis=1)
        ll = log_loss(y_test, proba, labels=_CLASSES)
        freq = np.bincount(y_train, minlength=3) / len(y_train)
        ll_majority = log_loss(y_test, np.tile(freq, (len(y_test), 1)), labels=_CLASSES)
        base = context_baseline(df_train)
        ll_context = log_loss(y_test, base.predict_proba(preprocess(df_test, context_columns(df_test))[0].values),
                              labels=_CLASSES)

        print("\n=== FINISH TYPE MODEL RESULTS ===")
        print(f"Test log-loss:            {ll:.4f}")
        print(f"  context only (div/title/rounds): {ll_context:.4f}")
        print(f"  class frequencies:               {ll_majority:.4f}")
        print(f"Test accuracy:            {accuracy_score(y_test, predictions):.2%}")
        print(f"Mean CV log-loss:         {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
        print("\nClass distribution in test set (mean predicted probability):")
        for i, name in enumerate(FINISH_CLASS_NAMES):
            n = (y_test == i).sum()
            print(f"  {name:12s}: {n:5d}  ({n/len(y_test):.1%})  predicted {proba[:, i].mean():.1%}")
        print("\nClassification Report (argmax; the output shows probabilities):")
        print(classification_report(y_test, predictions, labels=_CLASSES,
                                    target_names=FINISH_CLASS_NAMES, zero_division=0))

        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        print("\nTop 15 Features (XGBoost importance):")
        print(importances.nlargest(15).to_string())

        finish_path, finish_feats_path = MODEL_V1_FINISH_PATH, MODEL_V1_FINISH_FEATURES

    # Save
    log.info("Saving finish-type model to %s...", finish_path)
    joblib.dump(model, finish_path)
    joblib.dump(list(X_train.columns), finish_feats_path)
    log.info("Saved: %s, %s", finish_path.name, finish_feats_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the finish-type classifier.")
    parser.add_argument("--prod", action="store_true",
                         help="Train the production tier on 100%% of data (models_v1_prod/).")
    args = parser.parse_args()
    main(prod=args.prod)
