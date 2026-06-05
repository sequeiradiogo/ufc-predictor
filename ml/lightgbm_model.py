"""
lightgbm_model.py -- LightGBM model for UFC fight prediction.

Run:
    python ml/lightgbm_model.py
    python ml/lightgbm_model.py --tune --trials 100
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

# -- Project imports -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    CSV_WITH_ELO as CSV_PATH,
    TARGET_COL, META_COLS,
    TRAIN_TEST_SPLIT, RANDOM_STATE,
    MODEL_LGBM_PATH, MODEL_LGBM_FEATURES,
    LGBM_PARAMS,
    EXCLUDED_FEATURES,
)
from utils.logger import get_logger

DEFAULT_PARAMS: dict = {**LGBM_PARAMS, "random_state": RANDOM_STATE}

log = get_logger(__name__)


# -- Data loading --------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    log.info("Loading data from %s...", path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("Loaded %d rows x %d columns.", *df.shape)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop metadata columns, fill NaNs/infs with 0, return (X, y)."""
    X = df.drop(columns=META_COLS).fillna(0).replace([np.inf, -np.inf], 0)
    y = df[TARGET_COL]
    return X, y


# -- Train/test split ----------------------------------------------------------

def time_series_split(df: pd.DataFrame, split_ratio: float = TRAIN_TEST_SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 80/20 split -- no shuffling."""
    idx      = int(len(df) * split_ratio)
    df_train = df.iloc[:idx]
    df_test  = df.iloc[idx:]
    log.info("Train: %d fights  (up to %s)", len(df_train), df_train["date"].max().date())
    log.info("Test:  %d fights  (from %s)", len(df_test),  df_test["date"].min().date())
    return df_train, df_test


# -- Dataset symmetry ----------------------------------------------------------

def make_symmetric(df: pd.DataFrame) -> pd.DataFrame:
    """Flip Red <-> Blue to double the dataset and remove corner bias. Train set only."""
    df_flip = df.copy()
    df_flip[TARGET_COL] = 1 - df[TARGET_COL]
    diff_cols = [c for c in df.columns if "_diff" in c]
    df_flip[diff_cols] = df_flip[diff_cols] * -1
    if "striker_vs_wrestler" in df.columns and "wrestler_vs_striker" in df.columns:
        df_flip["striker_vs_wrestler"] = df["wrestler_vs_striker"]
        df_flip["wrestler_vs_striker"] = df["striker_vs_wrestler"]
    return pd.concat([df, df_flip], ignore_index=True).sort_values("date").reset_index(drop=True)


# -- Cross-validation ----------------------------------------------------------

def cross_validate(df: pd.DataFrame, params: dict | None = None, n_splits: int = 5) -> list[float]:
    """TimeSeriesSplit CV -- train on older fights, validate on newer."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    log.info("Running %d-fold TimeSeriesSplit cross-validation...", n_splits)
    tss   = TimeSeriesSplit(n_splits=n_splits)
    X_all, y_all = preprocess_data(df)
    scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(tss.split(X_all), 1):
        train_fold = make_symmetric(df.iloc[train_idx].copy())
        X_tr, y_tr = preprocess_data(train_fold)
        X_val = X_all.iloc[val_idx]
        y_val = y_all.iloc[val_idx]

        m = LGBMClassifier(**p, verbosity=-1)
        m.fit(X_tr, y_tr)
        score = accuracy_score(y_val, m.predict(X_val))
        scores.append(score)
        log.info("  Fold %d: %.2f%%", fold, score * 100)

    log.info("CV Mean: %.2f%%  +/- %.2f%%", np.mean(scores) * 100, np.std(scores) * 100)
    return scores


# -- Optuna hyperparameter tuning ----------------------------------------------

def tune_hyperparameters(df: pd.DataFrame, n_trials: int = 50) -> dict:
    """Use Optuna to find the best LightGBM hyperparameters via TimeSeriesSplit CV."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("Optuna not installed. Run: pip install optuna")
        return DEFAULT_PARAMS.copy()

    log.info("Starting Optuna hyperparameter search (%d trials)...", n_trials)

    tss = TimeSeriesSplit(n_splits=5)
    X_all, y_all = preprocess_data(df)

    def objective(trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state":     RANDOM_STATE,
        }

        fold_scores: list[float] = []
        for train_idx, val_idx in tss.split(X_all):
            train_fold = make_symmetric(df.iloc[train_idx].copy())
            X_tr, y_tr = preprocess_data(train_fold)
            X_val = X_all.iloc[val_idx]
            y_val = y_all.iloc[val_idx]

            m = LGBMClassifier(**params, verbosity=-1)
            m.fit(X_tr, y_tr)
            fold_scores.append(accuracy_score(y_val, m.predict(X_val)))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["random_state"] = RANDOM_STATE
    log.info("Best CV accuracy: %.2f%%", study.best_value * 100)
    log.info("Best params: %s", best)
    return best


# -- Visualisation -------------------------------------------------------------

def plot_model_performance(
    y_test: pd.Series,
    predictions: np.ndarray,
    probs: np.ndarray,
) -> None:
    """Confusion matrix + ROC curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax1,
                xticklabels=["Pred Blue", "Pred Red"],
                yticklabels=["Actual Blue", "Actual Red"])
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random guess")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


# -- Main ----------------------------------------------------------------------

def main(tune: bool = False, n_trials: int = 50) -> None:
    df = load_data(CSV_PATH)

    if tune:
        log.info("Hyperparameter tuning mode -- this may take several minutes.")
        best_params = tune_hyperparameters(df, n_trials=n_trials)
    else:
        best_params = DEFAULT_PARAMS.copy()

    cv_scores = cross_validate(df, params=best_params)

    train_df, test_df = time_series_split(df)
    train_df = make_symmetric(train_df)

    X_train, y_train = preprocess_data(train_df)
    X_test,  y_test  = preprocess_data(test_df)
    X_train = X_train.drop(columns=[f for f in EXCLUDED_FEATURES if f in X_train.columns])
    X_test  = X_test.drop(columns=[f for f in EXCLUDED_FEATURES if f in X_test.columns])
    feature_names = list(X_train.columns)

    log.info("Training LightGBM model...")
    model = LGBMClassifier(**best_params, verbosity=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probs       = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    log.info("Test Accuracy:    %.2f%%", acc * 100)
    log.info("Mean CV Accuracy: %.2f%% +/- %.2f%%", np.mean(cv_scores) * 100, np.std(cv_scores) * 100)

    print("\n=== RESULTS ===")
    print(f"Test Accuracy:    {acc:.2%}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2%} +/- {np.std(cv_scores):.2%}")
    if tune:
        print(f"Best params:      {best_params}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    plot_model_performance(y_test, predictions, probs)

    importances = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(10)
    print("\nTop 10 Features:")
    print(importances.to_string(index=False))

    actual = "Red" if y_test.iloc[-1] == 1 else "Blue"
    pred   = "Red" if predictions[-1]  == 1 else "Blue"
    conf   = probs[-1] if pred == "Red" else 1 - probs[-1]
    log.info("Sample prediction -- Actual: %s | Predicted: %s (%.1f%%)", actual, pred, conf * 100)

    log.info("Saving model artifacts to %s...", MODEL_LGBM_PATH.parent)
    joblib.dump(model,         MODEL_LGBM_PATH)
    joblib.dump(feature_names, MODEL_LGBM_FEATURES)
    log.info("Saved: %s", MODEL_LGBM_PATH.name)
    log.info("Saved: %s", MODEL_LGBM_FEATURES.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM UFC fight predictor.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search before training (slower).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50). Only used with --tune.",
    )
    args = parser.parse_args()
    main(tune=args.tune, n_trials=args.trials)
