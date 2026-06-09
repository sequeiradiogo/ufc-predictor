"""
mlp_model.py -- MLP neural network for UFC fight prediction.

Run:
    python ml/mlp_model.py
    python ml/mlp_model.py --tune --trials 100
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    CSV_WITH_ELO as CSV_PATH,
    TARGET_COL, META_COLS,
    TRAIN_TEST_SPLIT, RANDOM_STATE,
    MODEL_MLP_PATH, MODEL_MLP_FEATURES, MODEL_MLP_SCALER,
    MLP_PARAMS,
    EXCLUDED_FEATURES,
)
from ml.ML_data_preparation import compute_sample_weights
from ml.pytorch_mlp import PyTorchMLP
from utils.logger import get_logger

DEFAULT_PARAMS: dict = {**MLP_PARAMS, "random_state": RANDOM_STATE}

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
    X = X.drop(columns=[f for f in EXCLUDED_FEATURES if f in X.columns])
    y = df[TARGET_COL]
    return X, y


# -- Train/test split ----------------------------------------------------------

def time_series_split(df: pd.DataFrame, split_ratio: float = TRAIN_TEST_SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        sc = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)

        m = PyTorchMLP(**p)
        m.fit(X_tr_s, y_tr.values)
        score = accuracy_score(y_val, m.predict(X_val_s))
        scores.append(score)
        log.info("  Fold %d: %.2f%%", fold, score * 100)

    log.info("CV Mean: %.2f%%  +/- %.2f%%", np.mean(scores) * 100, np.std(scores) * 100)
    return scores


# -- Optuna hyperparameter tuning ----------------------------------------------

def tune_hyperparameters(df: pd.DataFrame, n_trials: int = 50) -> dict:
    """Use Optuna to find the best PyTorch MLP hyperparameters via TimeSeriesSplit CV."""
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
            "hidden_sizes": trial.suggest_categorical(
                "hidden_sizes",
                [(64, 64), (128, 64), (128, 128), (256, 128, 64), (256, 128, 64, 32)],
            ),
            "dropout":       trial.suggest_float("dropout",       0.1, 0.5),
            "lr":            trial.suggest_float("lr",            1e-4, 1e-2, log=True),
            "weight_decay":  trial.suggest_float("weight_decay",  1e-5, 1e-2, log=True),
            "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128]),
            "batch_norm":    trial.suggest_categorical("batch_norm", [True, False]),
            "max_epochs":    200,
            "patience":      15,
            "random_state":  RANDOM_STATE,
        }

        fold_scores: list[float] = []
        for train_idx, val_idx in tss.split(X_all):
            train_fold = make_symmetric(df.iloc[train_idx].copy())
            X_tr, y_tr = preprocess_data(train_fold)
            X_val = X_all.iloc[val_idx]
            y_val = y_all.iloc[val_idx]

            sc = StandardScaler()
            X_tr_s  = sc.fit_transform(X_tr)
            X_val_s = sc.transform(X_val)

            m = PyTorchMLP(**params)
            m.fit(X_tr_s, y_tr.values)
            fold_scores.append(accuracy_score(y_val, m.predict(X_val_s)))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))

    best = {**study.best_params, "random_state": RANDOM_STATE, "max_epochs": 300, "patience": 20}
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["Pred Blue", "Pred Red"],
                yticklabels=["Actual Blue", "Actual Red"])
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
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
    feature_names = list(X_train.columns)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    log.info("Training PyTorch MLP model...")
    base = PyTorchMLP(**best_params)
    base.fit(X_train_s, y_train.values)

    # Isotonic calibration on a holdout slice inside the training window
    cal_start  = int(len(train_df) * 0.80)
    X_cal_s    = X_train_s[cal_start:]
    y_cal      = y_train.iloc[cal_start:]
    raw_cal    = base.predict_proba(X_cal_s)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, y_cal.values)

    raw_probs   = base.predict_proba(X_test_s)[:, 1]
    probs       = calibrator.predict(raw_probs)
    predictions = (probs >= 0.5).astype(int)

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

    log.info("Saving model artifacts to %s...", MODEL_MLP_PATH.parent)
    joblib.dump({"base": base, "platt": calibrator}, MODEL_MLP_PATH)
    joblib.dump(scaler,        MODEL_MLP_SCALER)
    joblib.dump(feature_names, MODEL_MLP_FEATURES)
    log.info("Saved: %s", MODEL_MLP_PATH.name)
    log.info("Saved: %s", MODEL_MLP_SCALER.name)
    log.info("Saved: %s", MODEL_MLP_FEATURES.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP neural network UFC fight predictor.")
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
