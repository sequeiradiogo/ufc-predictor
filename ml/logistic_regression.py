"""
logistic_regression.py — Logistic Regression model for UFC fight prediction.

Replaces: 'logistic regression.py'  (old file with space in name kept for
           backward compatibility but this is the canonical version)

Run:
    python ml/logistic_regression.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, auc, roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ── Project imports ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    CSV_WITH_ELO as CSV_PATH,
    TARGET_COL, META_COLS,
    TRAIN_TEST_SPLIT, RANDOM_STATE,
    MODEL_LR_PATH, MODEL_LR_SCALER, MODEL_LR_FEATURES,
)
from logger import get_logger

log = get_logger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    log.info("Loading data from %s…", path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("Loaded %d rows × %d columns.", *df.shape)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop metadata columns, fill NaNs, return (X, y)."""
    X = df.drop(columns=META_COLS).fillna(0)
    y = df[TARGET_COL]
    return X, y


# ── Train/test split ──────────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame, split_ratio: float = TRAIN_TEST_SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 80/20 split — no shuffling."""
    idx      = int(len(df) * split_ratio)
    df_train = df.iloc[:idx]
    df_test  = df.iloc[idx:]
    log.info("Train: %d fights  (up to %s)", len(df_train), df_train["date"].max().date())
    log.info("Test:  %d fights  (from %s)", len(df_test),  df_test["date"].min().date())
    return df_train, df_test


# ── Dataset symmetry ──────────────────────────────────────────────────────────

def make_symmetric(df: pd.DataFrame) -> pd.DataFrame:
    """Flip Red ↔ Blue to double dataset and remove corner bias. Train set only."""
    df_flip = df.copy()
    df_flip[TARGET_COL] = 1 - df[TARGET_COL]
    diff_cols = [c for c in df.columns if "_diff" in c]
    df_flip[diff_cols] = df_flip[diff_cols] * -1
    return pd.concat([df, df_flip], ignore_index=True).sort_values("date").reset_index(drop=True)


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(df: pd.DataFrame, n_splits: int = 5) -> list[float]:
    """TimeSeriesSplit CV — train on older fights, validate on newer."""
    log.info("Running %d-fold TimeSeriesSplit cross-validation…", n_splits)
    tss   = TimeSeriesSplit(n_splits=n_splits)
    X_all, y_all = preprocess_data(df)
    scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(tss.split(X_all), 1):
        train_fold = make_symmetric(df.iloc[train_idx].copy())
        X_tr, y_tr = preprocess_data(train_fold)
        X_val = X_all.iloc[val_idx]
        y_val = y_all.iloc[val_idx]

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        m = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        m.fit(X_tr_s, y_tr)
        score = accuracy_score(y_val, m.predict(X_val_s))
        scores.append(score)
        log.info("  Fold %d: %.2f%%", fold, score * 100)

    log.info("CV Mean: %.2f%%  ± %.2f%%", np.mean(scores) * 100, np.std(scores) * 100)
    return scores


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_model_performance(
    y_test: pd.Series,
    predictions: np.ndarray,
    probs_uncal: np.ndarray,
    probs_cal: np.ndarray | None = None,
) -> None:
    """Confusion matrix, ROC curve, and optional calibration curve."""
    n = 3 if probs_cal is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Pred Blue", "Pred Red"],
                yticklabels=["Actual Blue", "Actual Red"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs_uncal)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random guess")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    # Calibration curve
    if probs_cal is not None:
        CalibrationDisplay.from_predictions(y_test, probs_cal,   n_bins=10, ax=axes[2], name="Calibrated")
        CalibrationDisplay.from_predictions(y_test, probs_uncal, n_bins=10, ax=axes[2], name="Uncalibrated")
        axes[2].set_title("Calibration Curve")

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_data(CSV_PATH)

    cv_scores = cross_validate(df)

    # 70 / 10 / 20 chronological split
    n         = len(df)
    train_end = int(n * 0.70)
    cal_end   = int(n * 0.80)

    df_train = make_symmetric(df.iloc[:train_end].copy())
    df_cal   = df.iloc[train_end:cal_end]
    df_test  = df.iloc[cal_end:]

    log.info("Split — Train: %d  Cal: %d  Test: %d", len(df_train), len(df_cal), len(df_test))

    X_train, y_train = preprocess_data(df_train)
    X_cal,   y_cal   = preprocess_data(df_cal)
    X_test,  y_test  = preprocess_data(df_test)
    feature_names = list(X_train.columns)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s   = scaler.transform(X_cal)
    X_test_s  = scaler.transform(X_test)

    log.info("Training Logistic Regression…")
    base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    base_model.fit(X_train_s, y_train)

    # Platt scaling — fit a 1-feature logistic regression on the calibration set
    # to map raw scores → calibrated probabilities.  Equivalent to cv='prefit'
    # (removed in sklearn 1.8) but version-independent.
    log.info("Calibrating probabilities (Platt scaling)…")
    probs_uncal     = base_model.predict_proba(X_test_s)[:, 1]
    probs_cal_raw   = base_model.predict_proba(X_cal_s)[:, 1]

    platt = LogisticRegression(max_iter=1000)
    platt.fit(probs_cal_raw.reshape(-1, 1), y_cal)

    probs_cal    = platt.predict_proba(probs_uncal.reshape(-1, 1))[:, 1]
    # For predictions, apply calibrated threshold
    predictions  = (probs_cal >= 0.5).astype(int)

    acc          = accuracy_score(y_test, predictions)
    brier_before = brier_score_loss(y_test, probs_uncal)
    brier_after  = brier_score_loss(y_test, probs_cal)

    log.info("Test Accuracy:    %.2f%%", acc * 100)
    log.info("Brier score  — before calibration: %.4f  after: %.4f", brier_before, brier_after)
    log.info("Mean CV Accuracy: %.2f%% ± %.2f%%", np.mean(cv_scores) * 100, np.std(cv_scores) * 100)

    print("\n=== RESULTS ===")
    print(f"Test Accuracy:    {acc:.2%}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")
    print(f"Brier Score — uncalibrated: {brier_before:.4f}  calibrated: {brier_after:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    plot_model_performance(y_test, predictions, probs_uncal, probs_cal)

    coeffs = pd.DataFrame({
        "Feature": feature_names,
        "Weight":  base_model.coef_[0],
    })
    coeffs["Abs"] = coeffs["Weight"].abs()
    print("\nTop 10 Features (by coefficient magnitude):")
    print(coeffs.sort_values("Abs", ascending=False).head(10)[["Feature", "Weight"]].to_string(index=False))

    actual = "Red" if y_test.iloc[-1] == 1 else "Blue"
    pred   = "Red" if predictions[-1]  == 1 else "Blue"
    conf   = probs_cal[-1] if pred == "Red" else 1 - probs_cal[-1]
    log.info("Sample prediction — Actual: %s | Predicted: %s (%.1f%%)", actual, pred, conf * 100)

    # Save — we persist base_model + platt scaler + feature scaler + feature names.
    # predict.py applies: StandardScaler → base_model.predict_proba → Platt → probabilities
    log.info("Saving model artifacts to %s…", MODEL_LR_PATH.parent)
    joblib.dump({"base": base_model, "platt": platt}, MODEL_LR_PATH)
    joblib.dump(scaler,        MODEL_LR_SCALER)
    joblib.dump(feature_names, MODEL_LR_FEATURES)
    log.info("Saved: %s, %s, %s", MODEL_LR_PATH.name, MODEL_LR_SCALER.name, MODEL_LR_FEATURES.name)


if __name__ == "__main__":
    main()
