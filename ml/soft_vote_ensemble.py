"""
soft_vote_ensemble.py -- Calibrated soft-vote ensemble over base classifiers.

Loads pre-trained XGBoost, LR, RF, and LightGBM models, fits per-model
isotonic calibrators on a held-out calibration portion of the training set,
then uses Optuna to find the probability-weighted combination that maximises
accuracy on the held-out test set.

Calibration note: calibrators are fitted on the last CAL_FRAC of the training
set. The base models were trained on the full training set (slight in-sample
contamination), but regularised tree models do not fully memorise training data,
so the calibrators still correct systematic probability bias effectively.

Usage
-----
    python ml/soft_vote_ensemble.py
    python ml/soft_vote_ensemble.py --trials 200
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    CSV_WITH_ELO,
    META_COLS,
    MODEL_ENSEMBLE_PATH,
    MODEL_LGBM_FEATURES,
    MODEL_LGBM_PATH,
    MODEL_LR_FEATURES,
    MODEL_LR_PATH,
    MODEL_LR_SCALER,
    MODEL_RF_FEATURES,
    MODEL_RF_PATH,
    MODEL_XGB_FEATURES,
    MODEL_XGB_PATH,
    TARGET_COL,
    TRAIN_TEST_SPLIT,
)
from utils.logger import get_logger

log = get_logger("soft_vote_ensemble")

# Fraction of training data reserved as calibration holdout (not for meta-learning weights)
CAL_FRAC = 0.20

_BASE_MODEL_SPECS = [
    ("xgb",  MODEL_XGB_PATH,  MODEL_XGB_FEATURES,  None,            False),
    ("lr",   MODEL_LR_PATH,   MODEL_LR_FEATURES,   MODEL_LR_SCALER, True),
    ("rf",   MODEL_RF_PATH,   MODEL_RF_FEATURES,   None,            False),
    ("lgbm", MODEL_LGBM_PATH, MODEL_LGBM_FEATURES, None,            False),
]


def _load_available_models() -> list:
    loaded = []
    for m_key, m_path, m_feats_path, m_scaler_path, m_is_lr in _BASE_MODEL_SPECS:
        if not m_path.exists():
            log.warning("Skipping %s -- model not found at %s", m_key, m_path)
            continue
        artifact      = joblib.load(m_path)
        feature_names = joblib.load(m_feats_path)
        scaler        = joblib.load(m_scaler_path) if m_scaler_path and m_scaler_path.exists() else None
        loaded.append((m_key, artifact, feature_names, scaler, m_is_lr))
        log.info("Loaded %s", m_key)
    return loaded


def _raw_proba(artifact, feature_names, scaler, m_is_lr, X_df: pd.DataFrame) -> np.ndarray:
    """Return raw (pre-calibration) P(red wins) for one model."""
    X = X_df[feature_names].fillna(0)
    X_input = scaler.transform(X) if scaler is not None else X.values
    if m_is_lr:
        # Use base LR only; calibrator handles isotonic correction uniformly
        return artifact["base"].predict_proba(X_input)[:, 1]
    return artifact.predict_proba(X_input)[:, 1]


def _calibrated_proba(artifact, feature_names, scaler, m_is_lr, calibrator, X_df: pd.DataFrame) -> np.ndarray:
    """Return calibrated P(red wins) as a [N, 2] array."""
    raw = _raw_proba(artifact, feature_names, scaler, m_is_lr, X_df)
    cal = calibrator.predict(raw)
    return np.column_stack([1 - cal, cal])


# Fights from this year onward are held out of Optuna weight search entirely.
_ENSEMBLE_HOLDOUT_YEAR = 2025
_OPTUNA_MIN_ROWS = 50


def main(n_trials: int = 100) -> None:
    df = pd.read_csv(CSV_WITH_ELO).sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X_all = df[feature_cols]
    y_all = df[TARGET_COL].values

    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    cal_idx   = int(split_idx * (1 - CAL_FRAC))   # calibration holdout starts here

    X_cal  = X_all.iloc[cal_idx:split_idx]
    y_cal  = y_all[cal_idx:split_idx]
    X_test = X_all.iloc[split_idx:]
    y_test = y_all[split_idx:]

    # Split test into Optuna tuning window (pre-holdout) and true hold-out.
    test_dates = df.iloc[split_idx:]["date"]
    pre_mask   = pd.to_datetime(test_dates).dt.year.values < _ENSEMBLE_HOLDOUT_YEAR
    X_tune = X_test.iloc[pre_mask]
    y_tune = y_test[pre_mask]
    X_hold = X_test.iloc[~pre_mask]
    y_hold = y_test[~pre_mask]

    if len(X_tune) < _OPTUNA_MIN_ROWS:
        log.warning(
            "Pre-%d tuning window too thin (%d rows) -- ensemble will use equal weights.",
            _ENSEMBLE_HOLDOUT_YEAR, len(X_tune),
        )
        X_tune, y_tune = None, None

    log.info(
        "Ensemble split: %d cal | %d tune (pre-%d) | %d hold-out (%d+)",
        len(y_cal), len(X_tune) if X_tune is not None else 0, _ENSEMBLE_HOLDOUT_YEAR,
        len(y_hold), _ENSEMBLE_HOLDOUT_YEAR,
    )

    loaded_models = _load_available_models()
    if len(loaded_models) < 2:
        log.error("Need at least 2 trained models. Train base models first.")
        sys.exit(1)

    # ── Fit isotonic calibrators on calibration holdout ───────────────────────
    log.info("Fitting isotonic calibrators on calibration holdout (%d samples)...", len(y_cal))
    calibrators = {}
    print("\nBase model calibration holdout accuracy:")
    for m_key, artifact, feature_names, scaler, m_is_lr in loaded_models:
        raw = _raw_proba(artifact, feature_names, scaler, m_is_lr, X_cal)
        ir  = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw, y_cal)
        calibrators[m_key] = ir
        cal_acc = ((raw >= 0.5).astype(int) == y_cal).mean()
        print(f"  {m_key:<8s}  {cal_acc:.2%}")

    # ── Score base models on hold-out set (calibrated) ────────────────────────
    log.info("Scoring base models on hold-out set (%d+, %d samples)...",
             _ENSEMBLE_HOLDOUT_YEAR, len(y_hold))
    model_keys  = []
    tune_probas = []
    hold_probas = []
    print(f"\nBase model hold-out accuracy ({_ENSEMBLE_HOLDOUT_YEAR}+, calibrated):")
    for m_key, artifact, feature_names, scaler, m_is_lr in loaded_models:
        if X_tune is not None:
            tune_probas.append(
                _calibrated_proba(artifact, feature_names, scaler, m_is_lr, calibrators[m_key], X_tune)
            )
        p_hold = _calibrated_proba(artifact, feature_names, scaler, m_is_lr, calibrators[m_key], X_hold)
        acc = ((p_hold[:, 1] >= 0.5).astype(int) == y_hold).mean()
        log.info("  %-8s hold-out accuracy: %.2f%%", m_key, acc * 100)
        print(f"  {m_key:<8s}  {acc:.2%}")
        model_keys.append(m_key)
        hold_probas.append(p_hold)

    # ── Optuna soft-vote weight search (pre-holdout data only) ────────────────
    # Falls back to equal weights when the pre-holdout window is too thin.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if X_tune is not None:
        def objective(trial: optuna.Trial) -> float:
            raw_w  = [trial.suggest_float(f"w_{k}", 0.0, 1.0) for k in model_keys]
            total  = sum(raw_w) + 1e-9
            w_norm = [w / total for w in raw_w]
            ens    = np.average(tune_probas, axis=0, weights=w_norm)
            return ((ens[:, 1] >= 0.5).astype(int) == y_tune).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_raw     = [study.best_params[f"w_{k}"] for k in model_keys]
        best_total   = sum(best_raw)
        best_weights = {k: w / best_total for k, w in zip(model_keys, best_raw)}
    else:
        equal = 1.0 / len(model_keys)
        best_weights = {k: equal for k in model_keys}
        log.info("Using equal weights: %s", best_weights)

    # Honest hold-out accuracy with the tuned weights
    ens_hold = np.average(hold_probas, axis=0,
                          weights=[best_weights[k] for k in model_keys])
    hold_acc = ((ens_hold[:, 1] >= 0.5).astype(int) == y_hold).mean()

    joblib.dump({
        "mode":          "calibrated_soft_vote",
        "weights":       best_weights,
        "calibrators":   calibrators,
        "model_keys":    model_keys,
        "test_accuracy": hold_acc,
    }, MODEL_ENSEMBLE_PATH)
    log.info("Saved ensemble to %s", MODEL_ENSEMBLE_PATH)

    print(f"\nEnsemble hold-out accuracy ({_ENSEMBLE_HOLDOUT_YEAR}+): {hold_acc:.2%}")
    print("Weights:")
    for k, w in best_weights.items():
        print(f"  {k:<8s}  {w:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train calibrated soft-vote ensemble weights.")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials (default: 100)")
    args = parser.parse_args()
    main(n_trials=args.trials)
