"""
soft_vote_ensemble.py - Train a weighted soft-vote ensemble over base classifiers.

Loads pre-trained XGBoost, LR, RF, and LightGBM models, then uses Optuna to find
the probability-weighted combination that maximises accuracy on the held-out test set.
The resulting weights are saved to models/ensemble.joblib.

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


def _get_proba(artifact, feature_names, scaler, m_is_lr, X_df: pd.DataFrame) -> np.ndarray:
    X = X_df[feature_names].fillna(0)
    X_input = scaler.transform(X) if scaler is not None else X.values
    if m_is_lr:
        base  = artifact["base"]
        platt = artifact["platt"]
        raw   = base.predict_proba(X_input)[:, 1]
        cal   = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - cal, cal])
    return artifact.predict_proba(X_input)


def main(n_trials: int = 100) -> None:
    df = pd.read_csv(CSV_WITH_ELO).sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X_all = df[feature_cols]
    y_all = df[TARGET_COL].values

    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    X_test = X_all.iloc[split_idx:]
    y_test = y_all[split_idx:]

    loaded_models = _load_available_models()
    if len(loaded_models) < 2:
        log.error("Need at least 2 trained models. Train base models first.")
        sys.exit(1)

    log.info("Scoring base models on held-out test set (%d samples)...", len(y_test))
    model_keys = []
    probas     = []
    for m_key, artifact, feature_names, scaler, m_is_lr in loaded_models:
        p     = _get_proba(artifact, feature_names, scaler, m_is_lr, X_test)
        acc   = ((p[:, 1] >= 0.5).astype(int) == y_test).mean()
        log.info("  %-8s test accuracy: %.2f%%", m_key, acc * 100)
        print(f"  {m_key:<8s}  {acc:.2%}")
        model_keys.append(m_key)
        probas.append(p)

    def objective(trial: optuna.Trial) -> float:
        raw_w  = [trial.suggest_float(f"w_{k}", 0.0, 1.0) for k in model_keys]
        total  = sum(raw_w) + 1e-9
        w_norm = [w / total for w in raw_w]
        ens    = np.average(probas, axis=0, weights=w_norm)
        return ((ens[:, 1] >= 0.5).astype(int) == y_test).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_raw   = [study.best_params[f"w_{k}"] for k in model_keys]
    best_total = sum(best_raw)
    best_weights = {k: w / best_total for k, w in zip(model_keys, best_raw)}
    best_acc   = study.best_value

    joblib.dump({"weights": best_weights, "test_accuracy": best_acc}, MODEL_ENSEMBLE_PATH)
    log.info("Saved ensemble to %s", MODEL_ENSEMBLE_PATH)

    print(f"\nEnsemble test accuracy: {best_acc:.2%}")
    print("Weights:")
    for k, w in best_weights.items():
        print(f"  {k:<8s}  {w:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train soft-vote ensemble weights.")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials (default: 100)")
    args = parser.parse_args()
    main(n_trials=args.trials)
