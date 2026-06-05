"""
feature_importance.py -- Measure permutation importance across all 4 base models
and flag low-signal features as pruning candidates.

Usage:
    python scripts/feature_importance.py

Outputs:
    feature_importance.csv  (project root)

After reviewing the CSV, populate EXCLUDED_FEATURES in config.py with any
prune_candidate=True features you want to drop, then retrain:
    python run_pipeline.py --steps 5,6,7,8,9,10
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import get_logger

log = get_logger(__name__)

_PRUNE_THRESHOLD = 0.001   # mean permutation importance below this -> low-signal
_MIN_MODELS_BELOW = 3      # must be below threshold in at least this many models


def _load_test_set():
    """Return (X_test, y_test) using the same 80/20 chronological split as training."""
    df = pd.read_csv(config.CSV_WITH_ELO)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    idx = int(len(df) * config.TRAIN_TEST_SPLIT)
    df_test = df.iloc[idx:]
    X = df_test.drop(columns=config.META_COLS).fillna(0).replace([np.inf, -np.inf], 0)
    y = df_test[config.TARGET_COL]
    log.info("Test set: %d fights (from %s)", len(df_test), df_test["date"].min().date())
    return X, y


def _perm_importance(model, X, y, feature_names, label):
    log.info("Computing permutation importance for %s (%d features)...", label, len(feature_names))
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return pd.Series(result.importances_mean, index=feature_names, name=label)


def main():
    X_test, y_test = _load_test_set()

    series = {}

    # -- XGBoost --
    xgb_features = joblib.load(config.MODEL_XGB_FEATURES)
    xgb_model    = joblib.load(config.MODEL_XGB_PATH)
    X_xgb = X_test[xgb_features].values
    log.info("XGBoost sanity accuracy: %.2f%%",
             accuracy_score(y_test, xgb_model.predict(X_xgb)) * 100)
    series["xgb_imp"] = _perm_importance(xgb_model, X_xgb, y_test, xgb_features, "XGBoost")

    # -- Logistic Regression (scaled; use base model, not Platt wrapper) --
    lr_features  = joblib.load(config.MODEL_LR_FEATURES)
    lr_artifact  = joblib.load(config.MODEL_LR_PATH)
    lr_scaler    = joblib.load(config.MODEL_LR_SCALER)
    lr_base      = lr_artifact["base"]
    X_lr_s = lr_scaler.transform(X_test[lr_features])
    log.info("LR sanity accuracy: %.2f%%",
             accuracy_score(y_test, lr_base.predict(X_lr_s)) * 100)
    series["lr_imp"] = _perm_importance(lr_base, X_lr_s, y_test, lr_features, "LogReg")

    # -- Random Forest --
    rf_features = joblib.load(config.MODEL_RF_FEATURES)
    rf_model    = joblib.load(config.MODEL_RF_PATH)
    X_rf = X_test[rf_features].values
    log.info("RF sanity accuracy: %.2f%%",
             accuracy_score(y_test, rf_model.predict(X_rf)) * 100)
    series["rf_imp"] = _perm_importance(rf_model, X_rf, y_test, rf_features, "RandomForest")

    # -- LightGBM --
    lgbm_features = joblib.load(config.MODEL_LGBM_FEATURES)
    lgbm_model    = joblib.load(config.MODEL_LGBM_PATH)
    X_lgbm = X_test[lgbm_features].values
    log.info("LightGBM sanity accuracy: %.2f%%",
             accuracy_score(y_test, lgbm_model.predict(X_lgbm)) * 100)
    series["lgbm_imp"] = _perm_importance(lgbm_model, X_lgbm, y_test, lgbm_features, "LightGBM")

    # -- Aggregate --
    all_features = sorted(
        set(xgb_features) | set(lr_features) | set(rf_features) | set(lgbm_features)
    )
    df_imp = pd.DataFrame(index=all_features)
    df_imp.index.name = "feature"
    for col, s in series.items():
        df_imp[col] = s.reindex(df_imp.index)

    imp_cols = list(series.keys())
    df_imp["mean_imp"]     = df_imp[imp_cols].mean(axis=1)
    df_imp["models_below"] = (df_imp[imp_cols] < _PRUNE_THRESHOLD).sum(axis=1)
    df_imp["prune_candidate"] = (
        (df_imp["mean_imp"] < _PRUNE_THRESHOLD) &
        (df_imp["models_below"] >= _MIN_MODELS_BELOW)
    )
    df_imp = df_imp.sort_values("mean_imp", ascending=False)

    out_path = ROOT_DIR / "feature_importance.csv"
    df_imp.to_csv(out_path)
    log.info("Saved: %s", out_path)

    print("\n=== TOP 20 FEATURES (by mean permutation importance) ===")
    print(df_imp.head(20)[imp_cols + ["mean_imp"]].to_string())

    prune = df_imp[df_imp["prune_candidate"]]
    print(f"\n=== PRUNING CANDIDATES ({len(prune)} features, threshold={_PRUNE_THRESHOLD}) ===")
    if len(prune) > 0:
        print(prune[imp_cols + ["mean_imp", "models_below"]].to_string())
        print("\nPaste into config.py EXCLUDED_FEATURES if you adopt the pruning:")
        print(sorted(prune.index.tolist()))
    else:
        print("No features below threshold across 3+ models -- nothing to prune.")

    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
