"""
train_v1_models.py -- Train all v1 (mdabbert) models and calibrated ensemble.

Trains XGBoost, Logistic Regression, Random Forest, LightGBM, and a
calibrated soft-vote ensemble from the v1 feature CSV. Saves all
artifacts to models_v1/.

Usage:
    python ml/train_v1_models.py
    python ml/train_v1_models.py --tune --trials 100
    python ml/train_v1_models.py --model xgb
    python ml/train_v1_models.py --model lr --tune --trials 50
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    CSV_V1_WITH_ELO,
    EXCLUDED_FEATURES,
    LGBM_PARAMS,
    LR_PARAMS,
    META_COLS,
    MODEL_V1_ENSEMBLE_PATH,
    MODEL_V1_LGBM_FEATURES,
    MODEL_V1_LGBM_PATH,
    MODEL_V1_LR_FEATURES,
    MODEL_V1_LR_PATH,
    MODEL_V1_LR_SCALER,
    MODEL_V1_RF_FEATURES,
    MODEL_V1_RF_PATH,
    MODEL_V1_XGB_FEATURES,
    MODEL_V1_XGB_PATH,
    MODELS_V1_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    TARGET_COL,
    TRAIN_TEST_SPLIT,
    XGB_PARAMS,
)
from ml.ML_data_preparation import compute_sample_weights
from ml.ML_data_preparation_v1 import make_symmetric
from utils.logger import get_logger

log = get_logger(__name__)

MODELS_V1_DIR.mkdir(exist_ok=True)

# Calibration holdout fraction of training data
CAL_FRAC = 0.20

_ALL_MODELS = ["xgb", "lr", "rf", "lgbm", "ensemble"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data(min_date: str | None = None) -> pd.DataFrame:
    if not CSV_V1_WITH_ELO.exists():
        log.error(
            "v1 feature CSV not found at '%s'. "
            "Run: python ml/ML_data_preparation_v1.py",
            CSV_V1_WITH_ELO,
        )
        sys.exit(1)
    df = pd.read_csv(CSV_V1_WITH_ELO)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if min_date:
        n_before = len(df)
        df = df[df["date"] >= min_date].reset_index(drop=True)
        log.info("Date filter %s: dropped %d rows, %d remaining", min_date, n_before - len(df), len(df))
    log.info("Loaded %d rows x %d columns from %s", *df.shape, CSV_V1_WITH_ELO.name)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=META_COLS).fillna(0).replace([np.inf, -np.inf], 0)
    X = X.drop(columns=[f for f in EXCLUDED_FEATURES if f in X.columns])
    y = df[TARGET_COL]
    return X, y


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx   = int(len(df) * TRAIN_TEST_SPLIT)
    train = df.iloc[:idx]
    test  = df.iloc[idx:]
    log.info("Train: %d fights up to %s", len(train), train["date"].max().date())
    log.info("Test:  %d fights from %s", len(test),  test["date"].min().date())
    return train, test


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgb(df: pd.DataFrame, tune: bool = False, n_trials: int = 50) -> None:
    log.info("=== XGBoost ===")
    params = _tune_xgb(df, n_trials) if tune else {**XGB_PARAMS, "random_state": RANDOM_STATE}

    train_df, test_df = time_split(df)
    train_sym = make_symmetric(train_df)
    X_tr, y_tr = preprocess(train_sym)
    X_te, y_te = preprocess(test_df)
    w = compute_sample_weights(train_sym["date"])
    feature_names = list(X_tr.columns)

    fit_p = {k: v for k, v in params.items() if k != "eval_metric"}
    model = XGBClassifier(**fit_p, eval_metric="logloss", early_stopping_rounds=50)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], sample_weight=w, verbose=False)

    acc = accuracy_score(y_te, model.predict(X_te))
    log.info("XGBoost test accuracy: %.2f%%", acc * 100)
    print(f"XGBoost test accuracy: {acc:.2%}")

    joblib.dump(model,         MODEL_V1_XGB_PATH)
    joblib.dump(feature_names, MODEL_V1_XGB_FEATURES)
    log.info("Saved xgboost.joblib + xgb_features.joblib")


def _tune_xgb(df: pd.DataFrame, n_trials: int) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log.info("Tuning XGBoost (%d trials)...", n_trials)
    tss = TimeSeriesSplit(n_splits=5)
    X_all, y_all = preprocess(df)

    def objective(trial) -> float:
        p = {
            "n_estimators":     trial.suggest_int("n_estimators",     100, 800),
            "learning_rate":    trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth",         2, 7),
            "subsample":        trial.suggest_float("subsample",       0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight",  1, 10),
            "gamma":            trial.suggest_float("gamma",           0.0, 2.0),
            "reg_alpha":        trial.suggest_float("reg_alpha",       0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda",      0.5, 5.0),
            "random_state":     RANDOM_STATE,
        }
        scores = []
        for tr_idx, va_idx in tss.split(X_all):
            sym = make_symmetric(df.iloc[tr_idx].copy())
            Xtr, ytr = preprocess(sym)
            Xva, yva = preprocess(df.iloc[va_idx])
            m = XGBClassifier(**p, eval_metric="logloss", verbosity=0)
            m.fit(Xtr, ytr, sample_weight=compute_sample_weights(sym["date"]))
            scores.append(accuracy_score(yva, m.predict(Xva)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = {**study.best_params, "random_state": RANDOM_STATE}
    log.info("XGBoost best CV: %.2f%%  params: %s", study.best_value * 100, best)
    return best


# ── Logistic Regression ───────────────────────────────────────────────────────

def train_lr(df: pd.DataFrame, tune: bool = False, n_trials: int = 50) -> None:
    log.info("=== Logistic Regression ===")
    params = _tune_lr(df, n_trials) if tune else LR_PARAMS

    # 70 / 10 / 20 split for train / Platt calibration / test
    n       = len(df)
    tr_end  = int(n * 0.70)
    cal_end = int(n * 0.80)

    train_sym  = make_symmetric(df.iloc[:tr_end].copy())
    df_cal     = df.iloc[tr_end:cal_end]
    df_test    = df.iloc[cal_end:]

    X_tr, y_tr   = preprocess(train_sym)
    X_cal, y_cal = preprocess(df_cal)
    X_te, y_te   = preprocess(df_test)
    feature_names = list(X_tr.columns)
    w = compute_sample_weights(train_sym["date"])

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_cal_s = scaler.transform(X_cal)
    X_te_s  = scaler.transform(X_te)

    base = LogisticRegression(**params)
    base.fit(X_tr_s, y_tr, sample_weight=w)

    probs_cal_raw = base.predict_proba(X_cal_s)[:, 1]
    platt = LogisticRegression(max_iter=1000)
    platt.fit(probs_cal_raw.reshape(-1, 1), y_cal)

    probs_te  = platt.predict_proba(base.predict_proba(X_te_s)[:, 1].reshape(-1, 1))[:, 1]
    acc = accuracy_score(y_te, (probs_te >= 0.5).astype(int))
    log.info("LR test accuracy: %.2f%%", acc * 100)
    print(f"LR test accuracy: {acc:.2%}")

    joblib.dump({"base": base, "platt": platt}, MODEL_V1_LR_PATH)
    joblib.dump(scaler,        MODEL_V1_LR_SCALER)
    joblib.dump(feature_names, MODEL_V1_LR_FEATURES)
    log.info("Saved logistic_regression.joblib + lr_scaler.joblib + lr_features.joblib")


def _tune_lr(df: pd.DataFrame, n_trials: int) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log.info("Tuning LR (%d trials)...", n_trials)
    tss = TimeSeriesSplit(n_splits=5)
    X_all, y_all = preprocess(df)
    scaler_all = StandardScaler()
    X_sc = scaler_all.fit_transform(X_all)

    def objective(trial) -> float:
        p = {
            "C":            trial.suggest_float("C", 0.001, 10.0, log=True),
            "solver":       trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter":     trial.suggest_int("max_iter", 200, 2000),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        scores = []
        for tr_idx, va_idx in tss.split(X_sc):
            sym = make_symmetric(df.iloc[tr_idx].copy())
            Xtr, ytr = preprocess(sym)
            Xtr_s = StandardScaler().fit_transform(Xtr)
            Xva_s = StandardScaler().fit(Xtr).transform(X_all[va_idx])
            m = LogisticRegression(**p)
            m.fit(Xtr_s, ytr, sample_weight=compute_sample_weights(sym["date"]))
            scores.append(accuracy_score(y_all[va_idx], m.predict(Xva_s)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info("LR best CV: %.2f%%  params: %s", study.best_value * 100, study.best_params)
    return study.best_params


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_rf(df: pd.DataFrame, tune: bool = False, n_trials: int = 50) -> None:
    log.info("=== Random Forest ===")
    params = _tune_rf(df, n_trials) if tune else {**RF_PARAMS, "random_state": RANDOM_STATE}

    train_df, test_df = time_split(df)
    train_sym = make_symmetric(train_df)
    X_tr, y_tr = preprocess(train_sym)
    X_te, y_te = preprocess(test_df)
    feature_names = list(X_tr.columns)
    w = compute_sample_weights(train_sym["date"])

    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=w)

    acc = accuracy_score(y_te, model.predict(X_te))
    log.info("RF test accuracy: %.2f%%", acc * 100)
    print(f"RF test accuracy: {acc:.2%}")

    joblib.dump(model,         MODEL_V1_RF_PATH)
    joblib.dump(feature_names, MODEL_V1_RF_FEATURES)
    log.info("Saved random_forest.joblib + rf_features.joblib")


def _tune_rf(df: pd.DataFrame, n_trials: int) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log.info("Tuning RF (%d trials)...", n_trials)
    tss = TimeSeriesSplit(n_splits=5)
    X_all, y_all = preprocess(df)

    def objective(trial) -> float:
        p = {
            "n_estimators":      trial.suggest_int("n_estimators",      100, 800),
            "max_depth":         trial.suggest_int("max_depth",          4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split",  2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",   1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight":      trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state":      RANDOM_STATE,
        }
        scores = []
        for tr_idx, va_idx in tss.split(X_all):
            sym = make_symmetric(df.iloc[tr_idx].copy())
            Xtr, ytr = preprocess(sym)
            m = RandomForestClassifier(**p)
            m.fit(Xtr, ytr, sample_weight=compute_sample_weights(sym["date"]))
            scores.append(accuracy_score(y_all[va_idx], m.predict(X_all.iloc[va_idx])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = {**study.best_params, "random_state": RANDOM_STATE}
    log.info("RF best CV: %.2f%%  params: %s", study.best_value * 100, best)
    return best


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lgbm(df: pd.DataFrame, tune: bool = False, n_trials: int = 50) -> None:
    log.info("=== LightGBM ===")
    params = _tune_lgbm(df, n_trials) if tune else {**LGBM_PARAMS, "random_state": RANDOM_STATE}

    train_df, test_df = time_split(df)
    train_sym = make_symmetric(train_df)
    X_tr, y_tr = preprocess(train_sym)
    X_te, y_te = preprocess(test_df)
    feature_names = list(X_tr.columns)
    w = compute_sample_weights(train_sym["date"])

    model = LGBMClassifier(**params, verbosity=-1)
    model.fit(X_tr.values, y_tr, sample_weight=w)

    acc = accuracy_score(y_te, model.predict(X_te.values))
    log.info("LightGBM test accuracy: %.2f%%", acc * 100)
    print(f"LightGBM test accuracy: {acc:.2%}")

    joblib.dump(model,         MODEL_V1_LGBM_PATH)
    joblib.dump(feature_names, MODEL_V1_LGBM_FEATURES)
    log.info("Saved lightgbm.joblib + lgbm_features.joblib")


def _tune_lgbm(df: pd.DataFrame, n_trials: int) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log.info("Tuning LightGBM (%d trials)...", n_trials)
    tss = TimeSeriesSplit(n_splits=5)
    X_all, y_all = preprocess(df)

    def objective(trial) -> float:
        p = {
            "n_estimators":     trial.suggest_int("n_estimators",     100, 800),
            "learning_rate":    trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth",         2, 8),
            "num_leaves":       trial.suggest_int("num_leaves",        16, 128),
            "subsample":        trial.suggest_float("subsample",       0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha",       0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda",      0.0, 3.0),
            "random_state":     RANDOM_STATE,
        }
        scores = []
        for tr_idx, va_idx in tss.split(X_all):
            sym = make_symmetric(df.iloc[tr_idx].copy())
            Xtr, ytr = preprocess(sym)
            m = LGBMClassifier(**p, verbosity=-1)
            m.fit(Xtr.values, ytr, sample_weight=compute_sample_weights(sym["date"]))
            scores.append(accuracy_score(y_all[va_idx], m.predict(X_all.iloc[va_idx].values)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = {**study.best_params, "random_state": RANDOM_STATE}
    log.info("LightGBM best CV: %.2f%%  params: %s", study.best_value * 100, best)
    return best


# Fights from this year onward are held out of Optuna weight search entirely.
# Optuna tunes weights on pre-holdout data; holdout accuracy is the honest metric.
_ENSEMBLE_HOLDOUT_YEAR = 2025
_OPTUNA_MIN_ROWS = 50   # fall back to full test set if tuning window is thinner


# ── Calibrated soft-vote ensemble ─────────────────────────────────────────────

def train_ensemble(df: pd.DataFrame, n_trials: int = 100) -> None:
    log.info("=== Calibrated Ensemble ===")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X_all = df[feature_cols]
    y_all = df[TARGET_COL].values

    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    cal_idx   = int(split_idx * (1 - CAL_FRAC))

    X_cal  = X_all.iloc[cal_idx:split_idx]
    y_cal  = y_all[cal_idx:split_idx]
    X_test = X_all.iloc[split_idx:]
    y_test = y_all[split_idx:]

    # Split test into Optuna tuning window (pre-holdout) and true hold-out.
    # This prevents the weight search from seeing the same fights the backtest
    # evaluates on, which would inflate reported accuracy.
    test_dates   = df.iloc[split_idx:]["date"].dt.year.values
    pre_mask     = test_dates < _ENSEMBLE_HOLDOUT_YEAR
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

    specs = [
        ("xgb",  MODEL_V1_XGB_PATH,  MODEL_V1_XGB_FEATURES,  None,              False),
        ("lr",   MODEL_V1_LR_PATH,   MODEL_V1_LR_FEATURES,   MODEL_V1_LR_SCALER, True),
        ("rf",   MODEL_V1_RF_PATH,   MODEL_V1_RF_FEATURES,   None,              False),
        ("lgbm", MODEL_V1_LGBM_PATH, MODEL_V1_LGBM_FEATURES, None,              False),
    ]

    loaded = []
    for m_key, m_path, f_path, s_path, is_lr in specs:
        if not m_path.exists() or not f_path.exists():
            log.warning("Skipping %s -- model not found at %s", m_key, m_path)
            continue
        artifact   = joblib.load(m_path)
        feat_names = joblib.load(f_path)
        scaler     = joblib.load(s_path) if s_path and s_path.exists() else None
        loaded.append((m_key, artifact, feat_names, scaler, is_lr))
        log.info("Loaded %s", m_key)

    if len(loaded) < 2:
        log.error("Need at least 2 trained base models. Train them first.")
        sys.exit(1)

    def _raw_proba(artifact, feat_names, scaler, is_lr, X_df) -> np.ndarray:
        Xm = X_df.reindex(columns=feat_names, fill_value=0).fillna(0)
        Xi = scaler.transform(Xm) if scaler else Xm.values
        base = artifact["base"] if is_lr else artifact
        return base.predict_proba(Xi)[:, 1]

    # Fit isotonic calibrators on calibration holdout (inside training window)
    log.info("Fitting isotonic calibrators (%d samples)...", len(y_cal))
    calibrators = {}
    print("\nBase model calibration accuracy:")
    for m_key, artifact, feat_names, scaler, is_lr in loaded:
        raw = _raw_proba(artifact, feat_names, scaler, is_lr, X_cal)
        ir  = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw, y_cal)
        calibrators[m_key] = ir
        acc = ((raw >= 0.5).astype(int) == y_cal).mean()
        print(f"  {m_key:<8s}  {acc:.2%}")

    # Score base models (calibrated) on the hold-out set
    model_keys  = []
    tune_probas = []   # used by Optuna (pre-holdout only); empty when X_tune is None
    hold_probas = []   # used for honest accuracy reporting
    print(f"\nBase model hold-out accuracy ({_ENSEMBLE_HOLDOUT_YEAR}+, calibrated):")
    for m_key, artifact, feat_names, scaler, is_lr in loaded:
        if X_tune is not None:
            raw_tune = _raw_proba(artifact, feat_names, scaler, is_lr, X_tune)
            cal_tune = calibrators[m_key].predict(raw_tune)
            tune_probas.append(np.column_stack([1 - cal_tune, cal_tune]))

        raw_hold = _raw_proba(artifact, feat_names, scaler, is_lr, X_hold)
        cal_hold = calibrators[m_key].predict(raw_hold)
        p_hold   = np.column_stack([1 - cal_hold, cal_hold])

        acc = ((p_hold[:, 1] >= 0.5).astype(int) == y_hold).mean()
        print(f"  {m_key:<8s}  {acc:.2%}")
        model_keys.append(m_key)
        hold_probas.append(p_hold)

    # Optuna weight search -- only sees pre-holdout data.
    # Falls back to equal weights when the pre-holdout window is too thin.
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if X_tune is not None:
        def objective(trial) -> float:
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

    # Report honest hold-out accuracy with the chosen weights
    ens_hold = np.average(hold_probas, axis=0,
                          weights=[best_weights[k] for k in model_keys])
    hold_acc = ((ens_hold[:, 1] >= 0.5).astype(int) == y_hold).mean()

    joblib.dump({
        "mode":          "calibrated_soft_vote",
        "weights":       best_weights,
        "calibrators":   calibrators,
        "model_keys":    model_keys,
        "test_accuracy": hold_acc,
    }, MODEL_V1_ENSEMBLE_PATH)

    print(f"\nEnsemble hold-out accuracy ({_ENSEMBLE_HOLDOUT_YEAR}+): {hold_acc:.2%}")
    print("Weights:")
    for k, w in best_weights.items():
        print(f"  {k:<8s}  {w:.4f}")
    log.info("Saved ensemble.joblib")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all v1 models and calibrated ensemble.")
    parser.add_argument(
        "--model",
        choices=_ALL_MODELS,
        default=None,
        help="Train a single model (default: all)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search before training (slow).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Optuna trials per model (default: 50; use 100 for production tuning).",
    )
    parser.add_argument(
        "--min-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Earliest fight date to include in training (default: all rows in CSV).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Sample weight decay: weight=exp(alpha*(year-max_year)). "
             "Overrides config.SAMPLE_WEIGHT_ALPHA. 0.0=uniform, 0.3=moderate, 0.5=strong.",
    )
    args = parser.parse_args()

    if args.alpha is not None:
        import config as _cfg
        _cfg.SAMPLE_WEIGHT_ALPHA = args.alpha
        log.info("Sample weight alpha set to %.3f", args.alpha)

    df = load_data(min_date=args.min_date)
    targets = [args.model] if args.model else _ALL_MODELS

    for target in targets:
        if target == "xgb":
            train_xgb(df, tune=args.tune, n_trials=args.trials)
        elif target == "lr":
            train_lr(df, tune=args.tune, n_trials=args.trials)
        elif target == "rf":
            train_rf(df, tune=args.tune, n_trials=args.trials)
        elif target == "lgbm":
            train_lgbm(df, tune=args.tune, n_trials=args.trials)
        elif target == "ensemble":
            train_ensemble(df, n_trials=args.trials if args.tune else 100)


if __name__ == "__main__":
    main()
