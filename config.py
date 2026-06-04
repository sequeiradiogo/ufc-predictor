"""
config.py — Central configuration for the UFC Predictor project.

All file paths and tunable constants live here so every script can import
them instead of hardcoding values.
"""

from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
DB_DIR     = ROOT_DIR / "db"
ML_DIR     = ROOT_DIR / "ml"
MODELS_DIR = ROOT_DIR / "models"      # persisted model artifacts
RAW_DIR    = ROOT_DIR / "raw_data"

# Create models dir on first import so saving never fails
MODELS_DIR.mkdir(exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
DB_UFCSTATS_PATH = DB_DIR / "ufc_ufcstats.db"    # UFCStats per-fight DB with rolling stats
DB_PATH          = DB_UFCSTATS_PATH               # active DB for ML pipeline (branch: feat/ufcstats-schema)
_DB_MDABBERT     = DB_DIR / "ufc_v2.db"           # mdabbert career-aggregate DB (kept for comparison)

# ── ML Datasets ───────────────────────────────────────────────────────────────
CSV_MASTER         = RAW_DIR / "ufc-master.csv"          # mdabbert source CSV with historical odds
CSV_WITH_ELO       = ML_DIR / "ufc_ml_data_with_debuts_and_elo.csv"
CSV_WITH_DEBUTS    = ML_DIR / "ufc_ml_data_with_debuts.csv"
CSV_WITHOUT_DEBUTS = ML_DIR / "ufc_ml_data_without_debuts.csv"

# ── Saved Model Artifacts ─────────────────────────────────────────────────────
MODEL_LR_PATH      = MODELS_DIR / "logistic_regression.joblib"
MODEL_LR_SCALER    = MODELS_DIR / "lr_scaler.joblib"
MODEL_LR_FEATURES  = MODELS_DIR / "lr_features.joblib"

MODEL_XGB_PATH     = MODELS_DIR / "xgboost.joblib"
MODEL_XGB_FEATURES = MODELS_DIR / "xgb_features.joblib"

MODEL_RF_PATH      = MODELS_DIR / "random_forest.joblib"
MODEL_RF_FEATURES  = MODELS_DIR / "rf_features.joblib"

MODEL_LGBM_PATH     = MODELS_DIR / "lightgbm.joblib"
MODEL_LGBM_FEATURES = MODELS_DIR / "lgbm_features.joblib"

MODEL_ENSEMBLE_PATH = MODELS_DIR / "ensemble.joblib"

# ── v1 (mdabbert) Model Artifacts ─────────────────────────────────────────────
MODELS_V1_DIR          = ROOT_DIR / "models_v1"
MODEL_V1_XGB_PATH      = MODELS_V1_DIR / "xgboost.joblib"
MODEL_V1_XGB_FEATURES  = MODELS_V1_DIR / "xgb_features.joblib"
MODEL_V1_LR_PATH       = MODELS_V1_DIR / "logistic_regression.joblib"
MODEL_V1_LR_SCALER     = MODELS_V1_DIR / "lr_scaler.joblib"
MODEL_V1_LR_FEATURES   = MODELS_V1_DIR / "lr_features.joblib"
MODEL_V1_RF_PATH       = MODELS_V1_DIR / "random_forest.joblib"
MODEL_V1_RF_FEATURES   = MODELS_V1_DIR / "rf_features.joblib"
MODEL_V1_LGBM_PATH     = MODELS_V1_DIR / "lightgbm.joblib"
MODEL_V1_LGBM_FEATURES = MODELS_V1_DIR / "lgbm_features.joblib"
MODEL_V1_ENSEMBLE_PATH = MODELS_V1_DIR / "ensemble.joblib"
DB_V1_PATH             = _DB_MDABBERT    # public alias for v1 DB

# ── ELO ───────────────────────────────────────────────────────────────────────
STARTING_ELO         = 1400   # slightly below 1500 to penalise unknowns
K_FACTOR_NORMAL      = 32
K_FACTOR_PROVISIONAL = 90     # large jumps for first few fights
PROVISIONAL_LIMIT    = 3      # fights before becoming "established"

# ── ML Training ───────────────────────────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.80        # 80 % history → train, 20 % recent → test
RANDOM_STATE     = 42
TARGET_COL       = "target"   # 1 = Red wins, 0 = Blue wins
META_COLS        = ["fight_id", "date", "division", "target"]

# ── XGBoost Hyperparameters (tuned via Optuna, 100 trials, 2026-06-04) ───────
# CV accuracy: 61.14% (v4 trajectory features)  |  run: python ml/XGBoost.py --tune --trials 100
XGB_PARAMS: dict = {
    "n_estimators":     364,
    "learning_rate":    0.041175,
    "max_depth":        2,
    "subsample":        0.9376,
    "colsample_bytree": 0.9525,
    "min_child_weight": 9,
    "gamma":            0.6687,
    "reg_alpha":        0.5384,
    "reg_lambda":       4.1889,
}

# ── Logistic Regression Hyperparameters (tuned via Optuna, 100 trials, 2026-06-04) ──
# CV accuracy: 61.12% (v4 trajectory features)  |  run: python ml/logistic_regression.py --tune --trials 100
LR_PARAMS: dict = {
    "C":            2.0860,
    "solver":       "saga",
    "max_iter":     517,
    "class_weight": "balanced",
}

# ── Random Forest Hyperparameters (tuned via Optuna, 100 trials, 2026-06-04) ──
# CV accuracy: 60.00% (v4 trajectory features)  |  run: python ml/random_forest.py --tune --trials 100
RF_PARAMS: dict = {
    "n_estimators":      225,
    "max_depth":         9,
    "min_samples_split": 8,
    "min_samples_leaf":  3,
    "max_features":      0.3,
    "class_weight":      "balanced",
}

# ── LightGBM Hyperparameters (tuned via Optuna, 100 trials, 2026-06-04) ───────
# CV accuracy: 60.98% (v4 trajectory features)  |  run: python ml/lightgbm_model.py --tune --trials 100
LGBM_PARAMS: dict = {
    "n_estimators":     354,
    "learning_rate":    0.017841,
    "max_depth":        4,
    "num_leaves":       53,
    "subsample":        0.8431,
    "colsample_bytree": 0.8923,
    "reg_alpha":        0.2022,
    "reg_lambda":       3.3206,
}

# ── Feature Engineering ───────────────────────────────────────────────────────
# Columns excluded from the automatic difference-feature loop
# Columns excluded from the automatic difference-feature loop.
# total_fight_time is a proxy for career length (wins+losses) used only
# for debutant detection — its diff is a linear combo of wins_diff + losses_diff.
# Raw per-fight counts (_landed/_atmpted are cumulative totals for ONE fight only,
# far noisier than the derived rolling accuracy/rate stats kept below).
EXCLUDE_STAT_KEYWORDS = [
    "stance", "dob", "opponent_id", "weight", "total_fight_time",
    "_landed", "_atmpted", "sub_att", "ctrl", "kd",
]

# Prior weight for shrinkage toward division mean in ML_data_preparation.
# A fighter needs ~SHRINKAGE_LAMBDA fights before their own stats dominate.
SHRINKAGE_LAMBDA = 5

# Minimum fight date included in the ML training set
MIN_FIGHT_DATE = "2005-01-01"

# Recent form window (number of prior fights to average)
RECENT_FORM_WINDOW = 3

# Opponents to average for strength-of-schedule (SOS) computation
SOS_WINDOW = 5

# KO vulnerability: fights to look back for recent KO/TKO stoppages
KO_VULN_WINDOW = 3

# EWMA span for time-decay striking/TD accuracy features
EWMA_SPAN = 5

# Fights to look back when computing trajectory/momentum slopes and streaks
TRAJECTORY_WINDOW = 5

# Known UFC weight divisions (lowercase, as stored in DB) in ascending weight order
DIVISIONS = [
    "women's strawweight",
    "women's flyweight",
    "women's bantamweight",
    "women's featherweight",
    "flyweight",
    "bantamweight",
    "featherweight",
    "lightweight",
    "welterweight",
    "middleweight",
    "light heavyweight",
    "heavyweight",
]

# Finish method → integer class (used by finish_type_model.py)
FINISH_METHOD_MAP: dict[str, int] = {
    "Decision - Unanimous": 0,
    "Decision - Split":     0,
    "Decision - Majority":  0,
    "KO/TKO":               1,
    "TKO - Doctor's Stoppage": 1,
    "Could Not Continue":   1,
    "Submission":           2,
}
FINISH_CLASS_NAMES = ["Decision", "KO/TKO", "Submission"]

# ── Finish Type Model Artifacts ───────────────────────────────────────────────
MODEL_FINISH_PATH     = MODELS_DIR / "finish_type.joblib"
MODEL_FINISH_FEATURES = MODELS_DIR / "finish_type_features.joblib"
