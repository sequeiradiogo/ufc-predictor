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
DB_PATH = DB_DIR / "ufc_v2.db"

# ── ML Datasets ───────────────────────────────────────────────────────────────
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

# ── XGBoost Hyperparameters (tuned via Optuna, 50 trials, 2026-05-26) ────────
# CV accuracy: 65.08% ± 1.54%  |  run: python ml/XGBoost.py --tune --trials 100
# to refresh these after adding new features.
XGB_PARAMS: dict = {
    "n_estimators":     367,
    "learning_rate":    0.02873,
    "max_depth":        4,
    "subsample":        0.6631,
    "colsample_bytree": 0.7104,
    "min_child_weight": 9,
    "gamma":            1.6733,
    "reg_alpha":        0.1539,
    "reg_lambda":       2.9687,
}

# ── Logistic Regression Hyperparameters (tuned via Optuna, 100 trials, 2026-06-01) ──
# CV accuracy: 62.43% +/- 2.45%  |  run: python ml/logistic_regression.py --tune --trials 100
LR_PARAMS: dict = {
    "C":            0.0005069,
    "solver":       "liblinear",
    "max_iter":     1883,
    "class_weight": "balanced",
}

# ── Random Forest Hyperparameters (tuned via Optuna, 100 trials, 2026-06-01) ──
# CV accuracy: 62.49% +/- 2.41%  |  run: python ml/random_forest.py --tune --trials 100
RF_PARAMS: dict = {
    "n_estimators":      203,
    "max_depth":         6,
    "min_samples_split": 9,
    "min_samples_leaf":  4,
    "max_features":      0.3,
    "class_weight":      None,
}

# ── LightGBM Hyperparameters (tuned via Optuna, 100 trials, 2026-06-01) ───────
# CV accuracy: 61.94% +/- 2.89%  |  run: python ml/lightgbm_model.py --tune --trials 100
LGBM_PARAMS: dict = {
    "n_estimators":     316,
    "learning_rate":    0.011995,
    "max_depth":        3,
    "num_leaves":       57,
    "subsample":        0.7433,
    "colsample_bytree": 0.5987,
    "reg_alpha":        0.3932,
    "reg_lambda":       0.004966,
}

# ── Feature Engineering ───────────────────────────────────────────────────────
# Columns excluded from the automatic difference-feature loop
# Columns excluded from the automatic difference-feature loop.
# total_fight_time is a proxy for career length (wins+losses) used only
# for debutant detection — its diff is a linear combo of wins_diff + losses_diff.
EXCLUDE_STAT_KEYWORDS = ["stance", "dob", "opponent_id", "weight", "total_fight_time"]

# Minimum fight date included in the ML training set
MIN_FIGHT_DATE = "2005-01-01"

# Recent form window (number of prior fights to average)
RECENT_FORM_WINDOW = 3

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
