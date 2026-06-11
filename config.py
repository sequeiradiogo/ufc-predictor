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
CSV_V1_WITH_ELO    = ML_DIR / "ufc_ml_data_v1.csv"

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

MODEL_MLP_PATH     = MODELS_DIR / "mlp.joblib"
MODEL_MLP_FEATURES = MODELS_DIR / "mlp_features.joblib"
MODEL_MLP_SCALER   = MODELS_DIR / "mlp_scaler.joblib"

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
MODEL_V1_MLP_PATH      = MODELS_V1_DIR / "mlp.joblib"
MODEL_V1_MLP_FEATURES  = MODELS_V1_DIR / "mlp_features.joblib"
MODEL_V1_MLP_SCALER    = MODELS_V1_DIR / "mlp_scaler.joblib"
MODEL_V1_STACKING_PATH = MODELS_V1_DIR / "stacking.joblib"

# Production models -- trained on 100% of data, used exclusively for inference
MODELS_V1_PROD_DIR          = ROOT_DIR / "models_v1_prod"
MODEL_V1_PROD_XGB_PATH      = MODELS_V1_PROD_DIR / "xgboost.joblib"
MODEL_V1_PROD_XGB_FEATURES  = MODELS_V1_PROD_DIR / "xgb_features.joblib"
MODEL_V1_PROD_LR_PATH       = MODELS_V1_PROD_DIR / "logistic_regression.joblib"
MODEL_V1_PROD_LR_SCALER     = MODELS_V1_PROD_DIR / "lr_scaler.joblib"
MODEL_V1_PROD_LR_FEATURES   = MODELS_V1_PROD_DIR / "lr_features.joblib"
MODEL_V1_PROD_RF_PATH       = MODELS_V1_PROD_DIR / "random_forest.joblib"
MODEL_V1_PROD_RF_FEATURES   = MODELS_V1_PROD_DIR / "rf_features.joblib"
MODEL_V1_PROD_LGBM_PATH     = MODELS_V1_PROD_DIR / "lightgbm.joblib"
MODEL_V1_PROD_LGBM_FEATURES = MODELS_V1_PROD_DIR / "lgbm_features.joblib"
MODEL_V1_PROD_ENSEMBLE_PATH = MODELS_V1_PROD_DIR / "ensemble.joblib"
MODEL_V1_PROD_MLP_PATH      = MODELS_V1_PROD_DIR / "mlp.joblib"
MODEL_V1_PROD_MLP_FEATURES  = MODELS_V1_PROD_DIR / "mlp_features.joblib"
MODEL_V1_PROD_MLP_SCALER    = MODELS_V1_PROD_DIR / "mlp_scaler.joblib"
DB_V1_PATH             = _DB_MDABBERT    # public alias for v1 DB

# ── ELO ───────────────────────────────────────────────────────────────────────
STARTING_ELO         = 1400   # slightly below 1500 to penalise unknowns
K_FACTOR_NORMAL      = 32
K_FACTOR_PROVISIONAL = 90     # large jumps for first few fights
PROVISIONAL_LIMIT    = 3      # fights before becoming "established"

# ── Glicko-2 ──────────────────────────────────────────────────────────────────
GLICKO_START_R     = 1500    # initial rating (Glicko-2 scale)
GLICKO_START_RD    = 350.0   # initial rating deviation (high = very uncertain)
GLICKO_START_SIGMA = 0.06    # initial volatility
GLICKO_TAU         = 0.5     # system constant; constrains how fast volatility changes

# ── ML Training ───────────────────────────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.80        # 80 % history → train, 20 % recent → test
RANDOM_STATE     = 42
TARGET_COL       = "target"   # 1 = Red wins, 0 = Blue wins
META_COLS        = ["fight_id", "date", "division", "target"]

# ── XGBoost Hyperparameters (tuned via Optuna, 100 trials, 2026-06-11) ───────
# 71 features (added head/body/leg acc, reach ratio, ewma rates)  |  68.1% ensemble 2025+
XGB_PARAMS: dict = {
    "n_estimators":     674,
    "learning_rate":    0.011268,
    "max_depth":        2,
    "subsample":        0.7935,
    "colsample_bytree": 0.4755,
    "min_child_weight": 8,
    "gamma":            1.6109,
    "reg_alpha":        0.3191,
    "reg_lambda":       2.6403,
}

# ── Logistic Regression Hyperparameters (tuned via Optuna, 100 trials, 2026-06-11) ──
# 71 features (added head/body/leg acc, reach ratio, ewma rates)  |  68.1% ensemble 2025+
LR_PARAMS: dict = {
    "C":            0.002849,
    "solver":       "lbfgs",
    "max_iter":     888,
    "class_weight": "balanced",
}

# ── Random Forest Hyperparameters (tuned via Optuna, 100 trials, 2026-06-11) ──
# 71 features (added head/body/leg acc, reach ratio, ewma rates)  |  68.1% ensemble 2025+
RF_PARAMS: dict = {
    "n_estimators":      323,
    "max_depth":         10,
    "min_samples_split": 7,
    "min_samples_leaf":  10,
    "max_features":      "sqrt",
    "class_weight":      None,
}

# ── LightGBM Hyperparameters (tuned via Optuna, 100 trials, 2026-06-11) ─────
# 71 features (added head/body/leg acc, reach ratio, ewma rates)  |  68.1% ensemble 2025+
LGBM_PARAMS: dict = {
    "n_estimators":     563,
    "learning_rate":    0.010007,
    "max_depth":        4,
    "num_leaves":       122,
    "subsample":        0.7463,
    "colsample_bytree": 0.9553,
    "reg_alpha":        0.9506,
    "reg_lambda":       1.6967,
}

# ── MLP Hyperparameters (PyTorch; tuned via Optuna, 100 trials, 2026-06-11) ──
# 71 features (added head/body/leg acc, reach ratio, ewma rates)  |  68.1% ensemble 2025+
MLP_PARAMS: dict = {
    "hidden_sizes": (64, 64),
    "dropout":      0.4775,
    "lr":           4.388e-3,
    "weight_decay": 4.476e-3,
    "batch_size":   64,
    "max_epochs":   300,
    "patience":     20,
    "batch_norm":   False,
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

# Features to drop at training and inference time after feature selection.
# Populated from scripts/feature_importance.py analysis (issue #43).
# Keep empty until the diagnostic run has been reviewed.
EXCLUDED_FEATURES: list[str] = [
    # Dead columns -- all zeros (phantom diffs from rolling stats join)
    "date_diff",
    "outcome_diff",
    # Zero importance across all 4 models
    "age_diff",
    # Glicko-2 features: near-zero target correlation (0.004 / -0.002); removing
    # them improves 2026 accuracy by +1.5pp (67.0% total vs 66.7% with Glicko).
    # ELO already captures the rating signal more cleanly.
    "glicko_diff",
    "glicko_rd_diff",
    # SHAP ablation (2026-06-11): negative permutation importance across 3-4 models.
    # str_def_diff: target corr=-0.033 (wrong direction) + r=0.49 with sapm_diff (collinear).
    # sos_diff: target corr=-0.049 (wrong direction) + redundant with career record features.
    # elo_diff: ALL 4 models negative PI; redundant with win/loss/streak features.
    # Ablation with equal-weight ensemble: dropping these 3 gives +0.44pp on 2025+ set.
    "str_def_diff",
    "sos_diff",
    "elo_diff",
]

# Prior weight for shrinkage toward division mean in ML_data_preparation.
# A fighter needs ~SHRINKAGE_LAMBDA fights before their own stats dominate.
SHRINKAGE_LAMBDA = 5

# Minimum fight date included in the ML training set.
# Set to 2018-01-01 based on adversarial validation (issue #47) -- pre-2018 fights
# show significant distribution shift and hurt out-of-sample accuracy.
MIN_FIGHT_DATE = "2018-01-01"

# Temporal sample weighting: weight = exp(-alpha * delta^beta) where delta = max_year - year.
# alpha=0.0 disables weighting entirely. beta=1.0 is flat exponential decay (original);
# beta>1.0 accelerates decay for older fights while keeping recent fights near full weight.
SAMPLE_WEIGHT_ALPHA = 0.01
SAMPLE_WEIGHT_BETA  = 1.5

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

# ── Division normalization constants ─────────────────────────────────────────
# Per-division reach std (cm) for reach_div_norm_diff feature.
# Computed from ufc_v2.db 2026-06 snapshot.
DIV_REACH_STD: dict[str, float] = {
    "bantamweight":           6.01,
    "featherweight":          5.64,
    "flyweight":              5.86,
    "heavyweight":            7.15,
    "light heavyweight":      6.48,
    "lightweight":            5.64,
    "middleweight":           5.99,
    "welterweight":           6.18,
    "women's bantamweight":   5.15,
    "women's featherweight":  5.11,
    "women's flyweight":      5.70,
    "women's strawweight":    5.66,
}
DIV_REACH_STD_FALLBACK = 6.5     # average within-division reach std

# Per-division sig-strike-per-minute std for splm_div_norm_diff feature.
DIV_SPLM_STD: dict[str, float] = {
    "bantamweight":           1.85,
    "featherweight":          2.98,
    "flyweight":              1.69,
    "heavyweight":            2.11,
    "light heavyweight":      2.20,
    "lightweight":            1.84,
    "middleweight":           2.17,
    "welterweight":           2.47,
    "women's bantamweight":   1.61,
    "women's featherweight":  2.95,
    "women's flyweight":      1.95,
    "women's strawweight":    3.00,  # capped; raw 5.7 is outlier-inflated
}
DIV_SPLM_STD_FALLBACK = 2.0      # conservative cross-division default

# ── Fighter name aliases ──────────────────────────────────────────────────────
# Maps alternate/historical names (lowercased) -> canonical UFCStats name.
# Covers Kaggle CSV typos, married name changes, nickname vs legal name, and
# transliteration differences between the Kaggle dataset and UFCStats.
# Keys must be lowercase; values must match the name in the fighters table.
NAME_ALIASES: dict[str, str] = {
    # Kaggle CSV typos
    "alekander volkov":     "Alexander Volkov",
    "caludia gadelha":      "Claudia Gadelha",
    "caludio puelles":      "Claudio Puelles",
    "krzystof jotko":       "Krzysztof Jotko",
    "vincente luque":       "Vicente Luque",
    "isabela de pauda":     "Isabela de Padua",
    "ode obsourne":         "Ode Osbourne",
    "youssef zalel":        "Youssef Zalal",
    "zhalgas zhamagulov":   "Zhalgas Zhumagulov",
    "nina ansaroff":        "Nina Nunes",
    "ariane lipski":        "Ariane da Silva",
    "brianna van buren":    "Brianna Fortino",
    "ulka sasaki":          "Yuta Sasaki",
    "roberto sanchez":      "Robert Sanchez",
    # Married name changes
    "ariane lipski":        "Ariane Carnelossi",
    "cheyanne buys":        "Cheyanne Vlismas",
    "joanne calderwood":    "Joanne Wood",
    "katlyn chookagian":    "Katlyn Cerminara",
    "michelle waterson":    "Michelle Waterson-Gomez",
    "nina ansaroff":        "Nina Nunes",
    "tecia torres":         "Tecia Pennington",
    # Nickname / ring name vs legal name
    "cris cyborg":          "Cristiane Justino",
    "mirko cro cop":        "Mirko Filipovic",
    "rampage jackson":      "Quinton Jackson",
    "minotauro nogueira":   "Antonio Rodrigo Nogueira",
    "patricio freire":      "Patricio Pitbull",
    # Name format differences (spacing, transliteration, Chinese name order)
    "weili zhang":          "Zhang Weili",
    "tiequan zhang":        "Zhang Tiequan",
    "na liang":             "Liang Na",
    "aori qileng":          "Aoriqileng",
    "rong zhu":             "Rongzhu",
    "su mudaerji":          "Sumudaerji",
    "wuliji buren":         "Wulijiburen",
    "heili alateng":        "Alatengheili",
    "an ying wang":         "Anying Wang",
    "seohee ham":           "Seo Hee Ham",
    "da un jung":           "Da Woon Jung",
    "da-un jung":           "Da Woon Jung",
    "jun yong park":        "JunYong Park",
    "chanmi jeon":          "Chan-Mi Jeon",
    "roldan sangcha-an":    "Roldan Sangcha'an",
    # Shortened / informal vs full name
    "alex munoz":           "Alexander Munoz",
    "alexandra albu":       "Aleksandra Albu",
    "ali qaisi":            "Ali AlQaisi",
    "benny alloway":        "Ben Alloway",
    "bradley scott":        "Brad Scott",
    "carlo pedersoli":      "Carlo Pedersoli Jr.",
    "costas philippou":     "Constantinos Philippou",
    "grigorii popov":       "Grigory Popov",
    "heather jo clark":     "Heather Clark",
    "ian garry":            "Ian Machado Garry",
    "jim crute":            "Jimmy Crute",
    "jimmy wallhead":       "Jim Wallhead",
    "joshua culibao":       "Josh Culibao",
    "joshua sampo":         "Josh Sampo",
    "kai kamaka":           "Kai Kamaka III",
    "kai kara france":      "Kai Kara-France",
    "luci pudilova":        "Lucie Pudilova",
    "montserrat conejo":    "Montserrat Conejo Ruiz",
    "montserrat rendon":    "Montse Rendon",
    "nico musoke":          "Nicholas Musoke",
    "peter yan":            "Petr Yan",
    "philip rowe":          "Phil Rowe",
    "phillip hawes":        "Phil Hawes",
    "rick glenn":           "Ricky Glenn",
    "rob whiteford":        "Robert Whiteford",
    "roberto sanchez":      "Robert Sanchez",
    "waldo cortes-acosta":  "Waldo Cortes Acosta",
    "zachary reese":        "Zach Reese",
}
