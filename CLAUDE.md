# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

---

## Commands

```bash
# Run tests (requires DB and trained models to exist)
python -m pytest tests/ -v

# Run a single test class
python -m pytest tests/test_pipeline.py::TestELO -v

# Full pipeline (ML steps only — DB already built)
python run_pipeline.py

# Full pipeline from scratch (new CSV source data)
python run_pipeline.py --full --csv path/to/UFC.csv

# Run specific pipeline steps
python run_pipeline.py --steps 4,5,6,7

# Train a single model directly
python ml/XGBoost.py
python ml/logistic_regression.py

# Tune XGBoost hyperparameters with Optuna (slow — update config.py XGB_PARAMS after)
python ml/XGBoost.py --tune --trials 100

# Predict a fight
python predict.py "Islam Makhachev" "Charles Oliveira"
python predict.py "Jones" "Miocic" --model lr --division "light heavyweight" --title

# Backtest model accuracy year-by-year (use --from-year 2022 for honest out-of-sample)
python backtest.py --from-year 2022
python backtest.py --model lr --save-csv results.csv

# Value-bet ROI simulation (requires odds_red/odds_blue populated in DB)
python backtest.py --odds
python backtest.py --odds --min-edge 0.05 --from-year 2020

# Start the REST API
uvicorn api:app --reload

# Add odds columns to DB (one-time migration)
python odds.py --migrate
```

---

## Architecture

### Pipeline overview

Raw CSV → SQLite DB → Rolling stats → ML feature CSV → Trained models → Predictions

The pipeline has 7 numbered steps (defined in `run_pipeline.py`):

| Steps | Layer | What happens |
|-------|-------|--------------|
| 1–3 | DB build | CSV → SQLite → rolling per-fight stats computed and upserted |
| 4 | Feature engineering | DB → ML feature CSV with ELO, form, age, style, division encoding |
| 5–7 | Training | Feature CSV → three saved `.joblib` models |

Steps 1–3 are run as subprocesses. Steps 4–7 are direct Python imports (faster, unified logging).

### Database schema (`db/ufc_v2.db`)

Three tables:

- **`fighters`** — one row per fighter: `fighter_id` (hex from UFCStats), `name`, `height`, `reach`, `stance`, `dob`, `splm`, `sapm`, `str_def`, `td_avg`
- **`fights`** — one row per fight: `fight_id`, `event_id`, `date`, `division`, `r_fighter_id`, `b_fighter_id`, `winner_id`, `method`, `title_fight`, `odds_red`, `odds_blue`
- **`fight_stats`** — two rows per fight (one per corner): per-round and total strike/TD/submission stats, plus the rolling columns added by `rolling.py`

Fighter IDs are hex strings sourced from UFCStats.com URLs (e.g. `c2299ec916bc7c56`).

### Rolling stats (`db/rolling.py`)

Reads `fight_stats`, sorts by date, applies `shift(1)` so each row only contains data from fights *before* the current one, then upserts the computed columns back into `fight_stats`. This is the critical leakage-prevention step — never skip the shift.

### ELO calculator (`ml/ELO_calculator.py`)

Replays all historical fights chronologically to produce pre-fight ELO ratings. Two modes:

- **Per-division** (used in training): keys ratings by `(fighter_id, division)` tuple — `build_elo_features()` and `get_current_ratings_by_division()`
- **Global** (fallback): `get_current_ratings()` — used when no division is specified

K-factor is `K_FACTOR_PROVISIONAL=90` for fighters with ≤3 fights, then `K_FACTOR_NORMAL=32`. Starting ELO is 1400 (config).

### Feature dataset (`ml/ML_data_preparation.py`)

Joins fights + fight_stats + ELO + recent form into a flat feature CSV. Key transformations:

- All stats become **diff columns** (`red_stat − blue_stat`) so Red/Blue assignments are symmetry-augmented during training (swap corners + flip target) to remove assignment bias
- **Recent form**: rolling win rate, finish rate, win streak over last 3 fights (configurable via `RECENT_FORM_WINDOW` in config)
- **Style matchup**: `striker_vs_wrestler` and `wrestler_vs_striker` interaction terms
- **Division**: 12-column one-hot encoding
- Debutant imputation function exists (`impute_debutant_stats`) but is **not called** — it was tested and hurt accuracy (zeros + `is_debutant_diff` flag outperforms division-average priors)

### Models

Three classifiers, all saved to `models/` as `.joblib` files:

| Model | Script | Artifacts | Notes |
|-------|--------|-----------|-------|
| XGBoost | `XGBoost.py` | `xgboost.joblib`, `xgb_features.joblib` | Default predictor; Optuna-tuned params in `config.XGB_PARAMS` |
| Logistic Regression | `logistic_regression.py` | `logistic_regression.joblib` (dict with `base`+`platt` keys), `lr_scaler.joblib`, `lr_features.joblib` | Platt-calibrated; better-calibrated probabilities than XGBoost |
| Finish type | `finish_type_model.py` | `finish_type.joblib`, `finish_type_features.joblib` | 3-class (Decision/KO-TKO/Submission); ~50% accuracy — use as soft signal only |

The LR artifact is a dict with `base` (raw model) and `platt` (calibration wrapper) — load with `artifact["base"]` and `artifact["platt"]`.

Training uses `TimeSeriesSplit(5)` with chronological ordering. The train/test split point is set by `TRAIN_TEST_SPLIT=0.80` (most recent 20% is test). For honest out-of-sample evaluation, use `backtest.py --from-year 2022`.

### Prediction (`predict.py` and `api.py`)

`predict.py` is the CLI; `api.py` is the FastAPI wrapper around the same logic. Both:

1. Resolve fighter names (fuzzy LIKE search against `fighters` table)
2. Pull latest stats from `fight_stats`
3. Compute current ELO by replaying fight history (`get_current_ratings_by_division`)
4. Compute recent form
5. Build the feature vector via `build_feature_vector()` — must match the feature names the model was trained on
6. Optionally compute value bets via `odds.py` if American moneyline odds are supplied

### Central config (`config.py`)

Single source of truth for all paths, constants, and hyperparameters. Always import from here — never hardcode paths. Key constants: `DB_PATH`, `XGB_PARAMS`, `STARTING_ELO`, `DIVISIONS`, `FINISH_METHOD_MAP`, `TRAIN_TEST_SPLIT`.

### Encoding note (Windows)

All Python source files must use ASCII-safe characters only. The Windows cp1252 console rejects Unicode symbols (→, —, ─, █). Use `->`, `--`, `-`, `#` instead.

---

## Before committing

Always untrack and gitignore generated/temporary files before committing. Files that must never be committed:

- `**/__pycache__/` and `*.pyc` — Python bytecode
- `models/*.joblib` — trained model artifacts (regenerate with `run_pipeline.py`)
- `db/ufc_v2.db` — SQLite database (regenerate with `db/ingest_mdabbert.py` + `run_pipeline.py`)
- `logs/` — runtime logs
- `ml/*.csv` — intermediate ML datasets
- `raw_data/*.db` — raw database files

Note: `raw_data/ufc-master.csv` IS tracked — it is the source of truth and contains
scraped data that cannot be re-downloaded from Kaggle.

If any of these were previously committed, untrack them with `git rm --cached <file>` (without deleting the local copy), then verify `.gitignore` covers them before staging the commit.

---

## Key invariants

- **No leakage**: every rolling stat uses `shift(1)`. ELO is computed *before* the fight is processed. Recent form excludes the current fight. Violating this inflates accuracy.
- **Symmetry augmentation**: during training, each fight is duplicated with corners swapped and target flipped. This is applied *per fold* to the training split only — never to the validation/test split.
- **SQLite TEXT affinity**: some numeric columns (e.g. `kd`, `ctrl`) are stored as TEXT. Always use `CAST(col AS REAL)` in numeric comparisons or aggregations.
- **Model performance ceiling**: stats-only models peak at ~65–68% on UFC fights. The naive "always pick Red" baseline is ~64% (red corner wins ~64% historically). Update `MODEL_RESULTS.md` after any significant retrain.
