# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

---

## Commands

```bash
# Run tests (requires DB and trained models to exist)
python -m pytest tests/ -v

# Run a single test class
python -m pytest tests/test_pipeline.py::TestELO -v

# Full pipeline (ML steps only -- DB already built)
python run_pipeline.py

# Run specific pipeline steps
python run_pipeline.py --steps 4,5,6,7

# Train a single model directly
python ml/XGBoost.py
python ml/logistic_regression.py
python ml/random_forest.py
python ml/lightgbm_model.py
python ml/soft_vote_ensemble.py

# Tune hyperparameters with Optuna (slow -- update config.py *_PARAMS after)
python ml/XGBoost.py --tune --trials 100
python ml/logistic_regression.py --tune --trials 100
python ml/random_forest.py --tune --trials 100
python ml/lightgbm_model.py --tune --trials 100
python ml/soft_vote_ensemble.py --trials 100

# Build the UFCStats per-fight DB from scratch (runs overnight ~8-10 hours)
python scripts/scrape_history.py --no-rolling
python -c "from db.rolling import main; from config import DB_UFCSTATS_PATH; main(db_path=DB_UFCSTATS_PATH)"

# Predict a fight
python predict.py "Islam Makhachev" "Charles Oliveira"
python predict.py "Islam Makhachev" "Charles Oliveira" --model ensemble
python predict.py "Jones" "Miocic" --model lr --division "light heavyweight" --title

# Backtest model accuracy year-by-year (use --from-year 2022 for honest out-of-sample)
python scripts/backtest.py --from-year 2022
python scripts/backtest.py --model lr --save-csv results.csv

# Value-bet ROI simulation (requires odds_red/odds_blue populated in DB)
python scripts/backtest.py --odds
python scripts/backtest.py --odds --min-edge 0.05 --from-year 2020

# Start the REST API
uvicorn api:app --reload

# Add odds columns to DB (one-time migration)
python utils/odds.py --migrate
```

---

## Architecture

### Pipeline overview

UFCStats scrape -> SQLite DB -> Rolling stats -> ML feature CSV -> Trained models -> Predictions

The pipeline has 10 numbered steps (defined in `run_pipeline.py`):

| Steps | Layer | What happens |
|-------|-------|--------------|
| 1-3 | DB build | Served by `scripts/scrape_history.py` for the UFCStats schema |
| 4 | Feature engineering | DB -> ML feature CSV with ELO, form, age, style, division encoding |
| 5-9 | Training | Feature CSV -> five saved `.joblib` model groups |
| 10 | Ensemble | Optuna-tuned soft-vote weights saved to `ensemble.joblib` |

Steps 4-10 are direct Python imports (faster, unified logging).

### Databases

Two databases exist side by side:

| DB | Path | Schema | Status |
|----|------|--------|--------|
| **UFCStats** | `db/ufc_ufcstats.db` | Per-fight granular stats + rolling windows | **Active** (`DB_PATH`) |
| mdabbert | `db/ufc_v2.db` | Career-aggregate pre-fight snapshots | Kept for comparison / v1 model predictions |

`DB_PATH` in `config.py` points to `DB_UFCSTATS_PATH` on this branch.

#### UFCStats DB schema (`db/ufc_ufcstats.db`)

Three tables:

- **`fighters`** -- one row per fighter: `fighter_id` (hex from UFCStats URL), `name`, `height`, `reach`, `stance`, `dob`
- **`fights`** -- one row per fight: `fight_id`, `event_id`, `date`, `division`, `r_fighter_id`, `b_fighter_id`, `winner_id`, `method`, `title_fight`, `odds_red`, `odds_blue`
- **`fight_stats`** -- two rows per fight (one per corner): raw per-fight stats (`kd`, `sig_str_landed`, `sig_str_atmpted`, `head_landed/atmpted`, `body_landed/atmpted`, `leg_landed/atmpted`, `dist/clinch/ground landed/atmpted`, `td_landed/atmpted`, `sub_att`, `ctrl`, `total_fight_time`) plus all rolling columns added by `rolling.py` (accuracies, rates, splm, sapm, str_def, td_avg, td_def, sub_avg, wins, losses, etc.)

Fighter IDs are hex strings from UFCStats.com URLs (e.g. `c2299ec916bc7c56`). Red/Blue corners are read from the fight detail page `div.b-fight-details__person` divs (first = Red, second = Blue) -- NOT from the event listing order which puts the winner first.

### Rolling stats (`db/rolling.py`)

Reads `fight_stats`, sorts by date, applies `shift(1)` so each row only contains data from fights *before* the current one, then upserts the computed columns back into `fight_stats`. This is the critical leakage-prevention step -- never skip the shift.

`main()` accepts an optional `db_path` parameter so it can target either DB.

### UFCStats scraper (`scrapers/ufcstats.py`)

`scrape_events_iter(since, existing_fighter_ids, skip_event_ids)` is a generator that yields `(event, data)` pairs. `scrape_history.py` uses it with checkpointing every 10 events. Resume support: already-ingested `event_id`s are skipped automatically.

### ELO calculator (`ml/ELO_calculator.py`)

Replays all historical fights chronologically to produce pre-fight ELO ratings. Two modes:

- **Per-division** (used in training): keys ratings by `(fighter_id, division)` tuple -- `build_elo_features()` and `get_current_ratings_by_division()`
- **Global** (fallback): `get_current_ratings()` -- used when no division is specified

K-factor is `K_FACTOR_PROVISIONAL=90` for fighters with <=3 fights, then `K_FACTOR_NORMAL=32`. Starting ELO is 1400 (config).

### Feature dataset (`ml/ML_data_preparation.py`)

Joins fights + fight_stats + ELO + recent form into a flat feature CSV. Key transformations:

- All stats become **diff columns** (`red_stat - blue_stat`) so Red/Blue assignments are symmetry-augmented during training (swap corners + flip target) to remove assignment bias
- **Feature selection**: raw per-fight counts (`_landed`, `_atmpted`, `sub_att`, `ctrl`, `kd`) are excluded from the diff loop via `EXCLUDE_STAT_KEYWORDS` -- these are too noisy (single-fight values). Only derived rolling stats (accuracies, rates, splm, sapm, etc.) are diffed.
- **Shrinkage toward division mean**: each fighter's rolling stats are blended toward their division average -- `smoothed = (n * raw + lambda * div_mean) / (n + lambda)` where `n` = prior fights and `lambda = SHRINKAGE_LAMBDA = 5`. Stabilises features for fighters with few fights.
- **Recent form**: rolling win rate and finish rate over last 3 fights (configurable via `RECENT_FORM_WINDOW` in config)
- **Style matchup**: `striker_vs_wrestler` and `wrestler_vs_striker` interaction terms
- **Division**: 12-column one-hot encoding
- Debutant imputation function exists (`impute_debutant_stats`) but is **not called** -- tested and hurt accuracy

### Models

All models are saved to `models/` as `.joblib` files and tracked in git.

| Model | Script | Artifacts | Test Acc | Notes |
|-------|--------|-----------|----------|-------|
| XGBoost | `ml/XGBoost.py` | `xgboost.joblib`, `xgb_features.joblib` | 63.02% | Optuna-tuned params in `config.XGB_PARAMS` |
| Logistic Regression | `ml/logistic_regression.py` | `logistic_regression.joblib`, `lr_scaler.joblib`, `lr_features.joblib` | 61.17% | Platt-calibrated; artifact is dict with `base`+`platt` keys |
| Random Forest | `ml/random_forest.py` | `random_forest.joblib`, `rf_features.joblib` | 61.82% | Optuna-tuned params in `config.RF_PARAMS` |
| LightGBM | `ml/lightgbm_model.py` | `lightgbm.joblib`, `lgbm_features.joblib` | 61.74% | Optuna-tuned params in `config.LGBM_PARAMS` |
| Ensemble | `ml/soft_vote_ensemble.py` | `ensemble.joblib` | **63.50%** | Soft-vote over XGB+LR+RF+LightGBM; Optuna-tuned weights; recommended |
| Finish type | `ml/finish_type_model.py` | `finish_type.joblib`, `finish_type_features.joblib` | ~51% | 3-class (Decision/KO-TKO/Submission); use as soft signal only |

Accuracy figures are on the held-out test set (fights from 2023-06-17 to 2026-05-30, ~1244 fights). These models are trained on the UFCStats rolling DB with no leakage.

The LR artifact is a dict with `base` (raw model) and `platt` (calibration wrapper) -- load with `artifact["base"]` and `artifact["platt"]`.

The ensemble artifact is a dict with `weights` (per-model float weights) and `test_accuracy`. Retrain it (step 10) whenever any base model is retrained.

Training uses `TimeSeriesSplit(5)` with chronological ordering. The train/test split point is set by `TRAIN_TEST_SPLIT=0.80` (most recent 20% is test). For honest out-of-sample evaluation, use `backtest.py --from-year 2022`.

### Prediction (`predict.py` and `api.py`)

`predict.py` is the CLI; `api.py` is the FastAPI wrapper around the same logic. Both:

1. Resolve fighter names (fuzzy LIKE search against `fighters` table)
2. Pull latest rolling stats from `fight_stats` (most recent fight row)
3. Compute current ELO by replaying fight history (`get_current_ratings_by_division`)
4. Compute recent form
5. Build the feature vector via `build_feature_vector()` -- must match the feature names the model was trained on
6. Optionally compute value bets via `odds.py` if American moneyline odds are supplied

Note: shrinkage is applied during training but NOT at inference time in `predict.py`. The effect is small for established fighters (10+ UFC fights).

### Central config (`config.py`)

Single source of truth for all paths, constants, and hyperparameters. Always import from here -- never hardcode paths. Key constants:

- `DB_PATH` / `DB_UFCSTATS_PATH` -- active DB (UFCStats rolling schema)
- `_DB_MDABBERT` -- legacy mdabbert DB kept for comparison
- `XGB_PARAMS`, `LR_PARAMS`, `RF_PARAMS`, `LGBM_PARAMS` -- Optuna-tuned hyperparameters
- `EXCLUDE_STAT_KEYWORDS` -- columns excluded from the diff feature loop
- `SHRINKAGE_LAMBDA` -- prior weight for division-mean shrinkage (default 5)
- `STARTING_ELO`, `DIVISIONS`, `FINISH_METHOD_MAP`, `TRAIN_TEST_SPLIT`

### Encoding note (Windows)

All Python source files must use ASCII-safe characters only. The Windows cp1252 console rejects Unicode symbols (->, --, -, #). Use `->`, `--`, `-`, `#` instead.

---

## Before committing

Files that must never be committed:

- `**/__pycache__/` and `*.pyc` -- Python bytecode
- `db/ufc_ufcstats.db` -- UFCStats SQLite DB (regenerate with `scripts/scrape_history.py`)
- `db/ufc_v2.db` -- mdabbert SQLite DB (legacy)
- `db/*_backup_*.db` -- rolling.py backup files
- `logs/` -- runtime logs
- `ml/*.csv` -- intermediate ML datasets
- `raw_data/*.db` -- raw database files

Files that ARE tracked:

- `raw_data/ufc-master.csv` -- source of truth for mdabbert pipeline
- `models/*.joblib` -- trained model artifacts; tracked so predictions work immediately after cloning without retraining

If any excluded files were previously committed, untrack them with `git rm --cached <file>` (without deleting the local copy), then verify `.gitignore` covers them before staging the commit.

---

## Key invariants

- **No leakage**: every rolling stat uses `shift(1)`. ELO is computed *before* the fight is processed. Recent form excludes the current fight. Violating this inflates accuracy.
- **Corner assignment**: Red/Blue corners come from the fight detail page (`div.b-fight-details__person`), NOT the event listing (which puts the winner first). Getting this wrong causes ~100% Red win rate in training data.
- **Symmetry augmentation**: during training, each fight is duplicated with corners swapped and target flipped. This is applied *per fold* to the training split only -- never to the validation/test split.
- **SQLite TEXT affinity**: some numeric columns (e.g. `kd`, `ctrl`) are stored as TEXT. Always use `CAST(col AS REAL)` in numeric comparisons or aggregations.
- **Model performance ceiling**: stats-only models peak at ~63-65% on UFC fights. The naive "always pick Red" baseline is ~55-60% on recent data (red corner wins ~55% in the UFCStats DB, reflecting more balanced matchmaking). Update `MODEL_RESULTS.md` after any significant retrain.
- **Shrinkage is training-only**: `apply_shrinkage()` in `ML_data_preparation.py` modifies the training CSV. It is NOT applied in `predict.py` at inference time -- acceptable approximation for established fighters.
