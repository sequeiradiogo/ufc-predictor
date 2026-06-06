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

# Train v2 models directly (UFCStats rolling DB)
python ml/XGBoost.py
python ml/logistic_regression.py
python ml/random_forest.py
python ml/lightgbm_model.py
python ml/soft_vote_ensemble.py

# Train v1 models (mdabbert career-aggregate DB) -- PRIMARY PREDICTION MODELS
python ml/train_v1_models.py                          # all models + ensemble
python ml/train_v1_models.py --model xgb              # single model
python ml/train_v1_models.py --tune --trials 100      # with Optuna tuning

# Rebuild v1 feature CSV
python ml/ML_data_preparation_v1.py
python ml/ML_data_preparation_v1.py --min-date 2018-01-01

# Tune v2 hyperparameters with Optuna (slow -- update config.py *_PARAMS after)
python ml/XGBoost.py --tune --trials 100
python ml/logistic_regression.py --tune --trials 100
python ml/random_forest.py --tune --trials 100
python ml/lightgbm_model.py --tune --trials 100
python ml/soft_vote_ensemble.py --trials 100

# Build the UFCStats per-fight DB from scratch (runs overnight ~8-10 hours)
python scripts/scrape_history.py --no-rolling
python -c "from db.rolling import main; from config import DB_UFCSTATS_PATH; main(db_path=DB_UFCSTATS_PATH)"

# Predict a fight (uses v1 DB + models by default)
python predict.py "Islam Makhachev" "Charles Oliveira"
python predict.py "Islam Makhachev" "Charles Oliveira" --model ensemble
python predict.py "Jones" "Miocic" --model lr --division "light heavyweight" --title

# Predict a full upcoming event card (scrapes UFCStats, outputs predictions/ MD)
python scripts/predict_event.py
python scripts/predict_event.py --model ensemble
python scripts/predict_event.py --output predictions/my-event.md

# Backtest v1 model accuracy year-by-year (--from-year 2025 for honest out-of-sample)
python scripts/backtest_v1.py --from-year 2025
python scripts/backtest_v1.py --model ensemble

# Backtest v2 model accuracy year-by-year
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

Two parallel pipelines exist. v1 is the active prediction pipeline.

**v1 pipeline (active):**
UFCStats scrape -> UFCStats DB (raw) -> mdabbert DB (career averages) -> v1 feature CSV -> v1 models -> Predictions

**v2 pipeline (reference):**
UFCStats scrape -> UFCStats DB -> Rolling stats -> v2 feature CSV -> v2 models

The v2 pipeline has 10 numbered steps (defined in `run_pipeline.py`):

| Steps | Layer | What happens |
|-------|-------|--------------|
| 1-3 | DB build | Served by `scripts/scrape_history.py` for the UFCStats schema |
| 4 | Feature engineering | DB -> ML feature CSV with ELO, form, age, style, division encoding |
| 5-9 | Training | Feature CSV -> five saved `.joblib` model groups |
| 10 | Ensemble | Optuna-tuned soft-vote weights saved to `ensemble.joblib` |

### Databases

| DB | Path | Schema | Status |
|----|------|--------|--------|
| **UFCStats** | `db/ufc_ufcstats.db` | Per-fight granular stats + rolling windows | Raw data source; updated by scraper |
| **mdabbert** | `db/ufc_v2.db` | Career-aggregate pre-fight snapshots | **Primary** -- used by v1 models for predictions |

`DB_PATH` in `config.py` points to `DB_UFCSTATS_PATH` (UFCStats). `DB_V1_PATH` points to `ufc_v2.db` (mdabbert).

**Important**: The mdabbert DB (`ufc_v2.db`) does not auto-update from the scraper. After each event, run the scraper to update the UFCStats DB, then manually update the mdabbert CSV. Issue #56 tracks building `scripts/sync_v1_from_v2.py` to automate this.

#### UFCStats DB schema (`db/ufc_ufcstats.db`)

Three tables:

- **`fighters`** -- one row per fighter: `fighter_id` (hex from UFCStats URL), `name`, `height`, `reach`, `stance`, `dob`
- **`fights`** -- one row per fight: `fight_id`, `event_id`, `date`, `division`, `r_fighter_id`, `b_fighter_id`, `winner_id`, `method`, `title_fight`, `odds_red`, `odds_blue`
- **`fight_stats`** -- two rows per fight (one per corner): raw per-fight stats (`kd`, `sig_str_landed`, `sig_str_atmpted`, `head_landed/atmpted`, `body_landed/atmpted`, `leg_landed/atmpted`, `dist/clinch/ground landed/atmpted`, `td_landed/atmpted`, `sub_att`, `ctrl`, `total_fight_time`) plus all rolling columns added by `rolling.py` (accuracies, rates, splm, sapm, str_def, td_avg, td_def, sub_avg, wins, losses, etc.)

Fighter IDs are hex strings from UFCStats.com URLs (e.g. `c2299ec916bc7c56`). Red/Blue corners are read from the fight detail page `div.b-fight-details__person` divs (first = Red, second = Blue) -- NOT from the event listing order which puts the winner first.

#### mdabbert DB schema (`db/ufc_v2.db`)

Three tables:

- **`fighters`** -- one row per fighter: `fighter_id` (MD5 hash of name), `name`, `height`, `reach`, `stance`, `dob`, `weight`
- **`fights`** -- one row per fight: `fight_id`, `event_id`, `date`, `r_fighter_id`, `b_fighter_id`, `winner_id`, `method`, `division`, `title_fight`
- **`fight_stats`** -- two rows per fight (one per corner): career-aggregate stats at the time of the fight (`avg_sig_str_pct`, `avg_td_pct`, `splm`, `td_avg`, `avg_sub_att`, `win_by_ko`, `win_by_sub`, `win_by_dec_unanimous`, `win_by_dec_split`, `wins`, `losses`, `career_win_streak`, `career_lose_streak`, `longest_win_streak`, `total_rounds_fought`, `total_title_bouts`, `total_fight_time`, `height`, `reach`, `stance`, `age`, `weightclass_rank`)

Note: mdabbert does NOT have `sapm`, `str_def`, or `td_def` natively. These are enriched at feature-build time from the UFCStats DB via `enrich_from_v2()` in `ML_data_preparation_v1.py`.

### Rolling stats (`db/rolling.py`)

Reads `fight_stats`, sorts by date, applies `shift(1)` so each row only contains data from fights *before* the current one, then upserts the computed columns back into `fight_stats`. This is the critical leakage-prevention step -- never skip the shift.

`main()` accepts an optional `db_path` parameter so it can target either DB.

### UFCStats scraper (`scrapers/ufcstats.py`)

`scrape_events_iter(since, existing_fighter_ids, skip_event_ids)` is a generator that yields `(event, data)` pairs. `scrape_history.py` uses it with checkpointing every 10 events. Resume support: already-ingested `event_id`s are skipped automatically.

### ELO / Glicko-2 calculator (`ml/ELO_calculator.py`)

Replays all historical fights chronologically to produce pre-fight ratings. Two rating systems:

**ELO** (original):
- Per-division: keys ratings by `(fighter_id, division)` tuple -- `build_elo_features()` and `get_current_ratings_by_division()`
- Global (fallback): `get_current_ratings()` -- used when no division is specified
- K-factor is `K_FACTOR_PROVISIONAL=90` for fighters with <=3 fights, then `K_FACTOR_NORMAL=32`. Starting ELO is 1400 (config).

**Glicko-2** (issue #40, additive alongside ELO):
- Same `(fighter_id, division)` key scheme
- Tracks rating `r`, deviation `rd`, and volatility `sigma` per fighter
- Fights grouped into calendar-quarter rating periods; fighters with no bouts get RD inflated (inactivity decay)
- Produces `glicko_diff` (rating gap) and `glicko_rd_diff` (uncertainty gap) features
- Public API: `build_glicko_features(conn)`, `get_current_glicko_by_division(conn)` -> `dict[(fighter_id, div), (r, rd, sigma)]`
- Constants: `GLICKO_START_R=1500`, `GLICKO_START_RD=350`, `GLICKO_START_SIGMA=0.06`, `GLICKO_TAU=0.5`

### v2 Feature dataset (`ml/ML_data_preparation.py`)

Joins fights + fight_stats + ELO + recent form into a flat feature CSV. Key transformations:

- All stats become **diff columns** (`red_stat - blue_stat`) so Red/Blue assignments are symmetry-augmented during training (swap corners + flip target) to remove assignment bias
- **Feature selection**: raw per-fight counts (`_landed`, `_atmpted`, `sub_att`, `ctrl`, `kd`) are excluded from the diff loop via `EXCLUDE_STAT_KEYWORDS` -- these are too noisy (single-fight values). Only derived rolling stats (accuracies, rates, splm, sapm, etc.) are diffed.
- **Shrinkage toward division mean**: each fighter's rolling stats are blended toward their division average -- `smoothed = (n * raw + lambda * div_mean) / (n + lambda)` where `n` = prior fights and `lambda = SHRINKAGE_LAMBDA = 5`. Stabilises features for fighters with few fights.
- **Recent form**: rolling win rate and finish rate over last 3 fights (configurable via `RECENT_FORM_WINDOW` in config)
- **Style matchup**: `striker_vs_wrestler` and `wrestler_vs_striker` interaction terms; `southpaw_adv_diff` (+1/-1/0) and `both_southpaw` (binary) stance features
- **Finish rates**: `ko_rate_diff`, `sub_rate_diff`, `dec_rate_diff` -- career win rates by method
- **Inactivity**: `days_since_last_diff` -- days since last fight
- **Strength of schedule**: `sos_diff` -- avg ELO of last 5 opponents (config: `SOS_WINDOW`)
- **KO vulnerability**: `ko_vuln_diff` -- times stopped by KO/TKO as loser in last 3 fights (config: `KO_VULN_WINDOW`)
- **Time-decay accuracy**: `ewma_str_acc_diff`, `ewma_td_acc_diff` -- EWMA of per-fight striking/TD accuracy (config: `EWMA_SPAN`); `str_acc_var_diff` -- rolling std of per-fight striking accuracy
- **Trajectory/momentum**: `win_streak_diff`, `loss_streak_diff` -- consecutive W/L run entering the fight; `str_acc_slope_diff`, `td_acc_slope_diff`, `splm_slope_diff` -- `np.polyfit` slope of per-fight metric over last `TRAJECTORY_WINDOW=5` fights (min_periods=2; 0-imputed for fighters with <2 prior fights)
- **Division**: 12-column one-hot encoding
- Debutant imputation function exists (`impute_debutant_stats`) but is **not called** -- tested and hurt accuracy

### v1 Feature dataset (`ml/ML_data_preparation_v1.py`)

Builds the v1 feature CSV from the mdabbert DB. Same diff-column and symmetry-augmentation approach as v2, but based on career averages instead of rolling windows. Key differences and additions:

- **Base stats from mdabbert**: `avg_sig_str_pct`, `avg_td_pct`, `splm`, `td_avg`, `avg_sub_att`, finish rates, wins, losses, streaks, `weightclass_rank`
- **Defensive stats from UFCStats DB** (cross-DB enrichment via `enrich_from_v2()`): `sapm`, `str_def`, `td_def` -- mdabbert lacks these natively. Name-matched (95.6% coverage) with `merge_asof(direction='backward')` to get the pre-fight value at each fight date.
- **Weightclass rank**: `weightclass_rank_diff` -- unranked fighters encoded as 16 (worse than any ranked fighter)
- **ELO + Glicko-2**: computed from mdabbert DB fight history (same `ELO_calculator.py`)
- **Recent form**: win rate and finish rate over last 3 fights
- **Style matchup**: same stance/style interaction terms as v2
- **SOS**: `sos_diff` -- avg ELO of last 5 opponents
- **Trajectory/momentum**: `win_streak_diff`, `loss_streak_diff`, slope features
- **Division**: 12-column one-hot encoding
- Training cutoff: `MIN_FIGHT_DATE = "2018-01-01"` (same as v2; adversarial validation confirmed pre-2018 distribution shift)
- Sample weighting: `SAMPLE_WEIGHT_ALPHA = 0.0` (disabled -- tested, consistently hurt accuracy vs hard cutoff)

### v1 Models

All v1 models are saved to `models_v1/` as `.joblib` files and tracked in git. These are the **active prediction models**.

| Model | Artifacts | Hold-out Acc (2025+) | Notes |
|-------|-----------|----------------------|-------|
| XGBoost | `xgboost.joblib`, `xgb_features.joblib` | -- | Optuna-tuned params in `config.XGB_PARAMS` |
| Logistic Regression | `logistic_regression.joblib`, `lr_scaler.joblib`, `lr_features.joblib` | -- | Platt-calibrated; artifact is dict with `base`+`platt` keys |
| Random Forest | `random_forest.joblib`, `rf_features.joblib` | -- | Optuna-tuned params in `config.RF_PARAMS` |
| LightGBM | `lightgbm.joblib`, `lgbm_features.joblib` | -- | Optuna-tuned params in `config.LGBM_PARAMS` |
| Ensemble | `ensemble.joblib` | **68.3%** | Calibrated soft-vote; Optuna weights tuned on pre-2025 test rows only |

Honest out-of-sample backtest (2025-2026, 586 fights): **68.3%** accuracy (ensemble). Naive Red baseline ~55%.

Note: `--from-year 2022` backtest numbers (82-90%) are inflated because 2022-2024 fights fall inside the training window with the 80/20 split. Always use `--from-year 2025` for honest evaluation.

**Ensemble leakage fix** (implemented in both `ml/train_v1_models.py` and `ml/soft_vote_ensemble.py`): Optuna weight search is restricted to pre-2025 test rows (`_ENSEMBLE_HOLDOUT_YEAR = 2025`). If the pre-2025 window has fewer than `_OPTUNA_MIN_ROWS = 50` rows, equal weights are used as fallback. The reported accuracy is always on the 2025+ holdout, not the Optuna tuning window.

### v2 Models (reference)

All v2 models are saved to `models/` as `.joblib` files and tracked in git.

| Model | Script | Artifacts | Test Acc | Notes |
|-------|--------|-----------|----------|-------|
| XGBoost | `ml/XGBoost.py` | `xgboost.joblib`, `xgb_features.joblib` | 62.54% | Optuna-tuned params in `config.XGB_PARAMS` |
| Logistic Regression | `ml/logistic_regression.py` | `logistic_regression.joblib`, `lr_scaler.joblib`, `lr_features.joblib` | 63.73% | Platt-calibrated; artifact is dict with `base`+`platt` keys |
| Random Forest | `ml/random_forest.py` | `random_forest.joblib`, `rf_features.joblib` | 62.69% | Optuna-tuned params in `config.RF_PARAMS` |
| LightGBM | `ml/lightgbm_model.py` | `lightgbm.joblib`, `lgbm_features.joblib` | 62.39% | Optuna-tuned params in `config.LGBM_PARAMS` |
| Ensemble | `ml/soft_vote_ensemble.py` | `ensemble.joblib` | **62.69%** | Calibrated soft-vote over XGB+LR+RF+LightGBM |
| Finish type | `ml/finish_type_model.py` | `finish_type.joblib`, `finish_type_features.joblib` | ~51% | 3-class (Decision/KO-TKO/Submission); use as soft signal only |

v2 out-of-sample backtest (2022-2026, 1832 fights): **66.5%** accuracy (XGBoost), +10.5% over naive Red baseline.

Training data cutoff: `MIN_FIGHT_DATE = "2018-01-01"` (issue #47 -- adversarial validation showed significant distribution shift pre-2018). Hyperparameters re-tuned on the 2018+ dataset.

The LR artifact is a dict with `base` (raw model) and `platt` (calibration wrapper) -- load with `artifact["base"]` and `artifact["platt"]`.

The ensemble artifact is a dict with `mode` (`"calibrated_soft_vote"`), `weights` (per-model float weights), `calibrators` (per-model `IsotonicRegression`), and `test_accuracy`. Retrain it (step 10) whenever any base model is retrained. Calibrators are fitted on the last 20% of training data (in-sample, but regularised models do not fully memorise it). At inference, raw probabilities from each base model are passed through their calibrator before the weighted average.

Training uses `TimeSeriesSplit(5)` with chronological ordering. The train/test split point is set by `TRAIN_TEST_SPLIT=0.80` (most recent 20% is test). For honest out-of-sample evaluation, use `backtest.py --from-year 2022`.

### Prediction (`predict.py` and `api.py`)

`predict.py` is the CLI; `api.py` is the FastAPI wrapper around the same logic. Both default to the v1 DB (`DB_V1_PATH`) and v1 models (`MODELS_V1_DIR`).

1. Resolve fighter names (fuzzy LIKE search against `fighters` table)
2. Pull latest career-aggregate stats from `fight_stats` (most recent fight row)
3. Compute current ELO by replaying fight history (`get_current_ratings_by_division`)
4. Compute recent form
5. Fetch `sapm`/`str_def`/`td_def` from the UFCStats DB (`_get_v2_defensive_stats`) -- these are not in mdabbert natively
6. Build the feature vector via `build_feature_vector()` -- must match the feature names the model was trained on
7. Optionally compute value bets via `odds.py` if American moneyline odds are supplied

Note: shrinkage is applied during v2 training but NOT at inference time. v1 uses career averages which are not shrunk.

### Event predictions (`scripts/predict_event.py`)

Scrapes the next upcoming UFC event from UFCStats (requires Playwright), runs v1 predictions for every non-debut fight, and writes a Markdown file to `predictions/`.

- Debut check uses the UFCStats DB (fighter IDs from the scraper match UFCStats hex IDs)
- Fighter name normalisation also uses UFCStats DB, then names are passed to v1 prediction by fuzzy search
- Output format: single-model table (no v2 comparison column)
- Predictions are stored in `predictions/<slug>.md` and tracked in git

### Central config (`config.py`)

Single source of truth for all paths, constants, and hyperparameters. Always import from here -- never hardcode paths. Key constants:

- `DB_PATH` / `DB_UFCSTATS_PATH` -- UFCStats rolling DB (raw per-fight data)
- `DB_V1_PATH` -- mdabbert career-aggregate DB (primary prediction source)
- `MODELS_V1_DIR` -- v1 model artifacts directory
- `CSV_V1_WITH_ELO` -- v1 feature CSV (`ml/ufc_ml_data_v1.csv`)
- `XGB_PARAMS`, `LR_PARAMS`, `RF_PARAMS`, `LGBM_PARAMS` -- Optuna-tuned hyperparameters (shared between v1 and v2 training)
- `MIN_FIGHT_DATE` -- training data cutoff (currently `"2018-01-01"`; see issue #47)
- `SAMPLE_WEIGHT_ALPHA` -- exponential time-decay weight for training rows; `0.0` = disabled (tested, hurt accuracy)
- `EXCLUDE_STAT_KEYWORDS` -- columns excluded from the diff feature loop
- `EXCLUDED_FEATURES` -- features dropped at training and inference time after permutation-importance analysis (issue #43); currently removes 3 dead columns (`date_diff`, `outcome_diff`, `age_diff`) that carry zero signal
- `SHRINKAGE_LAMBDA` -- prior weight for division-mean shrinkage (v2 training only, default 5)
- `STARTING_ELO`, `GLICKO_START_R`, `GLICKO_START_RD`, `GLICKO_START_SIGMA`, `GLICKO_TAU` -- rating system constants
- `DIVISIONS`, `FINISH_METHOD_MAP`, `TRAIN_TEST_SPLIT`

### Encoding note (Windows)

All Python source files must use ASCII-safe characters only. The Windows cp1252 console rejects Unicode symbols (->, --, -, #). Use `->`, `--`, `-`, `#` instead.

---

## Before committing

Files that must never be committed:

- `**/__pycache__/` and `*.pyc` -- Python bytecode
- `db/ufc_ufcstats.db` -- UFCStats SQLite DB (regenerate with `scripts/scrape_history.py`)
- `db/ufc_v2.db` -- mdabbert SQLite DB (regenerate from mdabbert CSV)
- `db/*_backup_*.db` -- rolling.py backup files
- `logs/` -- runtime logs
- `ml/*.csv` -- intermediate ML datasets (except `ufc_ml_data_with_debuts_and_elo.csv` which is explicitly un-ignored)
- `raw_data/*.db` -- raw database files

Files that ARE tracked:

- `raw_data/ufc-master.csv` -- source of truth for mdabbert pipeline
- `models/*.joblib` -- v2 trained model artifacts
- `models_v1/*.joblib` -- v1 trained model artifacts; tracked so predictions work immediately after cloning
- `predictions/*.md` -- event prediction files

If any excluded files were previously committed, untrack them with `git rm --cached <file>` (without deleting the local copy), then verify `.gitignore` covers them before staging the commit.

---

## Key invariants

- **No leakage**: every rolling stat uses `shift(1)`. ELO is computed *before* the fight is processed. Recent form excludes the current fight. Violating this inflates accuracy.
- **Corner assignment**: Red/Blue corners come from the fight detail page (`div.b-fight-details__person`), NOT the event listing (which puts the winner first). Getting this wrong causes ~100% Red win rate in training data.
- **Symmetry augmentation**: during training, each fight is duplicated with corners swapped and target flipped. This is applied *per fold* to the training split only -- never to the validation/test split.
- **SQLite TEXT affinity**: some numeric columns (e.g. `kd`, `ctrl`) are stored as TEXT. Always use `CAST(col AS REAL)` in numeric comparisons or aggregations.
- **Ensemble holdout**: Optuna weight search must only see pre-2025 test rows (`_ENSEMBLE_HOLDOUT_YEAR = 2025`). The reported accuracy must be on the 2025+ holdout. Violating this leaks the backtest evaluation window into weight tuning.
- **Backtest year for v1**: Use `--from-year 2025` for honest evaluation. The 80/20 split puts 2022-2024 fights inside the training window; `--from-year 2022` numbers are inflated.
- **v1 DB does not auto-update**: After each event, the UFCStats DB is updated by the scraper but the mdabbert DB must be updated separately. Issue #56 tracks automating this via `scripts/sync_v1_from_v2.py`.
- **Model performance ceiling**: v1 career-average models achieve ~68% on the 2025+ backtest (586 fights). The naive "always pick Red" baseline is ~55% on recent data. Update `MODEL_RESULTS.md` after any significant retrain.
- **Shrinkage is v2 training-only**: `apply_shrinkage()` in `ML_data_preparation.py` modifies the training CSV. It is NOT applied in `predict.py` at inference time. v1 uses raw career averages with no shrinkage.
