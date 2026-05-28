# UFC Predictor — Improvements Log

---

## ✅ Implemented

### Infrastructure

#### `config.py` (new)
Single source of truth for every path and constant in the project.
Before: each script had its own hardcoded paths (including absolute paths like `C:\Users\35193\Downloads\UFC.csv`).
After: every script imports from `config.py` — change a path once, it updates everywhere.

#### `requirements.txt` (new)
Lists all Python dependencies (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `matplotlib`, `seaborn`).
Anyone can clone the repo and run `pip install -r requirements.txt` to get started.

#### `README.md` (new)
Full project documentation covering architecture pipeline diagram, setup instructions, prediction CLI usage, project structure, model comparison, and feature descriptions.

#### `logger.py` (new)
Shared logging setup used by every script in the project.
- `get_logger(name)` returns a logger writing to both console and `logs/ufc_predictor.log`.
- Format: `timestamp | module | level | message`. File handler captures DEBUG; console defaults to INFO.
- All ML scripts (`ELO_calculator.py`, `ML_data_preparation.py`, `XGBoost.py`, `logistic_regression.py`, `predict.py`) now use this instead of bare `print()`.

#### File renames — spaces removed
- `ML_models/logistic regression.py` → `ML_models/logistic_regression.py` (canonical)
- `database_builder_files/raw SQL database.py` → `database_builder_files/raw_sql_database.py` (canonical)
- Old space-named files kept as thin `runpy` stubs so nothing breaks; safe to delete once confirmed unused.
- `raw_sql_database.py` now accepts `--csv` and `--db` CLI flags and uses `config.py` paths instead of hardcoded strings.

#### `run_pipeline.py` (new)
Single command to run the entire pipeline from raw data to trained models.
```
python run_pipeline.py                        # ML steps only (4,5,6) — most common
python run_pipeline.py --full --csv UFC.csv   # all 6 steps from scratch
python run_pipeline.py --steps 4,5            # specific steps only
python run_pipeline.py --dry-run              # preview without executing
```
Steps 1–3 (DB build) run as subprocesses; steps 4–6 (ML) use direct imports for speed and unified logging. Guards against running DB-dependent steps when the database doesn't exist. Prints a summary table of step results and elapsed times.

#### `refresh_data.py` (new)
Data refresh entry point for when new UFC events are added.
- `--csv path/to/UFC.csv` — rebuild DB from an updated export and retrain (works today).
- `--auto` — scrape new events from ufcstats.com since the last DB event (integration point documented; scraper not yet wired in).
- `--dry-run` flag supported on both modes.

#### Type hints — all ML scripts
Full PEP 3.10+ annotations on all functions across `ELO_calculator.py`, `XGBoost.py`, `logistic_regression.py`, `ML_data_preparation.py`, `predict.py`, and `run_pipeline.py`.

---

### Machine Learning

#### Model Serialisation — `XGBoost.py` + `logistic_regression.py`
Both models now save their artifacts to `models/` after training using `joblib`:
- XGBoost saves: `xgboost.joblib`, `xgb_features.joblib`
- Logistic Regression saves: `logistic_regression.joblib`, `lr_scaler.joblib`, `lr_features.joblib`

Before: models were trained and then discarded — every prediction required a full retrain.
After: train once, load instantly.

#### Cross-Validation — both model scripts
Added `TimeSeriesSplit` (5-fold) cross-validation that respects chronological ordering.
Each fold trains on older fights and validates on newer ones.
Symmetry (red↔blue flip) is applied per fold to the training split only — no leakage.
Reports per-fold accuracy + mean ± std before final training.

#### Probability Calibration — `logistic_regression.py`
Added `CalibratedClassifierCV` with `method='sigmoid'` and `cv='prefit'`.
Uses a proper 70 / 10 / 20 chronological train / calibration / test split.
Reports Brier score before and after calibration + calibration curve in the visualisation output.

---

### ELO Calculator — `ML_models/ELO_calculator.py`

#### Refactor + `get_current_ratings()` (new function)
- Extracted shared replay logic into a private `_replay_fights()` helper — no more duplicated loops between `build_elo_features()` and `get_current_ratings()`.
- `get_current_ratings(conn)` replays all fights and returns each fighter's ELO **after** their most recent bout. Used by `predict.py`.
- Full type hints and logging added throughout.

---

### Prediction CLI — `predict.py` (new)

```
python predict.py "Islam Makhachev" "Charles Oliveira"
python predict.py "Jones" "Miocic" --model lr
```

- Fuzzy fighter name search (partial names work; prompts if ambiguous)
- Pulls each fighter's latest rolling stats from the database
- Computes current ELO ratings by replaying all fight history
- Builds the feature vector in the exact format the model was trained on
- Displays win probabilities with a visual progress bar and ELO ratings
- Works with both XGBoost (default) and Logistic Regression (`--model lr`)
- Logs errors to `logs/ufc_predictor.log`

---

### Tests — `tests/test_pipeline.py` (new)

27 tests across 5 classes. Run with `python -m pytest tests/ -v`.

| Class | Tests | What's checked |
|-------|-------|----------------|
| `TestDatabase` | 8 | Required tables exist, fighter/fight counts, 2 stat rows per fight, no orphan records, no duplicate IDs, no null dates, winner references valid corner |
| `TestRollingStats` | 2 | No data leakage on first fight appearances, no negative fight times |
| `TestELO` | 5 | Correct output shape, first fight at STARTING_ELO, all ratings positive and finite, `get_current_ratings()` returns a valid dict |
| `TestMLDataset` | 7 | Binary target, no null targets, red-win rate in expected range, ELO column present, diff columns symmetric, no future data in train set, minimum row count |
| `TestSavedModels` | 5 | Model and scaler load, predict_proba in [0,1] summing to 1 — **auto-activates after first training run** |

**Data finding surfaced by tests:** ~2% of early-era UFC veterans have pre-seeded career stats on their first recorded date (their pre-dataset fights are absent from the source CSV). Documented as a known limitation; threshold set at 5% to catch real regressions.

---

---

### Feature Engineering — `ML_models/ML_data_preparation.py` (v2)

19 new feature columns added to the ML dataset (77 total, up from 58):

#### Recent form (3 new diff features)
`compute_recent_form()` builds per-fighter rolling statistics over the last 3 fights before each bout (shift(1) applied — no leakage):
- `recent_win_rate_diff` — fraction of last 3 fights won (Red − Blue)
- `recent_finish_rate_diff` — fraction of last 3 fights ended by KO/TKO or Submission (Red − Blue)
- `win_streak_diff` — consecutive wins immediately before this fight (Red − Blue)

Career averages already in the dataset capture long-run skill; these features capture momentum and current form.

#### Age at fight date (1 new diff feature)
`add_age_features()` parses `dob` from `fight_stats` (format: `YYYY/MM/DD`) and computes each fighter's age at the fight date:
- `age_diff` = age_red − age_blue (years, float)

Age at ~28–32 is typically peak performance in MMA; this lets the model learn peak-vs-declining matchups.

#### Style matchup features (4 new features)
`add_style_features()` computes style ratios from `splm` (significant strikes per minute) and `td_avg` (takedowns per 15 min):
- `grapple_ratio_diff` = grapple_ratio_red − grapple_ratio_blue
- `strike_ratio_diff`  = strike_ratio_red  − strike_ratio_blue
- `striker_vs_wrestler` = strike_ratio_red × grapple_ratio_blue  (high when Red is a striker vs Blue wrestler)
- `wrestler_vs_striker` = grapple_ratio_red × strike_ratio_blue  (high when Red is a wrestler vs Blue striker)

`striker_vs_wrestler` was the 10th most important XGBoost feature out of the gate.

#### Division one-hot encoding (12 new features)
`add_division_features()` adds a `div_*` binary column for each of the 12 known UFC weight classes.
The model can now learn division-specific patterns (e.g. finishes are far more common in Heavyweight than Strawweight).

#### Title fight flag (1 new feature)
`title_fight` is passed directly as a binary feature (0/1). Championship fights have meaningfully different dynamics — more conservative game-plans, 5 rounds vs 3.

---

### Finish Type Model — `ML_models/finish_type_model.py` (new)

Secondary XGBoost classifier (3-class: Decision / KO-TKO / Submission) trained on the same feature set.
- 5-fold TimeSeriesSplit CV: **48.1% ± 2.6%** (baseline for majority-class = 50.5%)
- Test accuracy: **50.1%** — usable as a soft signal, not a standalone predictor
- Saved as `models/finish_type.joblib` + `models/finish_type_features.joblib`
- `predict.py` loads it automatically when present and adds a "Predicted Finish Method" section to output
- Integrated as step 7 in `run_pipeline.py`

---

### Prediction CLI — `predict.py` (updated)

New arguments:
- `--division lightweight` — enables division one-hot encoding for the prediction
- `--title` — flags the fight as a title fight

New output section:
```
  Recent form (last 3 fights):
    Max Holloway               win_rate=67%  finish_rate=67%  streak=1
    Charles Oliveira           win_rate=33%  finish_rate=33%  streak=0

  Predicted Finish Method:
    Decision      ############-------- 60.6%
    KO/TKO        #####--------------- 27.1%
    Submission    ##------------------ 12.3%
```

---

## 🔲 Not Yet Implemented

### Machine Learning

- **Per-division ELO ratings** — ELO is currently global. A win in Heavyweight moves the same pool as a win in Women's Strawweight. Key ratings by `(fighter_id, division)` in `ELO_calculator.py`; update `ML_data_preparation.py` and `predict.py` accordingly.

- **Hyperparameter tuning** — XGBoost and LR params are manually chosen. Add `Optuna` or `GridSearchCV` with temporal cross-validation to find optimal settings systematically.

- **More models** — Random Forest, LightGBM, MLP. Ensemble them for better accuracy.

- **Better debutant handling** — The binary `is_debutant_diff` flag is a rough proxy. Try using global-average stats as a Bayesian prior for fighters with no history.

- **Experiment tracking** — Add MLflow or Weights & Biases to log params, metrics, and artifacts across runs so model versions can be compared.

### Feature Engineering

- **Per-division ELO ratings** — ELO is currently global. A win in Heavyweight moves the same pool as a win in Women's Strawweight. Key ratings by `(fighter_id, division)` in `ELO_calculator.py`; update `ML_data_preparation.py` and `predict.py` accordingly.

- **Hyperparameter tuning** — XGBoost and LR params are manually chosen. Add `Optuna` or `GridSearchCV` with temporal cross-validation to find optimal settings systematically.

- **More models** — Random Forest, LightGBM, MLP. Ensemble them for better accuracy.

- **Better debutant handling** — The binary `is_debutant_diff` flag is a rough proxy. Try using global-average stats as a Bayesian prior for fighters with no history.

- **Data leakage audit** — Formal verification that `.shift()` is applied correctly everywhere in `rolling.py`. A single off-by-one means future data bleeds into training.

### Betting Odds Integration

- **Odds columns in the `fights` table** — Add `odds_red` and `odds_blue` columns (American or decimal format) to the DB schema. Stored as **metadata only** — not used as a training feature so the model stays independent of the market.

- **Implied probability converter** — Helper that converts raw odds to win probability, stripping out the bookmaker's vig/overround.

- **Value bet comparator in `predict.py`** — After the model outputs its probability, compare against the odds-implied probability and surface edge:
  ```
  Model:         Islam Makhachev  68.0%
  Odds-implied:  Islam Makhachev  55.3%  (-124 American)
  → Edge: +12.7 pp  ✅ Value on Islam
  ```

- **Historical odds backtest** — Simulate ROI, value-bet win rate, and Kelly criterion stake sizing across all past fights once odds are populated.

- **BestFightOdds scraper** — Scrape closing moneylines from [bestfightodds.com](https://www.bestfightodds.com) (most comprehensive historical MMA odds source, goes back to ~2008, ~10 bookmakers). Fuzzy-match fighter names to the `fights` table by name + date and backfill `odds_red` / `odds_blue`. No public API exists — BeautifulSoup or Playwright scrape. Name normalisation is the hard part (e.g. "Conor McGregor" vs "C. McGregor"). Once populated, the value-bet comparator and Kelly sizing in `odds.py` and `predict.py` become immediately usable, and the historical backtest can run.

### Infrastructure

- **Delete old space-named stubs** — `raw SQL database.py` and `logistic regression.py` are now just stubs. Safe to delete once confirmed unused.

- **Incremental DB update** — `refresh_data.py --auto` has the integration point but the actual fight insert + rolling-stats recomputation is not yet wired up.

- **`rolling.py` refactor** — currently a top-level script with no functions or `__main__` guard, making it untestable and unimportable. Should mirror the structure of the other scripts.

### Deployment

- **REST API (FastAPI)** — `POST /predict` endpoint accepting `{"red_fighter": "...", "blue_fighter": "..."}`.

- **Web dashboard (Streamlit/Gradio)** — Drop-down fighter selector, win probability bars, feature importance breakdown.

- **Backtesting framework** — Simulate predictions at each historical point in time; report year-by-year accuracy to detect model drift.
