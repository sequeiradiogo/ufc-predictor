# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

---

## Commands

```bash
# Run tests (DB/model-dependent tests skip gracefully if those artifacts are absent -- see tests.yml)
python -m pytest tests/ -v

# Run a single test class
python -m pytest tests/test_refresh.py::TestParseAmericanOdds -v

# Train v1 models (mdabbert career-aggregate DB) -- PRIMARY PREDICTION MODELS
python ml/train_v1_models.py                          # all models (xgb/lr/rf/lgbm/mlp/ensemble) -- eval tier, models_v1/
python ml/train_v1_models.py --model xgb              # single model (choices: xgb, lr, rf, lgbm, mlp, ensemble, stacking)
python ml/train_v1_models.py --tune --trials 100      # with Optuna tuning (base models)
python ml/train_v1_models.py --model ensemble         # ensemble weights only (fast)
python ml/train_v1_models.py --model stacking         # stacking meta-model (not included in the default "all" run)
python ml/train_v1_models.py --prod                   # train production tier on 100% of data -- models_v1_prod/

# Train the finish-type (Decision/KO-TKO/Submission) classifier -- standalone, not part of "train all"
python ml/finish_type_model.py                        # eval tier -- models_v1/
python ml/finish_type_model.py --prod                 # production tier -- models_v1_prod/

# Rebuild v1 feature CSV
python ml/ML_data_preparation_v1.py
python ml/ML_data_preparation_v1.py --min-date 2018-01-01

# v1 CSV enrichment pipeline (run in order after scraping new fights)
python scripts/add_defensive_stats_to_csv.py          # add sapm/str_def/td_def from UFCStats DB
python scripts/add_rankings_to_csv.py                 # add R/B_match_weightclass_rank from rankings_history.csv
python scripts/add_computed_features_to_csv.py        # add ELO, Glicko, form, SOS, slopes, style, division one-hots

# Build the UFCStats per-fight DB from scratch (runs overnight ~8-10 hours)
python scripts/scrape_history.py --no-rolling
python -c "from db.rolling import main; from config import DB_UFCSTATS_PATH; main(db_path=DB_UFCSTATS_PATH)"

# Backfill reversals for historical fights (run once after scraper update, ~6 hours for ~8700 fights)
python scripts/backfill_reversals.py
python scripts/backfill_reversals.py --limit 100    # test first 100 fights

# Backfill per-round stats for historical fights (run once, ~6 hours for ~8700 fights)
python scripts/backfill_rounds.py
python scripts/backfill_rounds.py --limit 50        # test first 50 fights

# Sync v1 career-average DB from UFCStats DB after each event (replaces manual CSV update)
python scripts/sync_v1_from_v2.py
python scripts/sync_v1_from_v2.py --dry-run   # preview without writing

# Refresh the live UFC.com rankings snapshot into raw_data/rankings_history.csv (used by weekly-predictions.yml, before predict_event.py)
python scrapers/ufc_rankings_web.py
python scrapers/ufc_rankings_web.py --dry-run     # print the scraped snapshot without writing

# Incremental data refresh (scrape + enrich + ingest + rebuild CSV in one step; used by monthly-refresh.yml)
python scripts/refresh_data.py --auto
python scripts/refresh_data.py --auto --dry-run       # preview scrape, no writes

# Score a completed event's predictions against actual results + odds (used by monday-results.yml)
python scripts/score_event.py                          # auto-detect most recent unscored prediction
python scripts/score_event.py --json path/to/f.json
python scripts/score_event.py --min-confidence 0.55    # threshold for high-confidence breakout + P/L gating

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

# Start the REST API
uvicorn api:app --reload

# Add odds columns to DB (one-time migration)
python utils/odds.py --migrate
```

---

## Architecture

### Pipeline overview

The v1 pipeline (mdabbert career-average DB) is the active, only prediction pipeline. An earlier v2 pipeline (UFCStats rolling-stats DB, its own model set under `models/`) was reference-only and has been removed.

UFCStats scrape -> UFCStats DB (raw) -> ufc-master.csv enrichment -> mdabbert DB (career averages + pre-computed features) -> v1 feature CSV -> v1 models -> Predictions

The CSV enrichment step runs three scripts in order:
1. `scripts/add_defensive_stats_to_csv.py` -- adds `sapm`, `str_def`, `td_def` from UFCStats DB
2. `scripts/add_rankings_to_csv.py` -- adds `R/B_match_weightclass_rank` from `rankings_history.csv`
3. `scripts/add_computed_features_to_csv.py` -- adds ELO, Glicko-2, recent form, SOS, trajectory slopes, finish rates, inactivity, KO vulnerability, style matchup, stance features, division one-hots

After enrichment, `db/ingest_mdabbert.py` ingests everything into `ufc_v2.db`. `ML_data_preparation_v1.py` then reads the DB and builds the feature CSV as a pure diff-builder (no computation -- all features are pre-stored in the CSV/DB).

### Automation (GitHub Actions)

Four workflows in `.github/workflows/` automate the weekly/monthly cycle plus PR checks:

| Workflow | Schedule | What it does |
|----------|----------|---------------|
| `weekly-predictions.yml` | Fridays 12:00 UTC | Downloads DB artifacts + `raw_data/rankings_history.csv`, refreshes today's live rankings snapshot via `scrapers/ufc_rankings_web.py` and re-uploads the CSV to the release (both steps `continue-on-error` so a ufc.com markup change degrades to the existing stale CSV instead of blocking the run), runs `scripts/predict_event.py --model ensemble --skip-existing`, commits new `predictions/*.md` |
| `monday-results.yml` | Mondays 12:00 UTC | Runs `scripts/score_event.py --min-confidence 0.55` to score the weekend's event against actual results + BFO odds, commits the updated markdown |
| `monthly-refresh.yml` | 1st of month, 06:00 UTC | Full refresh: `scripts/refresh_data.py --auto` -> three CSV enrichment scripts -> `db/ingest_mdabbert.py` -> `ml/ML_data_preparation_v1.py` -> `ml/train_v1_models.py` (+ `--prod`) -> `ml/finish_type_model.py` (+ `--prod`), commits `raw_data/ufc-master.csv`, then re-uploads the DBs plus retrained `models_v1.tar.gz`/`models_v1_prod.tar.gz` to the release below (model artifacts are not committed to git -- see "DB distribution") |
| `tests.yml` | Every PR + push to `main` | Runs `python -m pytest tests/ -v` with no DB/model artifacts downloaded -- `test_inference.py`'s and `test_api.py`'s test classes `pytest.mark.skipif()` when `DB_PATH`/`DB_V1_PATH`/`MODELS_V1_DIR`/`MODELS_V1_PROD_DIR` don't exist, so CI always exercises the artifact-free unit tests in `test_refresh.py` plus whatever else can run, with no release-asset downloads or secrets needed (works on fork PRs) |

The three scheduled workflows also support `workflow_dispatch` for manual runs. **Scheduled GitHub Actions runs can be delayed by minutes to hours** (a shared-runner queueing behavior, not a repo bug) -- if a manual run and a delayed scheduled run overlap, whichever pushes second will fail with a non-fast-forward git error even though the underlying job succeeded; check the job output, not just the workflow conclusion, before assuming something didn't run.

**DB distribution**: `db/ufc_ufcstats.db` and `db/ufc_v2.db` are gitignored (too large to track) but CI needs them every run. They're stored in the `data-artifacts-latest` GitHub Release and downloaded at the start of each workflow (`gh release download data-artifacts-latest --dir db/ --pattern "*.db"`), then re-uploaded by `monthly-refresh.yml` after retraining (`gh release upload data-artifacts-latest ... --clobber`). `raw_data/rankings_history.csv` is also gitignored (not in the un-ignore list) and lives in the same release; `weekly-predictions.yml` downloads it separately (`gh release download data-artifacts-latest --dir raw_data/ --pattern "rankings_history.csv"`) since `predict.py`'s `_get_current_rank()` otherwise silently falls back to unranked (16) with no warning. Locally, keep your `db/` and `raw_data/rankings_history.csv` in sync by downloading the same release if you don't already have current copies.

**Model artifact distribution**: `models_v1/*.joblib` and `models_v1_prod/*.joblib` are gitignored, same rationale as the DBs -- binary joblib files don't diff, so committing a fresh copy every retrain bloats `.git` history indefinitely. They're stored in the same `data-artifacts-latest` release as two tarballs, one per dir since filenames collide across dirs (e.g. `ensemble.joblib`): `models_v1.tar.gz`, `models_v1_prod.tar.gz`. `weekly-predictions.yml` downloads+extracts `models_v1_prod.tar.gz` before `predict_event.py` runs (the only workflow that loads a model at runtime -- `monday-results.yml`'s `score_event.py` scores existing predictions and never loads one). `monthly-refresh.yml` tars and uploads both tiers after retraining instead of `git add`-ing them. Locally, download+extract the tarball(s) you need the same way (README "Setup"). History was rewritten with `git filter-repo` to purge previously-committed model blobs (2026-08-31) -- if `git pull` conflicts on `main`, re-clone instead.

### Databases

| DB | Path | Schema | Status |
|----|------|--------|--------|
| **UFCStats** | `db/ufc_ufcstats.db` | Per-fight granular stats + rolling windows | Raw data source; updated by scraper |
| **mdabbert** | `db/ufc_v2.db` | Career-aggregate pre-fight snapshots + pre-computed features | **Primary** -- used by v1 models for predictions |

`DB_PATH` in `config.py` points to `DB_UFCSTATS_PATH` (UFCStats). `DB_V1_PATH` points to `ufc_v2.db` (mdabbert).

**Important**: The mdabbert DB (`ufc_v2.db`) does not auto-update from the scraper on every event -- `monthly-refresh.yml` runs the full chain once a month, but there's no per-event trigger. To update sooner: (1) run `scripts/scrape_history.py` to update UFCStats DB, (2) run the three CSV enrichment scripts, (3) run `db/ingest_mdabbert.py`, (4) run `ml/ML_data_preparation_v1.py`. `scripts/refresh_data.py --auto` automates steps 1-2 (scrape + rankings) but the CSV enrichment / ingest / feature-rebuild steps still need to be run separately, same as `monthly-refresh.yml` does.

#### UFCStats DB schema (`db/ufc_ufcstats.db`)

Three tables:

- **`fighters`** -- one row per fighter: `fighter_id` (hex from UFCStats URL), `name`, `height`, `reach`, `stance`, `dob`
- **`fights`** -- one row per fight: `fight_id`, `event_id`, `date`, `division`, `r_fighter_id`, `b_fighter_id`, `winner_id`, `method`, `title_fight`, `odds_red`, `odds_blue`
- **`fight_stats`** -- two rows per fight (one per corner): raw per-fight stats (`kd`, `sig_str_landed`, `sig_str_atmpted`, `head_landed/atmpted`, `body_landed/atmpted`, `leg_landed/atmpted`, `dist/clinch/ground landed/atmpted`, `td_landed/atmpted`, `sub_att`, `reversals`, `ctrl`, `total_fight_time`) plus all rolling columns added by `rolling.py` (accuracies, rates, splm, sapm, str_def, td_avg, td_def, sub_avg, wins, losses, etc.). `reversals` was added by `backfill_reversals.py` after the initial scrape. `total_fight_time` is the CUMULATIVE seconds this fighter had fought **before** this fight (0 on debut), NOT this fight's own duration -- see the `total_fight_time` invariant below.

- **`fight_stats_rounds`** -- two rows per fight per round (one per corner): same columns as `fight_stats` per-fight counts but scoped to one round. Populated by `scripts/backfill_rounds.py` (historical backfill) and by the scraper going forward. Primary key `(fight_id, fighter_id, round)`. Round 0 is a sentinel row used by the backfill to mark fights where UFCStats had no round data. Used by `add_computed_features_to_csv.py` to compute R1 rolling features (`ewma_ctrl_r1`, `ewma_splm_r1`, `ewma_reversals_r1`).

Fighter IDs are hex strings from UFCStats.com URLs (e.g. `c2299ec916bc7c56`). Red/Blue corners are read from the fight detail page `div.b-fight-details__person` divs (first = Red, second = Blue) -- NOT from the event listing order which puts the winner first.

#### mdabbert DB schema (`db/ufc_v2.db`)

Three tables:

- **`fighters`** -- one row per fighter: `fighter_id` (MD5 hash of name), `name`, `height`, `reach`, `stance`, `dob`, `weight`
- **`fights`** -- one row per fight: `fight_id`, `event_id`, `date`, `r_fighter_id`, `b_fighter_id`, `winner_id`, `method`, `division`, `title_fight`, plus pre-computed fight-level features: `grapple_ratio_diff`, `striker_vs_wrestler`, `wrestler_vs_striker`, `southpaw_adv_diff`, `both_southpaw`, `weightclass_rank_diff`, and 12 `div_*` one-hot columns
- **`fight_stats`** -- two rows per fight (one per corner): career-aggregate stats at the time of the fight (`avg_sig_str_pct`, `avg_td_pct`, `splm`, `td_avg`, `avg_sub_att`, `win_by_ko`, `win_by_sub`, `win_by_dec_unanimous`, `win_by_dec_split`, `wins`, `losses`, `career_win_streak`, `career_lose_streak`, `longest_win_streak`, `total_rounds_fought`, `total_title_bouts`, `total_fight_time`, `height`, `reach`, `stance`, `age`, `weightclass_rank`) plus pre-computed features: `elo`, `glicko`, `glicko_rd`, `recent_win_rate`, `recent_finish_rate`, `sos`, `str_acc_slope`, `splm_slope`, `td_acc_slope`, `ko_rate`, `sub_rate`, `dec_rate`, `days_since_last`, `ko_vuln`, `kd_received`, `sapm`, `str_def`, `td_def`, `head_def`, `body_def`, `dist_def`, `ground_def`, `opp_adj_head_acc`, `opp_adj_body_acc`, `opp_adj_dist_acc`, `career_reversals`, `ewma_reversals`, `ewma_ctrl_r1`, `ewma_splm_r1`, `ewma_reversals_r1`

Note: `sapm`, `str_def`, `td_def` are enriched at CSV-build time from the UFCStats DB by `add_defensive_stats_to_csv.py`, then stored in `ufc-master.csv` and ingested into the mdabbert DB. `ML_data_preparation_v1.py` no longer does live cross-DB lookups.

### Rolling stats (`db/rolling.py`)

Reads `fight_stats`, sorts by date, applies `shift(1)` so each row only contains data from fights *before* the current one, then upserts the computed columns back into `fight_stats`. This is the critical leakage-prevention step -- never skip the shift.

`main()` accepts an optional `db_path` parameter so it can target either DB.

### UFCStats scraper (`scrapers/ufcstats.py`)

`scrape_events_iter(since, existing_fighter_ids, skip_event_ids)` is a generator that yields `(event, data)` pairs. `scrape_history.py` uses it with checkpointing every 10 events. Resume support: already-ingested `event_id`s are skipped automatically.

### ELO / Glicko-2 calculator (`ml/ELO_calculator.py`)

Replays all historical fights chronologically to produce pre-fight ratings. Two rating systems:

**ELO** (original):
- **Global** (current): `build_elo_features()` uses `_replay_fights()` which keys ratings by `fighter_id` only -- a single universal rating per fighter across all divisions. This fixes cold-start for division movers (fighters who move weight class no longer reset to 1400).
- Per-division variant still exists as `_replay_fights_by_division()` and `get_current_ratings_by_division()` -- used only in `predict.py` for inference on the mdabbert DB.
- K-factor is `K_FACTOR_PROVISIONAL=90` for fighters with <=3 fights, then `K_FACTOR_NORMAL=32`. Starting ELO is 1400 (config).

**Glicko-2** (issue #40, additive alongside ELO):
- Same global-per-fighter key scheme as ELO for training features
- Tracks rating `r`, deviation `rd`, and volatility `sigma` per fighter
- Fights grouped into calendar-quarter rating periods; fighters with no bouts get RD inflated (inactivity decay)
- Produces `glicko_diff` (rating gap) and `glicko_rd_diff` (uncertainty gap) features
- Public API: `build_glicko_features(conn)`, `get_current_glicko_by_division(conn)` -> `dict[(fighter_id, div), (r, rd, sigma)]`
- Constants: `GLICKO_START_R=1500`, `GLICKO_START_RD=350`, `GLICKO_START_SIGMA=0.06`, `GLICKO_TAU=0.5`

### v1 Feature dataset (`ml/ML_data_preparation_v1.py`)

Pure diff-builder: reads pre-computed feature columns from the mdabbert DB and converts them to `red - blue` diff columns. No computation is done here -- all features are pre-stored in `ufc-master.csv` via the three enrichment scripts and ingested into the DB by `db/ingest_mdabbert.py`.

Pre-computed features (stored in CSV/DB, diffed at build time):
- **Base stats from mdabbert**: `avg_sig_str_pct`, `avg_td_pct`, `splm`, `td_avg`, `avg_sub_att`, finish rates, wins, losses, streaks, `weightclass_rank`
- **Defensive stats** (from `add_defensive_stats_to_csv.py`): `sapm`, `str_def`, `td_def`, `head_acc`, `body_acc`, `leg_acc`, `head_def`, `body_def`, `dist_def`, `ground_def`
- **ELO** (global, from `add_computed_features_to_csv.py`): `elo_diff`
- **Glicko-2** (from `add_computed_features_to_csv.py`): `glicko_diff`, `glicko_rd_diff`
- **Recent form** (from `add_computed_features_to_csv.py`): `recent_win_rate_diff`, `recent_finish_rate_diff`
- **SOS** (from `add_computed_features_to_csv.py`): `sos_diff`
- **Finish rates** (from `add_computed_features_to_csv.py`): `ko_rate_diff`, `sub_rate_diff`, `dec_rate_diff`
- **Inactivity** (from `add_computed_features_to_csv.py`): `days_since_last_diff`
- **KO vulnerability** (from `add_computed_features_to_csv.py`): `ko_vuln_diff` -- cumulative KO/TKO losses (all history)
- **Knockdowns received** (from `add_computed_features_to_csv.py`): `kd_received_diff` -- cumulative knockdowns received; computed from UFCStats DB (has per-fight `kd`), mapped back to mdabbert IDs via MD5
- **Trajectory slopes** (from `add_computed_features_to_csv.py`): `str_acc_slope_diff`, `splm_slope_diff`, `td_acc_slope_diff`
- **Weightclass rank** (from `add_rankings_to_csv.py`): `weightclass_rank_diff` -- unranked fighters encoded as 16
- **Style matchup** (fight-level, from `add_computed_features_to_csv.py`): `grapple_ratio_diff`, `striker_vs_wrestler`, `wrestler_vs_striker`
- **Stance** (fight-level, from `add_computed_features_to_csv.py`): `southpaw_adv_diff`, `both_southpaw`
- **Division** (fight-level, from `add_computed_features_to_csv.py`): 12 `div_*` one-hot columns
- **Opponent-adjusted stats** (from `add_computed_features_to_csv.py`): `opp_adj_splm_diff`, `opp_adj_td_avg_diff`, `opp_adj_head_acc_diff`, `opp_adj_body_acc_diff`, `opp_adj_dist_acc_diff` -- output stats adjusted for opponent defense quality
- **Group E -- Reversals** (from `add_computed_features_to_csv.py`): `career_reversals_diff` -- cumulative reversals scored (all history); `ewma_reversals_diff` -- EWMA reversals per 15 min. Requires `backfill_reversals.py` to have been run; zero-filled until then.
- **Group F -- Round 1 stats** (from `add_computed_features_to_csv.py`): `ewma_ctrl_r1_diff`, `ewma_splm_r1_diff`, `ewma_reversals_r1_diff` -- EWMA of control time, striking output, and reversals in round 1 specifically. Requires `backfill_rounds.py` to have been run; zero-filled until then.
- Training cutoff: `MIN_FIGHT_DATE = "2018-01-01"` (adversarial validation confirmed pre-2018 distribution shift)
- Sample weighting: `SAMPLE_WEIGHT_ALPHA = 0.01`, `SAMPLE_WEIGHT_BETA = 1.5` -- power-law decay `exp(-alpha * delta^beta)`; beta=1.5 steepens decay for older fights while keeping recent fights near full weight (issue #54)

### v1 Models

All v1 models are saved to `models_v1/` as `.joblib` files, gitignored and distributed via the `data-artifacts-latest` release (see "Model artifact distribution" above). These are the **active prediction models**, used for backtesting and hyperparameter tuning.

| Model | Artifacts | 2025+ Acc | Notes |
|-------|-----------|-----------|-------|
| XGBoost | `xgboost.joblib`, `xgb_features.joblib` | 67.5% | Default params in `config.XGB_PARAMS` |
| Logistic Regression | `logistic_regression.joblib`, `lr_scaler.joblib`, `lr_features.joblib` | 65.3% | Platt-calibrated |
| Random Forest | `random_forest.joblib`, `rf_features.joblib` | 67.0% | Default params in `config.RF_PARAMS` |
| LightGBM | `lightgbm.joblib`, `lgbm_features.joblib` | 67.0% | Default params in `config.LGBM_PARAMS` |
| MLP | `mlp.joblib`, `mlp_scaler.joblib`, `mlp_features.joblib` | -- | PyTorch MLP (`ml/pytorch_mlp.py`); heavily weighted in the ensemble (~44%) despite not having its own published backtest row |
| Stacking | `stacking.joblib` | -- | Meta-model over base model predictions; trained separately via `--model stacking`, not part of the default "train all" run. Selectable in `predict.py`/`api.py` (`--model stacking`) but not in `predict_event.py`, which only supports `xgb`/`lr`/`rf`/`lgbm`/`ensemble` |
| Ensemble | `ensemble.joblib` | **69.2%** | Calibrated soft-vote; XGB+MLP-weighted (~40% XGB, ~44% MLP) -- **this is the model actually used for predictions** |

Honest out-of-sample backtest (2025-2026, 678 fights): **69.2%** accuracy (ensemble) as of the 2026-06-16 retrain; a later retrain on 2026-06-30 (7,323 fights) measured 69.6%. Monthly automated retrains (`monthly-refresh.yml`) may have moved this further -- check `git log --oneline | grep retrain` for the actual latest figure rather than trusting a pinned number here. Naive Red baseline ~55%.

Note: `--from-year 2022` backtest numbers (82-90%) are inflated because 2022-2024 fights fall inside the training window with the 80/20 split. Always use `--from-year 2025` for honest evaluation.

**Ensemble weight stability**: As of the 2026-06-16 retrain (91 features), XGB and MLP dominate ensemble weight (~40%/44%); LR dropped to ~15%. Params in `config.py` are the default values for this 91-feature set. Using `--tune` re-tunes base model hyperparameters via Optuna -- this has consistently hurt 2025+ accuracy (overfits to CV folds). Tested on the 86-feature set: tuning gave 68.6% vs 69.0% untuned. Use default params.

**Production tier (`models_v1_prod/`)**: A second, parallel set of the same model artifacts, trained on **100% of the data** (no train/test split) via `train_v1_models.py --prod`. This is what `predict.py` and `predict_event.py` actually load for live predictions -- the `models_v1/` eval tier exists solely for honest backtesting/tuning, since a model trained on all data can't be honestly evaluated on a held-out set. The prod ensemble borrows its per-model weights from the eval ensemble (to avoid in-sample leakage) and its calibrators are fit on eval-model predictions on the last 20% of data. `predict.py`/`predict_event.py` auto-select `models_v1_prod/` when it exists and is non-empty, falling back to `models_v1/` otherwise.

**Finish-type model**: `ml/finish_type_model.py` trains a separate 3-class classifier (Decision/KO-TKO/Submission) on the same v1 feature CSV, used as a soft signal in `predict.py`'s "Finish:" output. It is a standalone script, not part of `train_v1_models.py`'s "train all" run -- `monthly-refresh.yml` runs it explicitly (`python ml/finish_type_model.py` for the eval tier, `--prod` for the production tier) alongside `train_v1_models.py`. Saves `finish_type.joblib`/`finish_type_features.joblib` to `models_v1/` or `models_v1_prod/` respectively; `predict.py` loads it from whichever tier directory it resolved (`.exists()`-gated, fails soft if missing).

### Prediction (`predict.py` and `api.py`)

`predict.py` is the CLI; `api.py` is the FastAPI wrapper around the same logic. Both default to v1 (`DB_V1_PATH` for name resolution) and auto-select `MODELS_V1_PROD_DIR` over `MODELS_V1_DIR` when the prod tier exists (see "Production tier" above).

Career stats are **not** read from a stale mdabbert snapshot row -- they're recomputed live from the UFCStats DB at prediction time, closing the "one-fight-behind" staleness gap the mdabbert DB otherwise has for a fighter's most recent bout:

1. Resolve fighter names (fuzzy LIKE search against `fighters` table)
2. Recompute career-aggregate stats live from the UFCStats DB (`total_rounds_fought`, `longest_win_streak`, `win_by_dec_split`/`win_by_dec_unanimous` as exclusive buckets matching training, etc.) via `compute_live_career_stats()`. This also reconstructs each fighter's total career minutes fought (needed for `splm`/`sapm`/`td_avg`) by taking the most recent fight's `total_fight_time` (cumulative time *before* that fight) and adding that fight's own duration (`match_time_sec + (finish_round-1)*300`, same formula as `rolling.py`) -- it must NOT re-sum `total_fight_time` across rows, since each value is already a running total (see invariant below). It also pulls static `height`/`reach` from the `fighters` table so `height_diff`/`reach_diff` aren't silently zero.
3. Compute current ELO and Glicko-2 by replaying fight history from the UFCStats DB (`get_current_ratings_by_division`, `get_current_glicko_by_division`) -- both rating systems live in the UFCStats DB now, independent of the training-time computation frozen in `ufc-master.csv`
4. Compute recent form
5. Fetch `sapm`/`str_def`/`td_def` from the UFCStats DB (`_get_v2_defensive_stats`) -- live lookup, same as training-time enrichment
6. Read `weightclass_rank` live from `rankings_history.csv` instead of a stale mdabbert value
7. Build the feature vector via `build_feature_vector()` -- must match the feature names the model was trained on
8. Optionally compute value bets via `odds.py` if American moneyline odds are supplied

**`build_feature_vector()`'s generic `<stat>_diff` handler falls back to `red_stats.get(<stat>, 0)`/`blue_stats.get(<stat>, 0)` for any feature not explicitly handled** -- a real, non-excluded, trained feature will silently compute as `0 - 0 = 0` for every prediction if the live-stats dict (`compute_live_career_stats()`'s return value) doesn't include that key, with no error or warning. This is how `height_diff`/`reach_diff` went dead (fixed 2026-07-24) and is the general failure mode to check for whenever a new feature is added to training but not to `compute_live_career_stats()`.

No Contest fights (`winner_id IS NULL`) are excluded from win/loss/streak calculations but still count toward stat averages.

### Event predictions (`scripts/predict_event.py`)

Scrapes the next upcoming UFC event from UFCStats (requires Playwright), runs v1 predictions for every non-debut fight, and writes a Markdown file to `predictions/`.

- Debut check uses the UFCStats DB (fighter IDs from the scraper match UFCStats hex IDs)
- Fighter name normalisation also uses UFCStats DB, then names are passed to v1 prediction by fuzzy search
- Output format: single-model table
- Predictions are stored in `predictions/<slug>/<slug>.md` (+ matching `.json`/`.html`) and tracked in git. `_event_slug()` generates `<date>-<name>` (e.g. `2026-07-25-ufc-fight-night-ankalaev-vs-guskov`), date first -- GitHub's web/mobile UI has no configurable sort order and always lists folders alphabetically, so date-first is what makes that alphabetical order match chronological order. Older event folders predating this convention were one-time renamed to match; don't rename them back.

### Scoring & odds (`scripts/score_event.py`, `scrapers/bestfightodds.py`)

Run by `monday-results.yml` after a weekend event to grade the prior week's predictions against actual UFCStats results and bestfightodds.com (BFO) closing odds, then updates the prediction markdown in place (adds an Odds column, an Actual Result / Correct? column, and a Post-Event Summary + P/L section).

- `MIN_CONFIDENCE` (default `0.55`, override with `--min-confidence`) gates which fights count toward the "high-confidence" accuracy breakout and the P/L simulation (EUR 1 flat per pick); fights below the threshold still get a result recorded but aren't staked.
- **BFO event matching is date-based, not title-based**: BFO names events by location (e.g. `"UFC Oklahoma"`) while this repo names them by main-card fighters (e.g. `"UFC Fight Night: Du Plessis vs. Usman"`), so fuzzy-matching titles doesn't work. `scrape_bfo_odds()` matches BFO events to the prediction's `event_date` (+/-1 day) instead, searching both `_fetch_bfo_events()` (BFO homepage -- upcoming events only) and `_fetch_bfo_archive_events()` (BFO `/archive` -- completed events, what `score_event.py` actually needs since it always runs after the event). When a date has multiple candidate events (other orgs fight the same weekend), it disambiguates by scraping each candidate and keeping the one whose card actually contains the predicted fighters, not by fuzzy-matching event names.
- **Per-fighter odds come from a single consistent sportsbook column**, not independently "best price" per side -- BFO's per-book columns include prediction-market exchanges (Polymarket, Kalshi) alongside traditional sportsbooks, and picking each side's individually-best price mixes books, producing incoherent pairs (e.g. -194 / +1460 instead of a normal-vig -194 / +186).
- Fighter-name lookups fall back to fuzzy matching (`_fuzzy_odds_lookup`, floor 0.75/side) when the exact `_name_key` match fails -- BFO and UFCStats disagree on hyphenation (`Saint-Denis` vs `Saint Denis`), suffixes (`Kai Kamaka III`), and nicknames (`Zach` vs `Zachary`).
- BFO's HTML has changed at least once without notice (a `table.event-table` selector silently returned zero rows for months); if odds start coming back empty again, re-inspect the live page structure before assuming the matching logic is at fault.

### Central config (`config.py`)

Single source of truth for all paths, constants, and hyperparameters. Always import from here -- never hardcode paths. Key constants:

- `DB_PATH` / `DB_UFCSTATS_PATH` -- UFCStats rolling DB (raw per-fight data)
- `DB_V1_PATH` -- mdabbert career-aggregate DB (primary prediction source)
- `MODELS_V1_DIR` -- v1 eval-tier model artifacts directory (backtesting/tuning)
- `MODELS_V1_PROD_DIR` -- v1 production-tier model artifacts directory (trained on 100% of data; what `predict.py`/`predict_event.py` actually load)
- `CSV_V1_WITH_ELO` -- v1 feature CSV (`ml/ufc_ml_data_v1.csv`)
- `XGB_PARAMS`, `LR_PARAMS`, `RF_PARAMS`, `LGBM_PARAMS` -- hyperparameters; do not re-tune without a solid reason
- `MIN_FIGHT_DATE` -- training data cutoff (currently `"2018-01-01"`; see issue #47)
- `SAMPLE_WEIGHT_ALPHA` / `SAMPLE_WEIGHT_BETA` -- power-law decay weight: `exp(-alpha * delta^beta)`; `alpha=0.01, beta=1.5` in production (issue #54)
- `EXCLUDE_STAT_KEYWORDS` -- columns excluded from the diff feature loop
- `EXCLUDED_FEATURES` -- features dropped at training and inference time; currently removes 3 dead columns (`date_diff`, `outcome_diff`, `age_diff`) plus `glicko_diff` and `glicko_rd_diff` (near-zero target correlation; removing them improved ensemble accuracy by +0.3pp)
- `STARTING_ELO`, `GLICKO_START_R`, `GLICKO_START_RD`, `GLICKO_START_SIGMA`, `GLICKO_TAU` -- rating system constants
- `DIVISIONS`, `FINISH_METHOD_MAP`, `TRAIN_TEST_SPLIT`
- `NAME_ALIASES` -- maps alternate/historical fighter names (lowercase) to canonical UFCStats names. Covers Kaggle CSV typos, married name changes, nickname vs legal name, and transliteration differences. Keys must be lowercase; values must match the `fighters` table. Also duplicated locally in each enrichment script that needs it (`add_defensive_stats_to_csv.py`, `add_rankings_to_csv.py`, `append_new_fights.py`, etc.).

### Encoding note (Windows)

All Python source files must use ASCII-safe characters only. The Windows cp1252 console rejects Unicode symbols (->, --, -, #). Use `->`, `--`, `-`, `#` instead.

---

## Before committing

Files that must never be committed:

- `**/__pycache__/` and `*.pyc` -- Python bytecode
- `db/ufc_ufcstats.db` -- UFCStats SQLite DB (regenerate with `scripts/scrape_history.py`, or download the `data-artifacts-latest` GitHub Release for a current copy)
- `db/ufc_v2.db` -- mdabbert SQLite DB (regenerate from mdabbert CSV, or download the same release)
- `db/*_backup_*.db` -- rolling.py backup files
- `logs/` -- runtime logs
- `ml/*.csv` -- intermediate ML datasets
- `raw_data/*.db` -- raw database files
- `models_v1/*.joblib`, `models_v1_prod/*.joblib` -- trained model artifacts (distributed via the `data-artifacts-latest` release as tarballs, see "Model artifact distribution" above)

Files that ARE tracked:

- `raw_data/ufc-master.csv` -- source of truth for v1 pipeline; now 230 columns including all pre-computed features
- `predictions/*.md` -- event prediction files

If any excluded files were previously committed, untrack them with `git rm --cached <file>` (without deleting the local copy), then verify `.gitignore` covers them before staging the commit.

---

## Key invariants

- **No leakage**: every rolling stat uses `shift(1)`. ELO is computed *before* the fight is processed. Recent form excludes the current fight. Violating this inflates accuracy.
- **Corner assignment**: Red/Blue corners come from the fight detail page (`div.b-fight-details__person`), NOT the event listing (which puts the winner first). Getting this wrong causes ~100% Red win rate in training data.
- **Symmetry augmentation**: during training, each fight is duplicated with corners swapped and target flipped. This is applied *per fold* to the training split only -- never to the validation/test split.
- **SQLite TEXT affinity**: some numeric columns (e.g. `kd`, `ctrl`) are stored as TEXT. Always use `CAST(col AS REAL)` in numeric comparisons or aggregations.
- **Ensemble weights**: Optuna uses 5 independent restarts (100 trials each) on the first half of the test set; the reported hold-out accuracy is on the second half. XGB and MLP dominate on the 91-feature set (~40%/44%); weights shift between retrains -- do not hardcode them. Do not force balanced weights -- that hurts accuracy.
- **Do not use --tune for routine retraining**: `--tune` re-optimises base model hyperparameters via Optuna. This has consistently produced worse 2025+ accuracy than the default params in `config.py` (overfits to CV folds). Only use `--tune` if deliberately re-tuning after a major feature change, and always backtest before committing.
- **Backtest year for v1**: Use `--from-year 2025` for honest evaluation. The 80/20 split puts 2022-2024 fights inside the training window; `--from-year 2022` numbers are inflated.
- **v1 DB does not auto-update per event**: `monthly-refresh.yml` runs the full chain once a month; there's no per-event trigger. To update sooner, run the scraper, then the three CSV enrichment scripts, then `db/ingest_mdabbert.py`, then `ml/ML_data_preparation_v1.py`. Always backtest before committing retrained models. After retraining, also run `train_v1_models.py --prod` to update `models_v1_prod/` -- `predict.py`/`predict_event.py` load the prod tier, not the eval tier, so an eval-only retrain doesn't change live predictions.
- **BFO odds matching is date-based, not title-based**: bestfightodds.com names events by location (`"UFC Oklahoma"`), not by fighters like this repo does -- fuzzy-matching event titles silently fails every time. Match on event date (checking both the BFO homepage and `/archive`, since `score_event.py` always runs after the event has already dropped off the homepage) and disambiguate same-day collisions by checking which candidate's card actually contains the fighters, not by title similarity. See "Scoring & odds" above.
- **Scheduled GitHub Actions runs can be delayed hours, not minutes**: if a manual `workflow_dispatch` run and a delayed `schedule` run overlap, the second one to `git push` fails with a non-fast-forward error even though its job otherwise succeeded. A red X on a scheduled run doesn't necessarily mean the automation is broken -- check the job log before assuming so.
- **Sync script accuracy caveat**: `sync_v1_from_v2.py` produces correct per-minute `splm` from UFCStats rolling stats, while the Kaggle-sourced `ufc-master.csv` has a different `splm` scale for early-career fighters that happens to be more discriminative. After running sync, always backtest with `--from-year 2025` before committing. The current `ufc_v2.db` and `models_v1/` artifacts use the Kaggle-sourced pipeline (69.2% accuracy).
- **Model performance ceiling**: v1 career-average models are in the high-60s%/low-70s% range on the 2025+ backtest (91 features, default params) -- see "v1 Models" above for the last-known figure and how to check the current one. The naive "always pick Red" baseline is ~55% on recent data. There is no maintained results-history file -- run `python scripts/backtest_v1.py --from-year 2025` for the current number, or `git log --oneline | grep retrain` for iteration history.
- **Global ELO for training**: `build_elo_features()` uses a single universal ELO per fighter (not per division) to avoid cold-start when fighters change weight class. Prediction inference (`predict.py`) still calls `get_current_ratings_by_division()` for per-division ratings, which is a minor inconsistency to be aware of.
- **CSV is the source of truth**: `raw_data/ufc-master.csv` is enriched with all features before ingestion. Never compute ELO, Glicko, SOS, slopes, style matchup, or division one-hots inside `ML_data_preparation_v1.py` -- those must come from the CSV/DB. `ML_data_preparation_v1.py` is a pure diff-builder.
- **ELO train/inference split -- changing `ELO_calculator.py` requires a full pipeline re-run**: Training-time ELO is frozen in `raw_data/ufc-master.csv` (written by `add_computed_features_to_csv.py`). Inference-time ELO is computed live from the UFCStats DB by `api.py` and `predict.py`. If you change the ELO formula, inference immediately uses the new formula but the models were trained on the old one -- a silent train/inference skew. After any ELO formula change, always re-run: `add_computed_features_to_csv.py` -> `db/ingest_mdabbert.py` -> `ml/ML_data_preparation_v1.py` -> retrain and backtest before committing.
- **`fight_stats.total_fight_time` is cumulative pre-fight time, not a per-fight duration**: `rolling.py` writes it as seconds fought *before* each fight (0 on debut). Never re-sum it across rows (that double-cumulates); to get one fight's own duration use `match_time_sec + (finish_round-1)*300` via the `fights` table. `predict.py`'s `compute_live_career_stats()`/`compute_ewma_stats_single()` follow this correctly -- if you touch either, keep it that way.
  - **A code fix to how this column is written doesn't retroactively fix already-stored rows** -- it only takes effect the next time `rolling.py` runs a full rebuild against the DB. A merged fix and an actually-rebuilt shared DB are two different facts; don't assume the second from the first. **Detect staleness**: `SELECT MAX(total_fight_time) FROM fight_stats` -- capped at ~1500 (5 rounds x 300s, the max single-fight duration) across the whole table means it was never rebuilt post-fix. **Fix**: `python -c "from db.rolling import main; from config import DB_UFCSTATS_PATH; main(db_path=DB_UFCSTATS_PATH)"`, confirm `MAX(total_fight_time)` now reaches into the thousands, then `gh release upload data-artifacts-latest db/ufc_ufcstats.db --clobber`. Note `rolling.py`'s own `splm`/`sapm`/`td_avg` math is self-correcting regardless (it recomputes cumulative time in-memory from raw `match_time_sec`/`finish_round`, per `_compute_time_dependent()`) -- only the *persisted* column, and anything reading it directly (predict.py's live path), can go stale.
  - **A local `db/*.db` can silently diverge from the release asset even with matching size/timestamp** -- prior local testing/scraping can mutate it in place. Before treating a local repro as representative of CI (or vice versa), compare `sha256sum` against the release asset's digest (`gh release view data-artifacts-latest --json assets`).
- **`requirements.txt` pins the ML stack (scikit-learn/xgboost/lightgbm/torch/numpy/pandas) to exact versions, not `>=`** -- they must match whatever versions `models_v1(_prod)/*.joblib` were trained/pickled with, or a version drift silently corrupts calibrated ensemble probabilities on load (each calibrated component logs `InconsistentVersionWarning` when this happens -- that warning is a real bug signal here, not noise to suppress). After any retrain that bumps a library version locally, update these pins to match and re-verify predictions look sane in CI.
- **Live-inference feature parity is not enforced anywhere -- a feature can go silently dead**: `build_feature_vector()`'s fallback for any `<stat>_diff` not explicitly handled is `red_stats.get(<stat>, 0) - blue_stats.get(<stat>, 0)`. If the live-stats source (`compute_live_career_stats()` or another `extra_r`/`extra_b` provider) never populates that key, the feature silently computes as `0` for every prediction -- no error, no warning, and it can still carry real importance in the trained model (an entire audit-worth of features, including a top-2-importance one, went dead this way before being caught). Whenever you add a new trained feature, verify it's actually populated at inference -- don't assume the fallback will fail loudly, because it won't.
