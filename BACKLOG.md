# UFC Predictor — Backlog

Each item is sized for a single branch. Ordered by impact within each section.

---

## Data

### `data/kaggle-update` — Refresh fight data past September 2025
Download the updated Kaggle dataset (mdabbert or equivalent) and rebuild the database.
```
python run_pipeline.py --full --csv path/to/new_UFC.csv
```
Acceptance: `database_builder_files/ufc_v2.db` contains fights up to the most recent UFC event.

### `data/odds-backfill` — Scrape and backfill historical betting odds
Scrape closing moneylines from BestFightOdds.com (no public API — BeautifulSoup/Playwright).
- Match fighter names + date to the `fights` table (fuzzy match — names often differ slightly)
- Populate `odds_red` / `odds_blue` columns (already exist via `python odds.py --migrate`)
- Acceptance: >80% of fights from 2015+ have non-null odds. Schema migration already done.
- Note: name normalisation is the hard part (e.g. "Conor McGregor" vs "C. McGregor")

### `data/incremental-refresh` — Wire up `refresh_data.py --auto`
The skeleton is in place; the actual insert + rolling stats recomputation is not.
- Scrape new events from ufcstats.com since the last DB event date
- Insert into `fights` / `fight_stats` / `fighters`
- Rerun `rolling.py` for affected fighter rows only (incremental, not full rebuild)

---

## Machine Learning

### `ml/hyperparameter-lr` — Tune Logistic Regression with Optuna
XGBoost already has `--tune`. Add the same for LR:
- Params to search: C (regularisation), solver, max_iter, class_weight
- Objective: mean CV accuracy via TimeSeriesSplit(5)
- Save best params to `config.py` (mirror the `XGB_PARAMS` pattern)

### `ml/more-models` — Add Random Forest and LightGBM
Implement two new classifiers following the same pattern as `XGBoost.py`:
- TimeSeriesSplit CV, symmetry augmentation, `joblib` save, `--tune` flag
- Add to `run_pipeline.py` as steps 8 and 9
- Add model options to `predict.py` (`--model rf`, `--model lgbm`)
- Acceptance: both beat the LR baseline (63.3%) on hold-out test set

### `ml/ensemble` — Weighted probability ensemble
Combine XGBoost + LR (+ RF/LightGBM if available) via a soft-vote ensemble.
- Weight by hold-out accuracy (or tune weights with Optuna)
- Save as `models/ensemble.joblib`
- Add `--model ensemble` to `predict.py`

### `ml/experiment-tracking` — Add MLflow logging
Log params, CV metrics, test accuracy, and artifact paths for every training run.
- `mlflow.log_params(params)` + `mlflow.log_metric("test_accuracy", acc)`
- Saves across runs so model versions can be compared without reading MODEL_RESULTS.md manually
- `mlflow ui` to browse runs locally

---

## Betting Odds

> Requires `data/odds-backfill` to have meaningful data.

### `odds/backtest` — Historical value-bet backtest
Simulate ROI across all past fights where odds are populated:
- For each fight: compare model probability vs odds-implied fair probability
- Report: total bets placed, win rate, ROI, Kelly-staked ROI
- Add `python backtest.py --odds` flag to `backtest.py`

---

## API / Deployment

### `api/card-endpoint` — Upcoming card prediction endpoint
`GET /card` — scrape or accept a list of matchups and return predictions for a full event.
```json
[
  {"red_fighter": "Islam Makhachev", "blue_fighter": "Charles Oliveira"},
  ...
]
```
Could pull from a static upcoming-card JSON or accept a body list of matchups.

### `api/deploy-fly` — Deploy API to Fly.io
Containerise with Docker and deploy `uvicorn api:app` to Fly.io (free tier).
- Dockerfile, fly.toml
- DB baked into image or mounted as a volume
- Acceptance: `curl https://<app>.fly.dev/` returns `{"status": "ok"}`

---

## Infrastructure

### `infra/gitignore-lfs` — Move large files to Git LFS or exclude from history
`raw_data/UFC_with_rolling.csv` and `database_builder_files/ufc_v2.db` should not be in git history.
- Set up Git LFS for `*.csv` and `*.db` OR add to `.gitignore` and document download steps in README
- Rewrite history if already committed (requires force push — coordinate with collaborators)

### `infra/ci` — Add GitHub Actions CI
Run `pytest tests/ -v` on every push to main and on PRs.
- Matrix: Python 3.12
- Cache pip dependencies
- Gate: PR cannot merge if tests fail
