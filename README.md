# UFC Fight Predictor

A machine-learning system that predicts UFC fight outcomes with **65.7% accuracy** on honest out-of-sample data (2025-2026, 752 fights). Predictions are generated weekly via GitHub Actions and published to [`predictions/`](./predictions/).

---

## Results

**Backtest** (`python scripts/backtest_v1.py --from-year 2025`, ensemble, 2026-10-21 retrain):

| Period | Fights | Accuracy |
|--------|--------|----------|
| 2025 | 431 | 65.2% |
| 2026 | 321 | 66.4% |
| **Total (out-of-sample)** | **752** | **65.7%** |

**Live track record** (Jun-Oct 2026, 18 events, 166 fights with odds): model 56.0%, bookmaker favorite 71.7% on the same fights. The bookmaker favorite is the baseline that matters, and the model doesn't beat it yet. The naive "always pick Red corner" baseline is ~55%.

---

## How it works

```
UFCStats.com  -->  UFCStats DB (raw per-fight stats)
                       |
              CSV enrichment pipeline
              (ELO, Glicko-2, form, SOS, slopes, style)
                       |
               mdabbert DB (career averages + pre-computed features)
                       |
              v1 Feature CSV (69 diff features, Red - Blue)
                       |
    XGBoost / LR / RF / LightGBM / MLP  -->  Ensemble (65.7%)
                       |
                  Predictions
```

The ensemble is a calibrated soft-vote over five base models, followed by a single temperature scaling step. Weights are re-fit on every retrain and move around a lot. In the 2026-10-21 retrain Random Forest (~47%) and Logistic Regression (~41%) carry most of the weight, then MLP (~11%), with XGBoost and LightGBM under 1%. All features are pre-fight snapshots. `scripts/check_live_parity.py` checks that live inference builds the same features as training.

---

## Setup

**Clone and install:**
```bash
git clone https://github.com/sequeiradiogo/ufc-predictor.git
cd ufc-predictor
pip install -r requirements.txt
```

**Model artifacts and databases are distributed as GitHub Release assets** (too large to commit -- see [Repo size note](#repo-size-note)). Download and extract them before running predictions:
```bash
gh release download data-artifacts-latest --dir db/ --pattern "*.db"
gh release download data-artifacts-latest --pattern "models_v1_prod.tar.gz"
tar -xzf models_v1_prod.tar.gz
```

---

## Making a prediction

```bash
# Default: ensemble model
python predict.py "Islam Makhachev" "Charles Oliveira"

# Specific model
python predict.py "Jon Jones" "Stipe Miocic" --model xgb

# With division and title context
python predict.py "Jones" "Miocic" --model ensemble --division "heavyweight" --title
```

**Example output:**
```
Islam Makhachev  (Red)  vs  Charles Oliveira  (Blue)
------------------------------------------------------------
ELO:   Islam Makhachev 1624  |  Charles Oliveira 1598

Predicted winner: Islam Makhachev  (68.4% confidence)

  Islam Makhachev (Red)     ||||||||||||||||.... 68.4%
  Charles Oliveira (Blue)   |||||||............. 31.6%

Finish: Decision (54%) / Submission (27%) / KO-TKO (19%)
Model: Ensemble (Soft Vote)
```

---

## Predicting an upcoming event

Scrapes the next UFC event from UFCStats and generates predictions for every non-debut fight:
```bash
python scripts/predict_event.py
python scripts/predict_event.py --model ensemble
```

Output: a Markdown table + interactive HTML dashboard saved to `predictions/<event-slug>/`.

---

## REST API

```bash
uvicorn api:app --reload
```

Endpoints: `POST /predict`, `GET /fighters?q=`, `GET /` (health check). Full schema at `/docs` (Swagger UI) once the server is running.

---

## Keeping data current

After each UFC event, sync the database and retrain:
```bash
# 1. Scrape new fights into the UFCStats DB, then backfill round stats and rebuild rolling stats
python scripts/refresh_data.py --auto
python scripts/backfill_rounds.py
python -c "from db.rolling import main; from config import DB_UFCSTATS_PATH; main(db_path=DB_UFCSTATS_PATH)"
python scripts/check_db_health.py

# 2. Download the latest rankings (gitignored, lives in the GitHub Release)
gh release download data-artifacts-latest --dir raw_data/ --pattern "rankings_history.csv"

# 3. Enrich the v1 CSV (in this order)
python scripts/add_defensive_stats_to_csv.py
python scripts/add_rankings_to_csv.py
python scripts/rebuild_career_stats.py
python scripts/add_computed_features_to_csv.py

# 4. Rebuild the mdabbert DB and v1 feature dataset
python db/ingest_mdabbert.py
python ml/ML_data_preparation_v1.py

# 5. Retrain v1 models (eval tier, then production tier) and the finish-type model
python ml/train_v1_models.py
python ml/train_v1_models.py --prod
python ml/finish_type_model.py
python ml/finish_type_model.py --prod

# 6. Check before committing
python scripts/backtest_v1.py --from-year 2025
python scripts/check_live_parity.py --events 6
```

Or use the automated GitHub Actions workflow (runs on the 1st of each month).

---

## Automation

GitHub Actions workflows keep the system running without manual intervention:

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `weekly-predictions.yml` | Every Friday | Refreshes UFC.com rankings, checks DB health, predicts the upcoming card, commits to `predictions/` |
| `monday-results.yml` | Every Monday | Scores the weekend's predictions against results and closing odds (incl. the bookmaker-favorite baseline), updates accuracy and P/L |
| `monthly-refresh.yml` | 1st of month | Scrapes new fights, runs the full enrichment pipeline, retrains all models (eval + prod), gates on live-feature parity, uploads DBs and models to the release |
| `tests.yml` | Every PR + push to `main` | Runs the test suite (`pytest tests/ -v`) |

---

## Project structure

```
ufc-predictor/
├── predict.py                    -- Prediction CLI (start here)
├── api.py                        -- FastAPI REST wrapper
├── config.py                     -- All paths and constants
│
├── scripts/
│   ├── predict_event.py          -- Predict a full upcoming event card
│   ├── score_event.py            -- Score predictions against actual results + odds
│   ├── build_feed.py             -- Build predictions/feed.json (public feed for match-analyzer)
│   ├── refresh_data.py           -- Scrape new fights into the UFCStats DB
│   ├── backtest_v1.py            -- Out-of-sample accuracy evaluation
│   ├── check_db_health.py        -- Fail loudly on a DB that would corrupt live predictions
│   ├── check_live_parity.py      -- Compare live-inference features with training rows
│   ├── add_defensive_stats_to_csv.py
│   ├── add_rankings_to_csv.py
│   ├── rebuild_career_stats.py   -- Pre-fight career stats (same code path as inference)
│   └── add_computed_features_to_csv.py
│
├── ml/
│   ├── train_v1_models.py        -- Train all v1 models (XGB/LR/RF/LightGBM/MLP/ensemble)
│   ├── finish_type_model.py      -- Decision / KO-TKO / Submission classifier
│   ├── ML_data_preparation_v1.py -- Build v1 feature CSV from mdabbert DB
│   ├── ELO_calculator.py         -- ELO + Glicko-2 rating engine
│   ├── calibration.py            -- Ensemble temperature scaling
│   ├── opp_adjusted_defense.py   -- Opponent-adjusted striking defence
│   └── pytorch_mlp.py            -- PyTorch MLP
│
├── db/
│   ├── ingest_mdabbert.py        -- Rebuild mdabbert DB from enriched CSV
│   └── rolling.py                -- Compute rolling stats (leakage-safe shift)
│
├── scrapers/
│   ├── ufcstats.py               -- Fight results + stats from ufcstats.com
│   └── bestfightodds.py          -- Closing moneyline odds
│
├── models_v1/                    -- v1 eval-tier model artifacts (gitignored, download from Release)
├── models_v1_prod/               -- Production models (100% training data; gitignored, download from Release)
├── predictions/                  -- Weekly prediction outputs (MD + HTML)
├── raw_data/ufc-master.csv       -- Source CSV (2,330 fighters, 7,476 fights)
│
└── tests/                        -- Unit + integration tests
```

---

## Models

| Model | 2025+ Accuracy | Notes |
|-------|---------------|-------|
| XGBoost | 66.0% | Default params; tuning consistently hurts |
| Logistic Regression | 66.1% | Platt-calibrated |
| Random Forest | 66.8% | |
| LightGBM | 66.1% | |
| MLP | 65.0% | PyTorch |
| **Ensemble** | **65.7%** | Calibrated soft-vote + temperature; RF ~47%, LR ~41%, MLP ~11% |

Eval-tier models (`models_v1/`), 752 fights, 2026-10-21 retrain. On a single backtest run the individual models land within about a point of each other, so no one of them is clearly better. The ensemble is used for predictions because it gives better-calibrated probabilities (lower hold-out log-loss), not higher accuracy. Live predictions load the production tier (`models_v1_prod/`), which is trained on 100% of the data and borrows the eval ensemble's weights and temperature.

Training data: post-2018 fights only (adversarial validation confirmed distribution shift pre-2018). Evaluated with `--from-year 2025` because the chronological 80/20 split falls around Feb 2025, which puts 2024 and earlier in the training window.

---

## Data

- 2,330 fighters, 7,476 fights in `ufc-master.csv` (2010 - Sep 2026). The UFCStats DB has 8,911 fights back to 1994
- 3,496 fights in the model dataset (2018+, debuts excluded; includes the 20% test split)
- 69 engineered features per matchup for the win models. The finish-type model uses its own set of 91

The pipeline is fully reproducible -- rebuild the DB from scratch with `scripts/scrape_history.py` (takes 8-10 hours) or download the current snapshot from the GitHub Release.

### Credits

- **[mdabbert](https://www.kaggle.com/mdabbert)** -- the career-aggregate UFC dataset (`ufc-master.csv`) that powers the v1 prediction pipeline. The feature engineering and enrichment pipeline in this repo builds directly on top of his dataset.
- **[martj42](https://www.kaggle.com/datasets/martj42/ufc-rankings)** -- the UFC rankings dataset used to enrich fights with pre-fight weightclass rankings.
- **[UFCStats.com](http://www.ufcstats.com/)** -- primary source for per-fight statistics, scraped to build the UFCStats DB.

---

## Repo size note

Trained model artifacts (`.joblib`, up to ~27MB each) and SQLite databases are distributed via the `data-artifacts-latest` GitHub Release rather than committed to git -- binary artifacts don't diff, so committing a new copy on every retrain would bloat the repo's history indefinitely. `monthly-refresh.yml` uploads fresh artifacts to the release after each retrain instead of committing them.

## License

MIT -- see [LICENSE](./LICENSE).
