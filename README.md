# UFC Fight Predictor

A machine-learning system that predicts UFC fight outcomes with **69.6% accuracy** on honest out-of-sample data (2025-2026, 704 fights). Predictions are generated weekly via GitHub Actions and published to [`predictions/`](./predictions/).

---

## Results

| Period | Fights | Accuracy |
|--------|--------|----------|
| 2025 | 475 | 70.3% |
| 2026 | 229 | 68.1% |
| **Total (out-of-sample)** | **704** | **69.6%** |

Naive "always pick Red corner" baseline: ~55%. Full history in [MODEL_RESULTS.md](./MODEL_RESULTS.md).

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
              v1 Feature CSV (91 diff features, Red - Blue)
                       |
    XGBoost / LR / RF / LightGBM / MLP  -->  Ensemble (69.6%)
                       |
                  Predictions
```

The ensemble is a calibrated soft-vote over five base models. XGBoost and MLP carry the most weight (~40% and ~44% respectively). All features are pre-fight snapshots -- no leakage.

---

## Setup

**Clone and install:**
```bash
git clone https://github.com/sequeiradiogo/ufc-predictor.git
cd ufc-predictor
pip install -r requirements.txt
```

**Model artifacts are tracked in git**, so predictions work immediately after cloning -- no training required.

**Databases are distributed as GitHub Release assets** (too large to commit). Download them before running predictions:
```bash
gh release download data-artifacts-latest --dir db/ --pattern "*.db"
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

Endpoints: `POST /predict`, `GET /fighters`, `GET /health`. Full schema at `/docs` (Swagger UI) once the server is running.

---

## Keeping data current

After each UFC event, sync the database and retrain:
```bash
# 1. Scrape new fights and rebuild the UFCStats DB
python scripts/refresh_data.py --auto

# 2. Download the latest rankings (gitignored, lives in the GitHub Release)
gh release download data-artifacts-latest --dir raw_data/ --pattern "rankings_history.csv"

# 3. Enrich the v1 CSV
python scripts/add_defensive_stats_to_csv.py
python scripts/add_rankings_to_csv.py
python scripts/add_computed_features_to_csv.py

# 4. Rebuild the mdabbert DB and v1 feature dataset
python db/ingest_mdabbert.py
python ml/ML_data_preparation_v1.py

# 5. Retrain v1 models
python ml/train_v1_models.py

# 6. Backtest before committing
python scripts/backtest_v1.py --from-year 2025
```

Or use the automated GitHub Actions workflow (runs on the 1st of each month).

---

## Automation

Three GitHub Actions workflows keep the system running without manual intervention:

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `weekly-predictions.yml` | Every Friday | Scrapes the upcoming event card, generates predictions, commits to `predictions/` |
| `monday-results.yml` | Every Monday | Scrapes the weekend results, updates the prediction markdown with accuracy + P/L |
| `monthly-refresh.yml` | 1st of month | Scrapes new fights, retrains all v1 models, updates DB release assets |

---

## Project structure

```
ufc-predictor/
├── predict.py                    -- Prediction CLI (start here)
├── api.py                        -- FastAPI REST wrapper
├── config.py                     -- All paths and constants
├── run_pipeline.py               -- v2 pipeline orchestrator
│
├── scripts/
│   ├── predict_event.py          -- Predict a full upcoming event card
│   ├── score_event.py            -- Score predictions against actual results
│   ├── refresh_data.py           -- Scrape new fights and rebuild the DB
│   ├── backtest_v1.py            -- Out-of-sample accuracy evaluation
│   ├── add_defensive_stats_to_csv.py
│   ├── add_rankings_to_csv.py
│   └── add_computed_features_to_csv.py
│
├── ml/
│   ├── train_v1_models.py        -- Train all v1 models (XGB/LR/RF/LightGBM/MLP/ensemble)
│   ├── ML_data_preparation_v1.py -- Build v1 feature CSV from mdabbert DB
│   ├── ELO_calculator.py         -- ELO + Glicko-2 rating engine
│   └── ...                       -- Individual model scripts
│
├── db/
│   ├── ingest_mdabbert.py        -- Rebuild mdabbert DB from enriched CSV
│   └── rolling.py                -- Compute rolling stats (leakage-safe shift)
│
├── scrapers/
│   ├── ufcstats.py               -- Fight results + stats from ufcstats.com
│   └── bestfightodds.py          -- Closing moneyline odds
│
├── models_v1/                    -- v1 model artifacts (tracked in git)
├── models_v1_prod/               -- Production models (100% training data)
├── predictions/                  -- Weekly prediction outputs (MD + HTML)
├── raw_data/ufc-master.csv       -- Source CSV (2,271 fighters, 7,323 fights)
│
├── MODEL_RESULTS.md              -- Full model iteration history
└── tests/                        -- Unit + integration tests
```

---

## Models

| Model | 2025+ Accuracy | Notes |
|-------|---------------|-------|
| XGBoost | 67.5% | Default params; tuning consistently hurts |
| Logistic Regression | 65.3% | Platt-calibrated |
| Random Forest | 67.0% | |
| LightGBM | 67.0% | |
| **Ensemble** | **69.6%** | Calibrated soft-vote; XGB ~40%, MLP ~44% |

Training data: post-2018 fights only (adversarial validation confirmed distribution shift pre-2018). Evaluated on `--from-year 2025` to avoid the 80/20 split leaking 2022-2024 fights into the test window.

---

## Data

- 2,271 fighters
- 7,323 fights
- 91 engineered features per matchup

The pipeline is fully reproducible -- rebuild the DB from scratch with `scripts/scrape_history.py` (takes 8-10 hours) or download the current snapshot from the GitHub Release.

### Credits

- **[mdabbert](https://www.kaggle.com/mdabbert)** -- the career-aggregate UFC dataset (`ufc-master.csv`) that powers the v1 prediction pipeline. The feature engineering and enrichment pipeline in this repo builds directly on top of his dataset.
- **[martj42](https://www.kaggle.com/datasets/martj42/ufc-rankings)** -- the UFC rankings dataset used to enrich fights with pre-fight weightclass rankings.
- **[UFCStats.com](http://www.ufcstats.com/)** -- primary source for per-fight statistics, scraped to build the UFCStats DB.

---

## License

MIT -- see [LICENSE](./LICENSE).
