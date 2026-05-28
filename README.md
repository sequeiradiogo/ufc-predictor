# UFC Fight Predictor 🥊

A machine-learning system that predicts UFC fight outcomes using historical fighter statistics, rolling performance metrics, and ELO ratings.

## How it works

```
Raw Data (UFC.csv)
    ↓
SQLite Database      (3 normalised tables: fighters, fights, fight_stats)
    ↓
Rolling Stats        (cumulative per-fighter stats before each fight)
    ↓
ELO Ratings          (dynamic skill ratings updated after every fight)
    ↓
Feature Diffs        (Red − Blue for every stat)
    ↓
ML Models            (Logistic Regression + XGBoost)
    ↓
Predictions
```

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd ufc-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build database + train models in one command
#    (requires UFC.csv — download from Kaggle or ufcstats.com)
python run_pipeline.py --full --csv path/to/UFC.csv
```

If the database is already built and you only want to retrain the models:
```bash
python run_pipeline.py          # runs steps 4-6 (dataset → XGBoost → LR)
```

Run specific steps only:
```bash
python run_pipeline.py --steps 5        # XGBoost only
python run_pipeline.py --steps 5,6      # both models
python run_pipeline.py --dry-run        # preview without executing
```

---

## Making a Prediction

Once the models are trained, use the prediction CLI:

```bash
# XGBoost model (default)
python predict.py "Islam Makhachev" "Charles Oliveira"

# Logistic Regression
python predict.py "Conor McGregor" "Dustin Poirier" --model lr

# Partial names work — you'll be prompted if there are multiple matches
python predict.py "Jones" "Miocic"
```

**Example output:**
```
🥊  Islam Makhachev  (Red)  vs  Charles Oliveira  (Blue)
────────────────────────────────────────────────────
📊  ELO:  Islam Makhachev = 1659  |  Charles Oliveira = 1623

🏆  Predicted Winner: Islam Makhachev  (71.3% confidence)

    Islam Makhachev (Red)
    ████████████████░░░░ 71.3%

    Charles Oliveira (Blue)
    █████░░░░░░░░░░░░░░░ 28.7%

    Model: XGBoost
```

---

## Refreshing Data

When new UFC events happen, update and retrain with:
```bash
# From an updated UFC.csv export
python refresh_data.py --csv path/to/updated_UFC.csv

# Preview what would run
python refresh_data.py --csv path/to/updated_UFC.csv --dry-run
```

---

## Project Structure

```
ufc-predictor/
├── config.py                          ← All paths and constants (start here)
├── logger.py                          ← Shared logging setup
├── run_pipeline.py                    ← Pipeline orchestrator (run this)
├── refresh_data.py                    ← Update DB + retrain when new events drop
├── predict.py                         ← Prediction CLI
├── requirements.txt
├── README.md
├── IMPROVEMENTS.md                    ← Change log and roadmap
│
├── database_builder_files/
│   ├── raw_sql_database.py            ← Step 1: create DB from CSV
│   ├── keys.py                        ← Step 2: add foreign keys
│   ├── rolling.py                     ← Step 3: compute rolling stats
│   └── tests.py                       ← Basic DB validation queries
│
├── ML_models/
│   ├── ML_data_preparation.py         ← Step 4: build ML dataset from DB
│   ├── ELO_calculator.py              ← ELO rating engine
│   ├── XGBoost.py                     ← Step 5: XGBoost model
│   ├── logistic_regression.py         ← Step 6: Logistic Regression model
│   ├── check_elo.py                   ← ELO audit / visualisation
│   └── top_15_elo.py                  ← ELO leaderboard
│
├── models/                            ← Saved model artifacts (auto-created)
│   ├── xgboost.joblib
│   ├── xgb_features.joblib
│   ├── logistic_regression.joblib
│   ├── lr_scaler.joblib
│   └── lr_features.joblib
│
├── logs/                              ← Runtime logs (auto-created)
│   └── ufc_predictor.log
│
├── raw_data/                          ← Source CSV + intermediate databases
└── tests/
    └── test_pipeline.py               ← 27 unit + integration tests
```

---

## Models

| Model | Algorithm | Key Strength |
|-------|-----------|--------------|
| XGBoost | Gradient Boosting | Highest accuracy, handles non-linear relationships |
| Logistic Regression | Linear model | Interpretable feature weights, calibrated probabilities |

Both models are trained on **difference features** (Red stat − Blue stat), so the model sees only relative advantages rather than raw numbers. The dataset is also symmetrised (each fight appears from both corners' perspective) to remove the red-corner assignment bias.

---

## Key Features Used

- **Strike accuracy** — head, body, leg, clinch, distance, ground
- **Takedown stats** — accuracy, average per 15 min
- **Defense** — strike defense %, takedown defense %
- **Tempo** — significant strikes per minute, absorbed per minute
- **Experience** — total fight time, wins, losses
- **ELO rating** — dynamic skill rating computed from full fight history

---

## Running Tests

```bash
python -m pytest tests/ -v
```

22 tests pass immediately. 5 model tests activate automatically after the first training run.

---

## Data

The raw dataset comes from [UFC Stats](http://www.ufcstats.com/). The project currently includes:

- **2,611** unique fighters
- **8,337** fights
- **16,674** fight-level stat records (2 per fight)
