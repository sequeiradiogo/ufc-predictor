# Betting Simulation Results

Model: XGBoost | Dataset: mdabbert (2010-03-21 to 2026-03-28) | Run: 2026-05-28

---

## Model Accuracy (all fights, no odds required)

| Metric | Value |
|--------|-------|
| Overall accuracy | 67.68% (3,946 / 5,830 fights) |
| Mean confidence | 62.2% |
| Brier score | 0.2074 |
| Naive baseline (always pick Red) | 57.14% |
| Model edge over naive | +10.55% |

### Year-by-year accuracy

| Year | Fights | Correct | Accuracy | Avg Confidence |
|------|--------|---------|----------|----------------|
| 2010 | 159 | 104 | 65.4% | 61.8% |
| 2011 | 201 | 132 | 65.7% | 61.5% |
| 2012 | 250 | 168 | 67.2% | 60.6% |
| 2013 | 269 | 185 | 68.8% | 61.6% |
| 2014 | 330 | 216 | 65.5% | 61.2% |
| 2015 | 383 | 258 | 67.4% | 61.7% |
| 2016 | 381 | 262 | 68.8% | 60.9% |
| 2017 | 338 | 231 | 68.3% | 61.2% |
| 2018 | 369 | 259 | 70.2% | 60.8% |
| 2019 | 387 | 255 | 65.9% | 61.2% |
| 2020 | 355 | 243 | 68.5% | 63.3% |
| 2021 | 442 | 310 | 70.1% | 62.5% |
| 2022 | 466 | 328 | 70.4% | 62.9% |
| 2023 | 444 | 313 | 70.5% | 63.9% |
| 2024 | 478 | 291 | 60.9% | 62.9% |
| 2025 | 468 | 310 | 66.2% | 63.8% |
| 2026 | 110 | 81 | 73.6% | 66.7% |

### Accuracy by division (min 30 fights)

| Division | Fights | Accuracy |
|----------|--------|----------|
| Featherweight | 667 | 70.2% |
| Lightweight | 942 | 69.4% |
| Bantamweight | 600 | 68.7% |
| Women's Bantamweight | 174 | 68.4% |
| Heavyweight | 439 | 67.9% |
| Flyweight | 331 | 67.7% |
| Middleweight | 727 | 67.4% |
| Welterweight | 890 | 67.1% |
| Women's Strawweight | 294 | 66.0% |
| Light Heavyweight | 467 | 64.7% |
| Women's Flyweight | 227 | 63.0% |
| Catch Weight | 52 | 59.6% |

---

## Betting Simulations

Odds coverage: **5,603 / 5,830 fights (96%)** — American moneyline odds from mdabbert dataset.

---

### Section 1 — Bet 1 unit on every model pick

Bet 1 unit on whichever fighter the model predicts to win, for every fight where odds are available. No edge filter.

| Metric | Value |
|--------|-------|
| Fights | 5,603 |
| Wins | 3,804 (67.9%) |
| Total P&L | **+849.47 units** |
| ROI | **+15.2% per unit staked** |
| Outcome | PROFIT |

---

### Section 2 — Value bets only (edge >= 3%)

Only bet when the model's implied probability exceeds the vig-stripped market probability by at least 3 percentage points. Stake sizing: 1 unit flat OR quarter-Kelly on a 1,000-unit bankroll.

| Metric | Value |
|--------|-------|
| Bets placed | 4,723 (84% of odds-covered fights) |
| Win rate | 48.0% (2,267 / 4,723) |
| Avg edge | +13.9% |
| Avg decimal odds | 2.92x |
| Flat ROI | **+14.5% per unit staked** |
| Kelly ROI | **+7,915.8%** (on 1,000-unit bankroll, quarter-Kelly) |

### Year-by-year value bet breakdown

| Year | Bets | Wins | Win% | Flat ROI |
|------|------|------|------|----------|
| 2010 | 130 | 65 | 50.0% | +30.5% |
| 2011 | 167 | 62 | 37.1% | -2.3% |
| 2012 | 219 | 101 | 46.1% | +16.6% |
| 2013 | 227 | 91 | 40.1% | +5.3% |
| 2014 | 296 | 111 | 37.5% | +10.0% |
| 2015 | 340 | 168 | 49.4% | +28.6% |
| 2016 | 304 | 161 | 53.0% | +24.3% |
| 2017 | 289 | 140 | 48.4% | +22.9% |
| 2018 | 312 | 144 | 46.2% | +7.3% |
| 2019 | 322 | 168 | 52.2% | +18.9% |
| 2020 | 297 | 169 | 56.9% | +23.0% |
| 2021 | 369 | 221 | 59.9% | +31.1% |
| 2022 | 376 | 206 | 54.8% | +22.1% |
| 2023 | 301 | 159 | 52.8% | +15.2% |
| 2024 | 301 | 112 | 37.2% | -12.3% |
| 2025 | 377 | 155 | 41.1% | +0.9% |
| 2026 | 96 | 34 | 35.4% | -31.4% |

---

## Notes

- **In-sample caveat**: these simulations score the model against fights it was trained on (80% train / 20% test split). For honest out-of-sample ROI, run `python backtest.py --odds --from-year 2022`.
- **2024–2026 decline**: value bet win rate dropped sharply (37–41%), suggesting the betting market has grown more efficient or the model has drifted on recent fight styles. Retraining with newer data is the primary lever.
- **Kelly ROI is compounding**: the quarter-Kelly figure assumes profits are reinvested each fight — it is not comparable to the flat ROI figure.
- Section 1 win rate (67.9%) tracks model accuracy closely, as expected — it is a direct reflection of prediction quality, not market inefficiency.
