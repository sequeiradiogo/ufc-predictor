# Model Results History

---

## 2026-05-28 — mdabbert dataset (ufc-master.csv, data to 2026-03-28)

**Trigger:** Switched data source from original UFCStats hex-ID CSV (up to 2025-09-06) to mdabbert-format dataset (2010-03-21 to 2026-03-28). Wrote `ingest_mdabbert.py` adapter replacing pipeline steps 1-3.

**Dataset:** 5,830 fights x 42 columns (7,169 raw minus 1,339 debut fights excluded). Career-average stats (no per-fight raw strike counts). Starts 2010 (no pre-2010 data).

**Feature cleanup applied:**
- Dropped `avg_sig_str_landed_diff` (exact duplicate of `splm_diff`)
- Dropped `avg_td_landed_diff` (exact duplicate of `td_avg_diff`)
- Dropped `strike_ratio_diff` (r=-0.92 with `grapple_ratio_diff` for non-debutants)
- Dropped `total_fight_time_diff` (= `wins_diff + losses_diff` exactly)
- Dropped `is_debutant_diff` (always 0 after debutant exclusion)

**Exclusion filters:** debut fights (either fighter with 0 prior recorded fights) and pre-2005 fights.

### XGBoost
| Metric | Value |
|--------|-------|
| Test Accuracy | **64.67%** |
| CV Mean (5-fold TimeSeriesSplit) | 61.52% |
| CV Std | ± 3.36% |
| Train cutoff | up to 2023-06-03 |
| Test window | from 2023-06-03 (1,166 fights) |

### Logistic Regression
| Metric | Value |
|--------|-------|
| Test Accuracy | **62.95%** |
| CV Mean (5-fold TimeSeriesSplit) | 60.60% |
| CV Std | ± 2.27% |
| Brier Score (uncalibrated) | ~0.222 |
| Brier Score (calibrated, Platt) | ~0.222 |

### Finish Type Model (XGBoost, 3-class)
| Metric | Value |
|--------|-------|
| Test Accuracy | **53.61%** |
| CV Mean (5-fold TimeSeriesSplit) | 49.89% |
| CV Std | ± 3.16% |
| Majority-class baseline | ~51.5% (Decision) |

Class distribution in test set:
- Decision: 51.5%
- KO/TKO: 31.5%
- Submission: 17.0%

---

Baseline to compare against when retraining. Update this file after every significant retrain.

---

## 2026-05-26 — Feature Engineering v2

**Trigger:** Added 19 new features (recent form, age, style matchup, division one-hot, title fight flag)

**Dataset:** 8,190 fights × 77 columns (up from ~58)

### XGBoost
| Metric | Value |
|--------|-------|
| Test Accuracy | **65.87%** |
| CV Mean (5-fold TimeSeriesSplit) | 64.41% |
| CV Std | ± 1.30% |
| Train cutoff | up to 2022-06-25 |
| Test window | from 2022-06-25 |

**Top 10 features by importance:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | reach_diff | 0.0976 |
| 2 | elo_diff | 0.0460 |
| 3 | str_acc_diff | 0.0418 |
| 4 | age_diff | 0.0277 |
| 5 | grapple_ratio_diff | 0.0274 |
| 6 | str_def_diff | 0.0264 |
| 7 | head_atmpted_diff | 0.0257 |
| 8 | losses_diff | 0.0257 |
| 9 | strike_ratio_diff | 0.0249 |
| 10 | striker_vs_wrestler | 0.0204 |

### Logistic Regression
| Metric | Value |
|--------|-------|
| Test Accuracy | **63.31%** |
| CV Mean (5-fold TimeSeriesSplit) | 63.66% |
| CV Std | ± 1.98% |
| Brier Score (uncalibrated) | 0.2151 |
| Brier Score (calibrated, Platt) | 0.2189 |

### Finish Type Model (XGBoost, 3-class)
| Metric | Value |
|--------|-------|
| Test Accuracy | **50.12%** |
| CV Mean (5-fold TimeSeriesSplit) | 48.07% |
| CV Std | ± 2.56% |
| Majority-class baseline | ~50.5% (Decision) |

Class distribution in test set:
- Decision: 50.5%
- KO/TKO: 31.8%
- Submission: 17.7%

---

## Notes

- UFC fight prediction practical ceiling for stats-only models: **~65–68%** (matches expert human analysts)
- Red corner wins ~64% historically (stronger fighter assigned red) — a naive "always pick red" baseline sits at ~64%
- XGBoost consistently outperforms LR on test accuracy; LR has better-calibrated probabilities
- Finish type prediction is inherently noisy — 50% on a 3-class problem is usable as a soft signal only
- Next likely gains: per-division ELO, hyperparameter tuning (Optuna), betting odds integration
