"""
purged_cv.py -- Purged walk-forward cross-validation.

Adds a configurable time gap between the end of each training fold and
the start of validation. This prevents CV from being over-optimistic when
tuning hyperparameters: the gap approximates the real forecast horizon so
CV accuracy correlates better with held-out accuracy on future fights.

Why the plain TimeSeriesSplit overfits hyperparameters:
  - Validation folds immediately follow training folds in time.
  - Features/ELO for fighters in the val fold may be highly correlated with
    their last training-fold fight (sometimes just days apart).
  - Optuna finds params that exploit this near-boundary correlation; those
    params don't generalise to fights a year later.

With a 180-day gap, each val fold is separated from its training set by
the same horizon as the honest out-of-sample test (2025+ fights trained on
pre-2024 data). This makes --tune actually useful.
"""

import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    """
    Expanding-window walk-forward CV with a purge gap.

    Divides the timeline into n_splits+1 equal segments. Fold i trains on
    all fights up to the end of segment i, then validates on segment i+1
    after skipping gap_days from the training cutoff.
    """

    def __init__(self, n_splits: int = 5, gap_days: int = 180):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(
        self, dates: "pd.Series"
    ) -> "list[tuple[np.ndarray, np.ndarray]]":
        """
        Yield (train_indices, val_indices) pairs.
        dates must be sorted chronologically (same row order as the DataFrame).
        """
        dates      = pd.Series(dates).reset_index(drop=True)
        min_date   = dates.min()
        max_date   = dates.max()
        total_days = (max_date - min_date).days

        segment_days = total_days / (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_cutoff = min_date + pd.Timedelta(days=segment_days * i)
            val_start    = train_cutoff + pd.Timedelta(days=self.gap_days)
            val_end      = min_date + pd.Timedelta(days=segment_days * (i + 1))

            train_idx = np.where(dates <= train_cutoff)[0]
            val_idx   = np.where((dates > val_start) & (dates <= val_end))[0]

            if len(train_idx) >= 50 and len(val_idx) >= 20:
                yield train_idx, val_idx
