"""
calibration.py -- Corner-symmetric probabilities and the ensemble temperature.

`apply_temperature` scales the ensemble's final probability of Red:

    p' = sigmoid(logit(p) / T)

No intercept, so p'(Blue) = 1 - p'(Red) and the pick never changes. It reads
ensemble.joblib["temperature"] (artifacts without it behave as T = 1).
Until 2026-11 T was fit by log-loss on the first half of the test split,
together with Optuna weights on the same fights. Both were in-sample there, and
the plain mean of the calibrated models with T = 1 did better out of sample
(issue #73), so train_v1_models.py now stores T = 1. The transform stays so
older artifacts keep behaving as they were trained.

Shared by ml/train_v1_models.py, predict.py and scripts/backtest_v1.py so
training, live inference and backtesting apply the identical transform.
"""

import numpy as np

_EPS = 1e-6


def apply_temperature(p, temperature: float = 1.0):
    """Return p (scalar or array, probability of Red) scaled by *temperature*."""
    if temperature == 1.0:
        return p
    q = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    out = 1.0 / (1.0 + np.exp(-np.log(q / (1 - q)) / temperature))
    return float(out) if np.ndim(out) == 0 else out


def symmetric_proba(proba_red, X):
    """Corner-symmetric raw probability of Red: (f(x) + 1 - f(mirror(x))) / 2.

    *proba_red* maps an unscaled feature DataFrame to the model's raw P(Red);
    *X* is that DataFrame. Symmetry-augmented training only makes a model
    roughly symmetric -- Random Forest's bootstrap samples rarely hold both
    copies of a fight, so its raw outputs drifted ~0.05 off on lopsided
    matchups, and the steep top of its isotonic calibrator turned that into
    0.95 vs 0.75 for the same fight with corners swapped. Every raw base-model
    probability is taken through here (training, calibration, predict.py,
    backtest_v1.py), so with the mirrored calibrators (train_v1_models._mirror)
    and the intercept-free temperature the pick never depends on the corner.
    """
    from ml.ML_data_preparation_v1 import mirror_features

    return (np.asarray(proba_red(X), dtype=float)
            + 1 - np.asarray(proba_red(mirror_features(X)), dtype=float)) / 2

