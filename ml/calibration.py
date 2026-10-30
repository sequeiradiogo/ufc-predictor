"""
calibration.py -- Final temperature scaling of the ensemble probability.

The ensemble averages per-model isotonic-calibrated probabilities. Each
component is calibrated on its own, but nothing checks the *final* number:
on the 2025-26 hold-out the ensemble said 80-90% and won ~79%, said 90%+ and
won ~78%. A single temperature T shrinks the log-odds toward 50%:

    p' = sigmoid(logit(p) / T)

No intercept, so p'(Blue) = 1 - p'(Red) (corner-symmetric, like the
symmetry-augmented training) and the predicted side never changes -- only how
confident it is. T is fit by log-loss on the ensemble's tune split (out of
sample for every component model and calibrator) and stored in ensemble.joblib
under "temperature"; artifacts without it behave as T = 1.

Shared by ml/train_v1_models.py, predict.py and scripts/backtest_v1.py so
training, live inference and backtesting apply the identical transform.
"""

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-6
TEMPERATURE_BOUNDS = (0.5, 5.0)


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


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising the log-loss of apply_temperature(p, T) against y."""
    y = np.asarray(y, dtype=float)

    def _log_loss(t: float) -> float:
        q = np.clip(apply_temperature(p, t), _EPS, 1 - _EPS)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    return float(minimize_scalar(_log_loss, bounds=TEMPERATURE_BOUNDS, method="bounded").x)
