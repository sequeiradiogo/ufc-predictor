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


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising the log-loss of apply_temperature(p, T) against y."""
    y = np.asarray(y, dtype=float)

    def _log_loss(t: float) -> float:
        q = np.clip(apply_temperature(p, t), _EPS, 1 - _EPS)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    return float(minimize_scalar(_log_loss, bounds=TEMPERATURE_BOUNDS, method="bounded").x)
