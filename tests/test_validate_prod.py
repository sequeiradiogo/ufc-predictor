"""
tests/test_validate_prod.py -- scripts/validate_prod_retrain.py (issue #67): the
checks that gate a prod retrain, and the trial never writing to the live tiers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts import validate_prod_retrain as v  # noqa: E402


def _fights(n=150, seed=0):
    """Market probabilities and outcomes drawn from them (a well-priced market)."""
    rng = np.random.default_rng(seed)
    market = 1 / (1 + np.exp(-rng.normal(0, 0.9, n)))
    y = (rng.random(n) < market).astype(float)
    return market, y


class TestEvaluate:
    def test_market_like_model_passes(self):
        market, y = _fights()
        rng = np.random.default_rng(1)
        p = 1 / (1 + np.exp(-(np.log(market / (1 - market)) * 0.8 + rng.normal(0, 0.4, len(y)))))
        m, fails, warns = v.evaluate(p, y, market)
        assert fails == [] and warns == []
        assert m["market_corr"] > 0.7

    def test_constant_model_fails(self):
        market, y = _fights()
        _, fails, _ = v.evaluate(np.full(len(y), 0.5), y, market)
        assert any("collapsed" in f for f in fails)
        assert any("correlation" in f for f in fails)

    def test_flipped_model_fails(self):
        market, y = _fights()
        _, fails, _ = v.evaluate(1 - market, y, market)
        assert any("correlation" in f for f in fails)
        assert any("worse than the closing line" in f for f in fails)

    def test_far_better_than_the_market_is_flagged_as_leakage(self):
        market, y = _fights()
        p = np.where(y == 1, 0.85, 0.15)   # knows the result
        _, fails, _ = v.evaluate(p, y, market)
        assert any("leakage" in f for f in fails)

    def test_probabilities_at_the_bounds_fail(self):
        market, y = _fights()
        p = np.where(market >= 0.5, 0.95, 0.05)
        _, fails, _ = v.evaluate(p, y, market)
        assert any("bounds" in f for f in fails)

    def test_few_priced_fights_skip_market_checks(self):
        market, y = _fights()
        market[v.MIN_ODDS_FIGHTS - 1:] = np.nan
        m, fails, warns = v.evaluate(1 - np.nan_to_num(market, nan=0.5), y, market)
        assert "market_corr" not in m
        assert not any("correlation" in f for f in fails)
        assert warns and "market checks skipped" in warns[0]


class TestTrialIsolation:
    def test_every_model_path_is_redirected(self, tmp_path, monkeypatch):
        import ml.train_v1_models as tv
        # monkeypatch restores every constant after the test
        for name in dir(tv):
            if name.startswith(("MODEL_V1_", "MODELS_V1_")):
                monkeypatch.setattr(tv, name, getattr(tv, name))
        v.redirect_model_paths(tv, tmp_path / "eval", tmp_path / "prod")
        paths = {n: getattr(tv, n) for n in dir(tv)
                 if n.startswith(("MODEL_V1_", "MODELS_V1_")) and isinstance(getattr(tv, n), Path)}
        assert paths
        assert all(tmp_path in p.parents or p.parent == tmp_path for p in paths.values())
        assert tv.MODEL_V1_PROD_ENSEMBLE_PATH.parent == tmp_path / "prod"
        assert tv.MODEL_V1_ENSEMBLE_PATH.parent == tmp_path / "eval"

    def test_unredirected_path_is_refused(self, tmp_path, monkeypatch):
        import ml.train_v1_models as tv
        for name in dir(tv):
            if name.startswith(("MODEL_V1_", "MODELS_V1_")):
                monkeypatch.setattr(tv, name, getattr(tv, name))
        monkeypatch.setattr(tv, "MODELS_V1_EXTRA_DIR", Path("/somewhere/live"), raising=False)
        with pytest.raises(RuntimeError):
            v.redirect_model_paths(tv, tmp_path / "eval", tmp_path / "prod")
