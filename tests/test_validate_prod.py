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


class TestShadowV2:
    """Issue #76: V2 shadow scoring next to the prod validation."""

    @staticmethod
    def _trial(ids, dates, p_prod, p_v2, y):
        n = len(ids)
        return dict(fight_id=np.array(ids), date=np.array(dates), cutoff="2026-11-01",
                    y=np.asarray(y, float), market=np.full(n, np.nan),
                    p_prod=np.asarray(p_prod, float), p_v2=np.asarray(p_v2, float),
                    p_eval=np.full(n, 0.5))

    def test_each_fight_is_kept_from_the_first_run(self, tmp_path):
        h = tmp_path / "shadow.csv"
        t1 = self._trial(["a", "b"], ["2026-12-05", "2026-12-12"], [0.6, 0.6], [0.7, 0.7], [1, 0])
        t2 = self._trial(["b", "c"], ["2026-12-12", "2027-01-09"], [0.1, 0.4], [0.1, 0.4], [0, 1])
        assert v.append_shadow(h, t1, 42, "2026-12-01") == 2
        assert v.append_shadow(h, t2, 42, "2027-01-01") == 1
        import pandas as pd
        rows = pd.read_csv(h).set_index("fight_id")
        assert list(rows.index) == ["a", "b", "c"]
        assert rows.loc["b", "p_prod"] == pytest.approx(0.6)     # first run's value kept
        assert rows.loc["b", "run_date"] == "2026-12-01"

    def _history(self, tmp_path, n, v2_shift, date0="2026-12-01"):
        import pandas as pd
        rng = np.random.default_rng(0)
        market, y = _fights(n)
        p_prod = np.clip(market + rng.normal(0, 0.08, n), 0.05, 0.95)
        # V2 moves part of the way back to the well-priced market (better) or away (worse)
        p_v2 = np.clip(p_prod + v2_shift * (market - p_prod), 0.05, 0.95)
        dates = pd.date_range(date0, periods=max(1, n // 12), freq="7D").strftime("%Y-%m-%d")
        h = tmp_path / "h.csv"
        pd.DataFrame(dict(fight_id=[f"f{i}" for i in range(n)], date=np.resize(dates, n), run_date="x",
                          cutoff="x", seed=42, y=y, market=market, p_prod=p_prod, p_v2=p_v2,
                          p_eval=0.5)).to_csv(h, index=False)
        return h

    def test_collecting_before_the_first_look(self, tmp_path):
        s = v.shadow_summary(self._history(tmp_path, 300, 0.8))
        assert s["n"] == 300 and s["decision"].startswith("collecting")

    def test_clearly_better_v2_is_adopted(self, tmp_path):
        s = v.shadow_summary(self._history(tmp_path, v.SHADOW_FIRST_LOOK, 0.8))
        assert s["delta"] < 0 and s["ci90"][1] < 0
        assert s["decision"].startswith("adopt V2")

    def test_worse_v2_keeps_prod(self, tmp_path):
        s = v.shadow_summary(self._history(tmp_path, v.SHADOW_FIRST_LOOK, -0.8))
        assert s["delta"] > 0 and s["decision"].startswith("keep prod")

    def test_fights_before_shadow_since_are_ignored(self, tmp_path):
        s = v.shadow_summary(self._history(tmp_path, 120, 0.8, date0="2026-01-03"))
        assert s["n"] < 120

    def test_v2_split_falls_on_the_cutoff_and_is_restored(self, tmp_path, monkeypatch):
        import pandas as pd

        class FakeTV:
            TRAIN_TEST_SPLIT = 0.8
            seen = []

        def fit(df, prod=False):
            FakeTV.seen.append((len(df), int(len(df) * FakeTV.TRAIN_TEST_SPLIT), prod))
        for name in ("train_xgb", "train_lr", "train_rf", "train_lgbm", "train_mlp", "train_ensemble"):
            setattr(FakeTV, name, staticmethod(fit))
        monkeypatch.setattr(v, "redirect_model_paths", lambda *a, **k: None)
        train, hold = pd.DataFrame({"x": range(3417)}), pd.DataFrame({"x": range(141)})
        v.train_v2(FakeTV, train, hold, tmp_path / "v2")
        assert FakeTV.seen and all(s == (3558, 3417, False) for s in FakeTV.seen)
        assert FakeTV.TRAIN_TEST_SPLIT == 0.8
