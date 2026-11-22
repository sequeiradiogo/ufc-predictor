"""
tests/test_training.py
======================
Artifact-free tests for training-time helpers:
  - make_symmetric() corner-flip augmentation
  - bounded isotonic calibrators
  - final temperature scaling of the ensemble
  - eval-tier training / calibration windows
  - --tune: fit-window only, seeded, log-loss objective (issue #42)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.calibration import apply_temperature, symmetric_proba
from ml.ML_data_preparation_v1 import make_symmetric, mirror_features
from config import TRAIN_TEST_SPLIT
from ml.train_v1_models import CAL_PROB_MAX, CAL_PROB_MIN, _isotonic, _mirror, eval_fit_end, time_split
from ml.train_v1_models import preprocess as win_preprocess
import ml.train_v1_models as tv
from config import FINISH_ONLY_FEATURES


def _toy_fights() -> pd.DataFrame:
    return pd.DataFrame({
        "date":                pd.to_datetime(["2025-01-01", "2025-02-01"]),
        "target":              [1, 0],
        "elo_diff":            [50.0, -20.0],
        "td_offense_matchup":  [1.5, -0.4],
        "striker_vs_wrestler": [0.3, 0.1],
        "wrestler_vs_striker": [0.2, 0.6],
        "title_fight":         [1, 0],
    })


class TestMakeSymmetric:
    def test_every_antisymmetric_feature_is_negated(self):
        df = _toy_fights()
        sym = make_symmetric(df)
        assert len(sym) == 2 * len(df)
        for _, orig in df.iterrows():
            flip = sym[(sym["date"] == orig["date"]) & (sym["target"] == 1 - orig["target"])].iloc[0]
            assert flip["elo_diff"] == -orig["elo_diff"]
            # Red-minus-Blue without the _diff suffix -- was left un-negated until 2026-10
            assert flip["td_offense_matchup"] == -orig["td_offense_matchup"]
            assert flip["striker_vs_wrestler"] == orig["wrestler_vs_striker"]
            assert flip["wrestler_vs_striker"] == orig["striker_vs_wrestler"]
            assert flip["title_fight"] == orig["title_fight"]

    def test_antisymmetric_feature_keeps_its_signal(self):
        # A feature perfectly aligned with the target must stay aligned after
        # augmentation; an un-negated copy drives the correlation to exactly 0.
        df = _toy_fights()
        df["td_offense_matchup"] = np.where(df["target"] == 1, 1.0, -1.0)
        sym = make_symmetric(df)
        assert np.corrcoef(sym["td_offense_matchup"], sym["target"])[0, 1] > 0.99


class TestCornerSymmetry:
    """A prediction must not depend on which fighter is listed as Red.

    Until 2026-10 the calibrators were fit on real fights only (Red wins ~55%)
    and re-learned a red-corner prior: 19% of 2025+ picks flipped when the
    corners were swapped (Makhachev 95.7% as Red, 79.6% as Blue)."""

    def test_mirror_features_matches_make_symmetric(self):
        df = _toy_fights()
        flip = make_symmetric(df)
        flip = flip[flip["target"] != flip["date"].map(df.set_index("date")["target"])]
        mirrored = mirror_features(df).drop(columns="target")
        pd.testing.assert_frame_equal(
            flip.drop(columns="target").sort_values("date").reset_index(drop=True),
            mirrored.sort_values("date").reset_index(drop=True),
        )

    def test_mirrored_isotonic_is_point_symmetric(self):
        # Calibration data with a strong red prior
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.15, 0.85, 600)
        y = (rng.uniform(size=raw.size) < np.clip(raw + 0.08, 0, 1)).astype(int)
        grid = np.linspace(0, 1, 41)
        biased = _isotonic().fit(raw, y)
        assert (biased.predict(grid) + biased.predict(1 - grid) - 1).mean() > 0.05
        ir = _isotonic().fit(*_mirror(raw, y))
        np.testing.assert_allclose(ir.predict(grid) + ir.predict(1 - grid), 1.0, atol=1e-9)

    def test_symmetric_proba_is_exactly_antisymmetric(self):
        # A model that ignores corner symmetry entirely (red bias + nonlinearity)
        df = _toy_fights().drop(columns=["date", "target"])
        f = lambda X: 1 / (1 + np.exp(-(0.01 * X["elo_diff"] + X["striker_vs_wrestler"] + 0.4)))
        p = symmetric_proba(f, df)
        q = symmetric_proba(f, mirror_features(df))
        np.testing.assert_allclose(p + q, 1.0)


class TestBoundedCalibration:
    def test_isotonic_output_is_bounded(self):
        # Extreme bins that are all wins / all losses would otherwise map to 1.0 / 0.0
        raw = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        y = np.array([0, 0, 0, 1, 1, 1])
        ir = _isotonic().fit(raw, y)
        out = ir.predict(np.array([-5.0, 0.0, 0.5, 1.0, 5.0]))
        assert out.min() >= CAL_PROB_MIN
        assert out.max() <= CAL_PROB_MAX


class TestTemperature:
    def test_identity_at_one(self):
        p = np.array([0.1, 0.5, 0.87])
        assert np.array_equal(apply_temperature(p, 1.0), p)

    def test_shrinks_toward_half_without_changing_the_pick(self):
        p = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
        q = apply_temperature(p, 1.2)
        assert np.all(np.abs(q - 0.5) <= np.abs(p - 0.5))
        assert np.array_equal(q >= 0.5, p >= 0.5)
        assert q[2] == 0.5

    def test_corner_symmetric(self):
        # Swapping corners must give the complementary probability (no intercept)
        assert apply_temperature(0.8, 1.3) + apply_temperature(0.2, 1.3) == pytest.approx(1.0)

    def test_scalar_in_scalar_out(self):
        assert isinstance(apply_temperature(0.9, 1.2), float)


class TestEvalWindows:
    def test_calibration_window_is_out_of_sample(self):
        # Base models must stop training before the calibration window starts;
        # until 2026-10 the trees trained through it and were calibrated in-sample
        n = 3500
        fit_end, split = eval_fit_end(n), int(n * TRAIN_TEST_SPLIT)
        assert 0 < fit_end < split
        df = pd.DataFrame({"date": pd.date_range("2018-01-01", periods=n, freq="D")})
        train, test = time_split(df)
        assert len(train) == fit_end
        assert test.index[0] == split


def _toy_season(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Daily fights with a learnable signal, enough for 5 purged CV folds."""
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 100, n)
    return pd.DataFrame({
        "fight_id": [f"f{i}" for i in range(n)],
        "date":     pd.date_range("2018-01-01", periods=n, freq="3D"),
        "division": ["lightweight"] * n,
        "target":   (rng.uniform(size=n) < 1 / (1 + np.exp(-elo / 100))).astype(int),
        "elo_diff": elo,
        "reach_diff": rng.normal(0, 5, n),
    })


class TestTuning:
    """--tune (issue #42): eval tuning must not see the calibration window or the
    test split (until 2026-11 its CV ran over the whole CSV), the search is seeded,
    and the objective is log-loss on symmetrised probabilities unless asked otherwise."""

    def test_eval_tuning_sees_only_the_fit_window(self):
        df = _toy_season()
        assert len(tv._tune_window(df, prod=False)) == eval_fit_end(len(df))
        assert len(tv._tune_window(df, prod=True)) == len(df)

    def test_lr_search_is_reproducible_and_scores_logloss(self, monkeypatch):
        df = _toy_season()
        seen = []
        real = tv._tune

        def spy(name, d, n_trials, suggest, fit_proba, metric=None):
            seen.append(metric or tv.TUNE_METRIC)
            return real(name, d, n_trials, suggest, fit_proba, metric)

        monkeypatch.setattr(tv, "_tune", spy)
        a = tv._tune_lr(df, 3)
        b = tv._tune_lr(df, 3)
        assert a == b
        assert seen == ["logloss", "logloss"]

    def test_unknown_metric_is_rejected(self):
        with pytest.raises(ValueError):
            tv._tune("LR", _toy_season(), 1, lambda t: {}, lambda *a: None, metric="auc")


class TestFinishOnlyFeatures:
    """The finish-type model's corner-invariant columns live in the same CSV
    (issue #30) but must never reach the win models: make_symmetric() leaves
    them unchanged on a corner swap, which is right for "how does it end" and
    meaningless for "who wins"."""

    def test_win_models_drop_them(self):
        df = pd.DataFrame({
            "fight_id": ["a", "b"], "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "division": ["lightweight"] * 2, "target": [1, 0], "elo_diff": [10.0, -5.0],
            **{f: [1.0, 2.0] for f in FINISH_ONLY_FEATURES},
        })
        X, _ = win_preprocess(df)
        assert list(X.columns) == ["elo_diff"]
