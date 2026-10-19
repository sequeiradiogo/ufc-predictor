"""
tests/test_training.py
======================
Artifact-free tests for training-time helpers:
  - make_symmetric() corner-flip augmentation
  - bounded isotonic calibrators
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.ML_data_preparation_v1 import make_symmetric
from ml.train_v1_models import CAL_PROB_MAX, CAL_PROB_MIN, _isotonic


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


class TestBoundedCalibration:
    def test_isotonic_output_is_bounded(self):
        # Extreme bins that are all wins / all losses would otherwise map to 1.0 / 0.0
        raw = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        y = np.array([0, 0, 0, 1, 1, 1])
        ir = _isotonic().fit(raw, y)
        out = ir.predict(np.array([-5.0, 0.0, 0.5, 1.0, 5.0]))
        assert out.min() >= CAL_PROB_MIN
        assert out.max() <= CAL_PROB_MAX
