"""
tests/test_feature_vector.py
============================
Artifact-free checks of predict.build_feature_vector() edge cases.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import predict

_FEATS = ["height_diff", "reach_diff", "reach_div_norm_diff", "reach_ratio_diff"]


def _vector(red: dict, blue: dict) -> pd.Series:
    return predict.build_feature_vector(
        pd.Series(red), pd.Series(blue), 1500.0, 1500.0, {}, {}, "lightweight", 0, _FEATS,
    ).iloc[0]


class TestUnknownHeightReach:
    """UFCStats has no height/reach for some newcomers (stored as 0): the
    difference must be 0, not a ~180 cm gap (issue #77). Training applies the
    same rule in ML_data_preparation_v1.py."""

    def test_known_values_give_the_difference(self):
        v = _vector({"height": 180.0, "reach": 190.0}, {"height": 175.0, "reach": 185.0})
        assert v["height_diff"] == pytest.approx(5.0)
        assert v["reach_diff"] == pytest.approx(5.0)
        assert v["reach_div_norm_diff"] > 0

    @pytest.mark.parametrize("unknown", [0.0, None])
    def test_unknown_side_gives_zero(self, unknown):
        v = _vector({"height": 180.0, "reach": 190.0}, {"height": unknown, "reach": unknown})
        for f in _FEATS:
            assert v[f] == 0.0, f
        v = _vector({"height": unknown, "reach": unknown}, {"height": 175.0, "reach": 185.0})
        for f in _FEATS:
            assert v[f] == 0.0, f
