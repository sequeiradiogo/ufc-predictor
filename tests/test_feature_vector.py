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
from config import FINISH_ONLY_FEATURES

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


class TestFinishInvariantFeatures:
    """Finish-type model features (issue #30): sum and |difference| of the two
    fighters' values, from the same sources as the matching _diff features. How
    a fight ends doesn't depend on the corners, so a swap must change nothing."""

    RED = {"splm": 5.0, "td_avg": 0.5, "win_by_ko": 4, "total_rounds_fought": 20}
    BLUE = {"splm": 3.0, "td_avg": 2.5, "win_by_ko": 1, "total_rounds_fought": 12}
    EXTRA_R = {"ko_rate": 0.6, "sub_rate": 0.1, "sapm": 3.0}
    EXTRA_B = {"ko_rate": 0.2, "sub_rate": 0.5, "sapm": 4.0}
    FORM_R = {"recent_finish_rate": 0.8}
    FORM_B = {"recent_finish_rate": 0.2}

    def _v(self, swap: bool = False, five_rounds: int = 0, red: dict | None = None) -> pd.Series:
        r, b = (red or self.RED), self.BLUE
        fr, fb, er, eb = self.FORM_R, self.FORM_B, self.EXTRA_R, self.EXTRA_B
        if swap:
            r, b, fr, fb, er, eb = b, r, fb, fr, eb, er
        return predict.build_feature_vector(
            pd.Series(r), pd.Series(b), 1500.0, 1500.0, fr, fb, "lightweight", 0,
            FINISH_ONLY_FEATURES, extra_r=er, extra_b=eb, five_rounds=five_rounds,
        ).iloc[0]

    def test_sum_and_absdiff(self):
        v = self._v()
        assert v["splm_sum"] == pytest.approx(8.0)
        assert v["splm_absdiff"] == pytest.approx(2.0)
        assert v["ko_rate_sum"] == pytest.approx(0.8)            # from extra
        assert v["ko_rate_absdiff"] == pytest.approx(0.4)
        assert v["sapm_sum"] == pytest.approx(7.0)
        assert v["recent_finish_rate_sum"] == pytest.approx(1.0)  # from form
        assert v["recent_finish_rate_absdiff"] == pytest.approx(0.6)
        assert v["win_by_ko_absdiff"] == pytest.approx(3.0)

    def test_corner_swap_changes_nothing(self):
        pd.testing.assert_series_equal(self._v(), self._v(swap=True))

    def test_five_rounds_is_passed_through(self):
        assert self._v(five_rounds=1)["five_rounds"] == 1
        assert self._v()["five_rounds"] == 0

    def test_missing_value_counts_as_zero(self):
        # Training fills each corner's missing value with 0 before summing
        v = self._v(red={**self.RED, "splm": None})
        assert v["splm_sum"] == pytest.approx(3.0)
        assert v["splm_absdiff"] == pytest.approx(3.0)
