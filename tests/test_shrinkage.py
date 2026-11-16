"""
tests/test_shrinkage.py -- ml/shrinkage.py (issue #69): rate shrinkage toward the
league average, shared by the training CSV build and live inference.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.shrinkage import PRIORS, shrink, shrink_corner, shrink_wide


class TestShrink:
    def test_formula(self):
        # 3 rounds of 6.0 plus 6 pseudo-rounds of 3.0 -> (18 + 18) / 9
        assert shrink(6.0, 3, 3.0, 6) == pytest.approx(4.0)

    def test_no_rounds_gives_the_prior(self):
        assert shrink(17.0, 0, 1.5, 6) == pytest.approx(1.5)

    def test_more_rounds_keep_more_of_the_own_rate(self):
        few, many = shrink(10.0, 3, 2.0, 6), shrink(10.0, 60, 2.0, 6)
        assert 2.0 < few < many < 10.0

    def test_vectorised_and_nan_safe(self):
        out = shrink(np.array([6.0, np.nan]), np.array([3.0, 3.0]), 3.0, 6)
        assert out[0] == pytest.approx(4.0) and np.isnan(out[1])
        # NaN rounds count as no rounds
        assert shrink(6.0, np.nan, 3.0, 6) == pytest.approx(3.0)


class TestShrinkCorner:
    def test_k_zero_is_a_no_op(self):
        stats = {"splm": 9.0, "total_rounds_fought": 2}
        assert shrink_corner(stats, 2, k=0) is stats

    def test_only_prior_keys_change_and_missing_keys_stay_missing(self):
        stats = {"splm": 9.0, "wins": 2, "total_rounds_fought": 2}
        out = shrink_corner(stats, 2, k=6)
        assert out["splm"] == pytest.approx(shrink(9.0, 2, PRIORS["splm"], 6))
        assert out["wins"] == 2
        assert "td_avg" not in out
        assert stats["splm"] == 9.0  # input not mutated

    def test_series_in_series_out(self):
        s = pd.Series({"td_avg": 17.0, "stance": "Orthodox"})
        out = shrink_corner(s, 3, k=6)
        assert isinstance(out, pd.Series)
        assert out["td_avg"] == pytest.approx(shrink(17.0, 3, PRIORS["td_avg"], 6))
        assert out["stance"] == "Orthodox"

    def test_unparseable_value_left_alone(self):
        assert shrink_corner({"splm": None}, 3, k=6)["splm"] is None


class TestTrainingMatchesInference:
    """shrink_wide (ML_data_preparation_v1) and shrink_corner (predict.py) must agree."""

    def test_same_values_both_paths(self):
        red  = {"splm": 7.5, "td_avg": 17.0, "ko_rate": 1.0, "opp_adj_body_acc": 400.0, "total_rounds_fought": 3}
        blue = {"splm": 3.1, "td_avg": 0.4,  "ko_rate": 0.25, "opp_adj_body_acc": 66.0, "total_rounds_fought": 45}
        wide = pd.DataFrame([{**{f"r_{k}": v for k, v in red.items()},
                              **{f"b_{k}": v for k, v in blue.items()}}])
        shrink_wide(wide, k=6)
        live_r = shrink_corner(red, red["total_rounds_fought"], k=6)
        live_b = shrink_corner(blue, blue["total_rounds_fought"], k=6)
        for stat in ("splm", "td_avg", "ko_rate", "opp_adj_body_acc"):
            assert wide[f"r_{stat}"].iloc[0] == pytest.approx(live_r[stat])
            assert wide[f"b_{stat}"].iloc[0] == pytest.approx(live_b[stat])
        # rounds themselves are not shrunk
        assert wide["r_total_rounds_fought"].iloc[0] == 3

    def test_corner_swap_negates_the_diff(self):
        # Shrinkage is per corner, so swapping corners swaps the shrunk values
        a = {"splm": 7.5, "total_rounds_fought": 3}
        b = {"splm": 3.1, "total_rounds_fought": 45}
        d_ab = shrink_corner(a, 3, k=6)["splm"] - shrink_corner(b, 45, k=6)["splm"]
        d_ba = shrink_corner(b, 45, k=6)["splm"] - shrink_corner(a, 3, k=6)["splm"]
        assert d_ab == pytest.approx(-d_ba)

    def test_k_zero_leaves_wide_untouched(self):
        wide = pd.DataFrame([{"r_splm": 7.5, "r_total_rounds_fought": 3, "b_splm": 3.1, "b_total_rounds_fought": 45}])
        before = wide.copy()
        shrink_wide(wide, k=0)
        pd.testing.assert_frame_equal(wide, before)


class TestDivisionPriors:
    """config.SHRINK_BY_DIVISION: shrink toward the fight's division average."""

    def test_division_prior_differs_from_ufc_wide(self):
        from ml.shrinkage import DIV_PRIORS
        assert DIV_PRIORS["heavyweight"]["ko_rate"] > PRIORS["ko_rate"] > DIV_PRIORS["women's strawweight"]["ko_rate"]

    def test_zero_rounds_gives_the_division_prior(self):
        from ml.shrinkage import DIV_PRIORS
        out = shrink_corner({"ko_rate": 1.0}, 0, k=6, division="Heavyweight ", by_division=True)
        assert out["ko_rate"] == pytest.approx(DIV_PRIORS["heavyweight"]["ko_rate"])

    def test_unknown_or_small_division_falls_back_to_ufc_wide(self):
        for div in (None, "catch weight", "women's featherweight", "no such division"):
            out = shrink_corner({"ko_rate": 1.0}, 0, k=6, division=div, by_division=True)
            assert out["ko_rate"] == pytest.approx(PRIORS["ko_rate"])

    def test_by_division_off_ignores_division(self):
        out = shrink_corner({"ko_rate": 1.0}, 0, k=6, division="heavyweight", by_division=False)
        assert out["ko_rate"] == pytest.approx(PRIORS["ko_rate"])

    def test_training_matches_inference_per_division(self):
        fights = [("heavyweight", 3, 12), ("women's strawweight", 2, 40), ("catch weight", 5, 5)]
        wide = pd.DataFrame([{"division": d, "r_ko_rate": 0.9, "r_splm": 6.0, "r_total_rounds_fought": rr,
                              "b_ko_rate": 0.2, "b_splm": 3.0, "b_total_rounds_fought": br}
                             for d, rr, br in fights])
        shrink_wide(wide, k=6, by_division=True)
        for i, (d, rr, br) in enumerate(fights):
            live_r = shrink_corner({"ko_rate": 0.9, "splm": 6.0}, rr, k=6, division=d, by_division=True)
            live_b = shrink_corner({"ko_rate": 0.2, "splm": 3.0}, br, k=6, division=d, by_division=True)
            for stat in ("ko_rate", "splm"):
                assert wide[f"r_{stat}"].iloc[i] == pytest.approx(live_r[stat])
                assert wide[f"b_{stat}"].iloc[i] == pytest.approx(live_b[stat])


class TestPerStatK:
    """config.SHRINK_PER_STAT: one k per stat (ml/shrinkage.STAT_K)."""

    def test_every_stat_k_is_a_shrunk_stat_within_bounds(self):
        from ml.shrinkage import STAT_K, STAT_K_MAX
        assert set(STAT_K) <= set(PRIORS)
        assert all(2.0 <= v <= STAT_K_MAX for v in STAT_K.values())

    def test_uses_the_stat_k_and_falls_back_to_k(self):
        from ml.shrinkage import STAT_K
        out = shrink_corner({"td_avg": 5.0, "body_acc": 90.0, "ko_rate": 1.0}, 4, k=6, per_stat=True)
        assert out["td_avg"] == pytest.approx(shrink(5.0, 4, PRIORS["td_avg"], STAT_K["td_avg"]))
        assert out["body_acc"] == pytest.approx(shrink(90.0, 4, PRIORS["body_acc"], STAT_K["body_acc"]))
        assert "ko_rate" not in STAT_K
        assert out["ko_rate"] == pytest.approx(shrink(1.0, 4, PRIORS["ko_rate"], 6))

    def test_derived_stats_share_their_base_k(self):
        from ml.shrinkage import STAT_K
        assert STAT_K["ewma_splm"] == STAT_K["opp_adj_splm"] == STAT_K["splm"]
        assert STAT_K["ewma_head_acc"] == STAT_K["opp_adj_head_acc"] == STAT_K["head_acc"]

    def test_k_zero_is_still_off(self):
        stats = {"splm": 9.0}
        assert shrink_corner(stats, 2, k=0, per_stat=True) is stats

    def test_per_stat_off_is_the_single_k(self):
        a = shrink_corner({"splm": 9.0, "td_def": 80.0}, 3, k=6, per_stat=False)
        assert a["td_def"] == pytest.approx(shrink(80.0, 3, PRIORS["td_def"], 6))

    def test_training_matches_inference(self):
        fights = [("heavyweight", 3, 12), ("women's strawweight", 2, 40), ("catch weight", 5, 5)]
        stats = ("splm", "td_def", "ko_rate", "ewma_td_acc", "opp_adj_dist_acc")
        rv = {"splm": 6.0, "td_def": 90.0, "ko_rate": 0.9, "ewma_td_acc": 0.8, "opp_adj_dist_acc": 60.0}
        bv = {"splm": 3.0, "td_def": 40.0, "ko_rate": 0.1, "ewma_td_acc": 0.1, "opp_adj_dist_acc": 30.0}
        wide = pd.DataFrame([{"division": d, "r_total_rounds_fought": rr, "b_total_rounds_fought": br,
                              **{f"r_{s}": v for s, v in rv.items()}, **{f"b_{s}": v for s, v in bv.items()}}
                             for d, rr, br in fights])
        shrink_wide(wide, k=6, by_division=True, per_stat=True)
        for i, (d, rr, br) in enumerate(fights):
            live_r = shrink_corner(rv, rr, k=6, division=d, by_division=True, per_stat=True)
            live_b = shrink_corner(bv, br, k=6, division=d, by_division=True, per_stat=True)
            for s in stats:
                assert wide[f"r_{s}"].iloc[i] == pytest.approx(live_r[s])
                assert wide[f"b_{s}"].iloc[i] == pytest.approx(live_b[s])

    def test_build_feature_vector_follows_the_model_setting(self):
        import predict
        from ml.shrinkage import STAT_K
        red  = pd.Series({"td_def": 90.0, "total_rounds_fought": 2})
        blue = pd.Series({"td_def": 40.0, "total_rounds_fought": 40})
        args = (red, blue, 1500.0, 1500.0, {}, {}, "lightweight", 0, ["td_def_diff"])
        s = {"shrink_k": 6.0, "shrink_by_division": True, "shrink_per_stat": True}
        from ml.shrinkage import priors_for
        p = priors_for("lightweight", True)["td_def"]
        expected = shrink(90.0, 2, p, STAT_K["td_def"]) - shrink(40.0, 40, p, STAT_K["td_def"])
        assert predict.build_feature_vector(*args, shrink=s)["td_def_diff"].iloc[0] == pytest.approx(expected)


class TestModelSettings:
    """predict.py shrinks as the loaded model was trained, never from config.py."""

    def test_missing_settings_mean_off(self, tmp_path):
        from ml.shrinkage import OFF, read_settings
        assert read_settings(tmp_path / "nope.joblib") == OFF

    def test_roundtrip(self, tmp_path):
        from ml.shrinkage import read_settings, write_settings
        s = {"shrink_k": 6.0, "shrink_by_division": True, "shrink_per_stat": True}
        write_settings(tmp_path / "s.joblib", s)
        assert read_settings(tmp_path / "s.joblib") == s

    def test_settings_from_before_per_stat_read_as_per_stat_off(self, tmp_path):
        import joblib
        from ml.shrinkage import read_settings
        joblib.dump({"shrink_k": 6.0, "shrink_by_division": True}, tmp_path / "old.joblib")
        assert read_settings(tmp_path / "old.joblib")["shrink_per_stat"] is False

    def test_partial_retrain_with_other_settings_refused(self, tmp_path):
        from ml.shrinkage import stamp_models_dir
        stamp_models_dir(tmp_path, {"shrink_k": 0.0, "shrink_by_division": False}, partial=False)
        with pytest.raises(SystemExit):
            stamp_models_dir(tmp_path, {"shrink_k": 6.0, "shrink_by_division": True}, partial=True)
        # a full retrain may change them
        stamp_models_dir(tmp_path, {"shrink_k": 6.0, "shrink_by_division": True}, partial=False)

    def test_config_alone_does_not_change_live_features(self, monkeypatch):
        import config
        import predict
        monkeypatch.setattr(config, "SHRINK_K", 6.0)
        monkeypatch.setattr(config, "SHRINK_BY_DIVISION", True)
        red  = pd.Series({"splm": 9.0, "total_rounds_fought": 2})
        blue = pd.Series({"splm": 3.0, "total_rounds_fought": 40})
        args = (red, blue, 1500.0, 1500.0, {}, {}, "lightweight", 0, ["splm_diff"])
        # model without recorded settings -> no shrinkage, whatever config says
        assert predict.build_feature_vector(*args)["splm_diff"].iloc[0] == pytest.approx(6.0)
        # model trained with shrinkage -> shrunk exactly as in training
        s = {"shrink_k": 6.0, "shrink_by_division": True}
        expected = (shrink_corner({"splm": 9.0}, 2, k=6, division="lightweight", by_division=True)["splm"]
                    - shrink_corner({"splm": 3.0}, 40, k=6, division="lightweight", by_division=True)["splm"])
        assert predict.build_feature_vector(*args, shrink=s)["splm_diff"].iloc[0] == pytest.approx(expected)
