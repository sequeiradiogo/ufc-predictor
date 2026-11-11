"""Unit tests for the DWCS features (ml/pre_ufc.py) and the ESPN parsing helpers.

Artifact-free: build tiny bout tables in memory, no network.
"""

import pandas as pd
import pytest

from ml import pre_ufc
from scrapers.espn import _summary_stats, pair_fight, result_of


def _bouts(rows):
    cols = ["fighter_id", "date", "result", "method", "dur", "sl", "sa", "tl", "ctrl", "o_sl"]
    return pd.DataFrame(rows, columns=cols)


BOUTS = _bouts([
    ("A", "2023-08-01", "W", "KO/TKO", 5.0, 30.0, 50.0, 0.0, 10.0, 10.0),
    ("A", "2024-08-01", "L", "Decision - Unanimous", 15.0, 40.0, 100.0, 2.0, 120.0, 60.0),
    ("B", "2024-09-01", "W", "Decision - Split", 15.0, 50.0, 110.0, 1.0, 0.0, 45.0),
])


class TestFeaturesBefore:
    def test_only_bouts_strictly_before_the_date(self):
        f = pre_ufc.features_before("A", "2024-08-01", BOUTS)    # same-day bout excluded
        assert f["dwcs_fights"] == 1 and f["dwcs_wins"] == 1 and f["dwcs_finish_wins"] == 1
        f = pre_ufc.features_before("A", "2024-08-02", BOUTS)
        assert f["dwcs_fights"] == 2 and f["dwcs_wins"] == 1

    def test_no_bouts_gives_league_defaults(self):
        assert pre_ufc.features_before("A", "2020-01-01", BOUTS) == pre_ufc.DEFAULTS
        assert pre_ufc.features_before("Z", "2030-01-01", BOUTS) == pre_ufc.DEFAULTS
        assert pre_ufc.features_before(None, "2030-01-01", BOUTS) == pre_ufc.DEFAULTS

    def test_rates_are_shrunk_toward_the_league(self):
        f = pre_ufc.features_before("A", "2024-01-01", BOUTS)   # 30 landed in 5 min = 6.0/min raw
        expected = (30 + pre_ufc.RATE_PSEUDO_MIN * pre_ufc.LEAGUE_SLPM) / (5 + pre_ufc.RATE_PSEUDO_MIN)
        assert f["dwcs_splm"] == pytest.approx(expected)
        assert pre_ufc.LEAGUE_SLPM < f["dwcs_splm"] < 6.0
        assert f["dwcs_sapm"] < pre_ufc.LEAGUE_SLPM     # absorbed 10 in 5 min = 2.0/min

    def test_prefight_features_matches_features_before(self):
        pairs = pd.DataFrame({"fighter_id": ["A", "A", "B", "Z"],
                              "date": ["2024-08-01", "2025-01-01", "2024-09-01", "2025-01-01"]})
        out = pre_ufc.prefight_features(pairs, BOUTS)
        for i, (fid, d) in enumerate(zip(pairs["fighter_id"], pairs["date"])):
            exp = pre_ufc.features_before(fid, d, BOUTS)
            for f in pre_ufc.FEATURES:
                assert out.loc[i, f] == pytest.approx(exp[f]), (fid, d, f)

    def test_red_blue_swap_negates_the_diff(self):
        r = pre_ufc.features_before("A", "2025-01-01", BOUTS)
        b = pre_ufc.features_before("B", "2025-01-01", BOUTS)
        for f in pre_ufc.FEATURES:
            assert (r[f] - b[f]) == pytest.approx(-(b[f] - r[f]))


class TestEspnParsing:
    def test_summary_stats_split_landed_attempted(self):
        stats = [{"name": "sigStrikes", "value": 46, "displayValue": "46/101"},
                 {"name": "takedowns", "value": 2, "displayValue": "2/8"},
                 {"name": "knockDowns", "value": 1.0, "displayValue": "1"},
                 {"name": "timeInControl", "value": 78.0, "displayValue": "1:18"}]
        s = _summary_stats(stats)
        assert (s["sig_str_landed"], s["sig_str_atmpted"]) == (46, 101)
        assert (s["td_landed"], s["td_atmpted"]) == (2, 8)
        assert s["kd"] == 1 and s["ctrl"] == 78

    def test_result_of(self):
        row = {"r_winner": True, "b_winner": False, "method": "KO/TKO"}
        assert result_of(row, "r") == "W" and result_of(row, "b") == "L"
        row = {"r_winner": False, "b_winner": False, "method": "No Contest"}
        assert result_of(row, "r") == "NC"
        row = {"r_winner": False, "b_winner": False, "method": "Draw - Majority"}
        assert result_of(row, "b") == "D"

    def test_pair_fight_orients_corners_and_needs_both_names(self):
        cands = [("id1", "Lone'er Kavanagh", "id2", "An Tuan Ho"),
                 ("id3", "Anthony Johnson", "id4", "Glover Teixeira")]
        assert pair_fight("Lone'er Kavanagh", "An Tuan Ho", cands) == ("id1", "id2")
        assert pair_fight("An Tuan Ho", "Lone'er Kavanagh", cands) == ("id2", "id1")
        # one similar name is not enough
        assert pair_fight("Tony Johnson", "Someone Else", cands) is None


CAREER = pd.DataFrame([
    ("A", "2018-01-01", "other", "W", "KO/TKO"),
    ("A", "2018-06-01", "lfa", "L", "TKO (Punches)"),
    ("A", "2019-01-01", "other", "NC", "No Contest"),
    ("A", "2019-06-01", "cage-warriors", "W", "Submission (Rear Naked Choke)"),
    ("A", "2020-01-01", "other", "W", "Decision - Unanimous"),
], columns=["fighter_id", "date", "league", "result", "method"])
OTHER = pd.DataFrame([("A", "2021-01-01"), ("A", "2022-01-01")], columns=["fighter_id", "date"])


class TestCareerFeatures:
    def _one(self, fid, date):
        return pre_ufc.prefight_career(pd.DataFrame({"fighter_id": [fid], "date": [date]}), CAREER, OTHER).iloc[0]

    def test_record_before_the_date_only(self):
        f = self._one("A", "2019-06-01")          # the 2019-06-01 win is excluded
        assert (f["ext_fights"], f["ext_wins"], f["ext_losses"]) == (2, 1, 1)   # NC counts as nothing
        assert f["ext_ko_losses"] == 1
        f = self._one("A", "2030-01-01")
        assert (f["ext_fights"], f["ext_wins"], f["ext_losses"]) == (4, 3, 1)

    def test_streak_named_promotions_and_time(self):
        f = self._one("A", "2021-06-01")
        assert f["ext_win_streak"] == 2            # W, W after the loss (NC skipped)
        assert f["ext_named_promo_fights"] == 2    # lfa + cage-warriors
        assert f["pro_days_since_last"] == (pd.Timestamp("2021-06-01") - pd.Timestamp("2021-01-01")).days
        assert f["pro_years"] == pytest.approx((pd.Timestamp("2021-06-01") - pd.Timestamp("2018-01-01")).days / 365.25)

    def test_rates_shrunk_and_unknown_fighter_defaults(self):
        f = self._one("A", "2030-01-01")
        k = pre_ufc.CAREER_PSEUDO_FIGHTS
        assert f["ext_win_rate"] == pytest.approx((3 + k * pre_ufc.CAREER_WIN_PRIOR) / (4 + k))
        assert f["ext_finish_rate"] == pytest.approx((2 + k * pre_ufc.CAREER_FINISH_PRIOR) / (4 + k))
        z = self._one("Z", "2030-01-01")
        assert z["ext_fights"] == 0 and z["ext_win_rate"] == pytest.approx(pre_ufc.CAREER_WIN_PRIOR)
        assert z["pro_days_since_last"] == pre_ufc.NO_FIGHT_DAYS and z["pro_years"] == 0

    def test_tuf_exhibitions_are_dropped(self, tmp_path):
        csv = tmp_path / "career.csv"
        pd.DataFrame({
            "espn_id": ["1", "1"], "fighter_id": ["A", "A"], "date": ["2017-01-01", "2017-02-01"],
            "league": ["other", "other"], "event_name": ["The Ultimate Fighter 27 Semifinal", "LFA 1"],
            "result": ["W", "W"], "method": ["Decision", "KO/TKO"],
        }).to_csv(csv, index=False)
        c = pre_ufc.load_career(csv)
        assert len(c) == 1 and c.iloc[0]["date"] == "2017-02-01"


class TestOpponentQuality:
    BOUTS = pd.DataFrame([
        ("c1", "2018-01-01", "X", "espn:9", "X"),        # X beats a debutant
        ("c2", "2018-06-01", "espn:9", "espn:8", "espn:9"),
        ("c3", "2019-01-01", "A", "espn:9", ""),        # draw vs an opponent who has 1-1
        ("u1", "2020-01-01", "A", "X", "A"),             # UFC bout
    ], columns=["bout_id", "date", "a", "b", "winner"])

    def test_ratings_are_pre_bout_and_zero_sum_at_equal_k(self):
        r = pre_ufc.replay_ratings(self.BOUTS)
        first = r[r["bout_id"] == "c1"].set_index("node")
        assert first.loc["X", "elo"] == first.loc["espn:9", "elo"] == pre_ufc.MMA_ELO_START
        assert first.loc["X", "post_elo"] - pre_ufc.MMA_ELO_START == pytest.approx(
            pre_ufc.MMA_ELO_START - first.loc["espn:9", "post_elo"])
        c3 = r[(r["bout_id"] == "c3") & (r["node"] == "A")].iloc[0]
        assert (c3["opp_w"], c3["opp_l"]) == (1, 1)     # espn:9 lost c1, won c2

    def test_features_only_use_earlier_bouts(self):
        r = pre_ufc.replay_ratings(self.BOUTS)
        pairs = pd.DataFrame({"fighter_id": ["A", "A", "Z"], "date": ["2019-01-01", "2019-06-01", "2030-01-01"]})
        f = pre_ufc.prefight_opp_quality(pairs, r)
        assert f.loc[0, "mma_elo"] == pre_ufc.MMA_ELO_START                     # c3 not yet fought
        assert f.loc[0, "ext_opp_elo"] == pytest.approx(pre_ufc.MMA_ELO_START)  # no ext bouts -> prior
        c3_opp = r[(r["bout_id"] == "c3") & (r["node"] == "A")]["opp_elo"].iloc[0]
        k = pre_ufc.OPP_PSEUDO_FIGHTS
        assert f.loc[1, "ext_opp_elo"] == pytest.approx((c3_opp + k * pre_ufc.MMA_ELO_START) / (1 + k))
        assert f.loc[1, "ext_win_opp_elo"] == pytest.approx(pre_ufc.MMA_ELO_START)  # a draw isn't a win
        assert f.loc[2, "mma_elo"] == pre_ufc.MMA_ELO_START


class TestModelFeatures:
    """prefight_model_features(): the values the models train on and predict.py reads live."""

    CAREER = pd.DataFrame([
        ("A", "2015-01-01", "lfa", "W", "KO/TKO"),
        ("A", "2016-01-01", "other", "W", "Submission"),
        ("A", "2017-01-01", "other", "L", "Decision"),
    ], columns=["fighter_id", "date", "league", "result", "method"])
    EXT_GRAPH = pd.DataFrame([
        ("c1", "2015-01-01", "A", "espn:1", "A"),
        ("c2", "2016-01-01", "A", "espn:2", "A"),
        ("c3", "2017-01-01", "A", "espn:3", "espn:3"),
    ], columns=["bout_id", "date", "a", "b", "winner"])
    UFC = [  # fight_id, date, r, b, winner, method
        ("f1", "2018-01-01", "A", "B", "A", "KO/TKO"),
        ("f2", "2018-06-01", "A", "C", "A", "Decision - Unanimous"),
        ("f3", "2019-01-01", "B", "C", None, "Overturned"),       # NC: not a bout for the rating
        ("f4", "2019-06-01", "B", "C", None, "Decision - Split"),  # draw
    ]

    @pytest.fixture
    def conn(self, monkeypatch):
        import sqlite3
        empty_dwcs = pd.DataFrame(columns=["fighter_id", "date"])
        monkeypatch.setattr(pre_ufc, "_cached_offline", lambda: (self.CAREER, empty_dwcs, self.EXT_GRAPH))
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE fights (fight_id, date, r_fighter_id, b_fighter_id, winner_id, method)")
        c.executemany("INSERT INTO fights VALUES (?,?,?,?,?,?)", self.UFC)
        return c

    def test_ext_record_fades_with_ufc_fights(self, conn):
        pairs = pd.DataFrame({"fighter_id": ["A", "A", "A"], "date": ["2018-01-01", "2018-06-01", "2019-01-01"]})
        f = pre_ufc.prefight_model_features(pairs, conn)
        k = pre_ufc.FADE_K
        assert list(f["ext_fights"]) == pytest.approx([3.0, 3.0 * k / (k + 1), 3.0 * k / (k + 2)])
        prior = pre_ufc.NO_DATA["ext_win_rate"]
        raw = (2 + pre_ufc.CAREER_PSEUDO_FIGHTS * prior) / (3 + pre_ufc.CAREER_PSEUDO_FIGHTS)
        assert f.loc[2, "ext_win_rate"] == pytest.approx(prior + (raw - prior) * k / (k + 2))
        assert f.loc[0, "pro_years"] == pytest.approx(f.loc[2, "pro_years"] - 1.0, abs=0.01)   # not faded
        assert f.loc[1, "pro_days_since_last"] == 151.0      # last pro bout: UFC f1

    def test_unknown_fighter_gets_no_data_values(self, conn):
        f = pre_ufc.prefight_model_features(pd.DataFrame({"fighter_id": [None, "Z"], "date": ["2020-01-01"] * 2}), conn)
        for feat in pre_ufc.MODEL_FEATURES:
            assert list(f[feat]) == [pre_ufc.NO_DATA[feat]] * 2, feat

    def test_mma_elo_counts_ufc_bouts_and_skips_no_contests(self, conn):
        f = pre_ufc.prefight_model_features(pd.DataFrame({"fighter_id": ["B", "B"], "date": ["2019-06-01", "2020-01-01"]}), conn)
        assert f.loc[0, "ext_fights"] == 0 and f.loc[0, "mma_elo"] < pre_ufc.MMA_ELO_START   # lost f1, f3 ignored
        r = pre_ufc.replay_ratings(pre_ufc.graph_bouts(
            ufc_bouts=pd.DataFrame(self.UFC, columns=["fight_id", "date", "r_fighter_id", "b_fighter_id",
                                                      "winner_id", "method"]),
            opp_csv=pre_ufc.Path("-"), career_csv=pre_ufc.Path("-"), dwcs_csv=pre_ufc.Path("-"),
            map_csv=pre_ufc.Path("-")))
        assert set(r["bout_id"]) == {"uf1", "uf2", "uf4"}

    def test_regional_bouts_move_mma_elo_less_than_ufc_bouts(self, conn):
        f = pre_ufc.prefight_model_features(pd.DataFrame({"fighter_id": ["A"], "date": ["2018-01-01"]}), conn)
        full = pre_ufc.replay_ratings(self.EXT_GRAPH)
        full_a = full[full["node"] == "A"]["post_elo"].iloc[-1]
        assert abs(f.loc[0, "mma_elo"] - pre_ufc.MMA_ELO_START) < abs(full_a - pre_ufc.MMA_ELO_START)

    def test_live_call_on_a_db_cut_at_the_date_matches_training(self, conn):
        train = pre_ufc.prefight_model_features(pd.DataFrame({"fighter_id": ["A", "C"], "date": ["2019-06-01"] * 2}), conn)
        conn.execute("CREATE TEMP VIEW fights AS SELECT * FROM main.fights WHERE date < '2019-06-01'")
        live = pre_ufc.current_model_features(conn, ["A", "C"], before_date="2019-06-01")
        for i, fid in enumerate(["A", "C"]):
            for feat in pre_ufc.MODEL_FEATURES:
                assert live[fid][feat] == pytest.approx(train.loc[i, feat]), (fid, feat)
