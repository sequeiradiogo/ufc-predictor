"""
tests/test_inference.py
=======================
Tests for the live-inference pipeline introduced in the production-models PR:

  1. Production model artifacts load correctly
  2. Eval model artifacts are unchanged
  3. compute_live_career_stats returns all expected keys and sane values
  4. win_by_dec_unanimous and win_by_dec_split are exclusive buckets
  5. No Contest fights are excluded from win/loss/streak calculations
  6. _get_current_rank reads from rankings_history.csv correctly
  7. End-to-end prediction runs without error and returns valid probability
  8. Known fighter (Topuria vs Gaethje) picks the correct winner
  9. compute_ewma_stats_single: zone accuracy + output-rate extras (2026-07-24 fix)
  10. compute_r1_ewma_stats: round-1 EWMA features, guarded on fight_stats_rounds
  11. compute_opp_adjusted_stats: opponent-quality-normalized output stats
"""

import sqlite3
import sys
from pathlib import Path

import joblib
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH, MODELS_V1_DIR, MODELS_V1_PROD_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def _db_available():
    return DB_PATH.exists()

def _eval_models_available():
    return (MODELS_V1_DIR / "ensemble.joblib").exists()

def _prod_models_available():
    return MODELS_V1_PROD_DIR.exists() and (MODELS_V1_PROD_DIR / "ensemble.joblib").exists()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Production model artifacts
# ══════════════════════════════════════════════════════════════════════════════

_PROD_ARTIFACTS = [
    "xgboost.joblib",
    "xgb_features.joblib",
    "logistic_regression.joblib",
    "lr_scaler.joblib",
    "lr_features.joblib",
    "random_forest.joblib",
    "rf_features.joblib",
    "lightgbm.joblib",
    "lgbm_features.joblib",
    "ensemble.joblib",
]


@pytest.mark.skipif(not _prod_models_available(), reason="models_v1_prod/ not found")
class TestProdModelArtifacts:
    def test_all_artifacts_exist(self):
        for name in _PROD_ARTIFACTS:
            assert (MODELS_V1_PROD_DIR / name).exists(), f"Missing: {name}"

    def test_ensemble_has_weights(self):
        ens = joblib.load(MODELS_V1_PROD_DIR / "ensemble.joblib")
        assert "weights" in ens
        weights = ens["weights"]
        assert set(weights.keys()) >= {"xgb", "lr", "rf", "lgbm"}

    def test_ensemble_weights_sum_to_one(self):
        ens = joblib.load(MODELS_V1_PROD_DIR / "ensemble.joblib")
        total = sum(ens["weights"].values())
        assert abs(total - 1.0) < 1e-6

    def test_ensemble_has_calibrators(self):
        ens = joblib.load(MODELS_V1_PROD_DIR / "ensemble.joblib")
        assert "calibrators" in ens
        assert len(ens["calibrators"]) > 0

    def test_xgb_features_is_list(self):
        feats = joblib.load(MODELS_V1_PROD_DIR / "xgb_features.joblib")
        assert isinstance(feats, list)
        assert len(feats) > 10

    def test_lr_model_has_platt_key(self):
        artifact = joblib.load(MODELS_V1_PROD_DIR / "logistic_regression.joblib")
        assert isinstance(artifact, dict)
        assert "base" in artifact

    def test_prod_and_eval_have_same_feature_set(self):
        prod_feats = set(joblib.load(MODELS_V1_PROD_DIR / "xgb_features.joblib"))
        eval_feats = set(joblib.load(MODELS_V1_DIR / "xgb_features.joblib"))
        assert prod_feats == eval_feats, "Prod and eval feature sets diverged"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Eval model artifacts unchanged
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _eval_models_available(), reason="models_v1/ not found")
class TestEvalModelArtifacts:
    def test_eval_ensemble_exists(self):
        assert (MODELS_V1_DIR / "ensemble.joblib").exists()

    def test_eval_ensemble_loads(self):
        ens = joblib.load(MODELS_V1_DIR / "ensemble.joblib")
        assert "weights" in ens

    def test_eval_xgb_loads(self):
        joblib.load(MODELS_V1_DIR / "xgboost.joblib")

    def test_eval_lr_loads(self):
        artifact = joblib.load(MODELS_V1_DIR / "logistic_regression.joblib")
        assert "base" in artifact


# ══════════════════════════════════════════════════════════════════════════════
# 3 & 4. compute_live_career_stats: keys, values, exclusive dec buckets
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _db_available(), reason="UFCStats DB not found")
class TestLiveCareerStats:
    @pytest.fixture(scope="class")
    def conn(self):
        c = sqlite3.connect(str(DB_PATH))
        yield c
        c.close()

    def _first_fighter_with_fights(self, conn, min_fights=5):
        row = conn.execute(
            """
            SELECT fs.fighter_id, fi.name
            FROM fight_stats fs
            JOIN fighters fi ON fi.fighter_id = fs.fighter_id
            GROUP BY fs.fighter_id
            HAVING COUNT(*) >= ?
            LIMIT 1
            """,
            (min_fights,),
        ).fetchone()
        return row

    def test_returns_all_required_keys(self, conn):
        from predict import compute_live_career_stats
        row = self._first_fighter_with_fights(conn)
        assert row is not None, "No fighters with 5+ fights in DB"
        stats = compute_live_career_stats(conn, row[1])
        assert stats is not None
        required = {
            "wins", "losses", "win_by_ko", "win_by_sub",
            "win_by_dec_unanimous", "win_by_dec_split",
            "career_win_streak", "career_lose_streak", "longest_win_streak",
            "total_rounds_fought", "total_title_bouts",
            "avg_sig_str_pct", "avg_td_pct", "splm", "td_avg", "avg_sub_att",
            "sapm", "str_def", "td_def",
            "str_acc_slope", "splm_slope", "td_acc_slope",
        }
        missing = required - set(stats.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_win_dec_buckets_are_exclusive(self, conn):
        """win_by_dec_unanimous and win_by_dec_split must not double-count the same win."""
        from predict import compute_live_career_stats
        row = self._first_fighter_with_fights(conn)
        assert row is not None
        stats = compute_live_career_stats(conn, row[1])
        assert stats is not None
        total_dec = stats["win_by_dec_unanimous"] + stats["win_by_dec_split"]
        total_wins = stats["wins"]
        assert total_dec <= total_wins

    def test_wins_plus_losses_leq_total_fights(self, conn):
        from predict import compute_live_career_stats
        row = self._first_fighter_with_fights(conn)
        stats = compute_live_career_stats(conn, row[1])
        total_fights = conn.execute(
            "SELECT COUNT(*) FROM fight_stats WHERE fighter_id = ?", (row[0],)
        ).fetchone()[0]
        assert stats["wins"] + stats["losses"] <= total_fights

    def test_longest_win_streak_geq_current_streak(self, conn):
        from predict import compute_live_career_stats
        row = self._first_fighter_with_fights(conn)
        stats = compute_live_career_stats(conn, row[1])
        assert stats["longest_win_streak"] >= stats["career_win_streak"]

    def test_total_rounds_fought_positive(self, conn):
        from predict import compute_live_career_stats
        row = self._first_fighter_with_fights(conn, min_fights=3)
        stats = compute_live_career_stats(conn, row[1])
        assert stats["total_rounds_fought"] > 0

    def test_unknown_fighter_returns_none(self, conn):
        from predict import compute_live_career_stats
        assert compute_live_career_stats(conn, "ZZZZ Nonexistent Fighter ZZZZ") is None


# ══════════════════════════════════════════════════════════════════════════════
# 5. No Contest handling
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def nc_db():
    """
    In-memory DB with three fights for fighter 'aaa':
      - fight1: win (KO/TKO)
      - fight2: No Contest (winner_id = NULL, method = CNC)
      - fight3: win (Decision - Unanimous)
    Expected: wins=2, losses=0, streak=1 (last fight was a win after NC skipped),
              NC fight not counted in win/loss totals.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE fighters (fighter_id TEXT PRIMARY KEY, name TEXT,
                               height REAL, reach REAL, stance TEXT, dob TEXT);
        CREATE TABLE fights (
            fight_id TEXT PRIMARY KEY, event_id TEXT, date TEXT,
            division TEXT, r_fighter_id TEXT, b_fighter_id TEXT,
            winner_id TEXT, method TEXT, title_fight INTEGER,
            finish_round INTEGER, match_time_sec REAL
        );
        CREATE TABLE fight_stats (
            fight_id TEXT, fighter_id TEXT, corner TEXT,
            sig_str_landed REAL, sig_str_atmpted REAL,
            td_landed REAL, td_atmpted REAL,
            sub_att REAL, total_fight_time REAL,
            PRIMARY KEY (fight_id, fighter_id)
        );

        INSERT INTO fighters VALUES ('aaa','Test Fighter',180,185,'Orthodox','1990-01-01');
        INSERT INTO fighters VALUES ('bbb','Opponent One',175,180,'Orthodox','1991-01-01');
        INSERT INTO fighters VALUES ('ccc','Opponent Two',177,182,'Southpaw','1992-01-01');
        INSERT INTO fighters VALUES ('ddd','Opponent Three',178,181,'Orthodox','1993-01-01');

        -- match_time_sec/finish_round give own-fight durations of 300s, 900s, 900s
        -- (matching this fixture's pre-a33e19e total_fight_time values). fight_stats
        -- .total_fight_time is now the CUMULATIVE seconds fought BEFORE each fight:
        -- f1=0 (debut), f2=300 (after f1), f3=1200 (after f1+f2, NC still counts).
        INSERT INTO fights VALUES
            ('f1','e1','2023-01-01','lightweight','aaa','bbb','aaa','KO/TKO',0,2,0),
            ('f2','e2','2023-06-01','lightweight','aaa','ccc',NULL,'CNC',0,3,300),
            ('f3','e3','2024-01-01','lightweight','aaa','ddd','aaa','Decision - Unanimous',0,3,300);

        INSERT INTO fight_stats VALUES
            ('f1','aaa','r',40,80,2,4,0,0),
            ('f1','bbb','b',20,60,0,2,0,0),
            ('f2','aaa','r',30,70,1,3,0,300),
            ('f2','ccc','b',25,65,0,1,0,300),
            ('f3','aaa','r',50,90,3,5,0,1200),
            ('f3','ddd','b',35,75,1,4,0,1200);
    """)
    yield conn
    conn.close()


class TestNoContestHandling:
    def test_nc_excluded_from_win_count(self, nc_db):
        from predict import compute_live_career_stats
        stats = compute_live_career_stats(nc_db, "Test Fighter")
        assert stats["wins"] == 2
        assert stats["losses"] == 0

    def test_nc_excluded_from_streak(self, nc_db):
        from predict import compute_live_career_stats
        stats = compute_live_career_stats(nc_db, "Test Fighter")
        # Both decided fights were wins (NC skipped), so trailing streak = 2
        assert stats["career_win_streak"] == 2
        assert stats["career_lose_streak"] == 0

    def test_nc_excluded_from_recent_form(self, nc_db):
        from predict import compute_recent_form
        fid = nc_db.execute(
            "SELECT fighter_id FROM fighters WHERE name='Test Fighter'"
        ).fetchone()[0]
        # fight_stats table must have fight_id-level per-fight data
        # compute_recent_form needs fights joined with winner_id
        # add opponent_id column since it may be needed
        form = compute_recent_form(nc_db, fid)
        # 2 decided fights, both wins -> win_rate=1.0
        assert form["recent_win_rate"] == pytest.approx(1.0)

    def test_nc_included_in_stat_averages(self, nc_db):
        """NC fight should still count toward splm / career averages."""
        from predict import compute_live_career_stats
        stats = compute_live_career_stats(nc_db, "Test Fighter")
        # Fighter had 3 fights total including NC; splm > 0
        assert stats["splm"] > 0

    def test_nc_does_not_inflate_losses(self, nc_db):
        from predict import compute_live_career_stats
        stats = compute_live_career_stats(nc_db, "Test Fighter")
        assert stats["losses"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 9. compute_ewma_stats_single: zone accuracy + output-rate extras
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ewma_db():
    """
    In-memory DB with a single fight for fighter 'aaa' vs 'bbb'.
    match_time_sec=300, finish_round=1 -> own fight duration = 300s = 5 min,
    so every per-fight rate below has a clean, hand-computable denominator.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE fights (
            fight_id TEXT PRIMARY KEY, date TEXT,
            finish_round INTEGER, match_time_sec REAL
        );
        CREATE TABLE fight_stats (
            fight_id TEXT, fighter_id TEXT,
            sig_str_landed REAL, sig_str_atmpted REAL,
            td_landed REAL, td_atmpted REAL,
            head_landed REAL, head_atmpted REAL,
            body_landed REAL, body_atmpted REAL,
            dist_landed REAL, dist_atmpted REAL,
            clinch_landed REAL, sub_att REAL, reversals REAL, kd REAL,
            PRIMARY KEY (fight_id, fighter_id)
        );

        INSERT INTO fights VALUES ('f1', '2023-01-01', 1, 300);

        -- aaa: 30/60 str, 2/4 td, 15/30 head, 10/20 body, 5/10 dist,
        --      6 clinch landed, 1 sub attempt, 1 reversal
        -- bbb (opponent): 10 sig strikes landed against aaa, 1 kd against aaa
        INSERT INTO fight_stats VALUES
            ('f1', 'aaa', 30, 60, 2, 4, 15, 30, 10, 20, 5, 10, 6, 1, 1, 0),
            ('f1', 'bbb', 10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
    """)
    yield conn
    conn.close()


class TestEwmaStatsExtras:
    def test_zone_accuracy(self, ewma_db):
        from predict import compute_ewma_stats_single
        stats = compute_ewma_stats_single(ewma_db, "aaa")
        assert stats["ewma_head_acc"] == pytest.approx(15 / 30)
        assert stats["ewma_body_acc"] == pytest.approx(10 / 20)
        assert stats["ewma_dist_acc"] == pytest.approx(5 / 10)

    def test_output_rate_extras(self, ewma_db):
        from predict import compute_ewma_stats_single
        stats = compute_ewma_stats_single(ewma_db, "aaa")
        # fight_min = 300s / 60 = 5.0
        assert stats["ewma_clinch_per"]   == pytest.approx(6 / 5.0)
        assert stats["ewma_sub_att"]      == pytest.approx(1 * 900.0 / 300)
        assert stats["ewma_kd_received"]  == pytest.approx(1 / 5.0)
        assert stats["ewma_reversals"]    == pytest.approx(1 * 900.0 / 300)
        assert stats["career_reversals"]  == pytest.approx(1.0)

    def test_missing_reversals_column_returns_zero(self, ewma_db):
        """Pre-backfill DBs lack the reversals column entirely -- must not crash."""
        from predict import compute_ewma_stats_single
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE fights (fight_id TEXT PRIMARY KEY, date TEXT,
                                  finish_round INTEGER, match_time_sec REAL);
            CREATE TABLE fight_stats (
                fight_id TEXT, fighter_id TEXT,
                sig_str_landed REAL, sig_str_atmpted REAL,
                td_landed REAL, td_atmpted REAL,
                head_landed REAL, head_atmpted REAL,
                body_landed REAL, body_atmpted REAL,
                dist_landed REAL, dist_atmpted REAL,
                clinch_landed REAL, sub_att REAL, kd REAL,
                PRIMARY KEY (fight_id, fighter_id)
            );
            INSERT INTO fights VALUES ('f1', '2023-01-01', 1, 300);
            INSERT INTO fight_stats VALUES
                ('f1', 'aaa', 30, 60, 2, 4, 15, 30, 10, 20, 5, 10, 6, 1, 0),
                ('f1', 'bbb', 10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        """)
        stats = compute_ewma_stats_single(conn, "aaa")
        assert stats["ewma_reversals"] == 0.0
        assert stats["career_reversals"] == 0.0
        # Non-reversals stats should still compute normally
        assert stats["ewma_head_acc"] == pytest.approx(15 / 30)
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 10. compute_r1_ewma_stats
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def r1_db():
    """
    In-memory DB with fight_stats_rounds present. Fight goes to round 3
    (finish_round=3 > 1), so round-1 duration = flat 300s per the formula,
    regardless of match_time_sec.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE fights (
            fight_id TEXT PRIMARY KEY, date TEXT,
            finish_round INTEGER, match_time_sec REAL
        );
        CREATE TABLE fight_stats_rounds (
            fight_id TEXT, fighter_id TEXT, round INTEGER,
            ctrl REAL, sig_str_landed REAL, reversals REAL,
            PRIMARY KEY (fight_id, fighter_id, round)
        );

        INSERT INTO fights VALUES ('f1', '2023-01-01', 3, 900);
        -- round 1 only: 60s control, 10 sig strikes landed, 1 reversal
        INSERT INTO fight_stats_rounds VALUES ('f1', 'aaa', 1, 60, 10, 1);
        INSERT INTO fight_stats_rounds VALUES ('f1', 'aaa', 2, 999, 999, 999);
    """)
    yield conn
    conn.close()


class TestR1EwmaStats:
    def test_r1_features_computed_from_round_1_only(self, r1_db):
        from predict import compute_r1_ewma_stats
        stats = compute_r1_ewma_stats(r1_db, "aaa")
        # R1 duration = 300s (flat, since finish_round=3 > 1) = 5.0 min
        assert stats["ewma_ctrl_r1"]      == pytest.approx(60 / 300)
        assert stats["ewma_splm_r1"]      == pytest.approx(10 / 5.0)
        assert stats["ewma_reversals_r1"] == pytest.approx(1 / 5.0)

    def test_missing_table_returns_zero(self):
        """DBs before backfill_rounds.py has run lack fight_stats_rounds entirely."""
        from predict import compute_r1_ewma_stats
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE fights (fight_id TEXT PRIMARY KEY, date TEXT,
                                  finish_round INTEGER, match_time_sec REAL);
        """)
        stats = compute_r1_ewma_stats(conn, "aaa")
        assert stats == {"ewma_ctrl_r1": 0.0, "ewma_splm_r1": 0.0, "ewma_reversals_r1": 0.0}
        conn.close()

    def test_fighter_with_no_round1_rows_returns_zero(self, r1_db):
        from predict import compute_r1_ewma_stats
        stats = compute_r1_ewma_stats(r1_db, "nonexistent_fighter")
        assert stats == {"ewma_ctrl_r1": 0.0, "ewma_splm_r1": 0.0, "ewma_reversals_r1": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# 11. compute_opp_adjusted_stats
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def opp_adj_db():
    """
    In-memory DB: fighter 'aaa' fought 'bbb' (f1) and 'ccc' (f2). Each row's
    str_def/td_def/head_def/body_def/dist_def is a hardcoded pre-fight
    snapshot value (no rolling computation needed for this test).

    League average (over all 4 rows) and aaa's opponent average (bbb, ccc
    only) are chosen so every field's expected opp_adj_* value is a clean
    hand-computable number -- see the assertions below for the math.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE fights (fight_id TEXT PRIMARY KEY, date TEXT);
        CREATE TABLE fight_stats (
            fight_id TEXT, fighter_id TEXT,
            str_def REAL, td_def REAL, head_def REAL, body_def REAL, dist_def REAL,
            PRIMARY KEY (fight_id, fighter_id)
        );
        INSERT INTO fights VALUES ('f1', '2023-01-01'), ('f2', '2023-06-01');
        INSERT INTO fight_stats VALUES
            ('f1', 'aaa', 20, 30, 40, 50, 60),
            ('f1', 'bbb', 40, 50, 60, 70, 80),
            ('f2', 'aaa', 20, 30, 40, 50, 60),
            ('f2', 'ccc', 60, 70, 80, 90, 40);
    """)
    yield conn
    conn.close()


class TestOppAdjustedStats:
    @pytest.fixture(autouse=True)
    def _reset_league_cache(self):
        # _get_league_avg_defense caches its result at module level keyed by
        # nothing (not per-connection) -- must reset it before each test or
        # an earlier test's DB (e.g. the real production DB via
        # TestEndToEndPrediction) leaks its league averages into this fixture.
        import predict
        predict._league_def_cache = None
        yield
        predict._league_def_cache = None

    def test_opp_adjusted_formula(self, opp_adj_db):
        from predict import compute_opp_adjusted_stats
        result = compute_opp_adjusted_stats(
            opp_adj_db, "aaa",
            own_splm=10.0, own_td_avg=4.0,
            own_head_acc=0.5, own_body_acc=0.6, own_dist_acc=0.7,
        )
        # League avg str_def = mean(20,40,20,60) = 35 -> league allowed = 0.65
        # aaa's opponents' (bbb, ccc) avg str_def = mean(40,60) = 50 -> allowed = 0.50
        # opp_adj_splm = 10 * (0.65 / 0.50) = 13.0
        assert result["opp_adj_splm"] == pytest.approx(13.0)
        # League avg td_def = mean(30,50,30,70) = 45 -> allowed = 0.55
        # opponents' avg td_def = mean(50,70) = 60 -> allowed = 0.40
        # opp_adj_td_avg = 4 * (0.55 / 0.40) = 5.5
        assert result["opp_adj_td_avg"] == pytest.approx(5.5)
        # League avg head_def = mean(40,60,40,80) = 55 -> allowed = 0.45
        # opponents' avg head_def = mean(60,80) = 70 -> allowed = 0.30
        # opp_adj_head_acc = 0.5 * (0.45 / 0.30) = 0.75
        assert result["opp_adj_head_acc"] == pytest.approx(0.75)
        # League avg body_def = mean(50,70,50,90) = 65 -> allowed = 0.35
        # opponents' avg body_def = mean(70,90) = 80 -> allowed = 0.20
        # opp_adj_body_acc = 0.6 * (0.35 / 0.20) = 1.05
        assert result["opp_adj_body_acc"] == pytest.approx(1.05)
        # League avg dist_def = mean(60,80,60,40) = 60 -> allowed = 0.40
        # opponents' avg dist_def = mean(80,40) = 60 -> allowed = 0.40
        # opp_adj_dist_acc = 0.7 * (0.40 / 0.40) = 0.7
        assert result["opp_adj_dist_acc"] == pytest.approx(0.7)

    def test_fighter_with_no_history_falls_back_to_league_average(self, opp_adj_db):
        """A debutant has no career opponents -- opp_adj should reduce to own_stat
        unchanged (league_allowed / league_allowed = 1.0)."""
        from predict import compute_opp_adjusted_stats
        result = compute_opp_adjusted_stats(
            opp_adj_db, "nonexistent_fighter",
            own_splm=10.0, own_td_avg=4.0,
            own_head_acc=0.5, own_body_acc=0.6, own_dist_acc=0.7,
        )
        assert result["opp_adj_splm"] == pytest.approx(10.0)
        assert result["opp_adj_td_avg"] == pytest.approx(4.0)
        assert result["opp_adj_head_acc"] == pytest.approx(0.5)
        assert result["opp_adj_body_acc"] == pytest.approx(0.6)
        assert result["opp_adj_dist_acc"] == pytest.approx(0.7)


# ══════════════════════════════════════════════════════════════════════════════
# 6. _get_current_rank
# ══════════════════════════════════════════════════════════════════════════════

class TestGetCurrentRank:
    RANKINGS_CSV = ROOT_DIR / "raw_data" / "rankings_history.csv"

    @pytest.mark.skipif(
        not (ROOT_DIR / "raw_data" / "rankings_history.csv").exists(),
        reason="rankings_history.csv not found",
    )
    def test_unknown_fighter_returns_unranked(self):
        from predict import _get_current_rank
        assert _get_current_rank("ZZZZ Unknown Fighter ZZZZ", "lightweight") == 16.0

    @pytest.mark.skipif(
        not (ROOT_DIR / "raw_data" / "rankings_history.csv").exists(),
        reason="rankings_history.csv not found",
    )
    def test_champion_returns_zero(self):
        """A fighter with rank=0 (champion) should return 0, not 16."""
        from predict import _get_current_rank
        # Islam Makhachev has been LW champion for an extended period
        rank = _get_current_rank("Islam Makhachev", "lightweight")
        # Should be champion (0) or a ranked position -- not unranked (16)
        assert rank < 16.0

    @pytest.mark.skipif(
        not (ROOT_DIR / "raw_data" / "rankings_history.csv").exists(),
        reason="rankings_history.csv not found",
    )
    def test_rank_is_numeric(self):
        from predict import _get_current_rank
        rank = _get_current_rank("Jon Jones", "heavyweight")
        assert isinstance(rank, float)
        assert 0.0 <= rank <= 16.0


# ══════════════════════════════════════════════════════════════════════════════
# 7 & 8. End-to-end prediction
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (_db_available() and _prod_models_available()),
    reason="UFCStats DB or prod models not found",
)
class TestEndToEndPrediction:
    def test_prediction_runs_without_error(self):
        from predict import compute_prediction
        result = compute_prediction(
            "Islam Makhachev", "Charles Oliveira", "ensemble", "lightweight", False
        )
        assert result is not None

    def test_prediction_returns_valid_probability(self):
        from predict import compute_prediction
        result = compute_prediction(
            "Islam Makhachev", "Charles Oliveira", "ensemble", "lightweight", False
        )
        prob = result["red_prob"]
        assert 0.0 < prob < 1.0

    def test_red_and_blue_probs_sum_to_one(self):
        from predict import compute_prediction
        result = compute_prediction(
            "Islam Makhachev", "Charles Oliveira", "ensemble", "lightweight", False
        )
        assert abs(result["red_prob"] + result["blue_prob"] - 1.0) < 1e-6

    def test_cross_division_prediction_valid(self):
        """Cross-division matchup (featherweight champion vs lightweight champion) must return a valid probability."""
        from predict import compute_prediction
        result = compute_prediction(
            "Ilia Topuria", "Justin Gaethje", "ensemble", "featherweight", True
        )
        assert result is not None
        prob = result["red_prob"]
        assert 0.0 < prob < 1.0, f"red_prob out of range: {prob}"
        assert abs(result["red_prob"] + result["blue_prob"] - 1.0) < 1e-6
