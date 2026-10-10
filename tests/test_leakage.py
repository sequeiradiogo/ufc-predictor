"""
tests/test_leakage.py
=====================
Guards against the fight being predicted leaking into its own "pre-fight"
features.

The Kaggle-sourced ufc-master.csv rows (2010 - Mar 2026) stored career
averages that already included the current fight (found 2026-10), which
inflated the 2025+ backtest by ~4pp. Career stats are now rebuilt with
predict.compute_live_career_stats(before_date=...) by
scripts/rebuild_career_stats.py; these tests pin that behaviour.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH, NAME_ALIASES, RAW_DIR
from predict import compute_live_career_stats


# ══════════════════════════════════════════════════════════════════════════════
# 1. compute_live_career_stats(before_date=...) -- artifact-free
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def two_fight_db():
    """Fighter A: wins fight 1 (TD 1/2), loses fight 2 (TD 4/4)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE fighters (fighter_id TEXT, name TEXT, height REAL, reach REAL, dob TEXT, stance TEXT);
        CREATE TABLE fights (fight_id TEXT, date TEXT, method TEXT, winner_id TEXT, title_fight INTEGER,
                             finish_round INTEGER, match_time_sec INTEGER);
        CREATE TABLE fight_stats (fight_id TEXT, fighter_id TEXT, sig_str_landed INTEGER, sig_str_atmpted INTEGER,
                                  td_landed INTEGER, td_atmpted INTEGER, sub_att INTEGER, total_fight_time INTEGER,
                                  head_landed INTEGER, head_atmpted INTEGER, body_landed INTEGER, body_atmpted INTEGER,
                                  leg_landed INTEGER, leg_atmpted INTEGER, dist_landed INTEGER, dist_atmpted INTEGER,
                                  ground_landed INTEGER, ground_atmpted INTEGER);
        INSERT INTO fighters VALUES ('a', 'Fighter A', 180, 185, '1990-01-01', 'Southpaw'),
                                    ('b', 'Fighter B', 180, 185, NULL, 'Orthodox'),
                                    ('c', 'Fighter C', 180, 185, NULL, NULL);
        INSERT INTO fights VALUES ('f1', '2025-01-01', 'Decision - Unanimous', 'a', 0, 3, 300),
                                  ('f2', '2025-06-01', 'KO/TKO', 'c', 0, 1, 60);
        -- total_fight_time = cumulative seconds BEFORE the fight (rolling.py convention)
        -- zone columns: head, body, leg, dist, ground (landed, attempted)
        INSERT INTO fight_stats VALUES
            ('f1', 'a', 30, 60, 1, 2, 0, 0,   20, 40, 5, 10, 5, 10, 25, 50, 0, 0),
            ('f1', 'b', 20, 50, 0, 1, 0, 0,   10, 30, 5, 10, 5, 10, 15, 40, 5, 10),
            ('f2', 'a', 5, 10, 4, 4, 1, 900,  5, 10, 0, 0, 0, 0, 0, 0, 5, 10),
            ('f2', 'c', 9, 12, 0, 0, 0, 0,    9, 10, 0, 2, 0, 0, 9, 12, 0, 0);
    """)
    yield conn
    conn.close()


class TestBeforeDate:
    def test_excludes_fight_on_cutoff_date(self, two_fight_db):
        s = compute_live_career_stats(two_fight_db, "Fighter A", before_date="2025-06-01")
        assert (s["wins"], s["losses"]) == (1, 0)
        assert s["avg_td_pct"] == pytest.approx(0.5)      # fight 2's 4/4 must not count
        assert s["splm"] == pytest.approx(30 / 15)        # 30 landed over 15 minutes

    def test_includes_fight_after_cutoff(self, two_fight_db):
        s = compute_live_career_stats(two_fight_db, "Fighter A", before_date="2025-06-02")
        assert (s["wins"], s["losses"]) == (1, 1)
        assert s["avg_td_pct"] == pytest.approx(5 / 6)

    def test_no_cutoff_matches_latest(self, two_fight_db):
        latest = compute_live_career_stats(two_fight_db, "Fighter A")
        after = compute_live_career_stats(two_fight_db, "Fighter A", before_date="2099-01-01")
        assert latest["wins"] == after["wins"] and latest["splm"] == pytest.approx(after["splm"])

    def test_zone_stats_cover_every_prior_fight(self, two_fight_db):
        # rolling.py formulas over fights before the cutoff, including the most
        # recent one (reading the stored row for that fight lagged one fight)
        s = compute_live_career_stats(two_fight_db, "Fighter A", before_date="2025-06-02")
        assert s["head_acc"] == pytest.approx(25 / 50 * 100)
        assert s["head_def"] == pytest.approx((40 - 19) / 40 * 100)   # opponents: 10/30 + 9/10
        s = compute_live_career_stats(two_fight_db, "Fighter A", before_date="2025-06-01")
        assert s["head_acc"] == pytest.approx(20 / 40 * 100)

    def test_stance_returned(self, two_fight_db):
        # build_feature_vector() reads it for southpaw_adv_diff; missing -> always 0
        s = compute_live_career_stats(two_fight_db, "Fighter A")
        assert s["stance"] == "Southpaw"

    def test_age_taken_at_cutoff(self, two_fight_db):
        s = compute_live_career_stats(two_fight_db, "Fighter A", before_date="2020-01-01")
        assert s is None  # no fights before 2020 -> debut
        s = compute_live_career_stats(two_fight_db, "Fighter A", before_date="2025-06-01")
        assert s["age"] == pytest.approx(35.4, abs=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Stored training data is pre-fight only -- needs the UFCStats DB
# ══════════════════════════════════════════════════════════════════════════════

_MASTER = RAW_DIR / "ufc-master.csv"


@pytest.mark.skipif(not DB_UFCSTATS_PATH.exists() or not _MASTER.exists(),
                    reason="UFCStats DB not available (CI downloads no artifacts)")
class TestStoredCareerStatsArePreFight:
    """A random sample of ufc-master.csv rows must equal the live pre-fight
    value, not the post-fight one -- the check that would have caught the
    Kaggle leak."""

    COLS = {"avg_TD_pct": "avg_td_pct", "avg_SIG_STR_pct": "avg_sig_str_pct",
            "avg_SIG_STR_landed": "splm", "avg_TD_landed": "td_avg", "wins": "wins",
            "total_rounds_fought": "total_rounds_fought"}

    def test_sample_matches_pre_fight(self):
        df = pd.read_csv(_MASTER, low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        sample = df[df["date"] >= "2018-01-01"].sample(150, random_state=0)
        conn = sqlite3.connect(DB_UFCSTATS_PATH)
        canonical = {r[0].lower(): r[0] for r in conn.execute("SELECT name FROM fighters")}
        checked = mismatched = 0
        for _, row in sample.iterrows():
            key = str(row["R_fighter"]).lower().strip()
            name = NAME_ALIASES.get(key) or canonical.get(key)
            if name is None:
                continue
            pre = compute_live_career_stats(conn, name, before_date=row["date"].strftime("%Y-%m-%d"))
            if pre is None:
                continue
            checked += 1
            for col, key in self.COLS.items():
                if not np.isclose(float(row[f"R_{col}"]), pre[key], rtol=1e-3, atol=1e-3):
                    mismatched += 1
                    break
        conn.close()
        assert checked >= 100
        assert mismatched / checked <= 0.02, f"{mismatched}/{checked} rows differ from the pre-fight value"
