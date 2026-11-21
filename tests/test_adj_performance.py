"""
tests/test_adj_performance.py
=============================
Artifact-free tests for ml/adj_performance.py (opponent-adjusted landing ratio,
issue #68): pre-fight only, debut = 0, hand-computed value, opponent history
used, and training (prefight_adj_land_ratio) == live (current_adj_land_ratio).
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.adj_performance import (
    CLIP, DIV_MAD, FEATURE, K_OPP, PRIOR_MEAN, S_FIGHTER,
    current_adj_land_ratio, prefight_adj_land_ratio,
)

_DIV = "lightweight"


def _make_db(rows_fights, rows_stats):
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE fights (fight_id TEXT, date TEXT, division TEXT, match_time_sec INTEGER, finish_round INTEGER);
        CREATE TABLE fight_stats (fight_id TEXT, fighter_id TEXT, sig_str_landed INTEGER);
    """)
    conn.executemany("INSERT INTO fights VALUES (?, ?, ?, ?, ?)", rows_fights)
    conn.executemany("INSERT INTO fight_stats VALUES (?, ?, ?)", rows_stats)
    return conn


@pytest.fixture
def db():
    """f1: a beats debutant b 30-10; f2: c beats b 40-20; f3: a vs c, later."""
    conn = _make_db(
        [("f1", "2025-01-01", _DIV, 300, 3), ("f2", "2025-03-01", _DIV, 300, 3),
         ("f3", "2025-06-01", _DIV, 300, 3)],
        [("f1", "a", 30), ("f1", "b", 10), ("f2", "c", 40), ("f2", "b", 20),
         ("f3", "a", 25), ("f3", "c", 25)],
    )
    yield conn
    conn.close()


def _pre(conn):
    return prefight_adj_land_ratio(conn).set_index(["fighter_id", "fight_id"])[FEATURE]


def test_debut_is_zero(db):
    pre = _pre(db)
    assert pre[("a", "f1")] == 0.0 and pre[("b", "f1")] == 0.0
    assert current_adj_land_ratio(db, ["a", "zzz"], before_date="2025-01-01") == {
        "a": {FEATURE: 0.0}, "zzz": {FEATURE: 0.0},
    }


def test_hand_computed_value_vs_debutant_opponent(db):
    # Before f2, b has one fight (f1, landed 10 of 40 -> ratio 0.25) against a
    # debutant, whose baseline is the division prior: z = (0.25 - 0.5) / mad.
    mad = DIV_MAD[_DIV]
    z = (0.25 - PRIOR_MEAN) / mad
    # one fight: n_eff = 1 -> shrunk by 1 / (1 + S_FIGHTER)
    assert _pre(db)[("b", "f2")] == pytest.approx(z / (1 + S_FIGHTER))


def test_opponent_history_moves_the_baseline(db):
    # Going into f3, a's only fight is f1 against b, who had no prior fights at
    # f1 -> prior baseline. c's only fight (f2) was against b, who HAD history at
    # f2 (f1, where b's opponent a landed 75%): c's 0.667 in f2 is judged against
    # what b usually allows, so it counts for less than a's 0.75 against a debutant.
    pre = _pre(db)
    a_val, c_val = pre[("a", "f3")], pre[("c", "f3")]
    mad = DIV_MAD[_DIV]
    assert a_val == pytest.approx((0.75 - PRIOR_MEAN) / mad / (1 + S_FIGHTER))
    # b's history at f2: one fight, allowed 0.75, weight 0.5**(59/548), n_eff = 1
    a = 1.0 / (1.0 + K_OPP)
    mu = a * 0.75 + (1 - a) * PRIOR_MEAN
    sigma = max(a * 0.0 + (1 - a) * mad, 0.25 * mad)
    exp_c = np.clip((40 / 60 - mu) / sigma, -CLIP, CLIP) / (1 + S_FIGHTER)
    assert c_val == pytest.approx(exp_c)
    assert c_val < (40 / 60 - PRIOR_MEAN) / mad / (1 + S_FIGHTER)


def test_live_matches_training_and_excludes_the_fight(db):
    pre = _pre(db)
    live = current_adj_land_ratio(db, ["a", "c"], before_date="2025-06-01")
    assert live["a"][FEATURE] == pytest.approx(pre[("a", "f3")])
    assert live["c"][FEATURE] == pytest.approx(pre[("c", "f3")])
    # a later cutoff includes f3 (and a's value decays with the later date too)
    after = current_adj_land_ratio(db, ["a"], before_date="2025-06-02")
    assert after["a"][FEATURE] != pytest.approx(live["a"][FEATURE])


def test_later_fights_do_not_change_earlier_values(db):
    before = _pre(db)
    db.execute("INSERT INTO fights VALUES ('f4', '2025-09-01', ?, 300, 3)", (_DIV,))
    db.executemany("INSERT INTO fight_stats VALUES (?, ?, ?)", [("f4", "b", 50), ("f4", "a", 0)])
    after = _pre(db)
    for key in before.index:
        assert after[key] == pytest.approx(before[key]), key
