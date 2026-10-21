"""
tests/test_opp_adjusted_defense.py
==================================
Artifact-free tests for ml/opp_adjusted_defense.py: pre-fight only, debut = 0,
hand-computed values, and training (prefight_adj_defense) == live
(current_adj_defense) by construction.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.opp_adjusted_defense import (
    LEAGUE_SLPM, LEAGUE_STR_ACC, RESID_PSEUDO_MIN, current_adj_defense, prefight_adj_defense,
)


@pytest.fixture
def db():
    """a vs b (15 min, b lands 30/60 on a), then a vs c (5 min, c lands 10/20 on a)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE fights (fight_id TEXT, date TEXT, match_time_sec INTEGER, finish_round INTEGER);
        CREATE TABLE fight_stats (fight_id TEXT, fighter_id TEXT, sig_str_landed INTEGER, sig_str_atmpted INTEGER);
        INSERT INTO fights VALUES ('f1', '2025-01-01', 300, 3), ('f2', '2025-06-01', 300, 1);
        INSERT INTO fight_stats VALUES ('f1', 'a', 45, 90), ('f1', 'b', 30, 60),
                                       ('f2', 'a', 12, 30), ('f2', 'c', 10, 20);
    """)
    yield conn
    conn.close()


def test_debut_is_zero(db):
    pre = prefight_adj_defense(db).set_index(["fighter_id", "fight_id"])
    assert pre.loc[("a", "f1"), "adj_str_def"] == 0.0
    assert pre.loc[("a", "f1"), "adj_acc_def"] == 0.0
    assert current_adj_defense(db, ["a", "zzz"], before_date="2025-01-01") == {
        "a": {"adj_str_def": 0.0, "adj_acc_def": 0.0},
        "zzz": {"adj_str_def": 0.0, "adj_acc_def": 0.0},
    }


def test_hand_computed_value_excludes_current_fight(db):
    # Before f2, a has one fight: absorbed 30 in 15 min vs debutant b (league-average prior)
    pre = prefight_adj_defense(db).set_index(["fighter_id", "fight_id"])
    exp_str = (30 / 15 - LEAGUE_SLPM) * 15 / (15 + RESID_PSEUDO_MIN)
    exp_acc = (30 / 60 - LEAGUE_STR_ACC) * 15 / (15 + RESID_PSEUDO_MIN)
    assert pre.loc[("a", "f2"), "adj_str_def"] == pytest.approx(exp_str)
    assert pre.loc[("a", "f2"), "adj_acc_def"] == pytest.approx(exp_acc)


def test_live_matches_training(db):
    # Live value going into f2 must equal the stored pre-fight value for f2
    pre = prefight_adj_defense(db).set_index(["fighter_id", "fight_id"])
    live = current_adj_defense(db, ["a"], before_date="2025-06-01")["a"]
    for feat in ("adj_str_def", "adj_acc_def"):
        assert live[feat] == pytest.approx(pre.loc[("a", "f2"), feat])
    # ...and a later cutoff includes f2
    after = current_adj_defense(db, ["a"], before_date="2025-06-02")["a"]
    assert after["adj_str_def"] != pytest.approx(live["adj_str_def"])
