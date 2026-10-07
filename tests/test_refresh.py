"""
tests/test_refresh.py
=====================
Tests for the incremental refresh pipeline:
  - scraper helper functions (no HTTP)
  - _insert_new_data() against a temp SQLite DB
  - rolling.main(fighter_ids=...) incremental filter
"""

import re
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH


# ══════════════════════════════════════════════════════════════════════════════
# 1. UFCStats parser helpers
# ══════════════════════════════════════════════════════════════════════════════

from scrapers.ufcstats import (
    _parse_of,
    _ctrl_to_seconds,
    _fight_seconds,
    _parse_date,
    _normalize_division,
    _normalize_method,
    _id_from_url,
    _parse_height_cm,
    _parse_reach_cm,
)


class TestParseOf:
    def test_normal(self):
        assert _parse_of("50 of 100") == (50, 100)

    def test_zero(self):
        assert _parse_of("0 of 0") == (0, 0)

    def test_leading_space(self):
        assert _parse_of("  3 of 7  ") == (3, 7)

    def test_unparseable(self):
        assert _parse_of("--") == (0, 0)

    def test_empty(self):
        assert _parse_of("") == (0, 0)


class TestCtrlToSeconds:
    def test_normal(self):
        assert _ctrl_to_seconds("1:30") == 90

    def test_zero(self):
        assert _ctrl_to_seconds("0:00") == 0

    def test_longer(self):
        assert _ctrl_to_seconds("4:59") == 299

    def test_dash(self):
        assert _ctrl_to_seconds("--") == 0

    def test_empty(self):
        assert _ctrl_to_seconds("") == 0


class TestFightSeconds:
    def test_first_round_finish(self):
        # Round 1, 2:30 -> 150 seconds
        assert _fight_seconds(1, "2:30") == 150

    def test_third_round_finish(self):
        # Round 3, 4:32 -> 2*300 + 272 = 872
        assert _fight_seconds(3, "4:32") == 872

    def test_full_three_rounds(self):
        # Round 3, 5:00 -> decision
        assert _fight_seconds(3, "5:00") == 900

    def test_fifth_round(self):
        assert _fight_seconds(5, "5:00") == 1500


class TestParseDate:
    def test_with_period(self):
        assert _parse_date("May. 10, 2025") == "2025-05-10"

    def test_without_period(self):
        assert _parse_date("Jan 1, 2024") == "2024-01-01"

    def test_lowercase(self):
        assert _parse_date("march 22, 2025") == "2025-03-22"

    def test_bad_input(self):
        assert _parse_date("not a date") is None

    def test_zero_padded_day(self):
        assert _parse_date("Feb. 08, 2025") == "2025-02-08"


class TestNormalizeDivision:
    def test_lightweight(self):
        assert _normalize_division("Lightweight Bout") == "lightweight"

    def test_heavyweight(self):
        assert _normalize_division("HEAVYWEIGHT BOUT") == "heavyweight"

    def test_womens(self):
        assert _normalize_division("Women's Strawweight Bout") == "women's strawweight"

    def test_light_heavyweight(self):
        assert _normalize_division("Light Heavyweight Bout") == "light heavyweight"


class TestNormalizeMethod:
    def test_ko(self):
        assert _normalize_method("KO/TKO") == "KO/TKO"

    def test_tko(self):
        assert _normalize_method("TKO") == "KO/TKO"

    def test_submission(self):
        assert _normalize_method("Submission") == "Submission"

    def test_decision_unanimous(self):
        assert _normalize_method("Decision - Unanimous") == "Decision - Unanimous"

    def test_decision_split(self):
        assert _normalize_method("Decision - Split") == "Decision - Split"


class TestIdFromUrl:
    def test_fighter(self):
        assert _id_from_url("http://www.ufcstats.com/fighter-details/c2299ec916bc7c56") == "c2299ec916bc7c56"

    def test_fight(self):
        assert _id_from_url("http://www.ufcstats.com/fight-details/1c1afd4cf80c8506") == "1c1afd4cf80c8506"

    def test_trailing_slash(self):
        assert _id_from_url("http://www.ufcstats.com/fighter-details/abc123/") == "abc123"


class TestHeightReach:
    def test_height_six_one(self):
        cm = _parse_height_cm("6' 1\"")
        assert cm is not None
        assert abs(cm - 185.4) < 0.5

    def test_reach_72_inches(self):
        cm = _parse_reach_cm('72"')
        assert cm is not None
        assert abs(cm - 182.9) < 0.5

    def test_height_none(self):
        assert _parse_height_cm("--") is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. BFO parser helpers
# ══════════════════════════════════════════════════════════════════════════════

from scrapers.bestfightodds import _parse_american_odds, _name_key, _best_match


class TestParseAmericanOdds:
    def test_negative(self):
        assert _parse_american_odds("-150") == -150

    def test_positive(self):
        assert _parse_american_odds("+130") == 130

    def test_no_sign(self):
        assert _parse_american_odds("200") == 200

    def test_unicode_minus(self):
        assert _parse_american_odds("−150") == -150

    def test_invalid(self):
        assert _parse_american_odds("N/A") is None


class TestNameKey:
    def test_lowercase_strip(self):
        assert _name_key("Jon Jones") == "jon jones"

    def test_strip_punctuation(self):
        assert _name_key("Jon 'Bones' Jones") == "jon bones jones"

    def test_repeated_mononym(self):
        assert _name_key("Sumudaerji Sumudaerji") == "sumudaerji"


class TestBestMatch:
    def test_exact(self):
        assert _best_match("jon jones", ["jon jones", "israel adesanya"]) == "jon jones"

    def test_close(self):
        result = _best_match("jon jones", ["jon jones", "israel adesanya"])
        assert result == "jon jones"

    def test_no_match(self):
        assert _best_match("xyz qrs", ["jon jones", "israel adesanya"]) is None


# Trimmed from the real BFO UFC 332 page (2026-10): two exchange columns
# (Polymarket, Kalshi) ahead of the sportsbooks, Kalshi already near-settled.
_BFO_EVENT_HTML = """
<div class="table-scroller"><table class="odds-table">
<thead><tr>
  <th></th>
  <th data-b="28"><a>Polymarket</a>$20 Bonus</th>
  <th data-b="29"><a>Kalshi</a>Up to $500</th>
  <th data-b="21"><a>FanDuel</a></th>
  <th data-b="24"><a>Caesars</a></th>
  <th class="table-prop-header">Props</th>
</tr></thead>
<tbody>
  <tr><th><a><span class="t-b-fcc">Natalia Silva</span></a></th>
    <td></td>
    <td class="but-sg" data-li="[29,1,1]"><span id="a1">-3487</span></td>
    <td class="but-sg" data-li="[21,1,1]"><span id="a2">-198</span></td>
    <td class="but-sg" data-li="[24,1,1]"><span id="a3">-210</span></td>
    <td class="prop-cell"></td></tr>
  <tr><th><a><span class="t-b-fcc">Wang Cong</span></a></th>
    <td></td>
    <td class="but-sg" data-li="[29,2,1]"><span id="b1">+2236</span></td>
    <td class="but-sg" data-li="[21,2,1]"><span id="b2">+166</span></td>
    <td class="but-sg" data-li="[24,2,1]"><span id="b3">+175</span></td>
    <td class="prop-cell"></td></tr>
  <tr><th><a><span class="t-b-fcc">Fighter A</span></a></th>
    <td></td>
    <td></td>
    <td class="but-sg" data-li="[21,1,2]"><span id="c1">-150</span></td>
    <td class="but-sg" data-li="[24,1,2]"><span id="c2">-160</span></td>
    <td class="prop-cell"></td></tr>
  <tr><th><a><span class="t-b-fcc">Fighter B</span></a></th>
    <td></td>
    <td class="but-sg" data-li="[29,2,2]"><span id="d1">+900</span></td>
    <td></td>
    <td class="but-sg" data-li="[24,2,2]"><span id="d2">+140</span></td>
    <td class="prop-cell"></td></tr>
  <tr><th><a><span class="t-b-fcc">Exchange Only</span></a></th>
    <td></td>
    <td class="but-sg" data-li="[29,1,3]"><span id="e1">+105</span></td>
    <td></td><td></td><td class="prop-cell"></td></tr>
  <tr><th><a><span class="t-b-fcc">Exchange Other</span></a></th>
    <td></td>
    <td class="but-sg" data-li="[29,2,3]"><span id="f1">-126</span></td>
    <td></td><td></td><td class="prop-cell"></td></tr>
</tbody></table></div>
"""


class TestScrapeEventOdds:
    @pytest.fixture
    def matchups(self, monkeypatch):
        from bs4 import BeautifulSoup
        import scrapers.bestfightodds as bfo
        monkeypatch.setattr(bfo, "_get", lambda url: BeautifulSoup(_BFO_EVENT_HTML, "lxml"))
        return {m["r_name"]: m for m in bfo._scrape_event_odds("https://example/events/x")}

    def test_skips_prediction_market_columns(self, matchups):
        m = matchups["Natalia Silva"]
        assert (m["odds_red"], m["odds_blue"]) == (-198, 166)

    def test_pairs_by_book_not_position(self, matchups):
        # FanDuel only prices Fighter A, so both sides come from Caesars.
        m = matchups["Fighter A"]
        assert (m["odds_red"], m["odds_blue"]) == (-160, 140)

    def test_exchange_only_fight_has_no_odds(self, matchups):
        m = matchups["Exchange Only"]
        assert (m["odds_red"], m["odds_blue"]) == (None, None)


class TestParseFighterEvents:
    def test_event_headers(self):
        from bs4 import BeautifulSoup
        from scrapers.bestfightodds import _parse_fighter_events
        html = """<table class="team-stats-table">
          <tr class="event-header item-mobile-only-row"><td colspan="8" scope="row">
            <a href="/events/ufc-vegas-119-4226">UFC Vegas 119</a> Jun 21st 2026</td></tr>
          <tr class="main-row"><th><a href="/fighters/manel-kape-7510">Manel Kape</a></th></tr>
        </table>"""
        events = _parse_fighter_events(BeautifulSoup(html, "lxml"))
        assert events == [{
            "name":      "UFC Vegas 119",
            "url":       "https://www.bestfightodds.com/events/ufc-vegas-119-4226",
            "date_text": "Jun 21st 2026",
        }]


# ══════════════════════════════════════════════════════════════════════════════
# 2b. score_event.py re-scoring
# ══════════════════════════════════════════════════════════════════════════════

_UNSCORED_MD = """# UFC Test -- October 3, 2026

Model: Ensemble (Soft Vote) | Generated: 2026-10-02

Fighters making their UFC debut were excluded (no historical stats in DB).

> Interactive fighter comparison: [x.html](./x.html)

---

## Predictions

| Fight | Predicted Winner | Confidence | Likely Method |
|---|---|---|---|
| Ann Red vs Bea Blue | Ann Red | 70.0% | Decision (60%) / KO/TKO (30%) |

---

## Raw Model Output

### Ann Red vs Bea Blue
- Ensemble (Soft Vote): Ann Red 70.0% | Bea Blue 30.0%
"""

_PREDICTIONS = [{
    "red_name": "Ann Red", "blue_name": "Bea Blue", "winner": "Ann Red",
    "red_prob": 70.0, "blue_prob": 30.0, "finish_str": "Decision (60%) / KO/TKO (30%)",
}]
_RESULTS = [{"winner": "Ann Red", "loser": "Bea Blue", "method": "Decision - Unanimous", "round": 3}]


class TestRescore:
    def test_strip_undoes_scoring(self, tmp_path):
        from scripts.score_event import score_markdown, _strip_scoring
        md = tmp_path / "e.md"
        md.write_text(_UNSCORED_MD, encoding="utf-8")
        score_markdown(md, _PREDICTIONS, _RESULTS, {"Ann Red vs Bea Blue": (-3487, 2236)})
        table = re.compile(r"\| Fight \|.*?\n\|[-|]+\|\n.*?(?=\n\n|\n---)", re.DOTALL)
        stripped = _strip_scoring(md.read_text(encoding="utf-8"))
        assert table.sub("", stripped) == table.sub("", _UNSCORED_MD)

    def test_rescore_replaces_odds_and_summary(self, tmp_path):
        from scripts.score_event import score_markdown
        md = tmp_path / "e.md"
        md.write_text(_UNSCORED_MD, encoding="utf-8")
        score_markdown(md, _PREDICTIONS, _RESULTS, {"Ann Red vs Bea Blue": (-3487, 2236)})
        assert score_markdown(md, _PREDICTIONS, _RESULTS, {}) == "already scored"

        score_markdown(md, _PREDICTIONS, _RESULTS, {"Ann Red vs Bea Blue": (-198, 166)},
                       rescore=True)
        text = md.read_text(encoding="utf-8")
        assert "-198 / +166" in text and "-3487" not in text
        assert text.count("## Post-Event Summary") == 1
        assert text.count("**Result:") == 1
        assert text.count("| Scored:") == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. _insert_new_data against a temp DB
# ══════════════════════════════════════════════════════════════════════════════

from scripts.refresh_data import _insert_new_data

_FIGHTERS_DDL = """
CREATE TABLE fighters (
    fighter_id TEXT PRIMARY KEY,
    name TEXT, height REAL, reach REAL, stance TEXT, dob TEXT,
    splm REAL, sapm REAL, str_def REAL, td_avg REAL
)
"""
_FIGHTS_DDL = """
CREATE TABLE fights (
    fight_id TEXT PRIMARY KEY,
    event_id TEXT, date TEXT, division TEXT,
    r_fighter_id TEXT, b_fighter_id TEXT, winner_id TEXT,
    method TEXT, title_fight INTEGER,
    odds_red REAL, odds_blue REAL
)
"""
_STATS_DDL = """
CREATE TABLE fight_stats (
    fight_id TEXT, fighter_id TEXT,
    corner TEXT,
    kd INTEGER,
    sig_str_landed INTEGER, sig_str_atmpted INTEGER,
    total_str_landed INTEGER, total_str_atmpted INTEGER,
    td_landed INTEGER, td_atmpted INTEGER,
    sub_att INTEGER, ctrl INTEGER,
    head_landed INTEGER, head_atmpted INTEGER,
    body_landed INTEGER, body_atmpted INTEGER,
    leg_landed INTEGER, leg_atmpted INTEGER,
    dist_landed INTEGER, dist_atmpted INTEGER,
    clinch_landed INTEGER, clinch_atmpted INTEGER,
    ground_landed INTEGER, ground_atmpted INTEGER,
    total_fight_time INTEGER,
    PRIMARY KEY (fight_id, fighter_id)
)
"""


@pytest.fixture
def temp_db():
    conn = sqlite3.connect(":memory:")
    conn.execute(_FIGHTERS_DDL)
    conn.execute(_FIGHTS_DDL)
    conn.execute(_STATS_DDL)
    conn.commit()
    yield conn
    conn.close()


_SAMPLE_DATA = {
    "fighters": [
        {"fighter_id": "aaa", "name": "Fighter A", "height": 180.0, "reach": 185.0, "stance": "Orthodox", "dob": "1990-01-01"},
        {"fighter_id": "bbb", "name": "Fighter B", "height": 175.0, "reach": 180.0, "stance": "Southpaw",  "dob": "1992-03-15"},
    ],
    "fights": [
        {
            "fight_id": "fff", "event_id": "eee", "date": "2026-05-30",
            "division": "lightweight", "r_fighter_id": "aaa", "b_fighter_id": "bbb",
            "winner_id": "aaa", "method": "KO/TKO", "title_fight": 0,
            "odds_red": -150, "odds_blue": 130,
        }
    ],
    "fight_stats": [
        {
            "fight_id": "fff", "fighter_id": "aaa", "corner": "r",
            "kd": 1, "sig_str_landed": 50, "sig_str_atmpted": 80,
            "total_str_landed": 60, "total_str_atmpted": 90,
            "td_landed": 2, "td_atmpted": 3, "sub_att": 0, "ctrl": 90,
            "head_landed": 30, "head_atmpted": 40,
            "body_landed": 10, "body_atmpted": 20,
            "leg_landed": 10, "leg_atmpted": 20,
            "dist_landed": 45, "dist_atmpted": 70,
            "clinch_landed": 5, "clinch_atmpted": 10,
            "ground_landed": 0, "ground_atmpted": 0,
            "total_fight_time": 150,
        },
        {
            "fight_id": "fff", "fighter_id": "bbb", "corner": "b",
            "kd": 0, "sig_str_landed": 30, "sig_str_atmpted": 60,
            "total_str_landed": 35, "total_str_atmpted": 65,
            "td_landed": 0, "td_atmpted": 1, "sub_att": 0, "ctrl": 0,
            "head_landed": 20, "head_atmpted": 35,
            "body_landed": 5,  "body_atmpted": 15,
            "leg_landed": 5,   "leg_atmpted": 10,
            "dist_landed": 28, "dist_atmpted": 55,
            "clinch_landed": 2, "clinch_atmpted": 5,
            "ground_landed": 0, "ground_atmpted": 0,
            "total_fight_time": 150,
        },
    ],
}


class TestInsertNewData:
    def test_inserts_fighters(self, temp_db):
        _insert_new_data(_SAMPLE_DATA, temp_db)
        count = temp_db.execute("SELECT COUNT(*) FROM fighters").fetchone()[0]
        assert count == 2

    def test_inserts_fights(self, temp_db):
        _insert_new_data(_SAMPLE_DATA, temp_db)
        count = temp_db.execute("SELECT COUNT(*) FROM fights").fetchone()[0]
        assert count == 1

    def test_inserts_fight_stats(self, temp_db):
        _insert_new_data(_SAMPLE_DATA, temp_db)
        count = temp_db.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0]
        assert count == 2

    def test_returns_affected_fighter_ids(self, temp_db):
        affected = _insert_new_data(_SAMPLE_DATA, temp_db)
        assert affected == {"aaa", "bbb"}

    def test_idempotent(self, temp_db):
        # Running twice should not duplicate rows (INSERT OR IGNORE)
        _insert_new_data(_SAMPLE_DATA, temp_db)
        _insert_new_data(_SAMPLE_DATA, temp_db)
        count = temp_db.execute("SELECT COUNT(*) FROM fighters").fetchone()[0]
        assert count == 2

    def test_odds_stored(self, temp_db):
        _insert_new_data(_SAMPLE_DATA, temp_db)
        row = temp_db.execute("SELECT odds_red, odds_blue FROM fights WHERE fight_id='fff'").fetchone()
        assert row == (-150, 130)

    def test_empty_data_no_error(self, temp_db):
        affected = _insert_new_data({"fighters": [], "fights": [], "fight_stats": []}, temp_db)
        assert affected == set()


# ══════════════════════════════════════════════════════════════════════════════
# 4. rolling.main(fighter_ids=...) incremental filter
# ══════════════════════════════════════════════════════════════════════════════

class TestRollingIncremental:
    def test_main_accepts_fighter_ids(self):
        """rolling.main() should accept fighter_ids without raising a TypeError."""
        from db import rolling
        import inspect
        sig = inspect.signature(rolling.main)
        assert "fighter_ids" in sig.parameters

    def test_fighter_ids_default_none(self):
        from db import rolling
        import inspect
        sig = inspect.signature(rolling.main)
        assert sig.parameters["fighter_ids"].default is None

    @pytest.mark.skipif(not DB_PATH.exists(), reason="DB not found")
    def test_incremental_does_not_crash(self):
        """Run rolling.main() for a single known fighter -- should complete without error."""
        from db import rolling
        conn = sqlite3.connect(str(DB_PATH))
        # rolling.py requires per-fight strike columns; mdabbert DBs store career averages only
        db_cols = {row[1] for row in conn.execute("PRAGMA table_info(fight_stats)").fetchall()}
        if "body_landed" not in db_cols:
            conn.close()
            pytest.skip("DB missing per-fight strike columns (mdabbert format) -- rolling N/A")
        row = conn.execute(
            "SELECT fighter_id FROM fight_stats GROUP BY fighter_id HAVING COUNT(*)>=3 LIMIT 1"
        ).fetchone()
        conn.close()
        if row is None:
            pytest.skip("No fighters with 3+ fights in DB")
        rolling.main(fighter_ids={row[0]})


# ══════════════════════════════════════════════════════════════════════════════
# 5. refresh_data helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestRefreshHelpers:
    @pytest.mark.skipif(not DB_PATH.exists(), reason="DB not found")
    def test_get_last_event_date_returns_date(self):
        from scripts.refresh_data import get_last_event_date
        d = get_last_event_date()
        assert d is not None
        assert isinstance(d, date)

    @pytest.mark.skipif(not DB_PATH.exists(), reason="DB not found")
    def test_last_event_date_is_recent(self):
        from scripts.refresh_data import get_last_event_date
        d = get_last_event_date()
        assert d >= date(2020, 1, 1)

    @pytest.mark.skipif(not DB_PATH.exists(), reason="DB not found")
    def test_existing_fighter_ids_non_empty(self):
        from scripts.refresh_data import _existing_fighter_ids
        conn = sqlite3.connect(str(DB_PATH))
        ids = _existing_fighter_ids(conn)
        conn.close()
        assert len(ids) > 100
