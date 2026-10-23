"""
Tests for scripts/build_feed.py -- predictions/feed.json, the public contract read by match-analyzer.
Artifact-free: builds a throwaway predictions/ dir.
"""
import json

from scripts import build_feed, score_event
from tests.test_refresh import _PREDICTIONS, _RESULTS, _UNSCORED_MD


def _event(tmp_path, slug, date, md=None):
    d = tmp_path / slug
    d.mkdir()
    fights = [{**_PREDICTIONS[0], "finish": [0.6, 0.3, 0.1], "elo_red": 1550.0, "elo_blue": 1500.0,
               "radar": {"axes": []}}]
    (d / f"{slug}.json").write_text(json.dumps({"event": f"UFC {slug}", "date": date, "event_url": "u",
                                                "model_type": "ensemble", "fights": fights}), encoding="utf-8")
    if md is not None:
        (d / f"{slug}.md").write_text(md, encoding="utf-8")
    return d


class TestBuildFeed:
    def test_latest_event_and_track_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(score_event, "PREDICTIONS_DIR", tmp_path)
        old = _event(tmp_path, "2026-10-03-old", "2026-10-03", _UNSCORED_MD)
        _event(tmp_path, "2026-10-10-new", "2026-10-10")
        # Ann Red was the -198 favorite, the model picked her and she won
        score_event.score_markdown(old / "2026-10-03-old.md", _PREDICTIONS, _RESULTS,
                                   {"Ann Red vs Bea Blue": (-198, 166)})

        monkeypatch.setattr(build_feed, "TRACK_RECORD_SINCE", "2026-01-01")
        feed = json.loads(build_feed.write_feed().read_text(encoding="utf-8"))

        assert feed["schema"] == build_feed.SCHEMA == 1
        assert feed["finish_names"] == ["Decision", "KO/TKO", "Submission"]
        assert feed["event"]["date"] == "2026-10-10"          # the newest event, not the scored one
        fight = feed["event"]["fights"][0]
        assert set(fight) == set(build_feed.FIGHT_FIELDS)     # trimmed to the contract's fields
        assert (fight["red_prob"], fight["blue_prob"], fight["elo_red"]) == (70.0, 30.0, 1550.0)
        tr = feed["track_record"]
        assert (tr["events"], tr["fights"], tr["model_acc"], tr["favorite_acc"]) == (1, 1, 1.0, 1.0)
        # EUR 1 on a -198 winner (decimal 1.50 as score_markdown rounds it) returns +0.50
        assert (tr["model_staked"], tr["model_pl"], tr["model_roi"]) == (1.0, 0.5, 0.5)
        ev = feed["events"]
        assert [e["date"] for e in ev] == ["2026-10-03"]
        assert (ev[0]["model_ok"], ev[0]["n"], ev[0]["fav_ok"], ev[0]["fav_n"]) == (1, 1, 1, 1)
        assert ev[0]["fights"] == [{"red_name": "Ann Red", "blue_name": "Bea Blue", "pick": "Ann Red",
                                    "confidence": 0.7, "odds": [-198, 166], "result": ev[0]["fights"][0]["result"],
                                    "correct": True}]
        assert ev[0]["fights"][0]["result"].startswith("Ann Red")

    def test_track_record_ignores_events_before_cutoff(self, tmp_path, monkeypatch):
        monkeypatch.setattr(score_event, "PREDICTIONS_DIR", tmp_path)
        d = _event(tmp_path, "2026-10-03-old", "2026-10-03", _UNSCORED_MD)
        score_event.score_markdown(d / "2026-10-03-old.md", _PREDICTIONS, _RESULTS,
                                   {"Ann Red vs Bea Blue": (-198, 166)})
        assert build_feed.track_record(since="2026-10-04") is None
        assert build_feed.track_record(since="2026-10-03")["events"] == 1

    def test_empty_predictions_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(score_event, "PREDICTIONS_DIR", tmp_path)
        feed = json.loads(build_feed.write_feed().read_text(encoding="utf-8"))
        assert feed["event"] is None and feed["track_record"] is None and feed["events"] == []

    def test_summary_totals_match_summary_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(score_event, "PREDICTIONS_DIR", tmp_path)
        monkeypatch.setattr(score_event, "SUMMARY_PATH", tmp_path / "SUMMARY.md")
        d = _event(tmp_path, "2026-10-03-old", "2026-10-03", _UNSCORED_MD)
        score_event.score_markdown(d / "2026-10-03-old.md", _PREDICTIONS, _RESULTS,
                                   {"Ann Red vs Bea Blue": (-198, 166)})
        text = score_event.write_summary().read_text(encoding="utf-8")
        assert "**1 events: model 1/1 (100.0%)" in text
        assert score_event.summary_totals()["events"] == 1
