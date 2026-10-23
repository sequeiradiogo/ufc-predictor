"""
build_feed.py -- Write predictions/feed.json, the machine-readable summary of the latest event.

Read by external apps (match-analyzer's UFC tab) straight from GitHub's raw URL, so they
never parse markdown or guess slugs. It is a public contract: add fields freely, but bump
SCHEMA before removing or renaming one.

Called at the end of predict_event.py and score_event.py (both workflows commit predictions/).

Usage:
  python scripts/build_feed.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import FINISH_CLASS_NAMES
from scripts import score_event

SCHEMA = 1
SUMMARY_URL = "https://github.com/sequeiradiogo/ufc-predictor/blob/main/predictions/SUMMARY.md"
# The track record only counts events predicted by the fixed model: the 2026-10-19..21 retrains
# (Kaggle leak, calibration window, feature pruning) changed it enough that earlier events say
# little about it. SUMMARY.md keeps the full history.
TRACK_RECORD_SINCE = "2026-10-24"
FIGHT_FIELDS = ("red_name", "blue_name", "red_prob", "blue_prob", "winner",
                "finish_str", "finish", "elo_red", "elo_blue", "radar", "bars", "form")


def latest_event(predictions_dir: Path) -> dict | None:
    """The prediction JSON with the latest event date, trimmed to the feed's fields."""
    best = None
    for path in predictions_dir.glob("*/*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or "fights" not in meta:
            continue
        if best is None or meta.get("date", "") > best.get("date", ""):
            best = meta
    if best is None:
        return None
    return {
        "name":       best.get("event"),
        "date":       best.get("date"),
        "event_url":  best.get("event_url"),
        "model_type": best.get("model_type"),
        "fights":     [{k: f.get(k) for k in FIGHT_FIELDS} for f in best["fights"]],
    }


def track_record(min_confidence: float = score_event.MIN_CONFIDENCE,
                 since: str | None = None) -> dict | None:
    """Model vs bookmaker favorite over the scored events dated `since` (default TRACK_RECORD_SINCE)
    or later -- SUMMARY.md's counts over those events."""
    since = since or TRACK_RECORD_SINCE
    events = [e for e in score_event._scored_events(min_confidence) if e[0].parent.name[:10] >= since]
    t = score_event.summary_totals(min_confidence, events)
    if not t["events"]:
        return None
    ratio = lambda a, b: round(a / b, 4) if b else None
    return {
        "since":          since,
        "events":         t["events"],
        "fights":         t["n"],
        "model_acc":      ratio(t["model_ok"], t["n"]),
        "fights_with_odds": t["fav_n"],
        "model_acc_with_odds": ratio(t["fav_model_ok"], t["fav_n"]),
        "favorite_acc":   ratio(t["fav_ok"], t["fav_n"]),
        "min_confidence": min_confidence,
        "model_pl":       round(t["pl"], 2),
        "model_staked":   t["staked"],
        "model_roi":      ratio(t["pl"], t["staked"]),
        "favorite_roi":   ratio(t["fav_pl"], t["fav_staked"]),
    }


def scored_events(min_confidence: float = score_event.MIN_CONFIDENCE, since: str | None = None) -> list[dict]:
    """Per-event results since `since` (default TRACK_RECORD_SINCE), newest first: the model's and the
    favorite's hits plus every fight's pick, result and closing odds."""
    since = since or TRACK_RECORD_SINCE
    out = []
    for md_path, title, rec in score_event._scored_events(min_confidence):
        date = md_path.parent.name[:10]
        if date < since:
            continue
        out.append({"date": date, "name": title,
                    "model_ok": rec["model_ok"], "n": rec["n"],
                    "fav_model_ok": rec["fav_model_ok"], "fav_ok": rec["fav_ok"], "fav_n": rec["fav_n"],
                    "fights": score_event.event_fights(md_path.read_text(encoding="utf-8"))})
    return sorted(out, key=lambda e: e["date"], reverse=True)


def write_feed(min_confidence: float = score_event.MIN_CONFIDENCE) -> Path:
    predictions_dir = score_event.PREDICTIONS_DIR
    feed = {
        "schema":       SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "finish_names": FINISH_CLASS_NAMES,
        "event":        latest_event(predictions_dir),
        "track_record": track_record(min_confidence),
        "events":       scored_events(min_confidence),
        "summary_url":  SUMMARY_URL,
    }
    path = predictions_dir / "feed.json"
    path.write_text(json.dumps(feed, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"Wrote {write_feed()}")
