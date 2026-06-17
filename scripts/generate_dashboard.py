"""
generate_dashboard.py -- Regenerate the interactive HTML dashboard from a saved
fight data JSON (produced alongside each predictions/*.md by predict_event.py).

Usage:
    python scripts/generate_dashboard.py predictions/ufc-fight-night-kape-vs-horiguchi-2026-06-20.json
    python scripts/generate_dashboard.py predictions/ufc-fight-night-kape-vs-horiguchi-2026-06-20.json --open
    python scripts/generate_dashboard.py predictions/    # regenerate all JSONs in the folder
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.predict_event import generate_html, _format_date, MODEL_LABELS


def _load_json(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # Reconstruct the minimal event dict generate_html expects
    event = {"name": data["event"], "date": data["date"]}
    # Reconstruct results list from pre-serialised fight dicts
    results = []
    for f in data["fights"]:
        results.append({
            "red_name":    f["red_name"],
            "blue_name":   f["blue_name"],
            "winner":      f["winner"],
            "red_prob":    f["red_prob"]  / 100,
            "blue_prob":   f["blue_prob"] / 100,
            "confidence":  max(f["red_prob"], f["blue_prob"]) / 100,
            "elo_red":     f["elo_red"],
            "elo_blue":    f["elo_blue"],
            "finish_proba": f["finish"] or None,
            "form_red": {
                "recent_win_rate":    f["form"]["red_streak"],   # placeholder — not used by HTML
                "recent_finish_rate": f["form"]["red_finish"] / 100,
                "win_streak":         f["form"]["red_streak"],
            },
            "form_blue": {
                "recent_win_rate":    f["form"]["blue_streak"],
                "recent_finish_rate": f["form"]["blue_finish"] / 100,
                "win_streak":         f["form"]["blue_streak"],
            },
            "stats_red":  {
                "str_acc":  f["radar"]["red"][1]  / 100,
                "str_def":  f["radar"]["red"][2]  / 100,
                "td_acc":   f["radar"]["red"][3]  / 100,
                "splm":     f["radar"]["red"][4],
                "win_rate": f["radar"]["red"][5]  / 100,
            },
            "stats_blue": {
                "str_acc":  f["radar"]["blue"][1] / 100,
                "str_def":  f["radar"]["blue"][2] / 100,
                "td_acc":   f["radar"]["blue"][3] / 100,
                "splm":     f["radar"]["blue"][4],
                "win_rate": f["radar"]["blue"][5] / 100,
            },
            # The raw JSON already has all chart data; generate_html re-serialises it
            "_raw": f,
        })
    return event, results, data.get("model_type", "ensemble")


def regenerate(json_path: Path, open_after: bool = False) -> Path:
    event, results, model_type = _load_json(json_path)
    html_path = json_path.with_suffix(".html")
    generate_html(event, results, model_type, html_path)
    print(f"Dashboard written: {html_path}")
    if open_after:
        import subprocess, os
        os.startfile(str(html_path)) if sys.platform == "win32" else subprocess.run(["open", str(html_path)])
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate interactive HTML from prediction JSON.")
    parser.add_argument("target", help="Path to a .json file or a predictions/ directory")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in browser after generation")
    args = parser.parse_args()

    target = Path(args.target)

    if target.is_dir():
        jsons = sorted(target.glob("*.json"))
        if not jsons:
            print(f"[WARN] No .json files found in {target}")
            sys.exit(0)
        for j in jsons:
            regenerate(j, open_after=False)
        print(f"\nRegenerated {len(jsons)} dashboard(s).")
    elif target.suffix == ".json":
        regenerate(target, open_after=args.open)
    elif target.suffix == ".md":
        json_path = target.with_suffix(".json")
        if not json_path.exists():
            print(f"[ERROR] No companion JSON found at {json_path}")
            sys.exit(1)
        regenerate(json_path, open_after=args.open)
    else:
        print(f"[ERROR] Expected a .json or .md file, got: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
