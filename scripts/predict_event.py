"""
predict_event.py -- Scrape the next UFC event and generate predictions.

Usage:
    python scripts/predict_event.py
    python scripts/predict_event.py --model ensemble
    python scripts/predict_event.py --output predictions/my-event.md

Steps:
    1. Scrape the next upcoming event card from UFCStats (Playwright required).
    2. Look up fighters in the v2 DB by UFCStats ID (debut check + name normalisation).
    3. Skip fights where either fighter has no recorded fight history (debut).
    4. Run v1 (mdabbert) prediction for each fight.
    5. Write a Markdown file to the predictions/ folder.
"""

import argparse
import re
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH, DB_V1_PATH, MODELS_DIR, MODELS_V1_DIR, FINISH_CLASS_NAMES
from predict import compute_prediction, get_latest_stats
from scrapers.ufcstats import scrape_upcoming_event
from utils.logger import get_logger

log = get_logger(__name__)

PREDICTIONS_DIR = ROOT_DIR / "predictions"

MODEL_LABELS = {
    "xgb":      "XGBoost",
    "lr":       "Logistic Regression",
    "rf":       "Random Forest",
    "lgbm":     "LightGBM",
    "ensemble": "Ensemble (Soft Vote)",
}


# ── DB helpers ────────────────────────────────────────────────────────────────

def lookup_fighter(conn: sqlite3.Connection, fighter_id: str) -> tuple[str, str] | None:
    """Return (fighter_id, name) from DB, or None if not found."""
    row = conn.execute(
        "SELECT fighter_id, name FROM fighters WHERE fighter_id = ?",
        (fighter_id,),
    ).fetchone()
    return row if row else None


def has_fight_history(conn: sqlite3.Connection, fighter_id: str) -> bool:
    """Return True if the fighter has at least one fight_stats row."""
    row = conn.execute(
        "SELECT 1 FROM fight_stats WHERE fighter_id = ? LIMIT 1",
        (fighter_id,),
    ).fetchone()
    return row is not None


# ── Markdown formatter ────────────────────────────────────────────────────────

def _event_slug(name: str, event_date: str) -> str:
    """'UFC 317' + '2026-06-07' -> 'ufc-317-2026-06-07'"""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-{event_date}"


def _format_finish(finish_proba: list[float] | None) -> str:
    if not finish_proba:
        return "N/A"
    pairs = sorted(zip(FINISH_CLASS_NAMES, finish_proba), key=lambda x: -x[1])
    parts = [f"{name} ({p:.0%})" for name, p in pairs[:2]]
    return " / ".join(parts)


def _format_date(iso: str) -> str:
    """'2026-06-07' -> 'June 7, 2026'"""
    try:
        d = date.fromisoformat(iso)
        return f"{d.strftime('%B')} {d.day}, {d.year}"
    except Exception:
        return iso


def build_markdown(
    event: dict,
    results: list[dict],
    model_type: str,
) -> str:
    model_label    = MODEL_LABELS.get(model_type, model_type)
    generated      = date.today().isoformat()
    event_date_fmt = _format_date(event["date"])

    lines = [
        f"# {event['name']} -- {event_date_fmt}",
        "",
        f"Model: {model_label} | Generated: {generated}",
        "",
        "Fighters making their UFC debut were excluded (no historical stats in DB).",
        "",
        "---",
        "",
        "## Predictions",
        "",
        "| Fight | Predicted Winner | Confidence | Likely Method |",
        "|---|---|---|---|",
    ]

    for r in results:
        fight_label = f"{r['red_name']} vs {r['blue_name']}"
        finish_str  = _format_finish(r["finish_proba"])
        lines.append(
            f"| {fight_label} | {r['winner']} | {r['confidence']:.1%} | {finish_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Raw Model Output",
        "",
    ]

    for r in results:
        form_r     = r["form_red"]
        form_b     = r["form_blue"]
        finish_str = _format_finish(r["finish_proba"])

        lines += [
            f"### {r['red_name']} vs {r['blue_name']}",
            f"- ELO: {r['red_name']} {r['elo_red']:.0f} | {r['blue_name']} {r['elo_blue']:.0f}",
            (
                f"- Recent form: "
                f"{r['red_name']} win_rate={form_r['recent_win_rate']:.0%} "
                f"finish_rate={form_r['recent_finish_rate']:.0%} "
                f"streak={int(form_r['win_streak'])} | "
                f"{r['blue_name']} win_rate={form_b['recent_win_rate']:.0%} "
                f"finish_rate={form_b['recent_finish_rate']:.0%} "
                f"streak={int(form_b['win_streak'])}"
            ),
            f"- {model_label}: {r['red_name']} {r['red_prob']:.1%} | {r['blue_name']} {r['blue_prob']:.1%}",
            f"- Finish: {finish_str}",
            "",
        ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next UFC event card.")
    parser.add_argument("--model", default="ensemble",
                        choices=["xgb", "lr", "rf", "lgbm", "ensemble"],
                        help="Model to use (default: ensemble)")
    parser.add_argument("--output", default=None,
                        help="Output .md path (default: predictions/<slug>.md)")
    args = parser.parse_args()

    # 1. Scrape upcoming event
    log.info("Scraping next upcoming event from UFCStats...")
    event = scrape_upcoming_event()
    if not event:
        print("[ERROR]  No upcoming events found on UFCStats.")
        sys.exit(1)

    print(f"\nEvent: {event['name']}  ({event['date']})")
    print(f"Fights on card: {len(event['fights'])}\n")

    if not event["fights"]:
        print("[ERROR]  No fights found on the card. The page structure may have changed.")
        sys.exit(1)

    if not DB_V1_PATH.exists() or not MODELS_V1_DIR.exists():
        print("[ERROR]  v1 DB or models not found. Run the v1 training pipeline first.")
        sys.exit(1)

    # 2. Filter debuts (uses v2 UFCStats DB for ID-based lookup) and run v1 predictions
    conn    = sqlite3.connect(str(DB_PATH))
    results = []
    skipped = []

    for fight in event["fights"]:
        r_row = lookup_fighter(conn, fight["r_id"])
        b_row = lookup_fighter(conn, fight["b_id"])

        r_db_name = r_row[1] if r_row else None
        b_db_name = b_row[1] if b_row else None

        r_debut = (r_row is None) or (not has_fight_history(conn, fight["r_id"]))
        b_debut = (b_row is None) or (not has_fight_history(conn, fight["b_id"]))

        r_label = r_db_name or fight["r_name"]
        b_label = b_db_name or fight["b_name"]

        if r_debut or b_debut:
            debut_names = [n for n, d in [(r_label, r_debut), (b_label, b_debut)] if d]
            skipped.append(f"{r_label} vs {b_label} (debut: {', '.join(debut_names)})")
            log.info("Skipping %s vs %s -- debut(s): %s", r_label, b_label, debut_names)
            continue

        print(f"  Predicting: {r_db_name} vs {b_db_name} ({fight['division'] or 'unknown div'})")
        try:
            result = compute_prediction(
                red_name=r_db_name,
                blue_name=b_db_name,
                model_type=args.model,
                division=fight["division"] or None,
                title_fight=fight["title_fight"],
                db_path=DB_V1_PATH,
                models_dir=MODELS_V1_DIR,
            )
            results.append(result)
        except SystemExit:
            log.warning("compute_prediction exited for %s vs %s -- skipping", r_db_name, b_db_name)
            skipped.append(f"{r_db_name} vs {b_db_name} (prediction error)")

    conn.close()

    if skipped:
        print(f"\nSkipped {len(skipped)} fight(s):")
        for s in skipped:
            print(f"  - {s}")

    if not results:
        print("\n[WARN]  No fights to predict after filtering.")
        sys.exit(0)

    # 3. Write markdown
    md = build_markdown(event, results, args.model)

    if args.output:
        out_path = Path(args.output)
    else:
        PREDICTIONS_DIR.mkdir(exist_ok=True)
        slug     = _event_slug(event["name"], event["date"])
        out_path = PREDICTIONS_DIR / f"{slug}.md"

    out_path.write_text(md, encoding="utf-8")
    print(f"\nPredictions written to: {out_path}")


if __name__ == "__main__":
    main()
