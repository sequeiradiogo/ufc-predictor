"""
scripts/compare_predictions.py -- Side-by-side v1 vs v2 model comparison.

Usage
-----
    python scripts/compare_predictions.py "Fighter A" "Fighter B"
    python scripts/compare_predictions.py "Islam Makhachev" "Arman Tsarukyan" --division lightweight --title
    python scripts/compare_predictions.py "Sean O'Malley" "Merab Dvalishvili" --division bantamweight --event ufc-316-2026-06-07

Arguments
---------
    red_fighter   Red corner fighter name (partial OK)
    blue_fighter  Blue corner fighter name (partial OK)
    --model       Model type to use (default: ensemble)
    --division    Weight division
    --title       Flag as title fight
    --event       Event slug (e.g. ufc-316-2026-06-07) -- appends to predictions/<slug>.md

Notes
-----
- v2 uses ufc_ufcstats.db + models/ (rolling stats, no leakage)
- v1 uses ufc_v2.db + models_v1/ (career-aggregate stats, may have mild leakage)
- Fighter names are resolved from the v2 DB; the same names are reused for v1.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH, DB_V1_PATH, MODELS_DIR, MODELS_V1_DIR
from predict import compute_prediction
from utils.logger import get_logger

log = get_logger(__name__)

PREDICTIONS_DIR = ROOT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)


def _fmt_pct(p: float) -> str:
    return f"{p:.1%}"


def run_comparison(
    red_name: str,
    blue_name: str,
    model_type: str,
    division: str | None,
    title_fight: int,
    event_slug: str | None,
) -> None:

    print(f"\n[v2 - UFCStats rolling]")
    v2 = compute_prediction(
        red_name, blue_name,
        model_type=model_type,
        division=division,
        title_fight=title_fight,
        db_path=DB_PATH,
        models_dir=MODELS_DIR,
    )

    # Use resolved names from v2 for v1 so the interactive prompt fires only once
    r_name = v2["red_name"]
    b_name = v2["blue_name"]

    if DB_V1_PATH.exists():
        print(f"\n[v1 - mdabbert career-aggregate]")
        v1 = compute_prediction(
            r_name, b_name,
            model_type=model_type,
            division=division,
            title_fight=title_fight,
            db_path=DB_V1_PATH,
            models_dir=MODELS_V1_DIR,
        )
    else:
        log.warning("v1 DB not found at '%s' -- skipping v1 prediction.", DB_V1_PATH)
        v1 = None

    agree = (v1 is not None) and (v2["winner"] == v1["winner"])

    # ── Print comparison ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  COMPARISON: {r_name} (Red) vs {b_name} (Blue)")
    if division:
        print(f"  Division: {division.title()}" + ("  [TITLE]" if title_fight else ""))
    print("=" * 60)
    print()
    header = f"  {'Model':<28}  {'Winner':<22}  {'Confidence':>10}  {'Red%':>6}  {'Blue%':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    v2_winner_label = v2["winner"][:22]
    print(f"  {'v2 (UFCStats rolling)':<28}  {v2_winner_label:<22}  {_fmt_pct(v2['confidence']):>10}  {_fmt_pct(v2['red_prob']):>6}  {_fmt_pct(v2['blue_prob']):>6}")
    if v1:
        v1_winner_label = v1["winner"][:22]
        print(f"  {'v1 (mdabbert)':<28}  {v1_winner_label:<22}  {_fmt_pct(v1['confidence']):>10}  {_fmt_pct(v1['red_prob']):>6}  {_fmt_pct(v1['blue_prob']):>6}")
    print()
    print(f"  Models agree: {'YES' if agree else 'NO'}")
    print()

    # ── Append to event .md if requested ─────────────────────────────────────
    if event_slug:
        md_path = PREDICTIONS_DIR / f"{event_slug}.md"
        block = _build_md_block(r_name, b_name, v2, v1, agree, model_type)
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(block)
        print(f"  Appended comparison block to {md_path.relative_to(ROOT_DIR)}")


def _build_md_block(
    r_name: str,
    b_name: str,
    v2: dict,
    v1: dict | None,
    agree: bool,
    model_type: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n### {r_name} vs {b_name} -- Model Comparison\n",
        f"_Generated: {ts} | Model: {model_type}_\n",
        "\n",
        "| Model | Winner | Confidence | Red % | Blue % |\n",
        "|-------|--------|------------|-------|--------|\n",
        f"| v2 (UFCStats rolling) | {v2['winner']} | {_fmt_pct(v2['confidence'])} | {_fmt_pct(v2['red_prob'])} | {_fmt_pct(v2['blue_prob'])} |\n",
    ]
    if v1:
        lines.append(
            f"| v1 (mdabbert) | {v1['winner']} | {_fmt_pct(v1['confidence'])} | {_fmt_pct(v1['red_prob'])} | {_fmt_pct(v1['blue_prob'])} |\n"
        )
    lines += [
        "\n",
        f"Agree: {'Yes' if agree else 'No'}\n",
        "Result: _(fill in after)_\n",
        "v2 correct: _(fill in after)_\n",
        "v1 correct: _(fill in after)_\n",
    ]
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare v1 (mdabbert) and v2 (UFCStats) model predictions side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python scripts/compare_predictions.py "Islam Makhachev" "Arman Tsarukyan" --division lightweight --title
  python scripts/compare_predictions.py "Sean O'Malley" "Merab Dvalishvili" --division bantamweight --event ufc-316-2026-06-07
        """,
    )
    parser.add_argument("red_fighter",  help="Red corner fighter name (partial OK)")
    parser.add_argument("blue_fighter", help="Blue corner fighter name (partial OK)")
    parser.add_argument(
        "--model",
        choices=["xgb", "lr", "rf", "lgbm", "ensemble"],
        default="ensemble",
        help="Model type (default: ensemble)",
    )
    parser.add_argument("--division", default=None, help="Weight division")
    parser.add_argument("--title", action="store_true", default=False, help="Title fight flag")
    parser.add_argument(
        "--event",
        default=None,
        metavar="SLUG",
        help="Event slug (e.g. ufc-316-2026-06-07) -- appends block to predictions/<slug>.md",
    )

    args = parser.parse_args()
    run_comparison(
        args.red_fighter,
        args.blue_fighter,
        model_type=args.model,
        division=args.division,
        title_fight=int(args.title),
        event_slug=args.event,
    )


if __name__ == "__main__":
    main()
