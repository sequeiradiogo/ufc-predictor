"""
scripts/score_comparisons.py -- Score v1 vs v2 predictions from event .md files.

After filling in "v2 correct: 1" / "v1 correct: 0" (or "v1 correct: 1") in the
event prediction .md files, run this script to see how each model is doing.

Usage
-----
    python scripts/score_comparisons.py
    python scripts/score_comparisons.py --dir path/to/predictions
"""

import argparse
import re
import sys
from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = ROOT_DIR / "predictions"

# Matches lines like:
#   v2 correct: 1
#   v1 correct: 0
_CORRECT_RE = re.compile(r"^(v[12]) correct:\s*([01])", re.MULTILINE)
_AGREE_RE   = re.compile(r"^Agree:\s*(Yes|No)", re.MULTILINE | re.IGNORECASE)


def score_dir(predictions_dir: Path) -> None:
    md_files = sorted(predictions_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {predictions_dir}")
        sys.exit(0)

    v2_correct = v2_total = 0
    v1_correct = v1_total = 0
    agree_count = agree_total = 0

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")

        # Find all scored comparison blocks in this file
        for m in _CORRECT_RE.finditer(text):
            model, result = m.group(1), int(m.group(2))
            if model == "v2":
                v2_total   += 1
                v2_correct += result
            else:
                v1_total   += 1
                v1_correct += result

        for m in _AGREE_RE.finditer(text):
            agree_total += 1
            if m.group(1).lower() == "yes":
                agree_count += 1

    if v2_total == 0 and v1_total == 0:
        print("No scored comparisons found yet.")
        print("Fill in 'v2 correct: 1/0' and 'v1 correct: 1/0' lines in the event .md files.")
        return

    print()
    print(f"  {'Model':<8}  {'Fights':>8}  {'Correct':>8}  {'Accuracy':>10}")
    print("  " + "-" * 40)
    if v2_total:
        print(f"  {'v2':<8}  {v2_total:>8}  {v2_correct:>8}  {v2_correct/v2_total:>9.1%}")
    if v1_total:
        print(f"  {'v1':<8}  {v1_total:>8}  {v1_correct:>8}  {v1_correct/v1_total:>9.1%}")
    if agree_total:
        print(f"\n  Agree rate: {agree_count}/{agree_total} ({agree_count/agree_total:.1%})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tally v1 vs v2 prediction accuracy from event .md files.",
    )
    parser.add_argument(
        "--dir",
        default=str(PREDICTIONS_DIR),
        metavar="PATH",
        help=f"Directory containing event .md files (default: {PREDICTIONS_DIR})",
    )
    args = parser.parse_args()
    score_dir(Path(args.dir))


if __name__ == "__main__":
    main()
