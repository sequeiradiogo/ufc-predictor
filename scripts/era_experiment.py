"""
era_experiment.py -- Runs the four era-distribution experiment conditions from issue #47.

Conditions
----------
  Baseline  : MIN_FIGHT_DATE=2005-01-01, SAMPLE_WEIGHT_ALPHA=0.0
  Option A  : MIN_FIGHT_DATE=2018-01-01, SAMPLE_WEIGHT_ALPHA=0.0  (hard cutoff)
  Option B  : MIN_FIGHT_DATE=2005-01-01, SAMPLE_WEIGHT_ALPHA=0.5  (sample weighting)
  Option C  : MIN_FIGHT_DATE=2010-01-01, SAMPLE_WEIGHT_ALPHA=0.3  (hybrid)

For each condition the script:
  1. Patches MIN_FIGHT_DATE and SAMPLE_WEIGHT_ALPHA in config.py.
  2. Re-generates the feature CSV (step 4) only when the date cutoff changes.
  3. Retrains all base models + ensemble (steps 5, 6, 8, 9, 10).
  4. Runs the 2022+ backtest and captures accuracy.

Results are printed to stdout and saved to scripts/era_experiment_results.txt.
config.py is restored to the baseline values at the end.

Usage
-----
  python scripts/era_experiment.py
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.py"
RESULTS_PATH = ROOT / "scripts" / "era_experiment_results.txt"

CONDITIONS = [
    {"name": "Baseline", "date": "2005-01-01", "alpha": 0.0},
    {"name": "Option_A", "date": "2018-01-01", "alpha": 0.0},
    {"name": "Option_B", "date": "2005-01-01", "alpha": 0.5},
    {"name": "Option_C", "date": "2010-01-01", "alpha": 0.3},
]

BASELINE_DATE  = "2005-01-01"
BASELINE_ALPHA = 0.0


def patch_config(date: str, alpha: float) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = re.sub(
        r'MIN_FIGHT_DATE\s*=\s*"[^"]+"',
        f'MIN_FIGHT_DATE = "{date}"',
        text,
    )
    text = re.sub(
        r'SAMPLE_WEIGHT_ALPHA\s*=\s*[\d.]+',
        f'SAMPLE_WEIGHT_ALPHA = {alpha}',
        text,
    )
    CONFIG_PATH.write_text(text, encoding="utf-8")


def run_pipeline_steps(steps: str) -> tuple[bool, str]:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--steps", steps]
    result = subprocess.run(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return result.returncode == 0, result.stdout


def run_backtest() -> str:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [sys.executable, str(ROOT / "scripts" / "backtest.py"), "--from-year", "2022"]
    result = subprocess.run(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return result.stdout


def banner(msg: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {msg}\n{line}", flush=True)


def main() -> None:
    lines: list[str] = [f"Era Experiment -- {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    prev_date: str | None = None

    for cond in CONDITIONS:
        name  = cond["name"]
        date  = cond["date"]
        alpha = cond["alpha"]

        banner(f"{name}  |  date >= {date}  |  alpha = {alpha}")
        patch_config(date, alpha)

        # Regenerate feature CSV only when the date cutoff changes.
        if date != prev_date:
            print(f"  [step 4] Regenerating feature CSV for cutoff {date}...", flush=True)
            t0 = time.time()
            ok, out = run_pipeline_steps("4")
            print(f"  step 4 {'OK' if ok else 'FAILED'} ({time.time()-t0:.0f}s)", flush=True)
            if not ok:
                print(out[-2000:], flush=True)
                lines.append(f"\n=== {name} ===\nFAILED at step 4\n")
                continue

        prev_date = date

        # Train base models + ensemble.
        print("  [steps 5,6,8,9,10] Training models...", flush=True)
        t0 = time.time()
        ok, out = run_pipeline_steps("5,6,8,9,10")
        elapsed = time.time() - t0
        print(f"  training {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)", flush=True)
        # Print last 200 chars so the user can see test accuracy lines.
        print(out[-400:], flush=True)

        if not ok:
            lines.append(f"\n=== {name} ===\nFAILED during training\n{out[-1000:]}\n")
            continue

        # Backtest.
        print("  [backtest] Running 2022+ backtest...", flush=True)
        bt = run_backtest()
        print(bt[-600:], flush=True)

        lines.append(f"\n=== {name} (date>={date}, alpha={alpha}) ===\n{bt}\n")

    # Restore baseline config.
    patch_config(BASELINE_DATE, BASELINE_ALPHA)
    print("\nConfig restored to baseline.", flush=True)

    summary = "\n".join(lines)
    RESULTS_PATH.write_text(summary, encoding="utf-8")
    print(f"\nFull results saved to: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
