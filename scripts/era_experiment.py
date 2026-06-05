"""
era_experiment.py -- Re-run A vs C comparison with honest 2025 out-of-sample backtest (issue #51).

Conditions
----------
  Option A  : MIN_FIGHT_DATE=2018-01-01, SAMPLE_WEIGHT_ALPHA=0.0  (current hard cutoff)
  Option C  : MIN_FIGHT_DATE=2010-01-01, SAMPLE_WEIGHT_ALPHA=0.3  (2010 + exponential decay)

For each condition the script:
  1. Patches MIN_FIGHT_DATE and SAMPLE_WEIGHT_ALPHA in config.py.
  2. Re-generates the feature CSV (step 4) only when the date cutoff changes.
  3. Option A: retrains with existing tuned params (already Optuna-tuned on 2018+ data).
     Option C: Optuna-tunes all four base models (100 trials each) then retrains.
  4. Runs the 2025+ backtest (honest out-of-sample: all 2025 fights are post-training-cutoff).

Results are printed to stdout and saved to scripts/era_experiment_results_2025.txt.
config.py is restored to the original Option A values at the end.

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
RESULTS_PATH = ROOT / "scripts" / "era_experiment_results_2025.txt"

CONDITIONS = [
    {"name": "Option_A", "date": "2018-01-01", "alpha": 0.0, "tune": False},
    {"name": "Option_C", "date": "2010-01-01", "alpha": 0.3, "tune": True},
]

RESTORE_DATE  = "2018-01-01"
RESTORE_ALPHA = 0.0


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


def run_step(steps: str) -> tuple[bool, str]:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--steps", steps]
    result = subprocess.run(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return result.returncode == 0, result.stdout


def run_tune_and_train(n_trials: int = 100) -> tuple[bool, str]:
    """Optuna-tune each base model then retrain ensemble."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    model_scripts = [
        ROOT / "ml" / "XGBoost.py",
        ROOT / "ml" / "logistic_regression.py",
        ROOT / "ml" / "random_forest.py",
        ROOT / "ml" / "lightgbm_model.py",
    ]
    all_output: list[str] = []
    for script in model_scripts:
        cmd = [sys.executable, str(script), "--tune", "--trials", str(n_trials)]
        result = subprocess.run(
            cmd, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        all_output.append(result.stdout)
        if result.returncode != 0:
            return False, "\n".join(all_output)

    ok, out = run_step("10")
    all_output.append(out)
    return ok, "\n".join(all_output)


def run_backtest() -> str:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [sys.executable, str(ROOT / "scripts" / "backtest.py"), "--from-year", "2025"]
    result = subprocess.run(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return result.stdout


def banner(msg: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {msg}\n{line}", flush=True)


def main() -> None:
    lines: list[str] = [f"Era Experiment (issue #51) -- {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    lines.append("Evaluation metric: backtest --from-year 2025 (honest out-of-sample)\n")
    prev_date: str | None = None

    for cond in CONDITIONS:
        name  = cond["name"]
        date  = cond["date"]
        alpha = cond["alpha"]

        banner(f"{name}  |  date >= {date}  |  alpha = {alpha}")
        patch_config(date, alpha)

        if date != prev_date:
            print(f"  [step 4] Regenerating feature CSV for cutoff {date}...", flush=True)
            t0 = time.time()
            ok, out = run_step("4")
            print(f"  step 4 {'OK' if ok else 'FAILED'} ({time.time()-t0:.0f}s)", flush=True)
            if not ok:
                print(out[-2000:], flush=True)
                lines.append(f"\n=== {name} ===\nFAILED at step 4\n")
                continue

        prev_date = date

        if cond["tune"]:
            print("  [tune + train] Optuna (100 trials each) + retrain all models...", flush=True)
            t0 = time.time()
            ok, out = run_tune_and_train(n_trials=100)
            elapsed = time.time() - t0
            print(f"  tune+train {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)", flush=True)
        else:
            print("  [train] Retraining with existing tuned params + ensemble...", flush=True)
            t0 = time.time()
            ok, out = run_step("5,6,8,9,10")
            elapsed = time.time() - t0
            print(f"  train {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)", flush=True)
        print(out[-600:], flush=True)

        if not ok:
            lines.append(f"\n=== {name} ===\nFAILED during train\n{out[-1000:]}\n")
            continue

        print("  [backtest] Running 2025+ backtest...", flush=True)
        bt = run_backtest()
        print(bt[-600:], flush=True)

        lines.append(f"\n=== {name} (date>={date}, alpha={alpha}) ===\n{bt}\n")

    patch_config(RESTORE_DATE, RESTORE_ALPHA)
    print("\nConfig restored to Option A (2018-01-01, alpha=0.0).", flush=True)

    summary = "\n".join(lines)
    RESULTS_PATH.write_text(summary, encoding="utf-8")
    print(f"\nFull results saved to: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
