"""
tune_compare.py -- Re-tune all base models for Option A and Option C, then compare backtest.

Option A: MIN_FIGHT_DATE=2018-01-01, SAMPLE_WEIGHT_ALPHA=0.0
Option C: MIN_FIGHT_DATE=2010-01-01, SAMPLE_WEIGHT_ALPHA=0.3

For each condition:
  1. Patch config.py
  2. Regenerate feature CSV (step 4)
  3. Tune + train each base model (--tune --trials 100)
  4. Optimise ensemble weights (--trials 100)
  5. Run 2022+ backtest and capture results

Results saved to scripts/tune_compare_results.txt
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
CONFIG      = ROOT / "config.py"
RESULTS     = ROOT / "scripts" / "tune_compare_results.txt"
TRIALS      = 100

CONDITIONS = [
    {"name": "Option_C_tuned", "date": "2010-01-01", "alpha": 0.3},
]

MODEL_SCRIPTS = [
    ROOT / "ml" / "XGBoost.py",
    ROOT / "ml" / "logistic_regression.py",
    ROOT / "ml" / "random_forest.py",
    ROOT / "ml" / "lightgbm_model.py",
]
ENSEMBLE_SCRIPT = ROOT / "ml" / "soft_vote_ensemble.py"


def patch_config(date: str, alpha: float) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    text = re.sub(r'MIN_FIGHT_DATE\s*=\s*"[^"]+"', f'MIN_FIGHT_DATE = "{date}"', text)
    text = re.sub(r'SAMPLE_WEIGHT_ALPHA\s*=\s*[\d.]+', f'SAMPLE_WEIGHT_ALPHA = {alpha}', text)
    CONFIG.write_text(text, encoding="utf-8")


def run_script(script: Path, extra_args: list[str] | None = None) -> tuple[bool, str]:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [sys.executable, str(script)] + (extra_args or [])
    t0  = time.time()
    r   = subprocess.run(cmd, cwd=str(ROOT), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 0, r.stdout, time.time() - t0


def run_pipeline_step(step: str) -> tuple[bool, str]:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--steps", step]
    r   = subprocess.run(cmd, cwd=str(ROOT), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 0, r.stdout


def run_backtest() -> str:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cmd = [sys.executable, str(ROOT / "scripts" / "backtest.py"), "--from-year", "2022"]
    r   = subprocess.run(cmd, cwd=str(ROOT), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.stdout


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)


def main() -> None:
    lines = [f"Tune-compare experiment -- {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]

    for cond in CONDITIONS:
        name, date, alpha = cond["name"], cond["date"], cond["alpha"]
        banner(f"{name}  |  date >= {date}  |  alpha = {alpha}")

        patch_config(date, alpha)

        print("  [step 4] Generating feature CSV...", flush=True)
        ok, out = run_pipeline_step("4")
        rows = [l for l in out.splitlines() if "rows" in l.lower() or "remaining" in l.lower()]
        print(f"  CSV: {rows[-1].strip() if rows else '?'}", flush=True)

        tuning_lines = []
        for script in MODEL_SCRIPTS:
            model_name = script.stem
            print(f"  [tune] {model_name} ({TRIALS} trials)...", flush=True)
            ok, out, elapsed = run_script(script, ["--tune", "--trials", str(TRIALS)])
            status = "OK" if ok else "FAILED"
            print(f"  {model_name} {status} ({elapsed:.0f}s)", flush=True)

            # Extract best params and test accuracy from output
            for line in out.splitlines():
                if any(k in line for k in ("Best params", "Test Accuracy", "Ensemble test")):
                    print(f"    {line.strip()}", flush=True)
                    tuning_lines.append(f"  {model_name}: {line.strip()}")

        print("  [ensemble] Optimising weights...", flush=True)
        ok, ens_out, elapsed = run_script(ENSEMBLE_SCRIPT, ["--trials", str(TRIALS)])
        print(f"  ensemble {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)", flush=True)
        for line in ens_out.splitlines():
            if any(k in line for k in ("Ensemble test", "Weights", "xgb", "lr ", "rf ", "lgbm")):
                print(f"    {line.strip()}", flush=True)

        print("  [backtest] Running 2022+ backtest...", flush=True)
        bt = run_backtest()
        print(bt, flush=True)

        lines.append(f"\n=== {name} (date>={date}, alpha={alpha}) ===")
        lines.extend(tuning_lines)
        lines.append(ens_out[-800:])
        lines.append(bt)

    # Restore Option A as default
    patch_config("2018-01-01", 0.0)
    print("\nConfig restored to Option A (2018-01-01, alpha=0.0).", flush=True)

    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults saved to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
