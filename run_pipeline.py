"""
run_pipeline.py — UFC Predictor Pipeline Orchestrator
======================================================
Chains every pipeline step from raw data to trained models in one command.

Pipeline steps
--------------
  1  Build raw database from CSV          (db/raw_sql_database.py)
  2  Add foreign key constraints          (db/keys.py)
  3  Compute rolling statistics           (db/rolling.py)
  4  Generate ML feature dataset          (ml/ML_data_preparation.py)
  5  Train XGBoost model                  (ml/XGBoost.py)
  6  Train Logistic Regression model      (ml/logistic_regression.py)
  7  Train Finish-Type model              (ml/finish_type_model.py)
  8  Train Random Forest model            (ml/random_forest.py)
  9  Train LightGBM model                 (ml/lightgbm_model.py)

Usage
-----
  # Run only the ML steps (DB already built — most common case)
  python run_pipeline.py

  # Run everything from scratch
  python run_pipeline.py --full --csv path/to/UFC.csv

  # Run specific steps only
  python run_pipeline.py --steps 4,5,6,7

  # Dry run — show what would run without executing
  python run_pipeline.py --dry-run

Notes
-----
- Steps 1-3 only need to run once (or when you add new raw data).
- Steps 4-9 should be re-run whenever the DB is updated.
- Step 3 (rolling.py) is a standalone script with no functions, so it is
  invoked as a subprocess.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from utils.logger import get_logger

log = get_logger("pipeline")


# ── Step definitions ──────────────────────────────────────────────────────────

STEPS: dict[int, dict] = {
    1: {
        "name":   "Build raw database",
        "script": ROOT_DIR / "db" / "raw_sql_database.py",
        "mode":   "subprocess",
        "db_required": False,
    },
    2: {
        "name":   "Add foreign key constraints",
        "script": ROOT_DIR / "db" / "keys.py",
        "mode":   "subprocess",
        "db_required": False,
    },
    3: {
        "name":   "Compute rolling statistics",
        "script": ROOT_DIR / "db" / "rolling.py",
        "mode":   "subprocess",
        "db_required": False,
    },
    4: {
        "name":   "Generate ML feature dataset",
        "module": "ml.ML_data_preparation",
        "fn":     "main",
        "mode":   "import",
        "db_required": True,
    },
    5: {
        "name":   "Train XGBoost model",
        "module": "ml.XGBoost",
        "fn":     "main",
        "mode":   "import",
        "db_required": True,
    },
    6: {
        "name":   "Train Logistic Regression model",
        "module": "ml.logistic_regression",
        "fn":     "main",
        "mode":   "import",
        "db_required": True,
    },
    7: {
        "name":   "Train Finish-Type model",
        "module": "ml.finish_type_model",
        "fn":     "main",
        "mode":   "import",
        "db_required": True,
    },
    8: {
        "name":   "Train Random Forest model",
        "module": "ml.random_forest",
        "fn":     "main",
        "mode":   "import",
        "db_required": True,
    },
    9: {
        "name":   "Train LightGBM model",
        "module": "ml.lightgbm_model",
        "fn":     "main",
        "mode":   "import",
        "db_required": True,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_db() -> bool:
    """Return True if the main database exists."""
    try:
        from config import DB_PATH
        return DB_PATH.exists()
    except ImportError:
        return False


def _run_subprocess(script: Path, extra_args: list[str] | None = None) -> bool:
    """Run a Python script as a subprocess. Returns True on success."""
    cmd = [sys.executable, str(script)] + (extra_args or [])
    log.info("  $ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        log.error("Step failed with exit code %d.", result.returncode)
        return False
    return True


def _run_import(module: str, fn: str) -> bool:
    """Import a module and call a function from it. Returns True on success."""
    import importlib
    log.info("  Importing %s.%s()…", module, fn)
    try:
        mod  = importlib.import_module(module)
        func = getattr(mod, fn)
        func()
        return True
    except Exception as exc:
        log.error("Step failed: %s", exc, exc_info=True)
        return False


def _banner(step_num: int, name: str) -> None:
    log.info("")
    log.info("━" * 55)
    log.info("  Step %d / %d — %s", step_num, max(STEPS), name)
    log.info("━" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    steps:    list[int],
    dry_run:  bool       = False,
    csv_path: Path | None = None,
) -> None:
    """Execute *steps* in order, stopping on the first failure."""
    log.info("UFC Predictor Pipeline  —  steps: %s", steps)
    if dry_run:
        log.info("[DRY RUN] No steps will actually execute.")

    overall_start = time.time()
    results: dict[int, str] = {}

    for step_num in steps:
        step = STEPS[step_num]
        _banner(step_num, step["name"])

        # Guard: DB-dependent steps require the database to exist
        if step.get("db_required") and not _check_db():
            log.error(
                "Database not found — run steps 1-3 first "
                "(or use --full with --csv path/to/UFC.csv)"
            )
            results[step_num] = "SKIPPED (no DB)"
            continue

        if dry_run:
            log.info("  [DRY RUN] Would execute: %s", step["name"])
            results[step_num] = "DRY RUN"
            continue

        t0 = time.time()

        if step["mode"] == "subprocess":
            extra = []
            if step_num == 1 and csv_path:
                extra = ["--csv", str(csv_path)]
            ok = _run_subprocess(step["script"], extra)
        else:
            ok = _run_import(step["module"], step["fn"])

        elapsed = time.time() - t0
        results[step_num] = f"{'OK' if ok else 'FAILED'}  ({elapsed:.1f}s)"

        if not ok:
            log.error("Pipeline stopped at step %d.", step_num)
            break

    # Summary
    total = time.time() - overall_start
    log.info("")
    log.info("━" * 55)
    log.info("  Pipeline Summary  (%.1f s total)", total)
    log.info("━" * 55)
    for s, status in results.items():
        icon = "✓" if status.startswith("OK") else ("~" if "DRY" in status or "SKIP" in status else "✗")
        log.info("  %s  Step %d — %s  →  %s", icon, s, STEPS[s]["name"], status)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the UFC Predictor pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python run_pipeline.py                        # ML steps only (4-9)
  python run_pipeline.py --full --csv UFC.csv   # all steps from scratch
  python run_pipeline.py --steps 4,5            # specific steps
  python run_pipeline.py --dry-run              # preview without executing
        """,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all steps including DB build (1–6). Requires --csv.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers to run, e.g. '4,5,6'.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to UFC.csv (required for step 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    args = parser.parse_args()

    if args.steps:
        try:
            steps = [int(s.strip()) for s in args.steps.split(",")]
            invalid = [s for s in steps if s not in STEPS]
            if invalid:
                parser.error(f"Invalid step numbers: {invalid}. Valid: {list(STEPS)}")
        except ValueError:
            parser.error("--steps must be comma-separated integers, e.g. '4,5,6'")
    elif args.full:
        steps = list(STEPS)          # 1 through 9
    else:
        steps = [4, 5, 6, 7, 8, 9]  # default: ML steps only

    if 1 in steps and not args.csv and not args.dry_run:
        log.warning(
            "Step 1 requires a --csv path to UFC.csv. "
            "Defaulting to raw_data/UFC.csv — set --csv explicitly if different."
        )

    run_pipeline(steps, dry_run=args.dry_run, csv_path=args.csv)


if __name__ == "__main__":
    main()
