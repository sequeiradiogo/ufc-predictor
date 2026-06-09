"""
era_experiment.py -- Grid search over training cutoff, decay weight, and decay shape.

For each (MIN_FIGHT_DATE, SAMPLE_WEIGHT_ALPHA, SAMPLE_WEIGHT_BETA) candidate:
  1. Patches config.py in-place
  2. Rebuilds the v1 feature CSV  (ml/ML_data_preparation_v1.py)
  3. Retrains all v1 models       (ml/train_v1_models.py)
  4. Backtests from 2025          (scripts/backtest_v1.py --from-year 2025)
  5. Records per-year accuracy

Restores config.py to its original content in a finally block.

Weight formula: exp(-alpha * delta^beta), delta = max_year - year.
  beta=1.0  -> flat exponential (original behaviour)
  beta>1.0  -> decay steepens for older fights, gentler for recent ones

Usage:
    python scripts/era_experiment.py
    python scripts/era_experiment.py --tune
    python scripts/era_experiment.py --dry-run
    python scripts/era_experiment.py --cutoffs 2017 2018 --alphas 0.01 0.02 0.05 --betas 1.5 2.0 2.5
"""

import argparse
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.py"

BASELINE_CUTOFF = "2018-01-01"
BASELINE_ALPHA  = 0.0
BASELINE_BETA   = 1.0
BASELINE_ACC    = 66.4  # ensemble 2025+ pct, per CLAUDE.md

DEFAULT_CUTOFFS = ["2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"]
DEFAULT_ALPHAS  = [0.0, 0.1, 0.2, 0.3, 0.5]
DEFAULT_BETAS   = [1.0]  # flat -- beta grid only active when --betas is passed


def _build_grid(cutoffs, alphas, betas, no_filter=False):
    grid = []
    for cutoff, alpha, beta in product(cutoffs, alphas, betas):
        year = int(cutoff[:4])
        if not no_filter and alpha > 0.0 and year >= 2018 and beta == 1.0:
            continue  # flat decay + 2018 was already tested; skip unless beta varies
        grid.append((cutoff, alpha, beta))
    return grid


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

_DATE_RE  = re.compile(r'^(MIN_FIGHT_DATE\s*=\s*)"[^"]*"', re.MULTILINE)
_ALPHA_RE = re.compile(r'^(SAMPLE_WEIGHT_ALPHA\s*=\s*)[\d.]+', re.MULTILINE)
_BETA_RE  = re.compile(r'^(SAMPLE_WEIGHT_BETA\s*=\s*)[\d.]+', re.MULTILINE)


def _patch_config(original: str, cutoff: str, alpha: float, beta: float) -> str:
    text = _DATE_RE.sub(rf'\g<1>"{cutoff}"', original)
    text = _ALPHA_RE.sub(rf'\g<1>{alpha}', text)
    text = _BETA_RE.sub(rf'\g<1>{beta}', text)
    return text


def _write_config(text: str) -> None:
    CONFIG_PATH.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> tuple[int, str]:
    print(f"\n  [{label}] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=False, text=True)
    return result.returncode, ""


def _run_capture(cmd: list[str], label: str) -> tuple[int, str]:
    print(f"\n  [{label}] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode, result.stdout


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_YEAR_RE  = re.compile(r'^\s*(\d{4})\s+\d+\s+fights\s+([\d.]+)%')
_TOTAL_RE = re.compile(r'^\s*Total\s+\d+\s+fights\s+([\d.]+)%')


def _parse_backtest(stdout: str) -> dict:
    result = {}
    for line in stdout.splitlines():
        m = _YEAR_RE.match(line)
        if m:
            result[int(m.group(1))] = float(m.group(2))
        m2 = _TOTAL_RE.match(line)
        if m2:
            result["total"] = float(m2.group(1))
    return result


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def _candidate_label(cutoff: str, alpha: float, beta: float) -> str:
    return f"cutoff={cutoff}  alpha={alpha}  beta={beta}"


def _run_candidate(cutoff: str, alpha: float, beta: float, tune: bool) -> dict | None:
    rc, _ = _run([sys.executable, "ml/ML_data_preparation_v1.py"], "csv-rebuild")
    if rc != 0:
        print(f"  [ERROR] CSV rebuild failed (rc={rc}); skipping candidate.")
        return None

    retrain_args = [sys.executable, "ml/train_v1_models.py"]
    if tune:
        retrain_args += ["--tune"]
    rc, _ = _run(retrain_args, "retrain")
    if rc != 0:
        print(f"  [ERROR] Retrain failed (rc={rc}); skipping candidate.")
        return None

    bt_cmd = [sys.executable, "scripts/backtest_v1.py", "--from-year", "2025", "--model", "ensemble"]
    rc, stdout = _run_capture(bt_cmd, "backtest")
    if rc != 0:
        print(f"  [ERROR] Backtest failed (rc={rc}); skipping candidate.")
        return None

    return _parse_backtest(stdout)


def _print_table(results: list[tuple]) -> None:
    all_years = sorted(
        {yr for _, _, _, d in results for yr in d if isinstance(yr, int)},
        reverse=True,
    )

    header_years = "  ".join(f"{yr}" for yr in all_years)
    print(f"\n{'Cutoff':<14}  {'Alpha':<6}  {'Beta':<5}  {'Total':>6}  {header_years}  {'vs baseline':>11}")
    print("-" * (14 + 6 + 5 + 6 + len(all_years) * 8 + 22))

    def sort_key(row):
        _, _, _, d = row
        return d.get("total", d.get(max(k for k in d if isinstance(k, int)), 0))

    for cutoff, alpha, beta, d in sorted(results, key=sort_key, reverse=True):
        total   = d.get("total", float("nan"))
        yr_cols = "  ".join(f"{d.get(yr, float('nan')):>5.1f}%" for yr in all_years)
        delta   = total - BASELINE_ACC
        marker  = " *** BETTER" if delta > 0.5 else (" (+)" if delta > 0 else "")
        is_base = (cutoff == BASELINE_CUTOFF and alpha == BASELINE_ALPHA and beta == BASELINE_BETA)
        tag     = " [baseline]" if is_base else ""
        print(
            f"{cutoff:<14}  {alpha:<6}  {beta:<5}  {total:>5.1f}%  {yr_cols}  {delta:>+6.1f}pp{marker}{tag}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over cutoff, decay weight, and decay shape.")
    parser.add_argument("--cutoffs", nargs="+", default=None,
                        help="Cutoff dates, e.g. 2017 2018 or 2017-01-01 2018-01-01")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Alpha values, e.g. 0.01 0.02 0.05")
    parser.add_argument("--betas", nargs="+", type=float, default=None,
                        help="Beta values (decay shape), e.g. 1.0 1.5 2.0 2.5. beta=1 is flat exponential.")
    parser.add_argument("--tune", action="store_true",
                        help="Re-tune base model hyperparameters via Optuna (slow).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the candidate grid and exit without running anything.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable the pre-2018 flat-decay filter.")
    args = parser.parse_args()

    cutoffs = args.cutoffs if args.cutoffs else DEFAULT_CUTOFFS
    cutoffs = [c if "-" in c else f"{c}-01-01" for c in cutoffs]
    alphas  = args.alphas if args.alphas is not None else DEFAULT_ALPHAS
    betas   = args.betas  if args.betas  is not None else DEFAULT_BETAS

    grid = _build_grid(cutoffs, alphas, betas, no_filter=args.no_filter)

    print(f"\nEra experiment -- {len(grid)} candidates")
    print(f"Baseline: {BASELINE_CUTOFF}  alpha={BASELINE_ALPHA}  beta={BASELINE_BETA}  acc={BASELINE_ACC}%")
    if args.tune:
        print("Mode: full Optuna retune (slow)")
    print()

    for i, (cutoff, alpha, beta) in enumerate(grid, 1):
        print(f"  [{i}/{len(grid)}] {_candidate_label(cutoff, alpha, beta)}")
    if args.dry_run:
        print("\n--dry-run: exiting without running candidates.")
        return

    original_config = CONFIG_PATH.read_text(encoding="utf-8")
    results = []

    try:
        for i, (cutoff, alpha, beta) in enumerate(grid, 1):
            label = _candidate_label(cutoff, alpha, beta)
            print(f"\n{'='*60}")
            print(f"  Candidate {i}/{len(grid)}: {label}")
            print(f"{'='*60}")

            patched = _patch_config(original_config, cutoff, alpha, beta)
            _write_config(patched)

            acc = _run_candidate(cutoff, alpha, beta, tune=args.tune)
            if acc is not None:
                results.append((cutoff, alpha, beta, acc))
                yr_parts = " ".join(
                    f"{yr}={acc.get(yr, float('nan')):.1f}%"
                    for yr in sorted(k for k in acc if isinstance(k, int))
                )
                total = acc.get("total", float("nan"))
                delta = total - BASELINE_ACC
                print(
                    f"\nCANDIDATE_RESULT [{i}/{len(grid)}]: cutoff={cutoff} alpha={alpha} beta={beta} "
                    f"{yr_parts} total={total:.1f}% delta={delta:+.1f}pp",
                    flush=True,
                )
            else:
                print(
                    f"CANDIDATE_RESULT [{i}/{len(grid)}]: cutoff={cutoff} alpha={alpha} beta={beta} FAILED",
                    flush=True,
                )

    finally:
        print("\n  Restoring config.py to original values...")
        _write_config(original_config)
        print("  config.py restored.")

    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    _print_table(results)

    winners = [
        (c, a, b, d) for c, a, b, d in results
        if d.get("total", 0) - BASELINE_ACC > 0.5
        and not (c == BASELINE_CUTOFF and a == BASELINE_ALPHA and b == BASELINE_BETA)
    ]
    if winners:
        print(f"  {len(winners)} candidate(s) beat baseline by >0.5pp.")
        print("  Recommended next step: run with --tune for each winner before promoting.")
    else:
        print(f"  No candidate beat baseline ({BASELINE_ACC}%) by >0.5pp.")
        print("  Current config remains optimal.")


if __name__ == "__main__":
    main()
