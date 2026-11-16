"""
validate_prod_retrain.py -- Out-of-sample check of a production-tier retrain (issue #67).

models_v1_prod/ is trained on 100% of the data, so nothing in a retrain says
whether it is sound. This script replays the same retrain on data truncated
HOLD_MONTHS before the last fight -- eval tier, then prod tier, exactly as
train_v1_models.py does, written to a temporary directory (the real models_v1/
and models_v1_prod/ are never touched) -- and scores the trial prod ensemble on
the held-out fights, which no fit has seen.

What it can and can't tell (measured on 14 trial cutoffs x 3 seeds, 2023-05 to
2026-11 hold-outs, issue #67):
  * ~130 fights is far too few to see a change of a few thousandths of
    log-loss (the #69 / #73 effects need ~1,000 fights). It doesn't try to:
    whether the model got *better* is a question for multi-origin experiments.
  * The log-loss gap to the de-vigged closing line swings a lot by period
    (mean +0.027, sd 0.021, range -0.006..+0.074 per 3-month hold-out); in a
    hard quarter (2026-08..11, market 0.641) even a coin flip is only 0.05
    behind. So the gap only catches gross failures.
  * The correlation between the model's and the market's log-odds doesn't
    depend on results and is stable (mean 0.71, min 0.58, sd 0.07): a broken
    model (dead features, shuffled rows, a flipped label) falls far below it.

Failures (exit 1, nothing uploaded by monthly-refresh.yml):
  * correlation with the market below MIN_MARKET_CORR
  * log-loss gap to the market above MAX_MARKET_GAP, or below MIN_MARKET_GAP
    (far better than the closing line on unseen fights = almost surely leakage)
  * probabilities collapsed (std below MIN_PROB_SD) or piled at the calibrator
    bounds (share above MAX_AT_BOUNDS)
With fewer than MIN_ODDS_FIGHTS priced fights the market checks are skipped
with a warning instead of failing.

The report (metrics, thresholds, failures) is written to --report as JSON and,
on GitHub Actions, to the job summary.

Usage:
    python scripts/validate_prod_retrain.py
    python scripts/validate_prod_retrain.py --hold-months 6 --seed 1
    python scripts/validate_prod_retrain.py --report models_v1_prod/prod_validation.json
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.calibration import apply_temperature, symmetric_proba  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("validate_prod_retrain")

HOLD_MONTHS = 3
MIN_ODDS_FIGHTS = 40
MIN_MARKET_CORR = 0.40
MAX_MARKET_GAP = 0.15
MIN_MARKET_GAP = -0.04
MIN_PROB_SD = 0.08
MAX_AT_BOUNDS = 0.10

# Base-model artifacts as train_v1_models.py names them (same names in both tiers).
_SPECS = [
    ("xgb",  "xgboost.joblib",             "xgb_features.joblib",  None,                False),
    ("lr",   "logistic_regression.joblib", "lr_features.joblib",   "lr_scaler.joblib",  True),
    ("rf",   "random_forest.joblib",       "rf_features.joblib",   None,                False),
    ("lgbm", "lightgbm.joblib",            "lgbm_features.joblib", None,                False),
    ("mlp",  "mlp.joblib",                 "mlp_features.joblib",  "mlp_scaler.joblib", True),
]


def redirect_model_paths(tv, eval_dir: Path, prod_dir: Path) -> None:
    """Point every model path of the train_v1_models module *tv* into eval_dir / prod_dir.

    train_v1_models writes to module-level path constants; rebinding them is what keeps
    the trial away from the live artifacts. Refuses to continue if any path is left
    pointing elsewhere (e.g. a constant added later under another name).
    """
    for name in dir(tv):
        val = getattr(tv, name)
        if not isinstance(val, Path):
            continue
        if name.startswith("MODEL_V1_PROD_"):
            setattr(tv, name, prod_dir / val.name)
        elif name.startswith("MODEL_V1_"):
            setattr(tv, name, eval_dir / val.name)
        elif name == "MODELS_V1_PROD_DIR":
            setattr(tv, name, prod_dir)
        elif name == "MODELS_V1_DIR":
            setattr(tv, name, eval_dir)
    roots = (Path(eval_dir).resolve(), Path(prod_dir).resolve())
    for name in dir(tv):
        val = getattr(tv, name)
        if name.startswith(("MODEL_V1_", "MODELS_V1_")) and isinstance(val, Path):
            if not any(val.resolve() == r or r in val.resolve().parents for r in roots):
                raise RuntimeError(f"train_v1_models.{name} still points at {val}")


def score_ensemble(models_dir: Path, X: pd.DataFrame) -> np.ndarray:
    """P(Red wins) from the ensemble in *models_dir*, as predict.py computes it."""
    ens = joblib.load(models_dir / "ensemble.joblib")
    probs, weights = [], []
    for key, model_f, feat_f, scaler_f, is_lr in _SPECS:
        if key not in ens["weights"]:
            continue
        art, feats = joblib.load(models_dir / model_f), joblib.load(models_dir / feat_f)
        scaler = joblib.load(models_dir / scaler_f) if scaler_f else None
        base = art["base"] if is_lr else art
        Xm = X.reindex(columns=feats, fill_value=0).fillna(0)
        raw = symmetric_proba(
            lambda Z, b=base, s=scaler: b.predict_proba(s.transform(Z) if s else Z.values)[:, 1], Xm)
        probs.append(ens["calibrators"][key].predict(raw))
        weights.append(ens["weights"][key])
    p = np.average(probs, axis=0, weights=np.asarray(weights) / sum(weights))
    return apply_temperature(p, ens.get("temperature", 1.0))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0 when either side is constant (a collapsed model)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(p: np.ndarray, y: np.ndarray, market: np.ndarray) -> tuple[dict, list[str], list[str]]:
    """Metrics of predictions *p* on outcomes *y*; *market* is the de-vigged closing
    P(Red) (NaN without odds). Returns (metrics, failures, warnings)."""
    p, y, market = np.asarray(p, float), np.asarray(y, float), np.asarray(market, float)
    m = dict(n=int(len(y)), log_loss=_log_loss(p, y), accuracy=float(np.mean((p >= 0.5) == y)),
             prob_sd=float(np.std(p)), at_bounds=float(np.mean((p <= 0.051) | (p >= 0.949))))
    fails, warns = [], []
    if m["prob_sd"] < MIN_PROB_SD:
        fails.append(f"probabilities collapsed: std {m['prob_sd']:.3f} < {MIN_PROB_SD}")
    if m["at_bounds"] > MAX_AT_BOUNDS:
        fails.append(f"{m['at_bounds']:.0%} of probabilities at the calibrator bounds (> {MAX_AT_BOUNDS:.0%})")

    ok = ~np.isnan(market)
    m["n_odds"] = int(ok.sum())
    if m["n_odds"] < MIN_ODDS_FIGHTS:
        warns.append(f"only {m['n_odds']} held-out fights with odds (< {MIN_ODDS_FIGHTS}): market checks skipped")
        return m, fails, warns
    po, yo, mo = p[ok], y[ok], market[ok]
    m.update(log_loss_odds=_log_loss(po, yo), market_log_loss=_log_loss(mo, yo),
             accuracy_odds=float(np.mean((po >= 0.5) == yo)), favorite_accuracy=float(np.mean((mo >= 0.5) == yo)),
             market_corr=_corr(_logit(po), _logit(mo)),
             agrees_with_favorite=float(np.mean((po >= 0.5) == (mo >= 0.5))))
    m["market_gap"] = m["log_loss_odds"] - m["market_log_loss"]
    if not m["market_corr"] >= MIN_MARKET_CORR:
        fails.append(f"correlation with the market {m['market_corr']:.2f} < {MIN_MARKET_CORR}")
    if m["market_gap"] > MAX_MARKET_GAP:
        fails.append(f"log-loss {m['market_gap']:+.3f} worse than the closing line (> {MAX_MARKET_GAP})")
    if m["market_gap"] < MIN_MARKET_GAP:
        fails.append(f"log-loss {-m['market_gap']:.3f} better than the closing line on unseen fights "
                     f"(> {-MIN_MARKET_GAP}): check for leakage")
    return m, fails, warns


def run_trial(hold_months: int, seed: int) -> dict:
    """Retrain eval + prod tiers on data up to (last fight - hold_months) in a temp dir
    and score both on the held-out fights."""
    import ml.train_v1_models as tv
    from scripts.backtest_v1 import _experience_frame

    df_all = tv.load_data()
    cutoff = df_all["date"].max() - pd.DateOffset(months=hold_months)
    train = df_all[df_all["date"] <= cutoff].reset_index(drop=True)
    hold = df_all[df_all["date"] > cutoff].reset_index(drop=True)
    if hold.empty:
        raise SystemExit(f"No fights after {cutoff.date()} to hold out")

    tmp = Path(tempfile.mkdtemp(prefix="prod_trial_"))
    (tmp / "eval").mkdir()
    (tmp / "prod").mkdir()
    redirect_model_paths(tv, tmp / "eval", tmp / "prod")
    tv.RANDOM_STATE = seed
    log.info("Trial retrain on %d fights up to %s; holding out %d fights (%s to %s)",
             len(train), cutoff.date(), len(hold), hold["date"].min().date(), hold["date"].max().date())
    with contextlib.redirect_stdout(io.StringIO()):
        for prod in (False, True):   # the prod ensemble needs the eval tier first
            for fn in (tv.train_xgb, tv.train_lr, tv.train_rf, tv.train_lgbm, tv.train_mlp):
                fn(train, prod=prod)
            tv.train_ensemble(train, prod=prod)

    X = hold[[c for c in hold.columns if c not in tv.META_COLS]].replace([np.inf, -np.inf], 0)
    y = hold[tv.TARGET_COL].values
    market = (_experience_frame().set_index("fight_id")["market_red"]
              .reindex(hold["fight_id"]).values.astype(float))
    return dict(cutoff=str(cutoff.date()), n_train=len(train),
                hold_from=str(hold["date"].min().date()), hold_to=str(hold["date"].max().date()),
                y=y, market=market,
                p_prod=score_ensemble(tmp / "prod", X), p_eval=score_ensemble(tmp / "eval", X))


def _summary_md(report: dict) -> str:
    m, e = report["prod"], report["eval"]
    lines = [f"## Prod retrain validation: {'FAILED' if report['failures'] else 'passed'}",
             "",
             f"Trial retrain on fights up to {report['cutoff']} ({report['n_train']}), scored on "
             f"{m['n']} held-out fights ({report['hold_from']} to {report['hold_to']}, {m['n_odds']} with odds).",
             "",
             "| | prod (trial) | eval (trial) | closing line |",
             "|---|---|---|---|",
             f"| log-loss | {m['log_loss']:.4f} | {e['log_loss']:.4f} | |",
             f"| accuracy | {m['accuracy']:.1%} | {e['accuracy']:.1%} | |"]
    if "market_corr" in m:
        lines += [f"| log-loss, fights with odds | {m['log_loss_odds']:.4f} | {e['log_loss_odds']:.4f} | "
                  f"{m['market_log_loss']:.4f} |",
                  f"| accuracy, fights with odds | {m['accuracy_odds']:.1%} | {e['accuracy_odds']:.1%} | "
                  f"{m['favorite_accuracy']:.1%} (favorite) |",
                  f"| correlation with the market | {m['market_corr']:.2f} | {e['market_corr']:.2f} | |"]
    lines += [""] + [f"- **FAIL** {f}" for f in report["failures"]] + [f"- warning: {w}" for w in report["warnings"]]
    lines += ["", "_One trial on ~3 months can only catch a broken retrain, not a small change in quality (issue #67)._"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hold-months", type=int, default=HOLD_MONTHS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    t = run_trial(args.hold_months, args.seed)
    prod, fails, warns = evaluate(t["p_prod"], t["y"], t["market"])
    ev, _, _ = evaluate(t["p_eval"], t["y"], t["market"])
    report = dict(cutoff=t["cutoff"], n_train=t["n_train"], hold_from=t["hold_from"], hold_to=t["hold_to"],
                  seed=args.seed, prod=prod, eval=ev, failures=fails, warnings=warns,
                  thresholds=dict(min_market_corr=MIN_MARKET_CORR, max_market_gap=MAX_MARKET_GAP,
                                  min_market_gap=MIN_MARKET_GAP, min_prob_sd=MIN_PROB_SD,
                                  max_at_bounds=MAX_AT_BOUNDS, min_odds_fights=MIN_ODDS_FIGHTS))
    md = _summary_md(report)
    print(md)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
