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

Shadow scoring of the V2 prod design (issue #76, --shadow-history): the same
trial also trains V2, the eval pipeline with its test window moved past the
cutoff -- base models fit on the first 80% of the training data, calibrated on
the last 20% with their own predictions, nothing fit on the held-out fights
(equal weights, T = 1). V2 never gates the run. Every held-out fight not yet in
the history CSV is appended (outcome, closing line, p_prod / p_v2 / p_eval), so
each fight counts once, from the first refresh that scored it. The
pre-registered decision (issue #76) only uses fights dated SHADOW_SINCE or
later; --shadow-summary prints where it stands.

Usage:
    python scripts/validate_prod_retrain.py
    python scripts/validate_prod_retrain.py --hold-months 6 --seed 1
    python scripts/validate_prod_retrain.py --report models_v1_prod/prod_validation.json
    python scripts/validate_prod_retrain.py --shadow-history analytics/reports/prod_v2_shadow.csv
    python scripts/validate_prod_retrain.py --shadow-summary analytics/reports/prod_v2_shadow.csv
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

# Issue #76 shadow scoring. The 14 trial cutoffs of #67 held out fights up to
# 2026-11-07, so the pre-registered comparison only counts later fights; first
# look at SHADOW_FIRST_LOOK of them, final one at SHADOW_FINAL_LOOK.
SHADOW_SINCE = "2026-11-08"
SHADOW_FIRST_LOOK = 800
SHADOW_FINAL_LOOK = 1200
SHADOW_BOOT = 2000
_SHADOW_COLS = ["fight_id", "date", "run_date", "cutoff", "seed", "y", "market", "p_prod", "p_v2", "p_eval"]

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


def train_v2(tv, train: pd.DataFrame, hold: pd.DataFrame, out_dir: Path) -> None:
    """Issue #76's V2 prod design into *out_dir*: the eval pipeline run on train + hold
    with the split at the cutoff, so the base models fit on the first 80% of *train*,
    calibrate on its last 20% with their own predictions, and *hold* is only scored."""
    both = pd.concat([train, hold], ignore_index=True)
    old_split = tv.TRAIN_TEST_SPLIT
    tv.TRAIN_TEST_SPLIT = (len(train) + 0.5) / len(both)
    try:
        if int(len(both) * tv.TRAIN_TEST_SPLIT) != len(train):
            raise RuntimeError("V2 split does not fall on the cutoff")
        (out_dir / "unused_prod").mkdir(parents=True, exist_ok=True)
        redirect_model_paths(tv, out_dir, out_dir / "unused_prod")
        with contextlib.redirect_stdout(io.StringIO()):
            for fn in (tv.train_xgb, tv.train_lr, tv.train_rf, tv.train_lgbm, tv.train_mlp):
                fn(both, prod=False)
            tv.train_ensemble(both, prod=False)
    finally:
        tv.TRAIN_TEST_SPLIT = old_split


def run_trial(hold_months: int, seed: int, shadow: bool = False) -> dict:
    """Retrain eval + prod tiers on data up to (last fight - hold_months) in a temp dir
    and score both on the held-out fights (and V2 too with *shadow*)."""
    import ml.train_v1_models as tv
    from scripts.backtest_v1 import _experience_frame

    df_all = tv.load_data()
    cutoff = df_all["date"].max() - pd.DateOffset(months=hold_months)
    train = df_all[df_all["date"] <= cutoff].reset_index(drop=True)
    hold = df_all[df_all["date"] > cutoff].reset_index(drop=True)
    if hold.empty:
        raise SystemExit(f"No fights after {cutoff.date()} to hold out")

    X = hold[[c for c in hold.columns if c not in tv.META_COLS]].replace([np.inf, -np.inf], 0)
    y = hold[tv.TARGET_COL].values
    market = (_experience_frame().set_index("fight_id")["market_red"]
              .reindex(hold["fight_id"]).values.astype(float))
    out = dict(cutoff=str(cutoff.date()), n_train=len(train),
               hold_from=str(hold["date"].min().date()), hold_to=str(hold["date"].max().date()),
               y=y, market=market, fight_id=hold["fight_id"].values,
               date=hold["date"].dt.strftime("%Y-%m-%d").values)

    # The trial models (~20 MB per run) are only needed to score the hold-out
    with tempfile.TemporaryDirectory(prefix="prod_trial_", ignore_cleanup_errors=True) as tmp_name:
        tmp = Path(tmp_name)
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
        out["p_prod"] = score_ensemble(tmp / "prod", X)
        out["p_eval"] = score_ensemble(tmp / "eval", X)
        if shadow:
            log.info("Shadow: training V2 (fit on the first 80%%, own calibrators on the last 20%%)")
            train_v2(tv, train, hold, tmp / "v2")
            out["p_v2"] = score_ensemble(tmp / "v2", X)
    return out


def append_shadow(history: Path, t: dict, seed: int, run_date: str) -> int:
    """Append the trial's held-out fights that *history* doesn't have yet; returns how many.

    Each fight is kept from the first refresh that scored it: consecutive monthly
    trials hold out overlapping 3-month windows."""
    rows = pd.DataFrame({"fight_id": t["fight_id"], "date": t["date"], "run_date": run_date,
                         "cutoff": t["cutoff"], "seed": seed, "y": t["y"], "market": t["market"],
                         "p_prod": t["p_prod"], "p_v2": t["p_v2"], "p_eval": t["p_eval"]})[_SHADOW_COLS]
    old = pd.read_csv(history) if history.exists() else pd.DataFrame(columns=_SHADOW_COLS)
    new = rows[~rows["fight_id"].isin(old["fight_id"])]
    history.parent.mkdir(parents=True, exist_ok=True)
    out = new if old.empty else pd.concat([old, new], ignore_index=True)
    out.to_csv(history, index=False, float_format="%.6f")
    return len(new)


def _fight_ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def shadow_summary(history: Path, seed: int = 0) -> dict:
    """Where the pre-registered V2-vs-prod comparison (issue #76) stands.

    delta = mean per-fight log-loss of V2 minus prod over the fights dated
    SHADOW_SINCE or later; the 90% interval bootstraps whole event dates (the
    fights of one card share a retrain). At SHADOW_FIRST_LOOK fights: adopt V2 if
    the upper bound is < 0, keep prod if delta >= 0, otherwise wait for
    SHADOW_FINAL_LOOK, where V2 is adopted only if the upper bound is < 0.
    """
    h = pd.read_csv(history)
    h = h[h["date"] >= SHADOW_SINCE]
    res = dict(n=int(len(h)), first_look=SHADOW_FIRST_LOOK, final_look=SHADOW_FINAL_LOOK)
    if h.empty:
        return {**res, "decision": "no fights yet"}
    ll_prod, ll_v2 = _fight_ll(h["p_prod"], h["y"]), _fight_ll(h["p_v2"], h["y"])
    d = ll_v2 - ll_prod
    dates = h["date"].values
    by_date = [d[dates == u] for u in np.unique(dates)]
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([by_date[i] for i in rng.integers(0, len(by_date), len(by_date))]).mean()
             for _ in range(SHADOW_BOOT)]
    lo, hi = np.percentile(boots, [5, 95])
    res.update(delta=float(d.mean()), ci90=[float(lo), float(hi)], events=len(by_date),
               ll_prod=float(ll_prod.mean()), ll_v2=float(ll_v2.mean()))
    if len(h) < SHADOW_FIRST_LOOK:
        res["decision"] = f"collecting ({len(h)}/{SHADOW_FIRST_LOOK} fights before the first look)"
    elif hi < 0:
        res["decision"] = "adopt V2 (upper 90% bound < 0)"
    elif len(h) < SHADOW_FINAL_LOOK:
        res["decision"] = ("keep prod (V2 not better)" if d.mean() >= 0 else
                           f"inconclusive: keep collecting until {SHADOW_FINAL_LOOK} fights")
    else:
        res["decision"] = "keep prod (upper 90% bound >= 0 at the final look)"
    return res


def _shadow_md(s: dict, added: int | None = None) -> str:
    lines = ["### V2 shadow scoring (issue #76)", ""]
    if added is not None:
        lines.append(f"- {added} new held-out fights appended to the history.")
    if "delta" in s:
        lines.append(f"- Fights since {SHADOW_SINCE}: {s['n']} ({s['events']} events). "
                     f"Log-loss prod {s['ll_prod']:.4f}, V2 {s['ll_v2']:.4f}: "
                     f"V2 - prod {s['delta']:+.4f}, 90% CI [{s['ci90'][0]:+.4f}, {s['ci90'][1]:+.4f}].")
    lines.append(f"- Decision: **{s['decision']}**")
    return "\n".join(lines) + "\n"


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
    ap.add_argument("--shadow-history", type=Path,
                    help="also train V2 (issue #76) and append the new held-out fights to this CSV")
    ap.add_argument("--shadow-summary", type=Path, metavar="HISTORY",
                    help="only print where the V2-vs-prod comparison stands, then exit")
    args = ap.parse_args()

    if args.shadow_summary:
        print(_shadow_md(shadow_summary(args.shadow_summary)))
        return

    t = run_trial(args.hold_months, args.seed, shadow=args.shadow_history is not None)
    prod, fails, warns = evaluate(t["p_prod"], t["y"], t["market"])
    ev, _, _ = evaluate(t["p_eval"], t["y"], t["market"])
    report = dict(cutoff=t["cutoff"], n_train=t["n_train"], hold_from=t["hold_from"], hold_to=t["hold_to"],
                  seed=args.seed, prod=prod, eval=ev, failures=fails, warnings=warns,
                  thresholds=dict(min_market_corr=MIN_MARKET_CORR, max_market_gap=MAX_MARKET_GAP,
                                  min_market_gap=MIN_MARKET_GAP, min_prob_sd=MIN_PROB_SD,
                                  max_at_bounds=MAX_AT_BOUNDS, min_odds_fights=MIN_ODDS_FIGHTS))
    md = _summary_md(report)
    if args.shadow_history:
        # V2 never gates the run; a failed gate stops the workflow before the commit anyway
        report["v2"], _, _ = evaluate(t["p_v2"], t["y"], t["market"])
        added = append_shadow(args.shadow_history, t, args.seed, pd.Timestamp.now().strftime("%Y-%m-%d"))
        report["shadow"] = shadow_summary(args.shadow_history)
        md += "\n" + _shadow_md(report["shadow"], added)
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
