"""
check_live_parity.py -- Verify that live inference builds the same features
the models were trained on.

For each of the last N events in the v1 feature CSV, copy the UFCStats DB,
delete every fight from that date on, and run predict.compute_prediction()
(as_of = event date) on each fight -- i.e. exactly what predict_event.py would
have computed the day before. The captured feature vector is compared,
feature by feature, with that fight's training row.

Nothing else enforces this: build_feature_vector() silently falls back to 0
for any feature the live path doesn't populate, and a feature can also be
populated with a different definition or scale than training used. Both
happened (southpaw_adv_diff dead live; Kaggle career stats leaking; zone
accuracies on a different scale) and only showed up as lost accuracy.

Usage:
    python scripts/check_live_parity.py                  # last 4 events
    python scripts/check_live_parity.py --events 8 --min-match 0.9
Exits 1 if any feature matches its training value in fewer than
--min-match of the fights.
"""

import argparse
import logging
import shutil
import sqlite3
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import predict
from config import CSV_V1_WITH_ELO, DB_PATH, DB_V1_PATH, MODELS_V1_DIR, MODELS_V1_PROD_DIR


def _replay_event(day: pd.Timestamp, rows: pd.DataFrame, models_dir: Path, tmp: Path) -> list[dict]:
    """Live feature vectors for *rows* (one event), from a DB truncated before *day*."""
    d = day.strftime("%Y-%m-%d")
    db = tmp / "ufcstats_truncated.db"
    shutil.copy(DB_PATH, db)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM fights WHERE date >= ?", (d,))
    for table in ("fight_stats", "fight_stats_rounds"):
        try:
            conn.execute(f"DELETE FROM {table} WHERE fight_id NOT IN (SELECT fight_id FROM fights)")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()

    captured: list[pd.DataFrame] = []
    original_bfv = predict.build_feature_vector

    def _capture(*args, **kwargs):
        X = original_bfv(*args, **kwargs)
        captured.append(X)
        return X

    predict.build_feature_vector = _capture
    predict.DB_PATH = db
    out = []
    try:
        for _, f in rows.iterrows():
            captured.clear()
            try:
                predict.compute_prediction(
                    f["r_name"], f["b_name"], "ensemble", division=f["division"],
                    title_fight=int(f["title_fight"]), db_path=DB_V1_PATH,
                    models_dir=models_dir, as_of=d,
                )
            except (SystemExit, ValueError) as exc:
                print(f"  [skip] {f['r_name']} vs {f['b_name']}: {exc}")
                continue
            live = {}
            for X in captured:
                for col in X.columns:
                    live.setdefault(col, float(X[col].iloc[0]))
            out.append({"fight_id": f["fight_id"], **live})
    finally:
        predict.build_feature_vector = original_bfv
        predict.DB_PATH = DB_PATH
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Live-vs-training feature parity check.")
    parser.add_argument("--events", type=int, default=4, help="number of most recent events to replay")
    parser.add_argument("--min-match", type=float, default=0.9,
                        help="minimum share of fights whose live value equals the training value")
    parser.add_argument("--dump", type=Path, default=None,
                        help="write per-fight live vs training values to this CSV for debugging")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    models_dir = MODELS_V1_PROD_DIR if any(MODELS_V1_PROD_DIR.glob("*.joblib")) else MODELS_V1_DIR

    train = pd.read_csv(CSV_V1_WITH_ELO)
    train["date"] = pd.to_datetime(train["date"])
    v1 = sqlite3.connect(DB_V1_PATH)
    names = dict(v1.execute("SELECT fighter_id, name FROM fighters"))
    ids = pd.read_sql("SELECT fight_id, r_fighter_id, b_fighter_id FROM fights", v1)
    v1.close()
    train = train.merge(ids, on="fight_id")
    train["r_name"] = train["r_fighter_id"].map(names)
    train["b_name"] = train["b_fighter_id"].map(names)
    days = sorted(train["date"].unique())[-args.events:]

    live_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for day in days:
            rows = train[train["date"] == day]
            print(f"Replaying {pd.Timestamp(day).date()} ({len(rows)} fights) ...", flush=True)
            live_rows += _replay_event(pd.Timestamp(day), rows, models_dir, Path(tmp))
    if not live_rows:
        print("No fights replayed.")
        return 1

    live = pd.DataFrame(live_rows).set_index("fight_id")
    ref = train.set_index("fight_id").loc[live.index]
    # Only the win models' features gate the check; finish-type-only ones are
    # reported for information.
    win_feats = set()
    for key in ("xgb", "lr", "rf", "lgbm", "mlp"):
        path = models_dir / f"{key}_features.joblib"
        if path.exists():
            win_feats |= set(joblib.load(path))
    feats = [c for c in live.columns if c in ref.columns]
    if args.dump:
        side = live[feats].add_prefix("live_").join(ref[feats].add_prefix("train_"))
        side.join(ref[["date", "r_name", "b_name"]]).to_csv(args.dump)
        print(f"Wrote per-fight values to {args.dump}")
    report = []
    for f in feats:
        a = live[f].astype(float).values
        b = ref[f].astype(float).fillna(0).values
        # ufc-master.csv stores most columns rounded to 2 decimals, so a diff
        # of two rounded values can be off by 0.01 without a real mismatch.
        tol = np.maximum(0.011, 0.01 * np.abs(b))
        both_vary = a.std() > 0 and b.std() > 0
        report.append({
            "feature": f,
            "model": "win" if f in win_feats else "finish-only",
            "match": float((np.abs(a - b) <= tol).mean()),
            "corr": float(np.corrcoef(a, b)[0, 1]) if both_vary else np.nan,
            "live_zero": float((a == 0).mean()),
            "train_zero": float((b == 0).mean()),
            "scale": float(np.abs(a).mean() / np.abs(b).mean()) if np.abs(b).mean() > 0 else np.nan,
        })
    R = pd.DataFrame(report).sort_values("match")
    pd.set_option("display.width", 200)
    print(f"\n{len(live)} fights, {len(feats)} features. Worst first:\n")
    print(R.round(3).to_string(index=False))

    bad = R[(R["match"] < args.min_match) & (R["model"] == "win")]
    if len(bad):
        print(f"\n[FAIL] {len(bad)} feature(s) match training in < {args.min_match:.0%} of fights: "
              f"{', '.join(bad['feature'])}")
        return 1
    print(f"\n[ OK ] all features match training in >= {args.min_match:.0%} of fights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
