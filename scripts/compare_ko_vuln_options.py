"""
scripts/compare_ko_vuln_options.py

Compares two ko_vuln feature definitions without re-running the full pipeline:

  Option 0: baseline -- 3-fight window, KO losses only (current production)
  Option 1: combined -- KO losses + kd received, all history
  Option 2: separate -- KO losses only (all history) + kd_received as new feature

Trains XGBoost, LR, RF, LightGBM and a soft-vote ensemble for each option.
Train set: all fights before 2025. Test set: 2025+ (honest out-of-sample).
Ensemble weights tuned on first half of 2025+, accuracy reported on second half.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    CSV_V1_WITH_ELO,
    DB_PATH,
    DB_V1_PATH,
    EXCLUDED_FEATURES,
    FINISH_METHOD_MAP,
    LGBM_PARAMS,
    LR_PARAMS,
    META_COLS,
    RF_PARAMS,
    SAMPLE_WEIGHT_ALPHA,
    SAMPLE_WEIGHT_BETA,
    TARGET_COL,
    XGB_PARAMS,
    RANDOM_STATE,
)
from ml.ML_data_preparation import compute_sample_weights
from ml.ML_data_preparation_v1 import make_symmetric

optuna.logging.set_verbosity(optuna.logging.WARNING)

BACKTEST_FROM = "2025-01-01"
OPTUNA_TRIALS = 60   # per restart
OPTUNA_RESTARTS = 3


# ── Feature computation ──────────────────────────────────────────────────────

def _compute_kovuln_variants(conn: sqlite3.Connection) -> pd.DataFrame:
    fights_df = pd.read_sql_query(
        "SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id, method "
        "FROM fights ORDER BY date ASC, fight_id ASC",
        conn,
    )
    kd_df = pd.read_sql_query(
        "SELECT fight_id, fighter_id, CAST(kd AS INTEGER) AS kd "
        "FROM fight_stats WHERE kd IS NOT NULL",
        conn,
    )
    opp = kd_df.rename(columns={"fighter_id": "opp_id", "kd": "kd_received"})
    km = kd_df.merge(opp, on="fight_id", how="left")
    km = km[km["fighter_id"] != km["opp_id"]]
    kd_lookup = {(r["fight_id"], r["fighter_id"]): int(r["kd_received"] or 0)
                 for _, r in km.iterrows()}

    date_map = fights_df.set_index("fight_id")["date"].to_dict()
    long_rows = []
    for _, row in fights_df.iterrows():
        method_cls = FINISH_METHOD_MAP.get(row["method"], -1)
        winner_id  = row["winner_id"]
        for fid in (row["r_fighter_id"], row["b_fighter_id"]):
            is_win     = winner_id == fid
            ko_stopped = int(not is_win and winner_id is not None and method_cls == 1)
            kd_recv    = kd_lookup.get((row["fight_id"], fid), 0)
            long_rows.append({
                "ufc_fight_id": row["fight_id"],
                "ufc_fighter_id": fid,
                "date": date_map[row["fight_id"]],
                "ko_stopped": ko_stopped,
                "kd_received": kd_recv,
            })

    long = pd.DataFrame(long_rows)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["ufc_fighter_id", "date", "ufc_fight_id"]).reset_index(drop=True)
    grp = long.groupby("ufc_fighter_id", sort=False)
    long["ko_losses"]        = grp["ko_stopped"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    long["kd_received_cum"]  = grp["kd_received"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    long["ko_vuln_combined"] = long["ko_losses"] + long["kd_received_cum"]
    return long[["ufc_fight_id", "ufc_fighter_id", "ko_vuln_combined", "ko_losses", "kd_received_cum"]]


def _build_fight_mapping(mab_conn, ufc_conn) -> pd.DataFrame:
    mdf = pd.read_sql_query(
        "SELECT f.fight_id, f.date, r.name r_name, b.name b_name "
        "FROM fights f "
        "JOIN fighters r ON f.r_fighter_id = r.fighter_id "
        "JOIN fighters b ON f.b_fighter_id = b.fighter_id",
        mab_conn,
    )
    udf = pd.read_sql_query(
        "SELECT f.fight_id ufc_fight_id, f.date, r.name r_name, b.name b_name, "
        "f.r_fighter_id ufc_r_id, f.b_fighter_id ufc_b_id "
        "FROM fights f "
        "JOIN fighters r ON f.r_fighter_id = r.fighter_id "
        "JOIN fighters b ON f.b_fighter_id = b.fighter_id",
        ufc_conn,
    )
    for df in (mdf, udf):
        df["r_key"] = df["r_name"].str.lower().str.strip()
        df["b_key"] = df["b_name"].str.lower().str.strip()
    return mdf.merge(
        udf[["ufc_fight_id", "date", "r_key", "b_key", "ufc_r_id", "ufc_b_id"]],
        on=["date", "r_key", "b_key"], how="inner",
    )[["fight_id", "ufc_fight_id", "ufc_r_id", "ufc_b_id"]]


def _patch(ml_df: pd.DataFrame, mapping: pd.DataFrame,
           variants: pd.DataFrame, option: int) -> pd.DataFrame:
    df = ml_df.merge(mapping, on="fight_id", how="left")

    var_r = variants.rename(columns={
        "ufc_fighter_id": "ufc_r_id",
        "ko_vuln_combined": "ko_vuln_combined_r",
        "ko_losses": "ko_losses_r",
        "kd_received_cum": "kd_received_r",
    })
    var_b = variants.rename(columns={
        "ufc_fighter_id": "ufc_b_id",
        "ko_vuln_combined": "ko_vuln_combined_b",
        "ko_losses": "ko_losses_b",
        "kd_received_cum": "kd_received_b",
    })
    df = df.merge(
        var_r[["ufc_fight_id", "ufc_r_id", "ko_vuln_combined_r", "ko_losses_r", "kd_received_r"]],
        on=["ufc_fight_id", "ufc_r_id"], how="left")
    df = df.merge(
        var_b[["ufc_fight_id", "ufc_b_id", "ko_vuln_combined_b", "ko_losses_b", "kd_received_b"]],
        on=["ufc_fight_id", "ufc_b_id"], how="left")

    matched = df["ufc_fight_id"].notna().sum()
    print(f"    Matched {matched}/{len(df)} rows to UFCStats fights")

    if option == 1:
        diff = df["ko_vuln_combined_r"].fillna(0) - df["ko_vuln_combined_b"].fillna(0)
        df["ko_vuln_diff"] = np.where(df["ufc_fight_id"].notna(), diff, df["ko_vuln_diff"])
    elif option == 2:
        diff = df["ko_losses_r"].fillna(0) - df["ko_losses_b"].fillna(0)
        df["ko_vuln_diff"] = np.where(df["ufc_fight_id"].notna(), diff, df["ko_vuln_diff"])
        df["kd_received_diff"] = df["kd_received_r"].fillna(0) - df["kd_received_b"].fillna(0)

    drop = [c for c in df.columns if c in {
        "ufc_fight_id", "ufc_r_id", "ufc_b_id",
        "ko_vuln_combined_r", "ko_vuln_combined_b",
        "ko_losses_r", "ko_losses_b",
        "kd_received_r", "kd_received_b",
    }]
    return df.drop(columns=drop)


# ── Model training helpers ───────────────────────────────────────────────────

def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = set(META_COLS) | set(EXCLUDED_FEATURES) | {"glicko_diff", "glicko_rd_diff"}
    return [c for c in df.columns if c not in exclude and c != TARGET_COL]


def _preprocess(df: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[TARGET_COL].values
    return X.values, y


def _train_base_models(train_df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Train XGB, LR, RF, LGBM on training data. Returns dict of fitted models."""
    sym = make_symmetric(train_df)
    X_sym, y_sym = _preprocess(sym, feat_cols)
    w = compute_sample_weights(sym["date"])

    # LR: train on first 80% of sym, calibrate on last 20%
    n = len(sym)
    cal_cut = int(n * 0.80)
    X_lr_tr, y_lr_tr = X_sym[:cal_cut], y_sym[:cal_cut]
    X_lr_cal, y_lr_cal = X_sym[cal_cut:], y_sym[cal_cut:]
    w_lr = w[:cal_cut] if w is not None else None

    scaler = StandardScaler()
    X_lr_tr_s  = scaler.fit_transform(X_lr_tr)
    X_lr_cal_s = scaler.transform(X_lr_cal)

    base_lr = LogisticRegression(**LR_PARAMS)
    base_lr.fit(X_lr_tr_s, y_lr_tr, sample_weight=w_lr)
    raw_cal = base_lr.predict_proba(X_lr_cal_s)[:, 1]
    platt   = LogisticRegression(C=1.0, max_iter=200)
    platt.fit(raw_cal.reshape(-1, 1), y_lr_cal)

    xgb  = XGBClassifier(**{k: v for k, v in XGB_PARAMS.items() if k != "eval_metric"},
                         eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE)
    xgb.fit(X_sym, y_sym, sample_weight=w, verbose=False)

    rf = RandomForestClassifier(**RF_PARAMS, random_state=RANDOM_STATE)
    rf.fit(X_sym, y_sym, sample_weight=w)

    lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=RANDOM_STATE, verbosity=-1)
    lgbm.fit(X_sym, y_sym, sample_weight=w)

    return {
        "xgb":    xgb,
        "lr":     (base_lr, scaler, platt),
        "rf":     rf,
        "lgbm":   lgbm,
    }


def _get_proba(models: dict, X: np.ndarray, feat_cols: list[str]) -> dict[str, np.ndarray]:
    base_lr, scaler, platt = models["lr"]
    Xs = scaler.transform(X)
    raw_lr = base_lr.predict_proba(Xs)[:, 1]
    proba_lr = platt.predict_proba(raw_lr.reshape(-1, 1))[:, 1]

    return {
        "xgb":  models["xgb"].predict_proba(X)[:, 1],
        "lr":   proba_lr,
        "rf":   models["rf"].predict_proba(X)[:, 1],
        "lgbm": models["lgbm"].predict_proba(X)[:, 1],
    }


def _tune_ensemble(probas: dict, y: np.ndarray) -> dict[str, float]:
    """Optuna weight search on probas dict. Returns best weights summing to 1."""
    keys = list(probas.keys())
    P = np.column_stack([probas[k] for k in keys])

    best_acc   = -1.0
    best_w_raw = None

    for _ in range(OPTUNA_RESTARTS):
        study = optuna.create_study(direction="maximize")

        def obj(trial):
            raw = np.array([trial.suggest_float(k, 0.0, 1.0) for k in keys])
            if raw.sum() < 1e-9:
                return 0.0
            w = raw / raw.sum()
            proba = P @ w
            return float(((proba >= 0.5).astype(int) == y).mean()) if len(y) else 0.0

        study.optimize(obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        if study.best_value > best_acc:
            best_acc   = study.best_value
            best_w_raw = np.array([study.best_params[k] for k in keys])

    w_norm = best_w_raw / best_w_raw.sum()
    return {k: float(w_norm[i]) for i, k in enumerate(keys)}


def _eval_ensemble(probas: dict, weights: dict, y: np.ndarray) -> float:
    keys  = list(weights.keys())
    P     = np.column_stack([probas[k] for k in keys])
    w     = np.array([weights[k] for k in keys])
    proba = P @ w
    return float(((proba >= 0.5).astype(int) == y).mean())


# ── Main comparison ──────────────────────────────────────────────────────────

def run_option(label: str, df: pd.DataFrame) -> float:
    print(f"\n--- {label} ---")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feat_cols = _feature_cols(df)
    print(f"  Features: {len(feat_cols)}")

    train_df = df[df["date"] < BACKTEST_FROM].copy()
    test_df  = df[df["date"] >= BACKTEST_FROM].copy()

    tune_cut   = len(test_df) // 2
    tune_df    = test_df.iloc[:tune_cut]
    hold_df    = test_df.iloc[tune_cut:]

    print(f"  Train: {len(train_df)} | Tune: {len(tune_df)} | Hold: {len(hold_df)}")

    print("  Training base models (XGB, LR, RF, LGBM)...")
    models = _train_base_models(train_df, feat_cols)

    X_tune, y_tune = _preprocess(tune_df, feat_cols)
    X_hold, y_hold = _preprocess(hold_df, feat_cols)
    X_test, y_test = _preprocess(test_df, feat_cols)

    tune_probas = _get_proba(models, X_tune, feat_cols)
    hold_probas = _get_proba(models, X_hold, feat_cols)
    test_probas = _get_proba(models, X_test, feat_cols)

    # Individual model accuracies on full 2025+ set
    for name, p in test_probas.items():
        acc = ((p >= 0.5).astype(int) == y_test).mean()
        print(f"    {name:<6s} {acc:.1%}")

    print(f"  Tuning ensemble weights ({OPTUNA_RESTARTS}x{OPTUNA_TRIALS} Optuna trials)...")
    weights = _tune_ensemble(tune_probas, y_tune)
    for k, w in weights.items():
        print(f"    {k:<6s} {w:.3f}")

    hold_acc = _eval_ensemble(hold_probas, weights, y_hold)
    full_acc = _eval_ensemble(test_probas, weights, y_test)
    print(f"  Ensemble hold-out (2nd half 2025+): {hold_acc:.1%}")
    print(f"  Ensemble full 2025+ ({len(y_test)} fights):  {full_acc:.1%}")
    return hold_acc, full_acc


def main():
    print("Loading ML feature CSV...")
    ml_df = pd.read_csv(CSV_V1_WITH_ELO, low_memory=False)
    print(f"  {len(ml_df)} rows")

    print("\nConnecting to databases and computing ko_vuln variants...")
    ufc_conn = sqlite3.connect(str(DB_PATH))
    mab_conn = sqlite3.connect(str(DB_V1_PATH))
    variants = _compute_kovuln_variants(ufc_conn)
    mapping  = _build_fight_mapping(mab_conn, ufc_conn)
    ufc_conn.close()
    mab_conn.close()
    print(f"  Variants computed, {len(mapping)} fights mapped")

    h0, f0 = run_option("Option 0 -- baseline (3-fight window, no kd)", ml_df.copy())
    h1, f1 = run_option("Option 1 -- combined (KO losses + kd, all history)",
                        _patch(ml_df, mapping, variants, option=1))
    h2, f2 = run_option("Option 2 -- separate (KO losses all + kd_received_diff)",
                        _patch(ml_df, mapping, variants, option=2))

    print("\n" + "="*60)
    print("RESULTS (ensemble)")
    print("="*60)
    print(f"{'Option':<40s} {'Hold-out':>9} {'Full 2025+':>11}")
    print("-"*60)
    print(f"{'Baseline (3-fight window):':<40s} {h0:>9.1%} {f0:>11.1%}")
    print(f"{'Option 1 (combined KO+kd):':<40s} {h1:>9.1%} {f1:>11.1%}")
    print(f"{'Option 2 (split KO + kd_received):':<40s} {h2:>9.1%} {f2:>11.1%}")


if __name__ == "__main__":
    main()
