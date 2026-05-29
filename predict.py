"""
predict.py — UFC Fight Outcome Predictor CLI
============================================

Usage
-----
    python predict.py "Fighter A" "Fighter B"
    python predict.py "Islam Makhachev" "Charles Oliveira" --model lr
    python predict.py "Jones" "Miocic" --division "light heavyweight" --title

Arguments
---------
    red_fighter   Name of the Red corner fighter (partial names OK)
    blue_fighter  Name of the Blue corner fighter (partial names OK)
    --model       Model to use: 'xgb' (default) or 'lr'
    --division    Weight division (optional — for division feature encoding)
    --title       Flag if this is a title fight (default: False)

Notes
-----
- Models must have been trained first (XGBoost.py / logistic_regression.py).
- Fighter stats are taken from their most recent recorded fight.
- ELO ratings are computed by replaying all historical fights.
- Recent form (win rate, finish rate, win streak) is computed from fight history.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Allow imports from project root ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DB_PATH,
    STARTING_ELO, K_FACTOR_NORMAL, K_FACTOR_PROVISIONAL, PROVISIONAL_LIMIT,
    MODEL_XGB_PATH, MODEL_XGB_FEATURES,
    MODEL_LR_PATH, MODEL_LR_SCALER, MODEL_LR_FEATURES,
    MODEL_FINISH_PATH, MODEL_FINISH_FEATURES,
    FINISH_CLASS_NAMES,
    DIVISIONS,
    EXCLUDE_STAT_KEYWORDS,
    RECENT_FORM_WINDOW,
    FINISH_METHOD_MAP,
)
from ML_models.ELO_calculator import get_current_ratings_by_division
from odds import print_value_bet_summary
from logger import get_logger

log = get_logger(__name__)

_EPS = 1e-6


# ── Fighter Resolution ────────────────────────────────────────────────────────

def search_fighter(conn: sqlite3.Connection, name: str) -> list[tuple]:
    """Return (fighter_id, name) pairs whose name contains *name* (case-insensitive)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT fighter_id, name FROM fighters WHERE name LIKE ? ORDER BY name",
        (f"%{name}%",),
    )
    return cur.fetchall()


def resolve_fighter(conn: sqlite3.Connection, query: str) -> tuple[str, str]:
    """
    Find a fighter by partial name.  If multiple matches are found, prompt the
    user to choose.  Returns (fighter_id, full_name).
    """
    matches = search_fighter(conn, query)
    if not matches:
        print(f"\n[ERROR]  No fighter found matching '{query}'.")
        print("    Tip: try a shorter part of the name, e.g. 'McGregor' instead of 'Conor McGregor'.")
        sys.exit(1)

    if len(matches) == 1:
        return matches[0]

    # Exact match check (case-insensitive)
    exact = [m for m in matches if m[1].lower() == query.lower()]
    if len(exact) == 1:
        return exact[0]

    print(f"\nMultiple fighters found for '{query}':")
    display = matches[:10]
    for i, (fid, fname) in enumerate(display, 1):
        print(f"  {i:2}. {fname}")
    while True:
        try:
            choice = int(input("Enter number: ")) - 1
            if 0 <= choice < len(display):
                return display[choice]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid — enter a number from the list above.")


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_latest_stats(conn: sqlite3.Connection, fighter_id: str) -> pd.Series:
    """
    Fetch the most recent fight_stats row for a fighter.
    These are pre-fight rolling stats — best approximation of current skill.
    Returns an empty Series if the fighter has no recorded fights.
    """
    query = """
        SELECT fs.*
        FROM fight_stats AS fs
        JOIN fights AS f ON fs.fight_id = f.fight_id
        WHERE fs.fighter_id = ?
        ORDER BY f.date DESC
        LIMIT 1
    """
    df = pd.read_sql_query(query, conn, params=(fighter_id,))
    return df.iloc[0] if not df.empty else pd.Series(dtype=float)


# ── ELO ───────────────────────────────────────────────────────────────────────

def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 200))


def compute_current_elo(conn: sqlite3.Connection) -> dict[str, float]:
    """
    Replay every fight in chronological order and return each fighter's ELO
    rating *after* their most recent bout.
    """
    df = pd.read_sql_query(
        "SELECT r_fighter_id, b_fighter_id, winner_id FROM fights ORDER BY date ASC",
        conn,
    )
    ratings: dict[str, float] = {}
    counts:  dict[str, int]   = {}

    for _, row in df.iterrows():
        r_id, b_id, winner = row["r_fighter_id"], row["b_fighter_id"], row["winner_id"]

        r_elo = ratings.get(r_id, STARTING_ELO)
        b_elo = ratings.get(b_id, STARTING_ELO)
        k_r   = K_FACTOR_PROVISIONAL if counts.get(r_id, 0) < PROVISIONAL_LIMIT else K_FACTOR_NORMAL
        k_b   = K_FACTOR_PROVISIONAL if counts.get(b_id, 0) < PROVISIONAL_LIMIT else K_FACTOR_NORMAL

        exp_r   = _expected_score(r_elo, b_elo)
        score_r = 1.0 if winner == r_id else (0.0 if winner == b_id else 0.5)

        ratings[r_id] = r_elo + k_r * (score_r - exp_r)
        ratings[b_id] = b_elo + k_b * ((1 - score_r) - (1 - exp_r))
        counts[r_id]  = counts.get(r_id, 0) + 1
        counts[b_id]  = counts.get(b_id, 0) + 1

    return ratings


# ── Recent Form ───────────────────────────────────────────────────────────────

def _is_finish_method(method: str | None) -> int:
    if not isinstance(method, str):
        return 0
    return int(method in FINISH_METHOD_MAP and FINISH_METHOD_MAP[method] > 0)


def compute_recent_form(
    conn: sqlite3.Connection,
    fighter_id: str,
    window: int = RECENT_FORM_WINDOW,
) -> dict[str, float]:
    """
    Compute recent form stats for a single fighter from their full fight history.
    Returns: {recent_win_rate, recent_finish_rate, win_streak}
    """
    df = pd.read_sql_query(
        """
        SELECT f.date, f.winner_id, f.method,
               f.r_fighter_id, f.b_fighter_id
        FROM fights f
        WHERE f.r_fighter_id = ? OR f.b_fighter_id = ?
        ORDER BY f.date ASC, f.fight_id ASC
        """,
        conn,
        params=(fighter_id, fighter_id),
    )

    if df.empty:
        return {"recent_win_rate": 0.0, "recent_finish_rate": 0.0, "win_streak": 0}

    df["won"]      = (df["winner_id"] == fighter_id).astype(int)
    df["finished"] = df["method"].apply(_is_finish_method)

    # Recent window (last `window` fights excluding the hypothetical next fight)
    recent = df.tail(window)

    win_rate    = recent["won"].mean()
    finish_rate = recent["finished"].mean()

    # Win streak: count consecutive wins from the end
    streak = 0
    for w in reversed(df["won"].tolist()):
        if w == 1:
            streak += 1
        else:
            break

    return {
        "recent_win_rate":    float(win_rate),
        "recent_finish_rate": float(finish_rate),
        "win_streak":         float(streak),
    }


# ── Style Ratios ──────────────────────────────────────────────────────────────

def _style_ratios(stats: pd.Series) -> tuple[float, float]:
    """Return (grapple_ratio, strike_ratio) from a fighter's stats."""
    splm   = float(pd.to_numeric(stats.get("splm",   0), errors="coerce") or 0)
    td_avg = float(pd.to_numeric(stats.get("td_avg", 0), errors="coerce") or 0)
    denom  = splm + td_avg + _EPS
    return td_avg / denom, splm / denom


# ── Feature Vector ────────────────────────────────────────────────────────────

def build_feature_vector(
    red_stats:    pd.Series,
    blue_stats:   pd.Series,
    elo_r:        float,
    elo_b:        float,
    form_r:       dict[str, float],
    form_b:       dict[str, float],
    division:     str | None,
    title_fight:  int,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute all features the model expects.

    Handles:
      - Standard _diff features (Red − Blue rolling stats)
      - elo_diff
      - is_debutant_diff
      - recent_win_rate_diff / recent_finish_rate_diff / win_streak_diff
      - age_diff  (set to 0 — DOB not typically known for future fights)
      - grapple_ratio_diff / strike_ratio_diff / striker_vs_wrestler / wrestler_vs_striker
      - div_*  (one-hot division encoding)
      - title_fight  (direct binary feature)
    """
    grapple_r, strike_r = _style_ratios(red_stats)
    grapple_b, strike_b = _style_ratios(blue_stats)

    r_time = float(red_stats.get("total_fight_time",  0) or 0)
    b_time = float(blue_stats.get("total_fight_time", 0) or 0)

    # Division one-hot lookup
    div_lower = (division or "").lower().strip()

    def _div_col(name: str) -> str:
        return "div_" + name.replace(" ", "_").replace("'", "")

    row: dict = {}

    for feat in feature_names:

        # ── ELO ──────────────────────────────────────────────────────────────
        if feat == "elo_diff":
            row[feat] = elo_r - elo_b

        # ── Debutant ─────────────────────────────────────────────────────────
        elif feat == "is_debutant_diff":
            row[feat] = int(r_time == 0) - int(b_time == 0)

        # ── Recent form ───────────────────────────────────────────────────────
        elif feat in ("recent_win_rate_diff", "recent_finish_rate_diff", "win_streak_diff"):
            base = feat[: -len("_diff")]
            row[feat] = form_r.get(base, 0.0) - form_b.get(base, 0.0)

        # ── Age ───────────────────────────────────────────────────────────────
        elif feat == "age_diff":
            row[feat] = 0.0   # DOB unknown for hypothetical future fights

        # ── Style matchup ─────────────────────────────────────────────────────
        elif feat == "grapple_ratio_diff":
            row[feat] = grapple_r - grapple_b
        elif feat == "strike_ratio_diff":
            row[feat] = strike_r - strike_b
        elif feat == "striker_vs_wrestler":
            row[feat] = strike_r * grapple_b
        elif feat == "wrestler_vs_striker":
            row[feat] = grapple_r * strike_b

        # ── Division (one-hot) ────────────────────────────────────────────────
        elif feat.startswith("div_"):
            matched = bool(div_lower) and any(feat == _div_col(d) and div_lower == d for d in DIVISIONS)
            row[feat] = int(matched)

        # ── Title fight ───────────────────────────────────────────────────────
        elif feat == "title_fight":
            row[feat] = title_fight

        # ── Standard _diff features ───────────────────────────────────────────
        elif feat.endswith("_diff"):
            base  = feat[: -len("_diff")]
            r_val = pd.to_numeric(red_stats.get(base,  0), errors="coerce")
            b_val = pd.to_numeric(blue_stats.get(base, 0), errors="coerce")
            row[feat] = (r_val if not pd.isna(r_val) else 0) - (b_val if not pd.isna(b_val) else 0)

        else:
            row[feat] = 0  # unknown feature — default to 0

    return pd.DataFrame([row])[feature_names]   # enforce exact column order


# ── Core Prediction Logic ─────────────────────────────────────────────────────

def predict_fight(
    red_name:        str,
    blue_name:       str,
    model_type:      str = "xgb",
    division:        str | None = None,
    title_fight:     int = 0,
    odds_red:        float | None = None,
    odds_blue:       float | None = None,
) -> None:

    # ── Load saved artifacts ──────────────────────────────────────────────────
    if model_type == "xgb":
        model_path    = MODEL_XGB_PATH
        features_path = MODEL_XGB_FEATURES
        scaler_path   = None
        model_label   = "XGBoost"
    else:
        model_path    = MODEL_LR_PATH
        features_path = MODEL_LR_FEATURES
        scaler_path   = MODEL_LR_SCALER
        model_label   = "Logistic Regression"

    if not model_path.exists():
        script = "ML_models/XGBoost.py" if model_type == "xgb" else "ML_models/logistic_regression.py"
        log.error("No saved model found at '%s'. Run  python %s  first.", model_path, script)
        print(f"\n[ERROR]  No saved model found at '{model_path}'.")
        print(f"   Run  python {script}  first to train and save the model.")
        sys.exit(1)

    artifact      = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    scaler        = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None

    # LR is saved as {"base": model, "platt": calibrator}; XGBoost is saved directly
    if isinstance(artifact, dict):
        base_model  = artifact["base"]
        platt       = artifact["platt"]
        model       = None
    else:
        model       = artifact
        base_model  = None
        platt       = None

    # Finish-type model (optional)
    finish_model   = joblib.load(MODEL_FINISH_PATH)     if MODEL_FINISH_PATH.exists()     else None
    finish_feats   = joblib.load(MODEL_FINISH_FEATURES) if MODEL_FINISH_FEATURES.exists() else None

    # ── DB connection ─────────────────────────────────────────────────────────
    if not DB_PATH.exists():
        print(f"\n[ERROR]  Database not found at '{DB_PATH}'.")
        print("   Run the database builder scripts first.")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    # ── Resolve fighters ──────────────────────────────────────────────────────
    r_id, r_name = resolve_fighter(conn, red_name)
    b_id, b_name = resolve_fighter(conn, blue_name)

    if r_id == b_id:
        print("\n[WARN]  Both names resolved to the same fighter — please check the names.")
        conn.close()
        sys.exit(1)

    print(f"\n[FIGHT]  {r_name}  (Red)  vs  {b_name}  (Blue)")
    if division:
        print(f"         Division: {division.title()}" + ("  [TITLE FIGHT]" if title_fight else ""))
    print("-" * 56)

    # ── Get rolling stats ─────────────────────────────────────────────────────
    red_stats  = get_latest_stats(conn, r_id)
    blue_stats = get_latest_stats(conn, b_id)

    if red_stats.empty:
        print(f"[WARN]  No fight history for {r_name} -- all stats set to 0.")
    if blue_stats.empty:
        print(f"[WARN]  No fight history for {b_name} -- all stats set to 0.")

    # ── ELO ───────────────────────────────────────────────────────────────────
    log.info("Computing current ELO ratings…")
    div_lower = (division or "").lower().strip()
    if div_lower:
        # Per-division ELO — matches how the model was trained
        div_elo = get_current_ratings_by_division(conn)
        elo_r   = div_elo.get((r_id, div_lower), STARTING_ELO)
        elo_b   = div_elo.get((b_id, div_lower), STARTING_ELO)
    else:
        # Global ELO fallback when no division specified
        elo_ratings = compute_current_elo(conn)
        elo_r       = elo_ratings.get(r_id, STARTING_ELO)
        elo_b       = elo_ratings.get(b_id, STARTING_ELO)

    # ── Recent form ───────────────────────────────────────────────────────────
    log.info("Computing recent form…")
    form_r = compute_recent_form(conn, r_id)
    form_b = compute_recent_form(conn, b_id)

    conn.close()

    # ── Build & predict ───────────────────────────────────────────────────────
    X = build_feature_vector(
        red_stats, blue_stats,
        elo_r, elo_b,
        form_r, form_b,
        division, title_fight,
        feature_names,
    ).fillna(0)

    X_input = scaler.transform(X) if scaler is not None else X.values

    if model is not None:
        proba = model.predict_proba(X_input)[0]
    else:
        raw_prob   = base_model.predict_proba(X_input)[0, 1]
        calibrated = platt.predict_proba([[raw_prob]])[0, 1]
        proba      = [1 - calibrated, calibrated]

    red_win_prob  = float(proba[1])
    blue_win_prob = float(proba[0])
    winner_name   = r_name if red_win_prob >= 0.5 else b_name
    confidence    = max(red_win_prob, blue_win_prob)

    # ── Finish-type prediction ────────────────────────────────────────────────
    finish_proba = None
    if finish_model is not None and finish_feats is not None:
        # Use same feature vector but aligned to finish model's feature list
        X_fin = build_feature_vector(
            red_stats, blue_stats,
            elo_r, elo_b,
            form_r, form_b,
            division, title_fight,
            finish_feats,
        ).fillna(0)
        finish_proba = finish_model.predict_proba(X_fin.values)[0]

    # ── Display results ───────────────────────────────────────────────────────
    _bar = lambda p: "#" * int(p * 20) + "-" * (20 - int(p * 20))

    print(f"\n  ELO:  {r_name} = {elo_r:.0f}  |  {b_name} = {elo_b:.0f}")
    print(f"\n  Recent form (last {RECENT_FORM_WINDOW} fights):")
    print(f"    {r_name[:25]:25s}  win_rate={form_r['recent_win_rate']:.0%}  "
          f"finish_rate={form_r['recent_finish_rate']:.0%}  "
          f"streak={int(form_r['win_streak'])}")
    print(f"    {b_name[:25]:25s}  win_rate={form_b['recent_win_rate']:.0%}  "
          f"finish_rate={form_b['recent_finish_rate']:.0%}  "
          f"streak={int(form_b['win_streak'])}")

    print(f"\n  Predicted Winner: {winner_name}  ({confidence:.1%} confidence)")
    print()
    print(f"  {r_name} (Red)")
    print(f"  {_bar(red_win_prob)} {red_win_prob:.1%}")
    print()
    print(f"  {b_name} (Blue)")
    print(f"  {_bar(blue_win_prob)} {blue_win_prob:.1%}")
    print(f"\n  Model: {model_label}")

    if finish_proba is not None:
        print(f"\n  Predicted Finish Method:")
        for i, (name, p) in enumerate(zip(FINISH_CLASS_NAMES, finish_proba)):
            print(f"    {name:12s}  {_bar(p)} {p:.1%}")

    # ── Betting odds value-bet analysis ───────────────────────────────────────
    print_value_bet_summary(
        r_name, b_name,
        red_win_prob, blue_win_prob,
        odds_red_american=odds_red,
        odds_blue_american=odds_blue,
    )


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the outcome of a UFC fight.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python predict.py "Islam Makhachev" "Charles Oliveira"
  python predict.py "Conor McGregor" "Khabib Nurmagomedov" --model lr
  python predict.py "Jones" "Miocic" --division "light heavyweight" --title
  python predict.py "Holloway" "Poirier" --division lightweight
        """,
    )
    parser.add_argument("red_fighter",  help="Red corner fighter name (partial OK)")
    parser.add_argument("blue_fighter", help="Blue corner fighter name (partial OK)")
    parser.add_argument(
        "--model",
        choices=["xgb", "lr"],
        default="xgb",
        help="Model: 'xgb' = XGBoost (default), 'lr' = Logistic Regression",
    )
    parser.add_argument(
        "--division",
        default=None,
        help="Weight division (e.g. 'lightweight', 'welterweight'). Enables division encoding.",
    )
    parser.add_argument(
        "--title",
        action="store_true",
        default=False,
        help="Flag if this is a title fight.",
    )
    parser.add_argument(
        "--odds-red",
        type=float,
        default=None,
        help="American moneyline odds for Red corner (e.g. -150 or +200).",
    )
    parser.add_argument(
        "--odds-blue",
        type=float,
        default=None,
        help="American moneyline odds for Blue corner (e.g. -150 or +200).",
    )

    args = parser.parse_args()
    predict_fight(
        args.red_fighter,
        args.blue_fighter,
        args.model,
        division=args.division,
        title_fight=int(args.title),
        odds_red=args.odds_red,
        odds_blue=args.odds_blue,
    )


if __name__ == "__main__":
    main()
