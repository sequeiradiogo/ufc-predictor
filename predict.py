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
    --model       Model to use: 'xgb' (default), 'lr', 'rf', 'lgbm', or 'mlp'
    --division    Weight division (optional — for division feature encoding)
    --title       Flag if this is a title fight (default: False)

Notes
-----
- Models must have been trained first (XGBoost.py / logistic_regression.py / random_forest.py / lightgbm_model.py).
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
    DB_V1_PATH,
    MODELS_DIR,
    MODELS_V1_DIR,
    MODELS_V1_PROD_DIR,
    STARTING_ELO, K_FACTOR_NORMAL, K_FACTOR_PROVISIONAL, PROVISIONAL_LIMIT,
    GLICKO_START_R, GLICKO_START_RD,
    MODEL_XGB_PATH, MODEL_XGB_FEATURES,
    MODEL_LR_PATH, MODEL_LR_SCALER, MODEL_LR_FEATURES,
    MODEL_RF_PATH, MODEL_RF_FEATURES,
    MODEL_LGBM_PATH, MODEL_LGBM_FEATURES,
    MODEL_ENSEMBLE_PATH,
    MODEL_FINISH_PATH, MODEL_FINISH_FEATURES,
    FINISH_CLASS_NAMES,
    DIVISIONS,
    EXCLUDE_STAT_KEYWORDS,
    RECENT_FORM_WINDOW,
    FINISH_METHOD_MAP,
    SOS_WINDOW,
    KO_VULN_WINDOW,
    EWMA_SPAN,
    TRAJECTORY_WINDOW,
    NAME_ALIASES,
)
from ml.ELO_calculator import get_current_ratings_by_division, get_current_glicko_by_division
from utils.odds import print_value_bet_summary
from utils.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-6


# ── v2 defensive stats lookup (for v1 inference) ─────────────────────────────

def _get_v2_defensive_stats(conn_v2: sqlite3.Connection, fighter_name: str) -> dict:
    """
    Return the most recent pre-fight sapm/str_def/td_def for a fighter from
    the v2 (UFCStats) DB, matched by exact name.  Returns zeros on no match.
    """
    row = conn_v2.execute(
        "SELECT fighter_id FROM fighters WHERE name = ? LIMIT 1",
        (fighter_name,),
    ).fetchone()
    if not row:
        return {"sapm": 0.0, "str_def": 0.0, "td_def": 0.0}
    fid = row[0]
    stats = conn_v2.execute(
        """
        SELECT CAST(fs.sapm AS REAL), CAST(fs.str_def AS REAL), CAST(fs.td_def AS REAL)
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        WHERE fs.fighter_id = ?
        ORDER BY f.date DESC, f.fight_id DESC
        LIMIT 1
        """,
        (fid,),
    ).fetchone()
    if not stats:
        return {"sapm": 0.0, "str_def": 0.0, "td_def": 0.0}
    return {
        "sapm":    float(stats[0] or 0),
        "str_def": float(stats[1] or 0),
        "td_def":  float(stats[2] or 0),
    }


# ── Live career stat refresh from UFCStats DB ─────────────────────────────────

def _resolve_ufcstats_id(conn_v2: sqlite3.Connection, name: str) -> str | None:
    """Return the UFCStats hex fighter_id for an exact name match, or None."""
    row = conn_v2.execute(
        "SELECT fighter_id FROM fighters WHERE name = ? LIMIT 1", (name,)
    ).fetchone()
    return row[0] if row else None


def compute_live_career_stats(
    conn_v2: sqlite3.Connection,
    fighter_name: str,
    trajectory_window: int = TRAJECTORY_WINDOW,
) -> dict | None:
    """
    Recompute career aggregate stats and trajectory slopes from raw per-fight
    data in the UFCStats DB, including the fighter's most recent fight result.

    The mdabbert DB stores pre-fight snapshots, so its most recent row always
    lags one fight behind.  This function eliminates that gap by computing all
    career averages from scratch using cumulative sums over every fight on
    record.

    Slope computation matches the training pipeline: polyfit is applied to the
    series of *running career averages* (avg_sig_str_pct, splm, avg_td_pct)
    over the last trajectory_window fights -- NOT per-fight accuracy.

    Returns None if the fighter cannot be found in the UFCStats DB.
    """
    fid = _resolve_ufcstats_id(conn_v2, fighter_name)
    if fid is None:
        return None

    df = pd.read_sql_query(
        """
        SELECT f.date, f.method, f.winner_id,
               CAST(f.title_fight    AS INTEGER) AS title_fight,
               CAST(f.finish_round   AS INTEGER) AS finish_round,
               CAST(p.sig_str_landed  AS REAL)   AS p_sig_landed,
               CAST(p.sig_str_atmpted AS REAL)   AS p_sig_atmpted,
               CAST(p.td_landed       AS REAL)   AS p_td_landed,
               CAST(p.td_atmpted      AS REAL)   AS p_td_atmpted,
               CAST(p.sub_att         AS REAL)   AS p_sub_att,
               CAST(p.total_fight_time AS REAL)  AS fight_time,
               CAST(o.sig_str_landed  AS REAL)   AS o_sig_landed,
               CAST(o.sig_str_atmpted AS REAL)   AS o_sig_atmpted,
               CAST(o.td_landed       AS REAL)   AS o_td_landed,
               CAST(o.td_atmpted      AS REAL)   AS o_td_atmpted
        FROM fights f
        JOIN fight_stats p ON p.fight_id = f.fight_id AND p.fighter_id = ?
        JOIN fight_stats o ON o.fight_id = f.fight_id AND o.fighter_id != ?
        ORDER BY f.date ASC, f.fight_id ASC
        """,
        conn_v2,
        params=(fid, fid),
    )
    if df.empty:
        return None

    df["won"] = (df["winner_id"] == fid).astype(int)
    df["fight_time_min"] = df["fight_time"] / 60.0

    # Running cumulative sums for career average computation
    df["cum_sig_landed"]  = df["p_sig_landed"].cumsum()
    df["cum_sig_atmpted"] = df["p_sig_atmpted"].cumsum()
    df["cum_td_landed"]   = df["p_td_landed"].cumsum()
    df["cum_td_atmpted"]  = df["p_td_atmpted"].cumsum()
    df["cum_fight_time"]  = df["fight_time_min"].cumsum()

    # Career average after each fight (matches mdabbert snapshot format)
    df["career_str_acc"] = np.where(
        df["cum_sig_atmpted"] > 0,
        df["cum_sig_landed"] / df["cum_sig_atmpted"],
        0.0,
    )
    df["career_td_acc"] = np.where(
        df["cum_td_atmpted"] > 0,
        df["cum_td_landed"] / df["cum_td_atmpted"],
        0.0,
    )
    df["career_splm"] = np.where(
        df["cum_fight_time"] > 0,
        df["cum_sig_landed"] / df["cum_fight_time"],
        0.0,
    )

    # Slope of career averages over last trajectory_window fights
    window_df = df.tail(trajectory_window)

    def _slope(series: pd.Series) -> float:
        vals = series.values.astype(float)
        valid = vals[~np.isnan(vals)]
        if len(valid) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(valid), dtype=float), valid, 1)[0])

    str_acc_slope = _slope(window_df["career_str_acc"])
    splm_slope    = _slope(window_df["career_splm"])
    td_acc_slope  = _slope(window_df["career_td_acc"])

    # Final career averages (after all fights)
    total_time = df["fight_time_min"].sum()
    avg_sig_str_pct = float(df["career_str_acc"].iloc[-1])
    avg_td_pct      = float(df["career_td_acc"].iloc[-1])
    splm            = float(df["career_splm"].iloc[-1])
    sapm            = df["o_sig_landed"].sum() / total_time if total_time > 0 else 0.0
    td_avg          = df["p_td_landed"].sum() / (total_time / 15.0) if total_time > 0 else 0.0
    avg_sub_att     = float(df["p_sub_att"].mean()) if not df.empty else 0.0

    valid_o_str = df[df["o_sig_atmpted"] > 0]
    str_def = (
        (1.0 - valid_o_str["o_sig_landed"].sum() / valid_o_str["o_sig_atmpted"].sum()) * 100.0
        if not valid_o_str.empty else 0.0
    )
    valid_o_td = df[df["o_td_atmpted"] > 0]
    td_def = (
        (1.0 - valid_o_td["o_td_landed"].sum() / valid_o_td["o_td_atmpted"].sum()) * 100.0
        if not valid_o_td.empty else 0.0
    )

    # Win/loss counts and methods — exclude No Contest fights (winner_id IS NULL)
    df_decided = df[df["winner_id"].notna()].copy()
    wins   = int((df_decided["won"] == 1).sum())
    losses = int((df_decided["won"] == 0).sum())

    def _fc(m):
        return FINISH_METHOD_MAP.get(m, -1)

    win_by_ko  = int(((df_decided["won"] == 1) & (df_decided["method"].map(_fc) == 1)).sum())
    win_by_sub = int(((df_decided["won"] == 1) & (df_decided["method"].map(_fc) == 2)).sum())
    win_by_dec_split     = int(((df_decided["won"] == 1) & (df_decided["method"] == "Decision - Split")).sum())
    win_by_dec_unanimous = int(((df_decided["won"] == 1) & (df_decided["method"] == "Decision - Unanimous")).sum())

    # Trailing streak and longest win streak — skip NC fights
    streak = 0
    max_win_streak = 0
    cur_win_streak = 0
    for won in df_decided["won"].tolist():
        if won:
            cur_win_streak += 1
            max_win_streak = max(max_win_streak, cur_win_streak)
        else:
            cur_win_streak = 0
    # Trailing streak
    streak = 0
    for won in reversed(df_decided["won"].tolist()):
        if streak == 0:
            streak = 1 if won else -1
        elif (streak > 0 and won) or (streak < 0 and not won):
            streak += 1 if won else -1
        else:
            break
    career_win_streak  = max(0,  streak)
    career_lose_streak = max(0, -streak)
    longest_win_streak = max_win_streak

    total_rounds_fought = int(df["finish_round"].fillna(0).sum())
    total_title_bouts   = int(df["title_fight"].fillna(0).astype(int).sum())

    return {
        "wins":                 wins,
        "losses":               losses,
        "win_by_ko":            win_by_ko,
        "win_by_sub":           win_by_sub,
        "win_by_dec_unanimous": win_by_dec_unanimous,
        "win_by_dec_split":     win_by_dec_split,
        "career_win_streak":    career_win_streak,
        "career_lose_streak":   career_lose_streak,
        "longest_win_streak":   longest_win_streak,
        "total_rounds_fought":  total_rounds_fought,
        "avg_sig_str_pct":      avg_sig_str_pct,
        "avg_td_pct":           avg_td_pct,
        "splm":                 splm,
        "td_avg":               td_avg,
        "avg_sub_att":          avg_sub_att,
        "total_title_bouts":    total_title_bouts,
        "str_acc_slope":        str_acc_slope,
        "splm_slope":           splm_slope,
        "td_acc_slope":         td_acc_slope,
        "sapm":                 sapm,
        "str_def":              str_def,
        "td_def":               td_def,
    }


# ── Live rankings lookup ──────────────────────────────────────────────────────

_RANKINGS_CSV = ROOT_DIR / "raw_data" / "rankings_history.csv"
_rankings_cache: pd.DataFrame | None = None

def _get_current_rank(fighter_name: str, division: str | None) -> float:
    """
    Return the most recent UFC ranking for a fighter in the given division.
    Returns 16.0 (unranked encoding) if not found.
    """
    global _rankings_cache
    _UNRANKED = 16.0
    if not _RANKINGS_CSV.exists():
        return _UNRANKED
    if _rankings_cache is None:
        _rankings_cache = pd.read_csv(_RANKINGS_CSV)
    rh = _rankings_cache
    name_lower = fighter_name.lower()
    rh_lower = rh["fighter"].str.lower()
    mask = rh_lower == name_lower
    if not mask.any():
        return _UNRANKED
    matches = rh[mask].copy()
    # Filter by division weightclass if provided
    if division:
        div_norm = division.lower().replace("-", " ")
        div_mask = matches["weightclass"].str.lower().str.replace("-", " ").str.contains(div_norm)
        if div_mask.any():
            matches = matches[div_mask]
    # Most recent row
    matches = matches.sort_values("date", ascending=False)
    rank = matches.iloc[0]["rank"]
    try:
        r = float(rank)
        return r  # 0 = champion, 1-15 = ranked, keep as-is
    except (ValueError, TypeError):
        return _UNRANKED


# ── Fighter Resolution ────────────────────────────────────────────────────────

def _resolve_alias(name: str) -> str:
    """Substitute a known alternate name with its canonical UFCStats name."""
    return NAME_ALIASES.get(name.lower().strip(), name)


def search_fighter(conn: sqlite3.Connection, name: str) -> list[tuple]:
    """Return (fighter_id, name) pairs whose name contains *name* (case-insensitive)."""
    name = _resolve_alias(name)
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

    # Exclude No Contest fights (winner_id IS NULL) from win/loss/streak stats
    df = df[df["winner_id"].notna()].copy()
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


# ── Extra features: bio, finish rates, inactivity, SOS ───────────────────────

def get_fighter_bio(conn: sqlite3.Connection, fighter_id: str) -> dict[str, object]:
    """Return height (cm), reach (cm), and stance for a fighter."""
    cur = conn.cursor()
    cur.execute(
        "SELECT height, reach, stance FROM fighters WHERE fighter_id = ?",
        (fighter_id,),
    )
    row = cur.fetchone()
    if not row:
        return {"height": 0.0, "reach": 0.0, "stance": "orthodox"}
    h, r, s = row
    return {
        "height": float(h) if h is not None else 0.0,
        "reach":  float(r) if r is not None else 0.0,
        "stance": (s or "Orthodox").strip(),
    }


def compute_finish_rates_single(
    conn: sqlite3.Connection, fighter_id: str
) -> dict[str, float]:
    """Return career KO/sub/dec win rates for a fighter (all prior fights)."""
    df = pd.read_sql_query(
        """
        SELECT winner_id, method FROM fights
        WHERE r_fighter_id = ? OR b_fighter_id = ?
        ORDER BY date ASC, fight_id ASC
        """,
        conn,
        params=(fighter_id, fighter_id),
    )
    if df.empty:
        return {"ko_rate": 0.0, "sub_rate": 0.0, "dec_rate": 0.0}

    df = df[df["winner_id"].notna()]
    if df.empty:
        return {"ko_rate": 0.0, "sub_rate": 0.0, "dec_rate": 0.0}

    won = df["winner_id"] == fighter_id
    total_wins = int(won.sum())
    if total_wins == 0:
        return {"ko_rate": 0.0, "sub_rate": 0.0, "dec_rate": 0.0}

    method_cls = df["method"].map(FINISH_METHOD_MAP)
    ko_wins  = int((won & (method_cls == 1)).sum())
    sub_wins = int((won & (method_cls == 2)).sum())
    dec_wins = int((won & (method_cls == 0)).sum())
    return {
        "ko_rate":  ko_wins  / total_wins,
        "sub_rate": sub_wins / total_wins,
        "dec_rate": dec_wins / total_wins,
    }


def compute_inactivity_single(
    conn: sqlite3.Connection, fighter_id: str
) -> dict[str, float]:
    """Return days since the fighter's most recent fight relative to today."""
    df = pd.read_sql_query(
        """
        SELECT date FROM fights
        WHERE r_fighter_id = ? OR b_fighter_id = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        conn,
        params=(fighter_id, fighter_id),
    )
    if df.empty:
        return {"days_since_last": 365.0}

    last_fight = pd.to_datetime(df.iloc[0]["date"])
    today = pd.Timestamp.now().normalize()
    days = max(0, (today - last_fight).days)
    return {"days_since_last": float(days)}


def compute_sos_single(
    conn: sqlite3.Connection,
    fighter_id: str,
    elo_by_division: dict[tuple[str, str], float],
    window: int = SOS_WINDOW,
) -> dict[str, float]:
    """Return average ELO of the last `window` opponents (strength of schedule)."""
    df = pd.read_sql_query(
        """
        SELECT f.division,
               CASE WHEN f.r_fighter_id = ? THEN f.b_fighter_id
                    ELSE f.r_fighter_id END AS opp_id
        FROM fights f
        WHERE (f.r_fighter_id = ? OR f.b_fighter_id = ?)
        ORDER BY f.date DESC
        LIMIT ?
        """,
        conn,
        params=(fighter_id, fighter_id, fighter_id, window),
    )
    if df.empty:
        return {"sos": float(STARTING_ELO)}

    elo_vals = []
    for _, row in df.iterrows():
        div = str(row["division"]).lower().strip()
        opp_elo = elo_by_division.get((row["opp_id"], div), STARTING_ELO)
        elo_vals.append(opp_elo)

    return {"sos": float(np.mean(elo_vals))}


def compute_ko_vulnerability_single(
    conn: sqlite3.Connection,
    fighter_id: str,
    window: int = KO_VULN_WINDOW,
) -> dict[str, float]:
    """Return count of KO/TKO stoppages suffered in the last `window` fights."""
    df = pd.read_sql_query(
        """
        SELECT winner_id, method FROM fights
        WHERE r_fighter_id = ? OR b_fighter_id = ?
        ORDER BY date DESC, fight_id DESC
        LIMIT ?
        """,
        conn,
        params=(fighter_id, fighter_id, window),
    )
    if df.empty:
        return {"ko_vuln": 0.0}

    df = df[df["winner_id"].notna()]
    ko_stopped = 0
    for _, row in df.iterrows():
        if row["winner_id"] != fighter_id:
            method_cls = FINISH_METHOD_MAP.get(row["method"], -1)
            if method_cls == 1:
                ko_stopped += 1

    return {"ko_vuln": float(ko_stopped)}


def compute_ewma_stats_single(
    conn: sqlite3.Connection,
    fighter_id: str,
    span: int = EWMA_SPAN,
) -> dict[str, float]:
    """Return EWMA striking/TD accuracy and striking accuracy variance.
    Returns zeros on schema mismatch (e.g. v1 career-aggregate DB)."""
    try:
        df = pd.read_sql_query(
            """
            SELECT CAST(fs.sig_str_landed  AS REAL) AS str_land,
                   CAST(fs.sig_str_atmpted AS REAL) AS str_att,
                   CAST(fs.td_landed       AS REAL) AS td_land,
                   CAST(fs.td_atmpted      AS REAL) AS td_att
            FROM fight_stats fs
            JOIN fights f ON fs.fight_id = f.fight_id
            WHERE fs.fighter_id = ?
            ORDER BY f.date ASC, f.fight_id ASC
            """,
            conn,
            params=(fighter_id,),
        )
    except Exception:
        return {"ewma_str_acc": 0.0, "ewma_td_acc": 0.0, "str_acc_var": 0.0}
    if df.empty:
        return {"ewma_str_acc": 0.0, "ewma_td_acc": 0.0, "str_acc_var": 0.0}

    _eps = 1e-6
    df["pf_str_acc"] = df["str_land"] / (df["str_att"] + _eps)
    df.loc[df["str_att"] == 0, "pf_str_acc"] = 0.0
    df["pf_td_acc"]  = df["td_land"]  / (df["td_att"]  + _eps)
    df.loc[df["td_att"]  == 0, "pf_td_acc"]  = 0.0

    ewma_str = float(df["pf_str_acc"].ewm(span=span, min_periods=1).mean().iloc[-1])
    ewma_td  = float(df["pf_td_acc"].ewm(span=span, min_periods=1).mean().iloc[-1])
    var_str  = float(df["pf_str_acc"].rolling(span, min_periods=2).std().fillna(0).iloc[-1])

    return {"ewma_str_acc": ewma_str, "ewma_td_acc": ewma_td, "str_acc_var": var_str}


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
    extra_r:      dict | None = None,
    extra_b:      dict | None = None,
) -> pd.DataFrame:
    """
    Compute all features the model expects.

    Handles:
      - Standard _diff features (Red - Blue rolling stats)
      - elo_diff
      - is_debutant_diff
      - recent_win_rate_diff / recent_finish_rate_diff / win_streak_diff
      - age_diff  (set to 0 - DOB not typically known for future fights)
      - grapple_ratio_diff / strike_ratio_diff / striker_vs_wrestler / wrestler_vs_striker
      - div_*  (one-hot division encoding)
      - title_fight  (direct binary feature)
      - height_diff / reach_diff  (from extra_r/extra_b)
      - southpaw_adv_diff / both_southpaw  (from extra_r/extra_b stance)
      - ko_rate_diff / sub_rate_diff / dec_rate_diff  (from extra_r/extra_b)
      - days_since_last_diff  (from extra_r/extra_b)
      - sos_diff  (from extra_r/extra_b)
    """
    grapple_r, strike_r = _style_ratios(red_stats)
    grapple_b, strike_b = _style_ratios(blue_stats)

    _er = extra_r or {}
    _eb = extra_b or {}

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

        # ── Height / reach ────────────────────────────────────────────────────
        # height and reach live in fight_stats so they fall through to the
        # standard _diff handler below via red_stats.get("height") etc.

        # ── Stance matchup ────────────────────────────────────────────────────
        # stance is in fight_stats; use red_stats / blue_stats directly
        elif feat == "southpaw_adv_diff":
            r_st = str(red_stats.get("stance", "Orthodox") or "Orthodox").lower()
            b_st = str(blue_stats.get("stance", "Orthodox") or "Orthodox").lower()
            row[feat] = int(r_st == "southpaw" and b_st == "orthodox") - int(r_st == "orthodox" and b_st == "southpaw")
        elif feat == "both_southpaw":
            r_st = str(red_stats.get("stance", "Orthodox") or "Orthodox").lower()
            b_st = str(blue_stats.get("stance", "Orthodox") or "Orthodox").lower()
            row[feat] = int(r_st == "southpaw" and b_st == "southpaw")

        # ── Finish-method rates ───────────────────────────────────────────────
        elif feat == "ko_rate_diff":
            row[feat] = _er.get("ko_rate", 0.0) - _eb.get("ko_rate", 0.0)
        elif feat == "sub_rate_diff":
            row[feat] = _er.get("sub_rate", 0.0) - _eb.get("sub_rate", 0.0)
        elif feat == "dec_rate_diff":
            row[feat] = _er.get("dec_rate", 0.0) - _eb.get("dec_rate", 0.0)

        # ── Inactivity ────────────────────────────────────────────────────────
        elif feat == "days_since_last_diff":
            row[feat] = _er.get("days_since_last", 365.0) - _eb.get("days_since_last", 365.0)

        # ── Strength of schedule ──────────────────────────────────────────────
        elif feat == "sos_diff":
            row[feat] = _er.get("sos", float(STARTING_ELO)) - _eb.get("sos", float(STARTING_ELO))

        # ── Glicko-2 ──────────────────────────────────────────────────────────
        elif feat == "glicko_diff":
            row[feat] = _er.get("glicko", float(GLICKO_START_R)) - _eb.get("glicko", float(GLICKO_START_R))
        elif feat == "glicko_rd_diff":
            row[feat] = _er.get("glicko_rd", float(GLICKO_START_RD)) - _eb.get("glicko_rd", float(GLICKO_START_RD))

        # ── KO vulnerability ──────────────────────────────────────────────────
        elif feat == "ko_vuln_diff":
            row[feat] = _er.get("ko_vuln", 0.0) - _eb.get("ko_vuln", 0.0)

        # ── EWMA accuracy and variance ────────────────────────────────────────
        elif feat == "ewma_str_acc_diff":
            row[feat] = _er.get("ewma_str_acc", 0.0) - _eb.get("ewma_str_acc", 0.0)
        elif feat == "ewma_td_acc_diff":
            row[feat] = _er.get("ewma_td_acc", 0.0) - _eb.get("ewma_td_acc", 0.0)
        elif feat == "str_acc_var_diff":
            row[feat] = _er.get("str_acc_var", 0.0) - _eb.get("str_acc_var", 0.0)

        # ── v2 defensive metrics (extra_r/extra_b for v1; red_stats for v2) ──
        elif feat == "sapm_diff":
            r_val = _er["sapm"] if "sapm" in _er else float(pd.to_numeric(red_stats.get("sapm",    0), errors="coerce") or 0)
            b_val = _eb["sapm"] if "sapm" in _eb else float(pd.to_numeric(blue_stats.get("sapm",   0), errors="coerce") or 0)
            row[feat] = r_val - b_val
        elif feat == "str_def_diff":
            r_val = _er["str_def"] if "str_def" in _er else float(pd.to_numeric(red_stats.get("str_def",  0), errors="coerce") or 0)
            b_val = _eb["str_def"] if "str_def" in _eb else float(pd.to_numeric(blue_stats.get("str_def", 0), errors="coerce") or 0)
            row[feat] = r_val - b_val
        elif feat == "td_def_diff":
            r_val = _er["td_def"] if "td_def" in _er else float(pd.to_numeric(red_stats.get("td_def",  0), errors="coerce") or 0)
            b_val = _eb["td_def"] if "td_def" in _eb else float(pd.to_numeric(blue_stats.get("td_def", 0), errors="coerce") or 0)
            row[feat] = r_val - b_val

        # ── UFC ranking differential (v1 only; unranked encoded as 16) ───────
        elif feat == "weightclass_rank_diff":
            _UNRANKED = 16.0
            if "weightclass_rank" in _er:
                r_rank = float(_er["weightclass_rank"])
            else:
                r_val = float(pd.to_numeric(red_stats.get("weightclass_rank", _UNRANKED), errors="coerce") or _UNRANKED)
                r_rank = _UNRANKED if pd.isna(r_val) else r_val
            if "weightclass_rank" in _eb:
                b_rank = float(_eb["weightclass_rank"])
            else:
                b_val = float(pd.to_numeric(blue_stats.get("weightclass_rank", _UNRANKED), errors="coerce") or _UNRANKED)
                b_rank = _UNRANKED if pd.isna(b_val) else b_val
            row[feat] = r_rank - b_rank

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

def compute_prediction(
    red_name:    str,
    blue_name:   str,
    model_type:  str = "ensemble",
    division:    str | None = None,
    title_fight: int = 0,
    db_path:     Path | None = None,
    models_dir:  Path | None = None,
    r_fighter_id: str | None = None,
    b_fighter_id: str | None = None,
) -> dict:
    """
    Compute a fight prediction and return a result dict.

    Returns keys: red_name, blue_name, winner, red_prob, blue_prob,
                  confidence, elo_red, elo_blue, form_red, form_blue.

    db_path defaults to DB_PATH; models_dir defaults to MODELS_DIR.
    Pass r_fighter_id / b_fighter_id to bypass name resolution (useful when
    the caller already has the UFCStats fighter IDs).
    """
    db_path    = db_path    or DB_V1_PATH
    # Prefer production models (trained on 100% of data) when available
    if models_dir is None:
        models_dir = MODELS_V1_PROD_DIR if MODELS_V1_PROD_DIR.exists() and any(MODELS_V1_PROD_DIR.iterdir()) else MODELS_V1_DIR

    # ── Resolve artifact paths from models_dir ────────────────────────────────
    _paths = {
        "xgb":      (models_dir / "xgboost.joblib",            models_dir / "xgb_features.joblib",  None,                              False),
        "lr":       (models_dir / "logistic_regression.joblib", models_dir / "lr_features.joblib",   models_dir / "lr_scaler.joblib",   True),
        "rf":       (models_dir / "random_forest.joblib",       models_dir / "rf_features.joblib",   None,                              False),
        "lgbm":     (models_dir / "lightgbm.joblib",            models_dir / "lgbm_features.joblib", None,                              False),
        "mlp":      (models_dir / "mlp.joblib",                 models_dir / "mlp_features.joblib",  models_dir / "mlp_scaler.joblib",  True),
        "ensemble": (models_dir / "ensemble.joblib",            None,                                None,                              False),
        "stacking": (models_dir / "stacking.joblib",            None,                                None,                              False),
    }
    model_path, features_path, scaler_path, _is_lr = _paths[model_type]

    script_map = {
        "xgb":      "ml/XGBoost.py",
        "lr":       "ml/logistic_regression.py",
        "rf":       "ml/random_forest.py",
        "lgbm":     "ml/lightgbm_model.py",
        "mlp":      "ml/train_v1_models.py --model mlp",
        "ensemble": "ml/train_v1_models.py --model ensemble",
        "stacking": "ml/train_v1_models.py --model stacking",
    }
    if not model_path.exists():
        script = script_map[model_type]
        log.error("No saved model found at '%s'. Run  python %s  first.", model_path, script)
        print(f"\n[ERROR]  No saved model found at '{model_path}'.")
        print(f"   Run  python {script}  first to train and save the model.")
        sys.exit(1)

    artifact      = joblib.load(model_path)
    feature_names = joblib.load(features_path) if features_path is not None else None
    scaler        = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None

    if model_type == "ensemble":
        ensemble_weights     = artifact.get("weights", {})
        ensemble_calibrators = artifact.get("calibrators", {})
        model = base_model = platt = None
    elif model_type == "stacking":
        model = base_model = platt = None
        ensemble_weights = None
    elif isinstance(artifact, dict):
        base_model  = artifact["base"]
        platt       = artifact["platt"]
        model       = None
        ensemble_weights = None
    else:
        model       = artifact
        base_model  = None
        platt       = None
        ensemble_weights = None

    finish_path  = models_dir / "finish_type.joblib"
    finish_feats_path = models_dir / "finish_type_features.joblib"
    finish_model = joblib.load(finish_path)       if finish_path.exists()       else None
    finish_feats = joblib.load(finish_feats_path) if finish_feats_path.exists() else None

    # ── DB connection ─────────────────────────────────────────────────────────
    if not db_path.exists():
        print(f"\n[ERROR]  Database not found at '{db_path}'.")
        print("   Run the database builder scripts first.")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    # ── Resolve fighters ──────────────────────────────────────────────────────
    if r_fighter_id:
        r_id, r_name = r_fighter_id, red_name
    else:
        r_id, r_name = resolve_fighter(conn, red_name)

    if b_fighter_id:
        b_id, b_name = b_fighter_id, blue_name
    else:
        b_id, b_name = resolve_fighter(conn, blue_name)

    if r_id == b_id:
        print("\n[WARN]  Both names resolved to the same fighter — please check the names.")
        conn.close()
        sys.exit(1)

    # ── Get rolling stats ─────────────────────────────────────────────────────
    red_stats  = get_latest_stats(conn, r_id).copy()
    blue_stats = get_latest_stats(conn, b_id).copy()

    if red_stats.empty:
        print(f"[WARN]  No fight history for {r_name} -- all stats set to 0.")
    if blue_stats.empty:
        print(f"[WARN]  No fight history for {b_name} -- all stats set to 0.")

    # ── ELO ───────────────────────────────────────────────────────────────────
    log.info("Computing current ELO ratings...")
    div_lower   = (division or "").lower().strip()
    div_elo     = get_current_ratings_by_division(conn)   # used for SOS only
    elo_ratings = compute_current_elo(conn)               # global, matches training
    elo_r       = elo_ratings.get(r_id, STARTING_ELO)
    elo_b       = elo_ratings.get(b_id, STARTING_ELO)

    # ── Glicko-2 ──────────────────────────────────────────────────────────────
    log.info("Computing current Glicko-2 ratings...")
    div_glicko = get_current_glicko_by_division(conn)
    if div_lower and (r_id, div_lower) in div_glicko and (b_id, div_lower) in div_glicko:
        glicko_r_tuple = div_glicko[(r_id, div_lower)]
        glicko_b_tuple = div_glicko[(b_id, div_lower)]
    elif div_lower:
        # Partial miss -- use per-fighter fallback to any available division rating
        r_divs = [(k, v) for k, v in div_glicko.items() if k[0] == r_id]
        b_divs = [(k, v) for k, v in div_glicko.items() if k[0] == b_id]
        glicko_r_tuple = r_divs[0][1] if r_divs else (GLICKO_START_R, GLICKO_START_RD, 0.06)
        glicko_b_tuple = b_divs[0][1] if b_divs else (GLICKO_START_R, GLICKO_START_RD, 0.06)
    else:
        # Fall back to the most recent division the fighter appeared in
        r_divs = [(k, v) for k, v in div_glicko.items() if k[0] == r_id]
        b_divs = [(k, v) for k, v in div_glicko.items() if k[0] == b_id]
        glicko_r_tuple = r_divs[0][1] if r_divs else (GLICKO_START_R, GLICKO_START_RD, 0.06)
        glicko_b_tuple = b_divs[0][1] if b_divs else (GLICKO_START_R, GLICKO_START_RD, 0.06)

    # ── Recent form ───────────────────────────────────────────────────────────
    log.info("Computing recent form...")
    form_r = compute_recent_form(conn, r_id)
    form_b = compute_recent_form(conn, b_id)

    # ── Extra features ────────────────────────────────────────────────────────
    log.info("Computing extra features...")
    finish_r = compute_finish_rates_single(conn, r_id)
    finish_b = compute_finish_rates_single(conn, b_id)
    inact_r  = compute_inactivity_single(conn, r_id)
    inact_b  = compute_inactivity_single(conn, b_id)
    sos_r    = compute_sos_single(conn, r_id, div_elo)
    sos_b    = compute_sos_single(conn, b_id, div_elo)
    kovuln_r = compute_ko_vulnerability_single(conn, r_id)
    kovuln_b = compute_ko_vulnerability_single(conn, b_id)
    ewma_r   = compute_ewma_stats_single(conn, r_id)
    ewma_b   = compute_ewma_stats_single(conn, b_id)
    glicko_extra_r = {"glicko": glicko_r_tuple[0], "glicko_rd": glicko_r_tuple[1]}
    glicko_extra_b = {"glicko": glicko_b_tuple[0], "glicko_rd": glicko_b_tuple[1]}
    extra_r  = {**finish_r, **inact_r, **sos_r, **kovuln_r, **ewma_r, **glicko_extra_r}
    extra_b  = {**finish_b, **inact_b, **sos_b, **kovuln_b, **ewma_b, **glicko_extra_b}

    conn.close()

    # Refresh all stale stats from the UFCStats DB, which is updated by the
    # scraper after every event.  The mdabbert DB only updates when
    # ingest_mdabbert.py is run, so all features computed above from conn
    # (form, finish rates, inactivity, SOS, ko_vuln, ewma, career averages)
    # may lag by one event.  Recomputing from conn_v2 eliminates that gap.
    if DB_PATH.exists():
        try:
            conn_v2 = sqlite3.connect(str(DB_PATH))
            log.info("Refreshing live stats from UFCStats DB...")

            r_fid_v2 = _resolve_ufcstats_id(conn_v2, r_name)
            b_fid_v2 = _resolve_ufcstats_id(conn_v2, b_name)

            # Career averages + slopes (cumulative recomputation)
            live_r = compute_live_career_stats(conn_v2, r_name)
            live_b = compute_live_career_stats(conn_v2, b_name)

            if live_r:
                for k, v in live_r.items():
                    red_stats[k] = v
                extra_r["sapm"]    = live_r["sapm"]
                extra_r["str_def"] = live_r["str_def"]
                extra_r["td_def"]  = live_r["td_def"]
            else:
                log.warning("Live stat refresh failed for %s -- using stale mdabbert snapshot.", r_name)
                v2_def_r = _get_v2_defensive_stats(conn_v2, r_name)
                extra_r.update(v2_def_r)

            if live_b:
                for k, v in live_b.items():
                    blue_stats[k] = v
                extra_b["sapm"]    = live_b["sapm"]
                extra_b["str_def"] = live_b["str_def"]
                extra_b["td_def"]  = live_b["td_def"]
            else:
                log.warning("Live stat refresh failed for %s -- using stale mdabbert snapshot.", b_name)
                v2_def_b = _get_v2_defensive_stats(conn_v2, b_name)
                extra_b.update(v2_def_b)

            # ELO and Glicko from UFCStats (always current -- scraper updates this DB)
            elo_ratings_v2 = compute_current_elo(conn_v2)
            div_glicko_v2  = get_current_glicko_by_division(conn_v2)

            if r_fid_v2:
                elo_r = elo_ratings_v2.get(r_fid_v2, STARTING_ELO)
            if b_fid_v2:
                elo_b = elo_ratings_v2.get(b_fid_v2, STARTING_ELO)

            r_gid      = r_fid_v2 or r_id
            b_gid      = b_fid_v2 or b_id
            r_gid_divs = [(k, v) for k, v in div_glicko_v2.items() if k[0] == r_gid]
            b_gid_divs = [(k, v) for k, v in div_glicko_v2.items() if k[0] == b_gid]
            _def_glicko = (GLICKO_START_R, GLICKO_START_RD, 0.06)
            if div_lower:
                glicko_r_tuple = div_glicko_v2.get((r_gid, div_lower)) or (r_gid_divs[0][1] if r_gid_divs else _def_glicko)
                glicko_b_tuple = div_glicko_v2.get((b_gid, div_lower)) or (b_gid_divs[0][1] if b_gid_divs else _def_glicko)
            else:
                glicko_r_tuple = r_gid_divs[0][1] if r_gid_divs else _def_glicko
                glicko_b_tuple = b_gid_divs[0][1] if b_gid_divs else _def_glicko
            extra_r["glicko"]    = glicko_r_tuple[0]
            extra_r["glicko_rd"] = glicko_r_tuple[1]
            extra_b["glicko"]    = glicko_b_tuple[0]
            extra_b["glicko_rd"] = glicko_b_tuple[1]

            # Per-fight-history features: form, finish rates, inactivity, SOS,
            # ko_vuln, ewma -- all query fights/fight_stats so must use UFCStats IDs
            div_elo_v2 = get_current_ratings_by_division(conn_v2)

            if r_fid_v2:
                form_r   = compute_recent_form(conn_v2, r_fid_v2)
                finish_r = compute_finish_rates_single(conn_v2, r_fid_v2)
                inact_r  = compute_inactivity_single(conn_v2, r_fid_v2)
                sos_r    = compute_sos_single(conn_v2, r_fid_v2, div_elo_v2)
                kovuln_r = compute_ko_vulnerability_single(conn_v2, r_fid_v2)
                ewma_r   = compute_ewma_stats_single(conn_v2, r_fid_v2)
                extra_r.update({**finish_r, **inact_r, **sos_r, **kovuln_r, **ewma_r})
            else:
                log.warning("UFCStats ID not found for %s -- live feature refresh skipped.", r_name)

            if b_fid_v2:
                form_b   = compute_recent_form(conn_v2, b_fid_v2)
                finish_b = compute_finish_rates_single(conn_v2, b_fid_v2)
                inact_b  = compute_inactivity_single(conn_v2, b_fid_v2)
                sos_b    = compute_sos_single(conn_v2, b_fid_v2, div_elo_v2)
                kovuln_b = compute_ko_vulnerability_single(conn_v2, b_fid_v2)
                ewma_b   = compute_ewma_stats_single(conn_v2, b_fid_v2)
                extra_b.update({**finish_b, **inact_b, **sos_b, **kovuln_b, **ewma_b})
            else:
                log.warning("UFCStats ID not found for %s -- live feature refresh skipped.", b_name)

            conn_v2.close()
        except Exception as exc:
            log.warning("Live stat refresh failed: %s", exc)

    # ── Live rankings (always from rankings_history.csv) ─────────────────────
    extra_r["weightclass_rank"] = _get_current_rank(r_name, division)
    extra_b["weightclass_rank"] = _get_current_rank(b_name, division)

    # ── Build & predict ───────────────────────────────────────────────────────
    if model_type == "ensemble":
        _specs = [
            ("xgb",  models_dir / "xgboost.joblib",            models_dir / "xgb_features.joblib",  None,                              False),
            ("lr",   models_dir / "logistic_regression.joblib", models_dir / "lr_features.joblib",   models_dir / "lr_scaler.joblib",   True),
            ("rf",   models_dir / "random_forest.joblib",       models_dir / "rf_features.joblib",   None,                              False),
            ("lgbm", models_dir / "lightgbm.joblib",            models_dir / "lgbm_features.joblib", None,                              False),
            ("mlp",  models_dir / "mlp.joblib",                 models_dir / "mlp_features.joblib",  models_dir / "mlp_scaler.joblib",  True),
        ]
        model_probas   = []
        active_weights = []
        for m_key, m_path, m_feats_path, m_scaler_path, m_is_lr in _specs:
            if not m_path.exists():
                continue
            m_artifact      = joblib.load(m_path)
            m_feature_names = joblib.load(m_feats_path)
            m_scaler        = joblib.load(m_scaler_path) if m_scaler_path and m_scaler_path.exists() else None
            Xm = build_feature_vector(
                red_stats, blue_stats,
                elo_r, elo_b,
                form_r, form_b,
                division, title_fight,
                m_feature_names,
                extra_r=extra_r, extra_b=extra_b,
            ).fillna(0)
            Xm_input = m_scaler.transform(Xm) if m_scaler is not None else Xm.values
            if m_is_lr:
                m_base = m_artifact["base"]
                raw_p  = m_base.predict_proba(Xm_input)[0, 1]
                if m_key in ensemble_calibrators:
                    cal_p = float(ensemble_calibrators[m_key].predict([raw_p])[0])
                else:
                    m_platt = m_artifact["platt"]
                    if hasattr(m_platt, "predict_proba"):
                        cal_p = float(m_platt.predict_proba([[raw_p]])[0, 1])
                    else:
                        cal_p = float(m_platt.predict([raw_p])[0])
                m_proba = [1 - cal_p, cal_p]
            else:
                raw_p = m_artifact.predict_proba(Xm_input)[0, 1]
                if m_key in ensemble_calibrators:
                    cal_p   = float(ensemble_calibrators[m_key].predict([raw_p])[0])
                    m_proba = [1 - cal_p, cal_p]
                else:
                    m_proba = m_artifact.predict_proba(Xm_input)[0].tolist()
            model_probas.append(m_proba)
            active_weights.append(ensemble_weights.get(m_key, 1.0))
        if not model_probas:
            print("\n[ERROR]  No base models found for ensemble. Train base models first.")
            sys.exit(1)
        total_w        = sum(active_weights)
        active_weights = [w / total_w for w in active_weights]
        proba = list(np.average(model_probas, axis=0, weights=active_weights))
    elif model_type == "stacking":
        # Run each base model, collect calibrated probs, feed to meta-LR
        stk_specs  = artifact.get("specs", [])
        meta_lr    = artifact["meta"]
        meta_sc    = artifact["meta_scaler"]
        base_probs = []
        for m_key, m_path, f_path, s_path, m_is_lr in stk_specs:
            m_path = Path(m_path)
            f_path = Path(f_path)
            if not m_path.exists() or not f_path.exists():
                log.warning("Stacking: base model %s not found, using 0.5", m_key)
                base_probs.append(0.5)
                continue
            m_art  = joblib.load(m_path)
            m_feat = joblib.load(f_path)
            m_sc   = joblib.load(s_path) if s_path and Path(s_path).exists() else None
            Xm = build_feature_vector(
                red_stats, blue_stats, elo_r, elo_b, form_r, form_b,
                division, title_fight, m_feat, extra_r=extra_r, extra_b=extra_b,
            ).fillna(0)
            Xi = m_sc.transform(Xm) if m_sc is not None else Xm.values
            m_base = m_art["base"] if m_is_lr else m_art
            raw_p  = float(m_base.predict_proba(Xi)[0, 1])
            base_probs.append(raw_p)
        stacked  = np.array(base_probs).reshape(1, -1)
        stacked_s = meta_sc.transform(stacked)
        red_p    = float(meta_lr.predict_proba(stacked_s)[0, 1])
        proba    = [1 - red_p, red_p]
    else:
        X = build_feature_vector(
            red_stats, blue_stats,
            elo_r, elo_b,
            form_r, form_b,
            division, title_fight,
            feature_names,
            extra_r=extra_r, extra_b=extra_b,
        ).fillna(0)

        X_input = scaler.transform(X) if scaler is not None else X.values

        if model is not None:
            proba = model.predict_proba(X_input)[0]
        else:
            raw_prob = base_model.predict_proba(X_input)[0, 1]
            if hasattr(platt, "predict_proba"):
                calibrated = platt.predict_proba([[raw_prob]])[0, 1]
            else:
                calibrated = float(platt.predict([raw_prob])[0])
            proba = [1 - calibrated, calibrated]

    red_win_prob  = float(proba[1])
    blue_win_prob = float(proba[0])
    winner_name   = r_name if red_win_prob >= 0.5 else b_name
    confidence    = max(red_win_prob, blue_win_prob)

    finish_proba = None
    if finish_model is not None and finish_feats is not None:
        X_fin = build_feature_vector(
            red_stats, blue_stats,
            elo_r, elo_b,
            form_r, form_b,
            division, title_fight,
            finish_feats,
            extra_r=extra_r, extra_b=extra_b,
        ).fillna(0)
        finish_proba = finish_model.predict_proba(X_fin.values)[0].tolist()

    return {
        "red_name":    r_name,
        "blue_name":   b_name,
        "winner":      winner_name,
        "red_prob":    red_win_prob,
        "blue_prob":   blue_win_prob,
        "confidence":  confidence,
        "elo_red":     elo_r,
        "elo_blue":    elo_b,
        "form_red":    form_r,
        "form_blue":   form_b,
        "finish_proba": finish_proba,
    }


def predict_fight(
    red_name:        str,
    blue_name:       str,
    model_type:      str = "xgb",
    division:        str | None = None,
    title_fight:     int = 0,
    odds_red:        float | None = None,
    odds_blue:       float | None = None,
) -> None:

    model_labels = {
        "xgb":      "XGBoost",
        "lr":       "Logistic Regression",
        "rf":       "Random Forest",
        "lgbm":     "LightGBM",
        "mlp":      "MLP Neural Network",
        "ensemble": "Ensemble (Soft Vote)",
        "stacking": "Stacking Meta-Learner",
    }
    model_label = model_labels.get(model_type, model_type)

    result = compute_prediction(red_name, blue_name, model_type, division, title_fight)

    r_name        = result["red_name"]
    b_name        = result["blue_name"]
    red_win_prob  = result["red_prob"]
    blue_win_prob = result["blue_prob"]
    winner_name   = result["winner"]
    confidence    = result["confidence"]
    elo_r         = result["elo_red"]
    elo_b         = result["elo_blue"]
    form_r        = result["form_red"]
    form_b        = result["form_blue"]
    finish_proba  = result["finish_proba"]

    # ── Display results ───────────────────────────────────────────────────────
    print(f"\n[FIGHT]  {r_name}  (Red)  vs  {b_name}  (Blue)")
    if division:
        print(f"         Division: {division.title()}" + ("  [TITLE FIGHT]" if title_fight else ""))
    print("-" * 56)

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
        for name, p in zip(FINISH_CLASS_NAMES, finish_proba):
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
        choices=["xgb", "lr", "rf", "lgbm", "mlp", "ensemble", "stacking"],
        default="xgb",
        help="Model: 'xgb' = XGBoost (default), 'lr' = Logistic Regression, 'rf' = Random Forest, 'lgbm' = LightGBM, 'mlp' = MLP Neural Network, 'ensemble' = Soft-Vote Ensemble, 'stacking' = Stacking Meta-Learner",
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
