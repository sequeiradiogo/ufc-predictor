"""
ELO_calculator.py — Dynamic ELO rating engine for UFC fighters.

Exports
-------
build_elo_features()             → DataFrame with pre-fight ELO per fight (used in training)
                                   Uses per-division ratings (fighter_id × division).
get_current_ratings()            → dict[fighter_id, global_elo]   (global, fallback)
get_current_ratings_by_division()→ dict[(fighter_id, division), elo]  (division-specific)
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from config import (
        DB_PATH,
        STARTING_ELO, K_FACTOR_NORMAL, K_FACTOR_PROVISIONAL, PROVISIONAL_LIMIT,
        GLICKO_START_R, GLICKO_START_RD, GLICKO_START_SIGMA, GLICKO_TAU,
    )
    from logger import get_logger
except ImportError:
    DB_PATH              = Path(__file__).parent.parent / "db" / "ufc_v2.db"
    STARTING_ELO         = 1400
    K_FACTOR_NORMAL      = 32
    K_FACTOR_PROVISIONAL = 90
    PROVISIONAL_LIMIT    = 3
    GLICKO_START_R       = 1500
    GLICKO_START_RD      = 350.0
    GLICKO_START_SIGMA   = 0.06
    GLICKO_TAU           = 0.5
    import logging
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

log = get_logger(__name__)


# ── Core ELO math ─────────────────────────────────────────────────────────────

def get_expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for fighter A against fighter B (logistic curve)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 200))


def update_ratings(
    rating_r: float,
    rating_b: float,
    actual_r: float,
    k_r: float,
    k_b: float,
) -> tuple[float, float]:
    """
    Return updated (new_rating_red, new_rating_blue) after one fight.

    Parameters
    ----------
    rating_r / rating_b : pre-fight ELO for red / blue corner
    actual_r            : outcome from red's perspective (1=win, 0=loss, 0.5=draw)
    k_r / k_b           : K-factor for each fighter (provisional or normal)
    """
    expected_r = get_expected_score(rating_r, rating_b)
    expected_b = 1.0 - expected_r
    actual_b   = 1.0 - actual_r

    new_r = rating_r + k_r * (actual_r - expected_r)
    new_b = rating_b + k_b * (actual_b - expected_b)
    return new_r, new_b


# ── Internal replay helpers ───────────────────────────────────────────────────

def _replay_fights(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, int], list[float], list[float]]:
    """
    Global ELO replay — fighter_id only (no division key).

    Replay all fights in *df* (sorted by date ASC) and track:
      - fighter_ratings     : current ELO per fighter_id  after all fights
      - fighter_fight_counts: number of fights per fighter_id
      - red_elo_pre         : ELO for red  fighter BEFORE each fight
      - blue_elo_pre        : ELO for blue fighter BEFORE each fight
    """
    fighter_ratings:      dict[str, float] = {}
    fighter_fight_counts: dict[str, int]   = {}
    red_elo_pre:  list[float] = []
    blue_elo_pre: list[float] = []

    for _, row in df.iterrows():
        r_id   = row["r_fighter_id"]
        b_id   = row["b_fighter_id"]
        winner = row["winner_id"]

        r_curr  = fighter_ratings.get(r_id, STARTING_ELO)
        b_curr  = fighter_ratings.get(b_id, STARTING_ELO)
        r_count = fighter_fight_counts.get(r_id, 0)
        b_count = fighter_fight_counts.get(b_id, 0)

        red_elo_pre.append(r_curr)
        blue_elo_pre.append(b_curr)

        k_r = K_FACTOR_PROVISIONAL if r_count < PROVISIONAL_LIMIT else K_FACTOR_NORMAL
        k_b = K_FACTOR_PROVISIONAL if b_count < PROVISIONAL_LIMIT else K_FACTOR_NORMAL

        score_r = 1.0 if winner == r_id else (0.0 if winner == b_id else 0.5)
        new_r, new_b = update_ratings(r_curr, b_curr, score_r, k_r, k_b)

        fighter_ratings[r_id] = new_r
        fighter_ratings[b_id] = new_b
        fighter_fight_counts[r_id] = r_count + 1
        fighter_fight_counts[b_id] = b_count + 1

    return fighter_ratings, fighter_fight_counts, red_elo_pre, blue_elo_pre


def _replay_fights_by_division(
    df: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int], list[float], list[float]]:
    """
    Per-division ELO replay — key is (fighter_id, division).

    A fighter's ELO in Lightweight is independent of their ELO in Welterweight.
    On first appearance in a division they start at STARTING_ELO.
    The provisional K-factor is applied per-division (first PROVISIONAL_LIMIT
    fights *in that division*).

    Returns
    -------
    div_ratings     : dict[(fighter_id, division), elo]   after all fights
    div_counts      : dict[(fighter_id, division), n_fights_in_div]
    red_elo_pre     : division ELO for red  BEFORE each fight (list, same order as df)
    blue_elo_pre    : division ELO for blue BEFORE each fight
    """
    div_ratings: dict[tuple[str, str], float] = {}
    div_counts:  dict[tuple[str, str], int]   = {}
    red_elo_pre:  list[float] = []
    blue_elo_pre: list[float] = []

    for _, row in df.iterrows():
        r_id   = row["r_fighter_id"]
        b_id   = row["b_fighter_id"]
        winner = row["winner_id"]
        div    = str(row.get("division", "")).lower().strip()

        r_key = (r_id, div)
        b_key = (b_id, div)

        r_curr  = div_ratings.get(r_key, STARTING_ELO)
        b_curr  = div_ratings.get(b_key, STARTING_ELO)
        r_count = div_counts.get(r_key, 0)
        b_count = div_counts.get(b_key, 0)

        red_elo_pre.append(r_curr)
        blue_elo_pre.append(b_curr)

        k_r = K_FACTOR_PROVISIONAL if r_count < PROVISIONAL_LIMIT else K_FACTOR_NORMAL
        k_b = K_FACTOR_PROVISIONAL if b_count < PROVISIONAL_LIMIT else K_FACTOR_NORMAL

        score_r = 1.0 if winner == r_id else (0.0 if winner == b_id else 0.5)
        new_r, new_b = update_ratings(r_curr, b_curr, score_r, k_r, k_b)

        div_ratings[r_key] = new_r
        div_ratings[b_key] = new_b
        div_counts[r_key]  = r_count + 1
        div_counts[b_key]  = b_count + 1

    return div_ratings, div_counts, red_elo_pre, blue_elo_pre


# ── Public API ────────────────────────────────────────────────────────────────

def build_elo_features(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """
    Compute pre-fight ELO ratings for every fight in the database.

    Uses **per-division** ELO: a fighter's Lightweight rating is independent of
    their Welterweight rating.  Falls back cleanly to STARTING_ELO for a
    fighter's first appearance in a given division.

    Used by ML_data_preparation.py to add `elo_red`, `elo_blue`, and `elo_diff`.

    Returns
    -------
    DataFrame with columns: fight_id, elo_red, elo_blue
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(DB_PATH))

    query = """
        SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id
        FROM fights
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)

    if own_conn:
        conn.close()

    log.info("Calculating global ELO ratings for %d fights…", len(df))
    _, _, red_elo_pre, blue_elo_pre = _replay_fights(df)

    df["elo_red"]  = red_elo_pre
    df["elo_blue"] = blue_elo_pre

    log.info("ELO calculation complete.")
    return df[["fight_id", "elo_red", "elo_blue"]]


def get_current_ratings(conn: sqlite3.Connection | None = None) -> dict[str, float]:
    """
    **Global** ELO — one rating per fighter across all divisions.
    Kept for backward compatibility and as a fallback in predict.py when no
    division is specified.

    Returns
    -------
    dict mapping fighter_id → current global ELO (float)
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(DB_PATH))

    query = "SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id FROM fights ORDER BY date ASC"
    df = pd.read_sql_query(query, conn)

    if own_conn:
        conn.close()

    log.info("Computing global ELO ratings for %d fights…", len(df))
    fighter_ratings, _, _, _ = _replay_fights(df)
    log.info("Global ratings computed for %d fighters.", len(fighter_ratings))
    return fighter_ratings


def get_current_ratings_by_division(
    conn: sqlite3.Connection | None = None,
) -> dict[tuple[str, str], float]:
    """
    Per-division ELO — separate rating per (fighter_id, division) pair.

    Returns
    -------
    dict mapping (fighter_id, division_lower) → current ELO (float)
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(DB_PATH))

    query = """
        SELECT fight_id, date, division, r_fighter_id, b_fighter_id, winner_id
        FROM fights
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)

    if own_conn:
        conn.close()

    log.info("Computing per-division ELO ratings for %d fights…", len(df))
    div_ratings, _, _, _ = _replay_fights_by_division(df)
    log.info("Per-division ratings computed for %d (fighter, division) pairs.", len(div_ratings))
    return div_ratings


# ═══════════════════════════════════════════════════════════════════════════════
# Glicko-2 implementation
# Reference: Glickman, M.E. (2012). "Example of the Glicko-2 system."
# http://www.glicko.net/glicko/glicko2.pdf
# ═══════════════════════════════════════════════════════════════════════════════

import math as _math

_G2_SCALE = 173.7178  # converts between Glicko-1 and Glicko-2 internal scale


def _g(phi: float) -> float:
    return 1.0 / _math.sqrt(1.0 + 3.0 * phi ** 2 / _math.pi ** 2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + _math.exp(-_g(phi_j) * (mu - mu_j)))


def _glicko2_update(
    r: float,
    rd: float,
    sigma: float,
    outcomes: list[tuple[float, float, float]],
    tau: float = GLICKO_TAU,
) -> tuple[float, float, float]:
    """
    Apply one Glicko-2 rating-period update.

    Parameters
    ----------
    r, rd, sigma : current rating, deviation, volatility (Glicko-1 scale)
    outcomes     : list of (opponent_r, opponent_rd, score) for this period
                   score = 1.0 win, 0.0 loss, 0.5 draw
    tau          : system constant

    Returns
    -------
    (new_r, new_rd, new_sigma) on Glicko-1 scale
    """
    # Convert to Glicko-2 internal scale
    mu    = (r  - 1500.0) / _G2_SCALE
    phi   = rd / _G2_SCALE

    if not outcomes:
        # No bouts: inflate RD by volatility (inactivity decay), rating unchanged
        phi_star = _math.sqrt(phi ** 2 + sigma ** 2)
        return r, phi_star * _G2_SCALE, sigma

    # Step 3: compute v (estimated variance)
    v_inv = 0.0
    for opp_r, opp_rd, _s in outcomes:
        mu_j  = (opp_r  - 1500.0) / _G2_SCALE
        phi_j = opp_rd / _G2_SCALE
        g_j   = _g(phi_j)
        e_j   = _E(mu, mu_j, phi_j)
        v_inv += g_j ** 2 * e_j * (1.0 - e_j)
    v = 1.0 / v_inv

    # Step 4: compute delta (performance rating)
    delta_sum = 0.0
    for opp_r, opp_rd, s_j in outcomes:
        mu_j  = (opp_r  - 1500.0) / _G2_SCALE
        phi_j = opp_rd / _G2_SCALE
        g_j   = _g(phi_j)
        e_j   = _E(mu, mu_j, phi_j)
        delta_sum += g_j * (s_j - e_j)
    delta = v * delta_sum

    # Step 5: update volatility via Illinois algorithm (from Glickman 2012)
    a = _math.log(sigma ** 2)
    eps = 1e-6

    def _f(x: float) -> float:
        ex = _math.exp(x)
        d2 = phi ** 2 + v + ex
        return (ex * (delta ** 2 - d2) / (2.0 * d2 ** 2)
                - (x - a) / tau ** 2)

    # Bracket
    A = a
    if delta ** 2 > phi ** 2 + v:
        B = _math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while _f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    fA, fB = _f(A), _f(B)
    for _ in range(100):
        C  = A + (A - B) * fA / (fB - fA)
        fC = _f(C)
        if fB * fC < 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
        if abs(B - A) < eps:
            break
    new_sigma = _math.exp(A / 2.0)

    # Steps 6-7: update phi and mu
    phi_star = _math.sqrt(phi ** 2 + new_sigma ** 2)
    new_phi  = 1.0 / _math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    new_mu   = mu + new_phi ** 2 * delta_sum

    # Convert back to Glicko-1 scale
    return (new_mu * _G2_SCALE + 1500.0, new_phi * _G2_SCALE, new_sigma)


def _replay_fights_glicko_by_division(
    df: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], tuple[float, float, float]],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """
    Replay fights using Glicko-2 with calendar-quarter rating periods.

    Returns
    -------
    current_ratings : dict[(fighter_id, division), (r, rd, sigma)] after all periods
    red_r_pre       : pre-fight Glicko-2 rating for red fighter
    blue_r_pre      : pre-fight Glicko-2 rating for blue fighter
    red_rd_pre      : pre-fight RD for red
    blue_rd_pre     : pre-fight RD for blue
    """
    import numpy as np

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Assign each fight to a calendar quarter
    df["period"] = df["date"].dt.to_period("Q")
    periods = df["period"].unique()
    periods_sorted = sorted(periods)

    # Current ratings dict: key=(fighter_id, division), value=(r, rd, sigma)
    ratings: dict[tuple[str, str], tuple[float, float, float]] = {}

    # Pre-fight arrays indexed by fight row
    red_r_pre:  list[float] = [0.0] * len(df)
    blue_r_pre: list[float] = [0.0] * len(df)
    red_rd_pre: list[float] = [0.0] * len(df)
    blue_rd_pre: list[float] = [0.0] * len(df)

    # Track which fighters appear at all (to apply inactivity decay)
    all_keys: set[tuple[str, str]] = set()

    for period in periods_sorted:
        period_mask = df["period"] == period
        period_df   = df[period_mask]

        # Record pre-period ratings as pre-fight values; collect outcomes per fighter
        fighter_outcomes: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
        period_indices = period_df.index.tolist()

        for idx in period_indices:
            row   = df.loc[idx]
            r_id  = row["r_fighter_id"]
            b_id  = row["b_fighter_id"]
            div   = str(row.get("division", "")).lower().strip()
            r_key = (r_id, div)
            b_key = (b_id, div)

            r_curr = ratings.get(r_key, (GLICKO_START_R, GLICKO_START_RD, GLICKO_START_SIGMA))
            b_curr = ratings.get(b_key, (GLICKO_START_R, GLICKO_START_RD, GLICKO_START_SIGMA))

            all_keys.add(r_key)
            all_keys.add(b_key)

            # Record pre-fight (start-of-period) values
            red_r_pre[idx]  = r_curr[0]
            blue_r_pre[idx] = b_curr[0]
            red_rd_pre[idx] = r_curr[1]
            blue_rd_pre[idx] = b_curr[1]

            winner = row["winner_id"]
            score_r = 1.0 if winner == r_id else (0.0 if winner == b_id else 0.5)
            score_b = 1.0 - score_r

            if r_key not in fighter_outcomes:
                fighter_outcomes[r_key] = []
            fighter_outcomes[r_key].append((b_curr[0], b_curr[1], score_r))

            if b_key not in fighter_outcomes:
                fighter_outcomes[b_key] = []
            fighter_outcomes[b_key].append((r_curr[0], r_curr[1], score_b))

        # Apply updates for all fighters who fought this period
        for key, outcomes in fighter_outcomes.items():
            curr = ratings.get(key, (GLICKO_START_R, GLICKO_START_RD, GLICKO_START_SIGMA))
            ratings[key] = _glicko2_update(curr[0], curr[1], curr[2], outcomes)

        # Inactivity decay for fighters who did NOT fight this period
        for key in all_keys - set(fighter_outcomes.keys()):
            curr = ratings.get(key, (GLICKO_START_R, GLICKO_START_RD, GLICKO_START_SIGMA))
            ratings[key] = _glicko2_update(curr[0], curr[1], curr[2], [])

    return ratings, red_r_pre, blue_r_pre, red_rd_pre, blue_rd_pre


def get_elo_history_for_fighter(
    fighter_id: str,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """
    Replay all historical fights and return per-fight global ELO snapshots for
    one fighter. Uses the same global (cross-division) replay as build_elo_features()
    so values are consistent with what the v1 models were trained on.

    Returns
    -------
    List of dicts (chronological): date, opponent, opponent_id, result, method,
    division, elo_before, elo_after, elo_change
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(DB_PATH))

    query = """
        SELECT f.fight_id, f.date, f.division,
               f.r_fighter_id, f.b_fighter_id, f.winner_id, f.method,
               fr.name AS r_name, fb.name AS b_name
        FROM fights f
        JOIN fighters fr ON fr.fighter_id = f.r_fighter_id
        JOIN fighters fb ON fb.fighter_id = f.b_fighter_id
        ORDER BY f.date ASC
    """
    df = pd.read_sql_query(query, conn)

    if own_conn:
        conn.close()

    # Global ratings: keyed by fighter_id only, no division split
    ratings: dict[str, float] = {}
    counts:  dict[str, int]   = {}
    snapshots: list[dict] = []

    for _, row in df.iterrows():
        r_id = row["r_fighter_id"]
        b_id = row["b_fighter_id"]

        r_curr  = ratings.get(r_id, STARTING_ELO)
        b_curr  = ratings.get(b_id, STARTING_ELO)
        r_count = counts.get(r_id, 0)
        b_count = counts.get(b_id, 0)

        k_r = K_FACTOR_PROVISIONAL if r_count < PROVISIONAL_LIMIT else K_FACTOR_NORMAL
        k_b = K_FACTOR_PROVISIONAL if b_count < PROVISIONAL_LIMIT else K_FACTOR_NORMAL

        winner  = row["winner_id"]
        score_r = 1.0 if winner == r_id else (0.0 if winner == b_id else 0.5)
        new_r, new_b = update_ratings(r_curr, b_curr, score_r, k_r, k_b)

        ratings[r_id] = new_r
        ratings[b_id] = new_b
        counts[r_id]  = r_count + 1
        counts[b_id]  = b_count + 1

        if r_id == fighter_id or b_id == fighter_id:
            is_red     = r_id == fighter_id
            elo_before = r_curr if is_red else b_curr
            elo_after  = new_r  if is_red else new_b
            result     = ("win"  if winner == fighter_id
                          else "draw" if not winner
                          else "loss")
            snapshots.append({
                "date":        row["date"],
                "opponent":    row["b_name"] if is_red else row["r_name"],
                "opponent_id": b_id if is_red else r_id,
                "result":      result,
                "method":      row.get("method") or "",
                "division":    str(row.get("division", "")).lower().strip(),
                "elo_before":  round(elo_before, 1),
                "elo_after":   round(elo_after, 1),
                "elo_change":  round(elo_after - elo_before, 1),
            })

    return snapshots


# ── Glicko-2 public API ───────────────────────────────────────────────────────

def build_glicko_features(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """
    Compute pre-fight Glicko-2 ratings for every fight in the database.

    Returns
    -------
    DataFrame with columns: fight_id, glicko_red, glicko_blue, glicko_rd_red, glicko_rd_blue
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(DB_PATH))

    query = """
        SELECT fight_id, date, division, r_fighter_id, b_fighter_id, winner_id
        FROM fights
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)

    if own_conn:
        conn.close()

    log.info("Calculating Glicko-2 ratings for %d fights…", len(df))
    _, red_r, blue_r, red_rd, blue_rd = _replay_fights_glicko_by_division(df)

    df["glicko_red"]    = red_r
    df["glicko_blue"]   = blue_r
    df["glicko_rd_red"] = red_rd
    df["glicko_rd_blue"] = blue_rd

    log.info("Glicko-2 calculation complete.")
    return df[["fight_id", "glicko_red", "glicko_blue", "glicko_rd_red", "glicko_rd_blue"]]


def get_current_glicko_by_division(
    conn: sqlite3.Connection | None = None,
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """
    Replay all fights and return each fighter's current Glicko-2 state.

    Returns
    -------
    dict mapping (fighter_id, division_lower) -> (r, rd, sigma)
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(DB_PATH))

    query = """
        SELECT fight_id, date, division, r_fighter_id, b_fighter_id, winner_id
        FROM fights
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)

    if own_conn:
        conn.close()

    log.info("Computing current Glicko-2 ratings for %d fights…", len(df))
    current_ratings, _, _, _, _ = _replay_fights_glicko_by_division(df)
    log.info("Glicko-2 ratings computed for %d (fighter, division) pairs.", len(current_ratings))
    return current_ratings
