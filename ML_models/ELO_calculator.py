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
    from config import DB_PATH, STARTING_ELO, K_FACTOR_NORMAL, K_FACTOR_PROVISIONAL, PROVISIONAL_LIMIT
    from logger import get_logger
except ImportError:
    DB_PATH              = Path(__file__).parent.parent / "database_builder_files" / "ufc_v2.db"
    STARTING_ELO         = 1400
    K_FACTOR_NORMAL      = 32
    K_FACTOR_PROVISIONAL = 90
    PROVISIONAL_LIMIT    = 3
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
        SELECT fight_id, date, division, r_fighter_id, b_fighter_id, winner_id
        FROM fights
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)

    if own_conn:
        conn.close()

    log.info("Calculating per-division ELO ratings for %d fights…", len(df))
    _, _, red_elo_pre, blue_elo_pre = _replay_fights_by_division(df)

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
