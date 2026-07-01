"""
api.py — FastAPI REST endpoint for UFC fight predictions.

Start the server:
    uvicorn api:app --reload
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /                          Health check
GET  /fighters?q=<name>         Search fighters by partial name
POST /predict                   Predict a fight outcome
POST /predict/finish            Predict finish method
GET  /fighters/{id}/elo         Get current ELO for a fighter
POST /card                      Predict a full event card
GET  /fighters/{id}             Fighter profile, stats, and current ELO
GET  /fighters/{id}/elo-history Per-fight ELO snapshots (chronological)
GET  /fighters/{id}/recent-form Last N fights with result and method
GET  /compare?red=&blue=        Side-by-side stats, ELO diff, head-to-head history

Interactive docs: http://localhost:8000/docs
"""

import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DB_V1_PATH,
    STARTING_ELO,
    MODELS_V1_PROD_DIR, MODELS_V1_DIR,
    FINISH_CLASS_NAMES, DIVISIONS,
)
from predict import (
    compute_prediction,
    compute_current_elo,
)
from ml.ELO_calculator import get_current_ratings, get_current_ratings_by_division, get_elo_history_for_fighter
from utils.odds import american_to_prob, remove_vig, compute_edge, kelly_fraction

app = FastAPI(
    title="UFC Fight Predictor API",
    description="ML-powered UFC fight outcome predictions using historical stats and ELO ratings.",
    version="1.0.0",
)


# ── Startup: ELO cache (used by fighter profile / compare endpoints) ──────────

# ELO caches keyed by v1 DB fighter_id (MD5 of name).
# Populated at startup, refreshed via POST /admin/refresh-elo.
_elo_global: dict[str, float]            = {}  # fighter_id -> elo
_elo_div:    dict[tuple[str, str], float] = {}  # (fighter_id, division) -> elo


def _build_elo_caches() -> None:
    """Replay all historical fights from the v1 DB and populate ELO caches."""
    if not DB_V1_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_V1_PATH))
    try:
        t0 = time.monotonic()
        _elo_global.clear()
        _elo_global.update(get_current_ratings(conn))
        _elo_div.clear()
        _elo_div.update(get_current_ratings_by_division(conn))
        elapsed = time.monotonic() - t0
        import logging
        logging.getLogger(__name__).info(
            "ELO cache built: %d global ratings, %d division ratings in %.2fs",
            len(_elo_global), len(_elo_div), elapsed,
        )
    finally:
        conn.close()


@app.on_event("startup")
def startup() -> None:
    """Build ELO cache at startup. Models are loaded on demand by compute_prediction."""
    _build_elo_caches()


# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    red_fighter:  str = Field(...,  example="Islam Makhachev")
    blue_fighter: str = Field(...,  example="Charles Oliveira")
    model:        str = Field("xgb", pattern="^(xgb|lr|rf|lgbm|mlp|ensemble|stacking)$", description="'xgb', 'lr', 'rf', 'lgbm', 'mlp', 'ensemble', or 'stacking'")
    division:     Optional[str]   = Field(None, example="lightweight")
    title_fight:  Optional[int]   = Field(0,    ge=0, le=1)
    odds_red:     Optional[float] = Field(None, example=-150.0,
                                          description="American moneyline odds for Red (e.g. -150)")
    odds_blue:    Optional[float] = Field(None, example=130.0,
                                          description="American moneyline odds for Blue (e.g. +130)")


class FighterResult(BaseModel):
    fighter_id: str
    name:       str
    win_prob:   float
    elo:        float


class FinishProba(BaseModel):
    decision:   float
    ko_tko:     float
    submission: float


class ValueBet(BaseModel):
    fighter:      str
    model_prob:   float
    market_fair:  float
    edge:         float
    kelly_stake:  float
    value:        bool


class H2HFight(BaseModel):
    date:   str
    winner: str
    method: str


class PredictResponse(BaseModel):
    red:              FighterResult
    blue:             FighterResult
    predicted_winner: str
    confidence:       float
    model:            str
    finish_proba:     Optional[FinishProba]
    value_bets:       Optional[list[ValueBet]]
    prior_fights:     Optional[int]       = None
    head_to_head:     Optional[list[H2HFight]] = None


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    if not DB_V1_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Database not found: {DB_V1_PATH}")
    return sqlite3.connect(str(DB_V1_PATH))


def _search_fighters(conn: sqlite3.Connection, name: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT fighter_id, name FROM fighters WHERE name LIKE ? ORDER BY name LIMIT 20",
        (f"%{name}%",),
    )
    return [{"fighter_id": r[0], "name": r[1]} for r in cur.fetchall()]


def _get_h2h(conn: sqlite3.Connection, id_a: str, id_b: str) -> list[H2HFight]:
    """Return all fights between two fighters, newest first."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f.date, f.winner_id, f.method,
               f.r_fighter_id, fr.name AS r_name, fb.name AS b_name
        FROM fights f
        JOIN fighters fr ON fr.fighter_id = f.r_fighter_id
        JOIN fighters fb ON fb.fighter_id = f.b_fighter_id
        WHERE (f.r_fighter_id = ? AND f.b_fighter_id = ?)
           OR (f.r_fighter_id = ? AND f.b_fighter_id = ?)
        ORDER BY f.date DESC
        """,
        (id_a, id_b, id_b, id_a),
    )
    results = []
    for date, winner_id, method, r_fighter_id, r_name, b_name in cur.fetchall():
        # Resolve winner name from the original corner assignments, not the request order
        if winner_id == r_fighter_id:
            winner_name = r_name
        elif winner_id:
            winner_name = b_name
        else:
            winner_name = "Draw"
        results.append(H2HFight(
            date=date,
            winner=winner_name,
            method=method or "",
        ))
    return results


def _get_fighter_stats(conn: sqlite3.Connection, fighter_id: str) -> dict:
    """Return latest rolling stats and career record for a fighter."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT fs.splm, fs.sapm, fs.str_def, fs.td_avg, fs.td_def
        FROM fight_stats fs
        JOIN fights f ON f.fight_id = fs.fight_id
        WHERE fs.fighter_id = ?
        ORDER BY f.date DESC
        LIMIT 1
        """,
        (fighter_id,),
    )
    row = cur.fetchone()
    stats = {}
    if row:
        for key, val in zip(["splm", "sapm", "str_def", "td_avg", "td_def"], row):
            stats[key] = round(float(val), 3) if val is not None else None

    cur.execute(
        """
        SELECT
            COUNT(CASE WHEN winner_id = ? THEN 1 END),
            COUNT(CASE WHEN winner_id != ? AND winner_id IS NOT NULL AND winner_id != '' THEN 1 END)
        FROM fights WHERE r_fighter_id = ? OR b_fighter_id = ?
        """,
        (fighter_id, fighter_id, fighter_id, fighter_id),
    )
    wins, losses = cur.fetchone()
    return {"stats": stats, "record": f"{wins}-{losses}"}


# ── Prediction logic (shared) ─────────────────────────────────────────────────

def _run_prediction(req: PredictRequest) -> PredictResponse:
    _models_dir = (
        MODELS_V1_PROD_DIR
        if MODELS_V1_PROD_DIR.exists() and any(MODELS_V1_PROD_DIR.iterdir())
        else MODELS_V1_DIR
    )

    try:
        result = compute_prediction(
            red_name=req.red_fighter,
            blue_name=req.blue_fighter,
            model_type=req.model,
            division=req.division,
            title_fight=req.title_fight or 0,
            db_path=DB_V1_PATH,
            models_dir=_models_dir,
        )
    except SystemExit:
        raise HTTPException(status_code=404, detail=f"Fighter not found or prediction failed.")

    red_win_prob  = result["red_prob"]
    blue_win_prob = result["blue_prob"]

    # ── Fighter IDs + H2H from v1 DB ─────────────────────────────────────────
    conn = _get_conn()
    try:
        r_row = _search_fighters(conn, result["red_name"])
        b_row = _search_fighters(conn, result["blue_name"])
        r_id  = r_row[0]["fighter_id"] if r_row else ""
        b_id  = b_row[0]["fighter_id"] if b_row else ""
        h2h   = _get_h2h(conn, r_id, b_id) if r_id and b_id else []
    finally:
        conn.close()

    # ── Finish proba ──────────────────────────────────────────────────────────
    finish_proba_resp: Optional[FinishProba] = None
    if result.get("finish_proba"):
        fp = result["finish_proba"]
        finish_proba_resp = FinishProba(
            decision=float(fp[0]),
            ko_tko=float(fp[1]),
            submission=float(fp[2]),
        )

    # ── Odds / value bets ─────────────────────────────────────────────────────
    value_bets_resp: Optional[list[ValueBet]] = None
    if req.odds_red is not None and req.odds_blue is not None:
        raw_pr  = american_to_prob(req.odds_red)
        raw_pb  = american_to_prob(req.odds_blue)
        fair_pr, fair_pb = remove_vig(raw_pr, raw_pb)

        dec_r = 1 / raw_pr if raw_pr > 0 else 0
        dec_b = 1 / raw_pb if raw_pb > 0 else 0

        edge_r = compute_edge(red_win_prob, fair_pr)
        edge_b = compute_edge(blue_win_prob, fair_pb)

        value_bets_resp = [
            ValueBet(
                fighter=result["red_name"],
                model_prob=round(red_win_prob, 4),
                market_fair=round(fair_pr, 4),
                edge=round(edge_r, 4),
                kelly_stake=round(kelly_fraction(edge_r, dec_r), 4),
                value=edge_r >= 0.03,
            ),
            ValueBet(
                fighter=result["blue_name"],
                model_prob=round(blue_win_prob, 4),
                market_fair=round(fair_pb, 4),
                edge=round(edge_b, 4),
                kelly_stake=round(kelly_fraction(edge_b, dec_b), 4),
                value=edge_b >= 0.03,
            ),
        ]

    _model_labels = {
        "xgb": "XGBoost", "lr": "Logistic Regression", "rf": "Random Forest",
        "lgbm": "LightGBM", "mlp": "MLP Neural Network",
        "ensemble": "Ensemble (Soft Vote)", "stacking": "Stacking Meta-Learner",
    }

    return PredictResponse(
        red=FighterResult(
            fighter_id=r_id,
            name=result["red_name"],
            win_prob=round(red_win_prob, 4),
            elo=round(result["elo_red"], 1),
        ),
        blue=FighterResult(
            fighter_id=b_id,
            name=result["blue_name"],
            win_prob=round(blue_win_prob, 4),
            elo=round(result["elo_blue"], 1),
        ),
        predicted_winner=result["winner"],
        confidence=round(result["confidence"], 4),
        model=_model_labels.get(req.model, req.model),
        finish_proba=finish_proba_resp,
        value_bets=value_bets_resp,
        prior_fights=len(h2h) if h2h else None,
        head_to_head=h2h if h2h else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    """Returns API status and which models are loaded."""
    return {
        "status": "ok",
        "models_loaded": {
            "xgboost": "xgb" in _models,
            "logistic_regression": "lr_base" in _models,
            "finish_type": "finish" in _models,
        },
        "elo_cache": {
            "global_fighters": len(_elo_global),
            "division_pairs":  len(_elo_div),
        },
        "database": DB_V1_PATH.exists(),
        "docs": "/docs",
    }


@app.post("/admin/refresh-elo", summary="Rebuild ELO cache without restarting")
def refresh_elo():
    """
    Replay all historical fights and repopulate the ELO cache.
    Call this after ingesting new fight data so predictions use up-to-date ratings.
    """
    _build_elo_caches()
    return {
        "status": "ok",
        "global_fighters": len(_elo_global),
        "division_pairs":  len(_elo_div),
    }


@app.get("/fighters", summary="Search fighters by name")
def search_fighters(q: str = Query(..., min_length=2, description="Partial fighter name")):
    """Return up to 20 fighters whose name contains the query string."""
    conn = _get_conn()
    try:
        results = _search_fighters(conn, q)
    finally:
        conn.close()
    if not results:
        raise HTTPException(status_code=404, detail=f"No fighters found matching '{q}'")
    return {"query": q, "results": results, "count": len(results)}


@app.get("/fighters/{fighter_id}/elo", summary="Get fighter's current ELO")
def get_fighter_elo(
    fighter_id: str,
    division:   Optional[str] = Query(None, description="Division for division-specific ELO"),
):
    """Return a fighter's current ELO rating (global or per-division)."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM fighters WHERE fighter_id = ?", (fighter_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Fighter ID '{fighter_id}' not found")
        name = row[0]

        if division:
            div_elo = get_current_ratings_by_division(conn)
            elo = div_elo.get((fighter_id, division.lower()), STARTING_ELO)
        else:
            elos = compute_current_elo(conn)
            elo  = elos.get(fighter_id, STARTING_ELO)
    finally:
        conn.close()

    return {
        "fighter_id": fighter_id,
        "name":       name,
        "elo":        round(elo, 1),
        "division":   division or "global",
    }


@app.get("/fighters/{fighter_id}", summary="Get fighter profile")
def get_fighter_profile(fighter_id: str):
    """
    Return a fighter's profile: basic info, career record, latest rolling stats,
    and current ELO (division-specific for their most recent division).
    """
    conn = _get_conn()
    try:
        cur = conn.cursor()

        # Basic info
        cur.execute(
            "SELECT name, height, reach, stance, dob FROM fighters WHERE fighter_id = ?",
            (fighter_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Fighter '{fighter_id}' not found.")
        name, height, reach, stance, dob = row

        # Career record from fights table
        cur.execute(
            """
            SELECT
                COUNT(CASE WHEN winner_id = ? THEN 1 END),
                COUNT(CASE WHEN winner_id != ? AND winner_id IS NOT NULL AND winner_id != '' THEN 1 END),
                COUNT(CASE WHEN winner_id IS NULL OR winner_id = '' THEN 1 END)
            FROM fights
            WHERE r_fighter_id = ? OR b_fighter_id = ?
            """,
            (fighter_id, fighter_id, fighter_id, fighter_id),
        )
        wins, losses, draws = cur.fetchone()

        # Latest rolling stats from most recent fight_stats row
        cur.execute(
            """
            SELECT fs.splm, fs.sapm, fs.str_def, fs.td_avg, fs.td_def,
                   f.division, f.date
            FROM fight_stats fs
            JOIN fights f ON f.fight_id = fs.fight_id
            WHERE fs.fighter_id = ?
            ORDER BY f.date DESC
            LIMIT 1
            """,
            (fighter_id,),
        )
        stats_row = cur.fetchone()
    finally:
        conn.close()

    # Global ELO from cache -- consistent with the training CSV (build_elo_features
    # uses global ratings, not per-division)
    elo = _elo_global.get(fighter_id, STARTING_ELO)

    stats = {}
    if stats_row:
        for key, val in zip(["splm", "sapm", "str_def", "td_avg", "td_def"], stats_row[:5]):
            stats[key] = round(float(val), 3) if val is not None else None

    return {
        "fighter_id": fighter_id,
        "name":       name,
        "height":     height,
        "reach":      reach,
        "stance":     stance,
        "dob":        dob,
        "record":     {"wins": wins, "losses": losses, "draws": draws},
        "stats":      stats,
        "elo": {
            "current": round(elo, 1),
            "scope":   "global",
        },
    }


@app.get("/fighters/{fighter_id}/elo-history", summary="Get per-fight ELO history")
def get_fighter_elo_history(fighter_id: str):
    """
    Return per-fight ELO snapshots for a fighter in chronological order.
    Shows elo_before, elo_after, and elo_change for every UFC fight on record.
    Uses the same per-division ELO replay as predict.py.
    """
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM fighters WHERE fighter_id = ?", (fighter_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Fighter '{fighter_id}' not found.")
        history = get_elo_history_for_fighter(fighter_id, conn)
    finally:
        conn.close()

    return {"fighter_id": fighter_id, "name": row[0], "history": history}


@app.get("/fighters/{fighter_id}/recent-form", summary="Get recent fight results")
def get_fighter_recent_form(
    fighter_id: str,
    n: int = Query(5, ge=1, le=20, description="Number of recent fights to return"),
):
    """
    Return the last N fights for a fighter with result, method, opponent, and division.
    """
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM fighters WHERE fighter_id = ?", (fighter_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Fighter '{fighter_id}' not found.")
        name = row[0]

        cur.execute(
            """
            SELECT f.date, f.division, f.method, f.title_fight,
                   f.winner_id,
                   fr.name AS r_name, fb.name AS b_name,
                   f.r_fighter_id, f.b_fighter_id
            FROM fights f
            JOIN fighters fr ON fr.fighter_id = f.r_fighter_id
            JOIN fighters fb ON fb.fighter_id = f.b_fighter_id
            WHERE f.r_fighter_id = ? OR f.b_fighter_id = ?
            ORDER BY f.date DESC
            LIMIT ?
            """,
            (fighter_id, fighter_id, n),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    fights = []
    for date, division, method, title_fight, winner_id, r_name, b_name, r_id, b_id in rows:
        opponent = b_name if r_id == fighter_id else r_name
        opp_id   = b_id   if r_id == fighter_id else r_id
        result   = ("win"  if winner_id == fighter_id
                    else "draw" if not winner_id
                    else "loss")
        fights.append({
            "date":        date,
            "opponent":    opponent,
            "opponent_id": opp_id,
            "result":      result,
            "method":      method or "",
            "division":    (division or "").lower(),
            "title_fight": bool(title_fight),
        })

    return {"fighter_id": fighter_id, "name": name, "fights": fights}


@app.get("/compare", summary="Compare two fighters side-by-side")
def compare_fighters(
    red:  str = Query(..., description="Red fighter name (partial OK)"),
    blue: str = Query(..., description="Blue fighter name (partial OK)"),
):
    """
    Return a side-by-side stat comparison, ELO diff, record, and full
    head-to-head fight history for two fighters.
    Diffs follow the model convention: red - blue (positive = red advantage).
    """
    conn = _get_conn()
    try:
        r_matches = _search_fighters(conn, red)
        b_matches = _search_fighters(conn, blue)

        if not r_matches:
            raise HTTPException(status_code=404, detail=f"Fighter not found: '{red}'")
        if not b_matches:
            raise HTTPException(status_code=404, detail=f"Fighter not found: '{blue}'")

        def _best(matches: list[dict], query: str) -> dict:
            for m in matches:
                if m["name"].lower() == query.lower():
                    return m
            return matches[0]

        r = _best(r_matches, red)
        b = _best(b_matches, blue)

        if r["fighter_id"] == b["fighter_id"]:
            raise HTTPException(status_code=400, detail="Both names resolved to the same fighter.")

        r_data = _get_fighter_stats(conn, r["fighter_id"])
        b_data = _get_fighter_stats(conn, b["fighter_id"])
        h2h    = _get_h2h(conn, r["fighter_id"], b["fighter_id"])
    finally:
        conn.close()

    elo_r = _elo_global.get(r["fighter_id"], STARTING_ELO)
    elo_b = _elo_global.get(b["fighter_id"], STARTING_ELO)

    stat_keys = ["splm", "sapm", "str_def", "td_avg", "td_def"]
    r_stats   = r_data["stats"]
    b_stats   = b_data["stats"]

    diff = {}
    for k in stat_keys:
        rv = r_stats.get(k)
        bv = b_stats.get(k)
        diff[k] = round(rv - bv, 3) if rv is not None and bv is not None else None
    diff["elo"] = round(elo_r - elo_b, 1)

    return {
        "red": {
            "fighter_id": r["fighter_id"],
            "name":       r["name"],
            "elo":        round(elo_r, 1),
            "record":     r_data["record"],
            "stats":      r_stats,
        },
        "blue": {
            "fighter_id": b["fighter_id"],
            "name":       b["name"],
            "elo":        round(elo_b, 1),
            "record":     b_data["record"],
            "stats":      b_stats,
        },
        "diff":        diff,
        "head_to_head": h2h,
    }


@app.post("/predict", response_model=PredictResponse, summary="Predict fight outcome")
def predict(req: PredictRequest):
    """
    Predict the winner of a UFC fight.

    - Partial names OK (first match is used — use /fighters to verify)
    - Optionally include American moneyline odds to get value-bet analysis
    - Finish type probabilities (Decision / KO-TKO / Submission) included if model is loaded
    """
    return _run_prediction(req)


@app.post("/predict/batch", summary="Predict multiple fights at once")
def predict_batch(requests: list[PredictRequest]):
    """Run up to 10 predictions in one call."""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 fights per batch request.")
    return [_run_prediction(r) for r in requests]


# ── /card models ──────────────────────────────────────────────────────────────

class CardMatchup(BaseModel):
    red_fighter:  str
    blue_fighter: str
    division:     Optional[str]   = None
    title_fight:  Optional[int]   = Field(0, ge=0, le=1)
    odds_red:     Optional[float] = None
    odds_blue:    Optional[float] = None


class CardRequest(BaseModel):
    fights:     list[CardMatchup] = Field(..., min_length=1, max_length=15)
    model:      str               = Field("ensemble", pattern="^(xgb|lr|rf|lgbm|mlp|ensemble|stacking)$")
    event_name: Optional[str]     = None
    event_date: Optional[str]     = None


class CardFightResult(BaseModel):
    red_fighter:  str
    blue_fighter: str
    prediction:   Optional[PredictResponse] = None
    error:        Optional[str]             = None


class CardResponse(BaseModel):
    event_name:  Optional[str]
    event_date:  Optional[str]
    model:       str
    fight_count: int
    fights:      list[CardFightResult]


@app.post("/card", response_model=CardResponse, summary="Predict a full event card")
def predict_card(req: CardRequest):
    """
    Predict outcomes for every fight on an event card.

    - Supply up to 15 matchups with optional division, title-fight flag, and odds.
    - A single model is used across all fights (default: ensemble).
    - Fights where a fighter cannot be resolved are returned with an error field
      rather than failing the entire request.
    - Optionally pass event_name and event_date for context in the response.
    """
    results: list[CardFightResult] = []
    for matchup in req.fights:
        pred_req = PredictRequest(
            red_fighter=matchup.red_fighter,
            blue_fighter=matchup.blue_fighter,
            model=req.model,
            division=matchup.division,
            title_fight=matchup.title_fight or 0,
            odds_red=matchup.odds_red,
            odds_blue=matchup.odds_blue,
        )
        try:
            prediction = _run_prediction(pred_req)
            results.append(CardFightResult(
                red_fighter=matchup.red_fighter,
                blue_fighter=matchup.blue_fighter,
                prediction=prediction,
            ))
        except HTTPException as exc:
            results.append(CardFightResult(
                red_fighter=matchup.red_fighter,
                blue_fighter=matchup.blue_fighter,
                error=exc.detail,
            ))

    return CardResponse(
        event_name=req.event_name,
        event_date=req.event_date,
        model=req.model,
        fight_count=len(results),
        fights=results,
    )
