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

Interactive docs: http://localhost:8000/docs
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DB_PATH,
    STARTING_ELO,
    MODEL_XGB_PATH, MODEL_XGB_FEATURES,
    MODEL_LR_PATH, MODEL_LR_SCALER, MODEL_LR_FEATURES,
    MODEL_FINISH_PATH, MODEL_FINISH_FEATURES,
    FINISH_CLASS_NAMES, DIVISIONS,
)
from predict import (
    resolve_fighter,
    get_latest_stats,
    compute_current_elo,
    compute_recent_form,
    build_feature_vector,
)
from ML_models.ELO_calculator import get_current_ratings_by_division
from odds import american_to_prob, remove_vig, compute_edge, kelly_fraction

app = FastAPI(
    title="UFC Fight Predictor API",
    description="ML-powered UFC fight outcome predictions using historical stats and ELO ratings.",
    version="1.0.0",
)


# ── Startup: load models once ─────────────────────────────────────────────────

_models: dict = {}

@app.on_event("startup")
def load_models() -> None:
    """Pre-load model artifacts at startup so predictions are fast."""
    if MODEL_XGB_PATH.exists():
        _models["xgb"]          = joblib.load(MODEL_XGB_PATH)
        _models["xgb_features"] = joblib.load(MODEL_XGB_FEATURES)

    if MODEL_LR_PATH.exists():
        artifact                = joblib.load(MODEL_LR_PATH)
        _models["lr_base"]      = artifact["base"]
        _models["lr_platt"]     = artifact["platt"]
        _models["lr_scaler"]    = joblib.load(MODEL_LR_SCALER)
        _models["lr_features"]  = joblib.load(MODEL_LR_FEATURES)

    if MODEL_FINISH_PATH.exists():
        _models["finish"]          = joblib.load(MODEL_FINISH_PATH)
        _models["finish_features"] = joblib.load(MODEL_FINISH_FEATURES)


# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    red_fighter:  str = Field(...,  example="Islam Makhachev")
    blue_fighter: str = Field(...,  example="Charles Oliveira")
    model:        str = Field("xgb", pattern="^(xgb|lr)$", description="'xgb' or 'lr'")
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


class PredictResponse(BaseModel):
    red:              FighterResult
    blue:             FighterResult
    predicted_winner: str
    confidence:       float
    model:            str
    finish_proba:     Optional[FinishProba]
    value_bets:       Optional[list[ValueBet]]


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Database not found: {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


def _search_fighters(conn: sqlite3.Connection, name: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT fighter_id, name FROM fighters WHERE name LIKE ? ORDER BY name LIMIT 20",
        (f"%{name}%",),
    )
    return [{"fighter_id": r[0], "name": r[1]} for r in cur.fetchall()]


# ── Prediction logic (shared) ─────────────────────────────────────────────────

def _run_prediction(req: PredictRequest) -> PredictResponse:
    # ── Model selection ───────────────────────────────────────────────────────
    if req.model == "xgb":
        if "xgb" not in _models:
            raise HTTPException(status_code=503, detail="XGBoost model not loaded. Train it first.")
        model         = _models["xgb"]
        feature_names = _models["xgb_features"]
        scaler        = None
        platt         = None
        model_label   = "XGBoost"
    else:
        if "lr_base" not in _models:
            raise HTTPException(status_code=503, detail="LR model not loaded. Train it first.")
        model         = None
        feature_names = _models["lr_features"]
        scaler        = _models["lr_scaler"]
        platt         = _models["lr_platt"]
        base_model    = _models["lr_base"]
        model_label   = "Logistic Regression"

    # ── DB queries ────────────────────────────────────────────────────────────
    conn = _get_conn()
    try:
        # Resolve fighters
        r_matches = _search_fighters(conn, req.red_fighter)
        b_matches = _search_fighters(conn, req.blue_fighter)

        if not r_matches:
            raise HTTPException(status_code=404, detail=f"Fighter not found: '{req.red_fighter}'")
        if not b_matches:
            raise HTTPException(status_code=404, detail=f"Fighter not found: '{req.blue_fighter}'")

        # Exact match first; otherwise take first result
        def _best(matches: list[dict], query: str) -> dict:
            for m in matches:
                if m["name"].lower() == query.lower():
                    return m
            return matches[0]

        r = _best(r_matches, req.red_fighter)
        b = _best(b_matches, req.blue_fighter)

        if r["fighter_id"] == b["fighter_id"]:
            raise HTTPException(status_code=400, detail="Both names resolved to the same fighter.")

        # Stats
        red_stats  = get_latest_stats(conn, r["fighter_id"])
        blue_stats = get_latest_stats(conn, b["fighter_id"])

        # ELO
        div_lower = (req.division or "").lower().strip()
        if div_lower:
            div_elo = get_current_ratings_by_division(conn)
            elo_r   = div_elo.get((r["fighter_id"], div_lower), STARTING_ELO)
            elo_b   = div_elo.get((b["fighter_id"], div_lower), STARTING_ELO)
        else:
            elos  = compute_current_elo(conn)
            elo_r = elos.get(r["fighter_id"], STARTING_ELO)
            elo_b = elos.get(b["fighter_id"], STARTING_ELO)

        # Recent form
        form_r = compute_recent_form(conn, r["fighter_id"])
        form_b = compute_recent_form(conn, b["fighter_id"])

    finally:
        conn.close()

    # ── Feature vector ────────────────────────────────────────────────────────
    X = build_feature_vector(
        red_stats, blue_stats,
        elo_r, elo_b,
        form_r, form_b,
        req.division, req.title_fight or 0,
        feature_names,
    ).fillna(0)

    X_input = scaler.transform(X) if scaler is not None else X.values

    if platt is None:
        proba = model.predict_proba(X_input)[0]
    else:
        raw_prob   = base_model.predict_proba(X_input)[0, 1]
        calibrated = platt.predict_proba([[raw_prob]])[0, 1]
        proba      = [1 - calibrated, calibrated]

    red_win_prob  = float(proba[1])
    blue_win_prob = float(proba[0])
    winner_name   = r["name"] if red_win_prob >= 0.5 else b["name"]
    confidence    = max(red_win_prob, blue_win_prob)

    # ── Finish type ───────────────────────────────────────────────────────────
    finish_proba_resp: Optional[FinishProba] = None
    if "finish" in _models:
        X_fin = build_feature_vector(
            red_stats, blue_stats,
            elo_r, elo_b,
            form_r, form_b,
            req.division, req.title_fight or 0,
            _models["finish_features"],
        ).fillna(0)
        fp = _models["finish"].predict_proba(X_fin.values)[0]
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

        edge_r = compute_edge(red_win_prob,  fair_pr)
        edge_b = compute_edge(blue_win_prob, fair_pb)

        value_bets_resp = [
            ValueBet(
                fighter=r["name"],
                model_prob=round(red_win_prob, 4),
                market_fair=round(fair_pr, 4),
                edge=round(edge_r, 4),
                kelly_stake=round(kelly_fraction(edge_r, dec_r), 4),
                value=edge_r >= 0.03,
            ),
            ValueBet(
                fighter=b["name"],
                model_prob=round(blue_win_prob, 4),
                market_fair=round(fair_pb, 4),
                edge=round(edge_b, 4),
                kelly_stake=round(kelly_fraction(edge_b, dec_b), 4),
                value=edge_b >= 0.03,
            ),
        ]

    return PredictResponse(
        red=FighterResult(
            fighter_id=r["fighter_id"],
            name=r["name"],
            win_prob=round(red_win_prob, 4),
            elo=round(elo_r, 1),
        ),
        blue=FighterResult(
            fighter_id=b["fighter_id"],
            name=b["name"],
            win_prob=round(blue_win_prob, 4),
            elo=round(elo_b, 1),
        ),
        predicted_winner=winner_name,
        confidence=round(confidence, 4),
        model=model_label,
        finish_proba=finish_proba_resp,
        value_bets=value_bets_resp,
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
        "database": DB_PATH.exists(),
        "docs": "/docs",
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
