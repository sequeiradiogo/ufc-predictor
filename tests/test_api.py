"""
tests/test_api.py
==================
Smoke tests for the FastAPI REST layer (api.py).

These exercise the app through fastapi.testclient.TestClient (no live
uvicorn process needed) so a broken endpoint -- like the undefined
`_models` NameError that used to crash GET / -- fails CI instead of
going unnoticed until someone hits it manually.

DB/model-dependent tests skip gracefully when those artifacts are
absent, matching the pattern in test_inference.py.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH, DB_V1_PATH, MODELS_V1_DIR, MODELS_V1_PROD_DIR
from api import app


def _db_v1_available():
    return DB_V1_PATH.exists()


def _models_available():
    return (MODELS_V1_DIR / "ensemble.joblib").exists() or (
        MODELS_V1_PROD_DIR.exists() and (MODELS_V1_PROD_DIR / "ensemble.joblib").exists()
    )


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ══════════════════════════════════════════════════════════════════════════════
# Health check -- always runnable, no artifacts required
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthCheck:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_shape(self, client):
        body = client.get("/").json()
        assert body["status"] == "ok"
        assert "models_dir" in body
        assert set(body["models_loaded"]) == {
            "xgboost", "logistic_regression", "ensemble", "finish_type",
        }
        assert all(isinstance(v, bool) for v in body["models_loaded"].values())
        assert "elo_cache" in body
        assert isinstance(body["database"], bool)


# ══════════════════════════════════════════════════════════════════════════════
# Fighter search -- validation always runnable; DB-backed lookups skip w/o DB
# ══════════════════════════════════════════════════════════════════════════════

class TestFighterSearch:
    def test_query_too_short_returns_422(self, client):
        resp = client.get("/fighters", params={"q": "a"})
        assert resp.status_code == 422

    @pytest.mark.skipif(not _db_v1_available(), reason="mdabbert DB not found")
    def test_known_fighter_found(self, client):
        resp = client.get("/fighters", params={"q": "Makhachev"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        assert any("Makhachev" in r["name"] for r in body["results"])

    @pytest.mark.skipif(not _db_v1_available(), reason="mdabbert DB not found")
    def test_unknown_fighter_returns_404(self, client):
        resp = client.get("/fighters", params={"q": "Zzzznonexistentfighter"})
        assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# Prediction -- needs both the DB and trained model artifacts
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (DB_PATH.exists() and _db_v1_available() and _models_available()),
    reason="DB and/or v1 model artifacts not found",
)
class TestPredict:
    def test_predict_returns_valid_response(self, client):
        resp = client.post(
            "/predict",
            json={"red_fighter": "Islam Makhachev", "blue_fighter": "Charles Oliveira"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_winner"] in (body["red"]["name"], body["blue"]["name"])
        assert 0.0 <= body["red"]["win_prob"] <= 1.0
        assert abs(body["red"]["win_prob"] + body["blue"]["win_prob"] - 1.0) < 1e-6

    def test_predict_unknown_fighter_returns_error(self, client):
        resp = client.post(
            "/predict",
            json={"red_fighter": "Zzzznonexistentfighter", "blue_fighter": "Charles Oliveira"},
        )
        assert resp.status_code >= 400
