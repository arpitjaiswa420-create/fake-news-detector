"""
Unit and Integration tests for FastAPI endpoints.
"""

import io
import pytest
from starlette.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    # Browser request receives interactive HTML dashboard
    response_html = client.get("/")
    assert response_html.status_code == 200
    assert "<!DOCTYPE html>" in response_html.text or "TruthPulse" in response_html.text

    # API JSON client receives sitemap JSON
    response_json = client.get("/", headers={"Accept": "application/json"})
    assert response_json.status_code == 200
    data = response_json.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "best_model_name" in data
    assert "models" in data


def test_predict_text_endpoint(client):
    payload = {
        "title": "Treasury announces new bond issuance targets",
        "text": "The Department of the Treasury released quarterly borrowing estimates and confirmed auction schedules for institutional dealers.",
        "explain": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["REAL", "FAKE"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "metadata_signals" in data
    assert "disclaimer" in data


def test_predict_empty_payload(client):
    payload = {"title": "", "text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_batch_predict_endpoint(client):
    csv_content = (
        "title,text\n"
        "Official statement on economic growth,The bureau reported gross domestic product expanded in Q3.\n"
        "BOMBSHELL: Proof of secret biolab conspiracy exposed!!,Shocking leak proves politicians hid the truth.\n"
    )
    files = {"file": ("test_articles.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/batch-predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_articles"] == 2
    assert "predicted_fake" in data
    assert "predicted_real" in data
    assert len(data["preview_results"]) == 2
