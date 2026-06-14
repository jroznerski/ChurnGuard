"""Integration tests for the FastAPI endpoints (no trained model required)."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 79.85,
    "total_charges": 958.20,
}


class TestHealthEndpoints:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "ChurnGuard API"

    def test_health(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "uptime_seconds" in body
        assert "model_ready" in body

    def test_response_time_header(self):
        r = client.get("/api/v1/health")
        assert "x-response-time" in r.headers or "X-Response-Time" in r.headers


class TestPredictionSchema:
    def test_invalid_gender_rejected(self):
        bad = {**SAMPLE_CUSTOMER, "gender": "Unknown"}
        r = client.post("/api/v1/predictions/predict", json=bad)
        assert r.status_code == 422

    def test_negative_tenure_rejected(self):
        bad = {**SAMPLE_CUSTOMER, "tenure": -1}
        r = client.post("/api/v1/predictions/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_contract_rejected(self):
        bad = {**SAMPLE_CUSTOMER, "contract": "Quarterly"}
        r = client.post("/api/v1/predictions/predict", json=bad)
        assert r.status_code == 422


class TestBatchSchema:
    def test_empty_batch_rejected(self):
        r = client.post("/api/v1/predictions/predict/batch", json={"customers": []})
        assert r.status_code == 422

    def test_valid_batch_schema(self):
        payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
        r = client.post("/api/v1/predictions/predict/batch", json=payload)
        # 503 if no model, but schema should pass
        assert r.status_code in (200, 503)
