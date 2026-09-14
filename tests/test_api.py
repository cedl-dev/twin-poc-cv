"""Unit tests for the FastAPI inference service.

The real MLflow-loaded model is replaced with a small model trained
in-memory for the duration of each test (via monkeypatching `load_model`),
so the test suite never depends on a pre-existing trained/registered model
on disk — it must pass on a fresh clone before `microtwin-train` has ever
been run.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import IsolationForest

from microtwin.api import main
from microtwin.api.main import _most_deviant_channel
from microtwin.api.schemas import SensorReading
from microtwin.data.generate import generate_sensor_data


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    df = generate_sensor_data(n_rows=2_000, anomaly_rate=0.05, seed=0)
    model = IsolationForest(contamination=0.05, n_estimators=50, random_state=0)
    model.fit(df[["temperature_c", "beam_intensity_ua"]])

    monkeypatch.setattr(main, "load_model", lambda: model)

    with TestClient(main.app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_nominal_reading_is_not_flagged(client: TestClient) -> None:
    response = client.post("/predict", json={"temperature_c": 18.0, "beam_intensity_ua": 50.0})

    assert response.status_code == 200
    body = response.json()
    assert body["is_anomaly"] is False
    assert body["sensor_status"] == "NOMINAL"
    assert body["originating_pv"] is None


def test_predict_extreme_temperature_is_flagged_as_critical_on_the_temperature_pv(
    client: TestClient,
) -> None:
    response = client.post("/predict", json={"temperature_c": 200.0, "beam_intensity_ua": 50.0})

    assert response.status_code == 200
    body = response.json()
    assert body["is_anomaly"] is True
    assert body["sensor_status"] == "CRITICAL"
    assert body["originating_pv"] == "GANIL:CRYO:TEMP01"


def test_predict_extreme_beam_intensity_is_flagged_on_the_beam_pv(client: TestClient) -> None:
    response = client.post("/predict", json={"temperature_c": 18.0, "beam_intensity_ua": 5000.0})

    assert response.status_code == 200
    body = response.json()
    assert body["is_anomaly"] is True
    assert body["originating_pv"] == "GANIL:BEAM:INT01"


def test_predict_rejects_non_numeric_payload(client: TestClient) -> None:
    response = client.post(
        "/predict", json={"temperature_c": "not-a-number", "beam_intensity_ua": 50.0}
    )

    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    response = client.post("/predict", json={"temperature_c": 18.0})

    assert response.status_code == 422


def test_most_deviant_channel_picks_the_larger_z_score() -> None:
    reading = SensorReading(temperature_c=18.0, beam_intensity_ua=5000.0)

    column, z_score = _most_deviant_channel(reading)

    assert column == "beam_intensity_ua"
    assert z_score > 0
