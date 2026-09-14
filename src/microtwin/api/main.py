"""FastAPI inference service for the MicroTwin anomaly detector.

The model is loaded once at startup (via the `lifespan` handler) and kept
resident in memory for the app's lifetime, so `/predict` never touches disk
or the MLflow registry on a per-request basis.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI

from microtwin import config
from microtwin.api.schemas import PredictionResponse, SensorReading, SensorStatus

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["temperature_c", "beam_intensity_ua"]

_state: dict[str, Any] = {}


def load_model() -> Any:
    """Load the latest registered model version from the MLflow registry."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{config.MLFLOW_REGISTERED_MODEL_NAME}/latest"
    return mlflow.sklearn.load_model(model_uri)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Loading model '%s' from the MLflow registry...", config.MLFLOW_REGISTERED_MODEL_NAME
    )
    _state["model"] = load_model()
    logger.info("Model loaded, ready to serve.")
    yield
    _state.clear()


app = FastAPI(title="MicroTwin Inference API", lifespan=lifespan)


def _most_deviant_channel(reading: SensorReading) -> tuple[str, float]:
    """Return the (column, z-score) of whichever channel deviates most from its nominal point.

    The nominal mean/std are the same ones used to generate the training
    data (`config.py`); using them as the reference here is only valid
    because this is a closed synthetic system, not a real calibrated sensor.
    """
    temperature_deviation = reading.temperature_c - config.TEMPERATURE_MEAN_C
    temperature_z = abs(temperature_deviation) / config.TEMPERATURE_STD_C

    beam_deviation = reading.beam_intensity_ua - config.BEAM_INTENSITY_MEAN_UA
    beam_z = abs(beam_deviation) / config.BEAM_INTENSITY_STD_UA

    if temperature_z >= beam_z:
        return "temperature_c", temperature_z
    return "beam_intensity_ua", beam_z


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading) -> PredictionResponse:
    model = _state["model"]
    features = pd.DataFrame(
        [[reading.temperature_c, reading.beam_intensity_ua]], columns=FEATURE_COLUMNS
    )

    is_anomaly = bool(model.predict(features)[0] == -1)
    anomaly_score = float(model.decision_function(features)[0])

    sensor_status = SensorStatus.NOMINAL
    originating_pv = None
    if is_anomaly:
        column, z_score = _most_deviant_channel(reading)
        originating_pv = config.SENSOR_PV_NAMES[column]
        sensor_status = (
            SensorStatus.CRITICAL
            if z_score >= config.ANOMALY_Z_SCORE_CRITICAL_THRESHOLD
            else SensorStatus.WARNING
        )

    return PredictionResponse(
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        sensor_status=sensor_status,
        originating_pv=originating_pv,
    )
