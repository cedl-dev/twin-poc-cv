"""Pydantic request/response models for the inference API.

Pydantic validates incoming JSON against `SensorReading` before it ever
reaches the model, rejecting malformed payloads with a 422 at no compute
cost.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """A single timestamped multivariate sensor reading to classify."""

    temperature_c: float = Field(..., description="Cryostat temperature reading, in Celsius.")
    beam_intensity_ua: float = Field(..., description="Beam intensity reading, in microamperes.")


class SensorStatus(str, Enum):
    """Business-facing status tag, simulating a control-system (EPICS/TANGO) alarm level."""

    NOMINAL = "NOMINAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PredictionResponse(BaseModel):
    """Result of classifying one `SensorReading`."""

    is_anomaly: bool
    anomaly_score: float = Field(
        ..., description="Isolation Forest decision function score; lower means more anomalous."
    )
    sensor_status: SensorStatus
    originating_pv: Optional[str] = Field(
        None, description="EPICS-style PV name of the channel most responsible for the anomaly."
    )
