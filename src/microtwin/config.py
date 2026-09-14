"""Centralized configuration for the MicroTwin pipeline.

Every setting is a module-level constant, overridable via an environment
variable of the same name prefixed with `MICROTWIN_` (12-factor style), so
the pipeline can be reconfigured for containerized or HPC deployment without
touching code.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Filesystem layout ---
DATA_RAW_DIR = Path(os.getenv("MICROTWIN_DATA_RAW_DIR", str(PROJECT_ROOT / "data" / "raw")))
DATA_PROCESSED_DIR = Path(
    os.getenv("MICROTWIN_DATA_PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed"))
)
MODELS_DIR = Path(os.getenv("MICROTWIN_MODELS_DIR", str(PROJECT_ROOT / "models")))
PROCESSED_DATA_FILENAME = os.getenv("MICROTWIN_PROCESSED_DATA_FILENAME", "sensor_data.parquet")

# --- MLflow ---
# A SQLite-backed store (rather than a plain "file:" store) is required for
# the Model Registry (versioned model artifacts) to work locally.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "microtwin-anomaly-detection")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "microtwin-isolation-forest"
)

# --- Sensor data generation (Step 1) ---
# Maps each generated data column to the EPICS-style process variable it
# represents. Used by the API (Step 4) to report the originating PV in
# prediction responses.
SENSOR_PV_NAMES = {
    "temperature_c": "GANIL:CRYO:TEMP01",
    "beam_intensity_ua": "GANIL:BEAM:INT01",
}

N_ROWS = int(os.getenv("MICROTWIN_N_ROWS", "1000000"))
ANOMALY_RATE = float(os.getenv("MICROTWIN_ANOMALY_RATE", "0.01"))
RANDOM_SEED = int(os.getenv("MICROTWIN_RANDOM_SEED", "42"))

TEMPERATURE_MEAN_C = float(os.getenv("MICROTWIN_TEMPERATURE_MEAN_C", "18.0"))
TEMPERATURE_STD_C = float(os.getenv("MICROTWIN_TEMPERATURE_STD_C", "0.5"))
BEAM_INTENSITY_MEAN_UA = float(os.getenv("MICROTWIN_BEAM_INTENSITY_MEAN_UA", "50.0"))
BEAM_INTENSITY_STD_UA = float(os.getenv("MICROTWIN_BEAM_INTENSITY_STD_UA", "2.0"))

# --- Isolation Forest hyperparameters (Step 3) ---
CONTAMINATION = float(os.getenv("MICROTWIN_CONTAMINATION", str(ANOMALY_RATE)))
N_ESTIMATORS = int(os.getenv("MICROTWIN_N_ESTIMATORS", "200"))
