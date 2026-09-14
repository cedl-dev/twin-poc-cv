"""Isolation Forest training with MLflow experiment tracking.

Trains an unsupervised anomaly detector on the processed sensor Parquet
dataset and logs the run — hyperparameters, metrics, and the fitted model as
a versioned Model Registry artifact — to MLflow. The inference API (Step 4)
then always loads a traceable, reproducible model version instead of a
loose pickle file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from microtwin import config

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["temperature_c", "beam_intensity_ua"]


def load_training_data(path: Path | None = None) -> pd.DataFrame:
    """Load the processed Parquet dataset produced by the ETL pipeline (Step 2)."""
    default_path = config.DATA_PROCESSED_DIR / config.PROCESSED_DATA_FILENAME
    source = path if path is not None else default_path
    return pd.read_parquet(source)


def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = config.CONTAMINATION,
    n_estimators: int = config.N_ESTIMATORS,
    random_state: int = config.RANDOM_SEED,
) -> IsolationForest:
    """Fit an Isolation Forest on the feature columns of `df`.

    Unsupervised: only `FEATURE_COLUMNS` are seen by the model. A ground-truth
    `is_anomaly` column, if present, is never used for training — it exists
    only because this is synthetic data, and is used solely for evaluation
    in `evaluate_against_ground_truth`.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(df[FEATURE_COLUMNS])
    return model


def evaluate_against_ground_truth(df: pd.DataFrame, predictions) -> dict[str, float]:
    """Compare model predictions to the synthetic ground-truth anomaly flag.

    A real unsupervised deployment would not have this luxury; it is only
    possible here because Step 1 generates and keeps the true `is_anomaly`
    label alongside the data, purely for model validation. Returns an empty
    dict if that column isn't available.
    """
    if "is_anomaly" not in df.columns:
        return {}

    y_true = df["is_anomaly"].to_numpy()
    y_pred = predictions == -1  # IsolationForest: -1 means anomaly, 1 means normal

    return {
        "ground_truth_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "ground_truth_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ground_truth_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_training(data_path: Path | None = None) -> str:
    """Train the model and log a full MLflow run. Returns the MLflow run ID."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    df = load_training_data(data_path)

    with mlflow.start_run() as run:
        start = time.perf_counter()
        model = train_isolation_forest(df)
        training_duration_s = time.perf_counter() - start

        predictions = model.predict(df[FEATURE_COLUMNS])
        detected_anomaly_rate = float((predictions == -1).mean())

        mlflow.log_params(
            {
                "contamination": config.CONTAMINATION,
                "n_estimators": config.N_ESTIMATORS,
                "random_state": config.RANDOM_SEED,
                "n_training_rows": len(df),
                "feature_columns": ",".join(FEATURE_COLUMNS),
            }
        )
        mlflow.log_metrics(
            {
                "training_duration_s": training_duration_s,
                "detected_anomaly_rate": detected_anomaly_rate,
                **evaluate_against_ground_truth(df, predictions),
            }
        )
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=config.MLFLOW_REGISTERED_MODEL_NAME,
            input_example=df[FEATURE_COLUMNS].head(5),
        )

        logger.info(
            "Run %s: trained on %d rows in %.2fs, detected anomaly rate %.4f",
            run.info.run_id,
            len(df),
            training_duration_s,
            detected_anomaly_rate,
        )
        return run.info.run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_training()
