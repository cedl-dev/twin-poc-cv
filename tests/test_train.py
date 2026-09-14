"""Unit tests for the Isolation Forest training functions.

Deliberately does not exercise `run_training` (the full MLflow-logging
path) here: MLflow run/registry side effects are exercised manually via the
`microtwin-train` CLI against the real tracking store, not in the fast unit
test suite, to keep tests hermetic and independent of any tracking backend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microtwin.data.etl import save_to_parquet
from microtwin.data.generate import generate_sensor_data
from microtwin.training.train import (
    evaluate_against_ground_truth,
    load_training_data,
    train_isolation_forest,
)


def test_train_isolation_forest_fits_and_flags_approximately_contamination_share() -> None:
    df = generate_sensor_data(n_rows=5_000, anomaly_rate=0.05, seed=0)

    model = train_isolation_forest(df, contamination=0.05, n_estimators=100, random_state=0)
    predictions = model.predict(df[["temperature_c", "beam_intensity_ua"]])

    assert set(np.unique(predictions)).issubset({-1, 1})
    detected_rate = (predictions == -1).mean()
    assert detected_rate == pytest.approx(0.05, abs=0.02)


def test_evaluate_against_ground_truth_recovers_injected_anomalies() -> None:
    df = generate_sensor_data(n_rows=5_000, anomaly_rate=0.05, seed=1)
    model = train_isolation_forest(df, contamination=0.05, n_estimators=100, random_state=1)
    predictions = model.predict(df[["temperature_c", "beam_intensity_ua"]])

    metrics = evaluate_against_ground_truth(df, predictions)

    assert set(metrics) == {"ground_truth_precision", "ground_truth_recall", "ground_truth_f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
    # Injected anomalies are extreme outliers (8-30 std devs away), so a
    # well-fit model should recover most of them.
    assert metrics["ground_truth_recall"] > 0.7


def test_evaluate_against_ground_truth_returns_empty_dict_without_label() -> None:
    df = pd.DataFrame({"temperature_c": [1.0, 2.0], "beam_intensity_ua": [3.0, 4.0]})
    predictions = np.array([1, -1])

    assert evaluate_against_ground_truth(df, predictions) == {}


def test_load_training_data_reads_a_parquet_file(tmp_path) -> None:
    df = generate_sensor_data(n_rows=200, seed=2)
    path = save_to_parquet(df, tmp_path / "data.parquet")

    loaded = load_training_data(path)

    pd.testing.assert_frame_equal(loaded, df)
