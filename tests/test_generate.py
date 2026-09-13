"""Unit tests for the vectorized sensor data generator."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from microtwin.data.generate import generate_sensor_data


def test_returns_expected_shape_and_dtypes() -> None:
    df = generate_sensor_data(n_rows=1_000, anomaly_rate=0.05, seed=0)

    assert len(df) == 1_000
    assert list(df.columns) == ["timestamp", "temperature_c", "beam_intensity_ua", "is_anomaly"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert pd.api.types.is_float_dtype(df["temperature_c"])
    assert pd.api.types.is_float_dtype(df["beam_intensity_ua"])
    assert pd.api.types.is_bool_dtype(df["is_anomaly"])


def test_anomaly_rate_is_approximately_respected() -> None:
    n_rows = 200_000
    anomaly_rate = 0.02
    df = generate_sensor_data(n_rows=n_rows, anomaly_rate=anomaly_rate, seed=1)

    observed_rate = df["is_anomaly"].mean()
    assert observed_rate == pytest.approx(anomaly_rate, abs=0.005)


def test_generation_is_reproducible_given_a_seed() -> None:
    df1 = generate_sensor_data(n_rows=500, seed=42)
    df2 = generate_sensor_data(n_rows=500, seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_generates_millions_of_rows_in_well_under_a_second() -> None:
    start = time.perf_counter()
    df = generate_sensor_data(n_rows=5_000_000, seed=7)
    elapsed = time.perf_counter() - start

    assert len(df) == 5_000_000
    assert elapsed < 1.0


def test_invalid_anomaly_rate_raises() -> None:
    with pytest.raises(ValueError):
        generate_sensor_data(n_rows=10, anomaly_rate=1.5)
