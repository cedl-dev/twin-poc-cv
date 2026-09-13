"""Unit tests for the ETL pipeline: cleaning, downcasting, and Parquet persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microtwin.data.etl import clean_sensor_data, downcast_dtypes, run_etl, save_to_parquet
from microtwin.data.generate import generate_sensor_data


def _raw_dataframe_with_invalid_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01"),
                pd.NaT,
                pd.Timestamp("2026-01-01 00:00:02"),
                pd.Timestamp("2026-01-01 00:00:03"),
            ],
            "temperature_c": [18.0, 18.1, np.nan, 18.3],
            "beam_intensity_ua": [50.0, 50.1, 50.2, np.inf],
            "is_anomaly": [False, False, False, False],
        }
    )


def test_clean_sensor_data_drops_null_timestamp_and_non_finite_values() -> None:
    raw = _raw_dataframe_with_invalid_rows()

    cleaned = clean_sensor_data(raw)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "temperature_c"] == 18.0
    assert list(cleaned.index) == [0]


def test_downcast_dtypes_converts_float64_to_float32() -> None:
    df = generate_sensor_data(n_rows=1_000, seed=0)
    assert df["temperature_c"].dtype == np.float64

    downcast = downcast_dtypes(df)

    assert downcast["temperature_c"].dtype == np.float32
    assert downcast["beam_intensity_ua"].dtype == np.float32


def test_downcast_dtypes_reduces_memory_usage() -> None:
    df = generate_sensor_data(n_rows=100_000, seed=0)

    before_bytes = df.memory_usage(deep=True).sum()
    downcast = downcast_dtypes(df)
    after_bytes = downcast.memory_usage(deep=True).sum()

    assert after_bytes < before_bytes


def test_downcast_dtypes_leaves_non_float_columns_untouched() -> None:
    df = generate_sensor_data(n_rows=100, seed=0)

    downcast = downcast_dtypes(df)

    assert downcast["is_anomaly"].dtype == df["is_anomaly"].dtype
    assert downcast["timestamp"].dtype == df["timestamp"].dtype


def test_save_to_parquet_round_trip_preserves_dtypes(tmp_path) -> None:
    df = downcast_dtypes(generate_sensor_data(n_rows=500, seed=0))
    target = tmp_path / "sensor_data.parquet"

    written_path = save_to_parquet(df, target)
    read_back = pd.read_parquet(written_path)

    assert written_path == target
    pd.testing.assert_frame_equal(read_back, df)


def test_run_etl_end_to_end_produces_a_clean_downcast_parquet_file(tmp_path) -> None:
    raw = generate_sensor_data(n_rows=10_000, anomaly_rate=0.02, seed=0)
    target = tmp_path / "output.parquet"

    output_path = run_etl(raw, output_path=target)
    result = pd.read_parquet(output_path)

    assert output_path == target
    assert output_path.exists()
    assert result["temperature_c"].dtype == np.float32
    assert result["beam_intensity_ua"].dtype == np.float32
    assert len(result) == len(raw)


def test_downcast_on_empty_dataframe_does_not_divide_by_zero() -> None:
    empty = pd.DataFrame({"temperature_c": pd.Series(dtype="float64")})

    result = downcast_dtypes(empty)

    assert result.empty
    assert result["temperature_c"].dtype == np.float32
