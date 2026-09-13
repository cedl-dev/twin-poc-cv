"""Vectorized synthetic sensor data generator.

Simulates two GANIL accelerator telemetry channels — temperature and beam
intensity — sampled at 1 Hz, with a configurable fraction of injected
anomalies (correlated spikes/drops on both channels). Every array below is
built with NumPy vectorized operations: there is no Python-level loop over
rows, which is what makes generating millions of samples take milliseconds
instead of minutes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microtwin import config


def generate_sensor_data(
    n_rows: int = config.N_ROWS,
    anomaly_rate: float = config.ANOMALY_RATE,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic, timestamped multivariate sensor log.

    Args:
        n_rows: number of 1 Hz samples to generate.
        anomaly_rate: fraction of rows (in [0, 1)) where both channels
            receive a correlated spike/drop, simulating a fault event.
        seed: seed for the random generator, for reproducible datasets.

    Returns:
        A DataFrame with columns: timestamp, temperature_c,
        beam_intensity_ua, is_anomaly. See `config.SENSOR_PV_NAMES` for the
        column-to-PV mapping.
    """
    if not 0.0 <= anomaly_rate < 1.0:
        raise ValueError(f"anomaly_rate must be in [0, 1), got {anomaly_rate}")

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start="2026-01-01", periods=n_rows, freq="s")

    temperature = rng.normal(config.TEMPERATURE_MEAN_C, config.TEMPERATURE_STD_C, size=n_rows)
    beam_intensity = rng.normal(
        config.BEAM_INTENSITY_MEAN_UA, config.BEAM_INTENSITY_STD_UA, size=n_rows
    )

    anomaly_mask = rng.random(n_rows) < anomaly_rate
    temperature_shift = rng.choice([-1.0, 1.0], size=n_rows) * rng.uniform(8.0, 15.0, size=n_rows)
    beam_shift = rng.choice([-1.0, 1.0], size=n_rows) * rng.uniform(15.0, 30.0, size=n_rows)

    temperature = np.where(anomaly_mask, temperature + temperature_shift, temperature)
    beam_intensity = np.where(anomaly_mask, beam_intensity + beam_shift, beam_intensity)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": temperature,
            "beam_intensity_ua": beam_intensity,
            "is_anomaly": anomaly_mask,
        }
    )
