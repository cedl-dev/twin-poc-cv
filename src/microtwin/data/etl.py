"""ETL pipeline: clean the raw sensor DataFrame, downcast dtypes, and persist to Parquet.

Three single-responsibility steps (`clean_sensor_data`, `downcast_dtypes`,
`save_to_parquet`), composed by `run_etl`. Each one takes and returns a
DataFrame (or, for the last one, writes it out) so they can be tested and
reasoned about independently of the others.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from microtwin import config

logger = logging.getLogger(__name__)


def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with an unparseable timestamp or a non-finite sensor reading.

    A real control-system archiver can emit a null timestamp or a NaN/inf
    reading on a transient read failure; this mirrors that reality instead
    of assuming the raw feed is always clean. Returns a new DataFrame with a
    fresh 0-based index; the input is not modified in place.
    """
    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")

    numeric_columns = ["temperature_c", "beam_intensity_ua"]
    finite_mask = clean[numeric_columns].apply(np.isfinite).all(axis=1)
    valid_mask = clean["timestamp"].notna() & finite_mask

    dropped = len(clean) - int(valid_mask.sum())
    if dropped:
        logger.warning("Dropping %d invalid row(s) out of %d", dropped, len(clean))

    return clean.loc[valid_mask].reset_index(drop=True)


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 sensor columns to float32.

    pandas defaults every numeric column to float64; that precision is far
    beyond what a physical temperature/beam-intensity sensor reports, so
    float32 is an effectively lossless way to roughly halve memory use.
    Logs a before/after `memory_usage()` comparison so the gain is visible
    rather than merely asserted.
    """
    before_bytes = int(df.memory_usage(deep=True).sum())

    downcast = df.copy()
    float_columns = downcast.select_dtypes(include="float64").columns
    downcast[float_columns] = downcast[float_columns].astype("float32")

    after_bytes = int(downcast.memory_usage(deep=True).sum())
    logger.info(
        "Downcast memory usage: %.2f MB -> %.2f MB (-%.1f%%)",
        before_bytes / 1e6,
        after_bytes / 1e6,
        100 * (1 - after_bytes / before_bytes) if before_bytes else 0.0,
    )
    return downcast


def save_to_parquet(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist `df` to a Parquet file, creating parent directories as needed.

    Defaults to `config.DATA_PROCESSED_DIR / config.PROCESSED_DATA_FILENAME`
    when `path` is not given.
    """
    default_path = config.DATA_PROCESSED_DIR / config.PROCESSED_DATA_FILENAME
    target = path if path is not None else default_path
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, engine="pyarrow", index=False)
    return target


def run_etl(raw_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Run the full pipeline (clean -> downcast -> persist) and return the output path."""
    cleaned = clean_sensor_data(raw_df)
    downcast = downcast_dtypes(cleaned)
    return save_to_parquet(downcast, output_path)
