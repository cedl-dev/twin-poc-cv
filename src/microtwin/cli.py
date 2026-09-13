"""Command-line entry points for the MicroTwin pipeline.

Each function here is a thin CLI wrapper around the library code in
`microtwin.data` / `microtwin.training` / etc., so the exact same logic can
be invoked as a single Slurm/PBS job step (no notebook state) or called
directly from Python (e.g. from tests) with an explicit `argv` list.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from microtwin import config
from microtwin.data.etl import run_etl
from microtwin.data.generate import generate_sensor_data

logger = logging.getLogger(__name__)


def _parse_build_dataset_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sensor data and run it through the ETL pipeline."
    )
    parser.add_argument("--n-rows", type=int, default=config.N_ROWS)
    parser.add_argument("--anomaly-rate", type=float, default=config.ANOMALY_RATE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override the default Parquet output path (data/processed/sensor_data.parquet).",
    )
    return parser.parse_args(argv)


def build_dataset(argv: list[str] | None = None) -> Path:
    """Generate synthetic sensor data and persist it through the ETL pipeline.

    Registered as the `microtwin-build-dataset` console script; `argv`
    defaults to `sys.argv[1:]` when called from the command line, and can be
    passed explicitly (e.g. from tests) otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_build_dataset_args(argv)

    raw_df = generate_sensor_data(
        n_rows=args.n_rows, anomaly_rate=args.anomaly_rate, seed=args.seed
    )
    output_path = run_etl(raw_df, output_path=args.output_path)

    logger.info("Dataset written to %s (%d rows)", output_path, len(raw_df))
    return output_path


if __name__ == "__main__":
    build_dataset()
