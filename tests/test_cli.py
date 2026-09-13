"""Unit tests for the CLI entry points."""

from __future__ import annotations

import pandas as pd

from microtwin.cli import build_dataset


def test_build_dataset_writes_a_parquet_file_with_the_requested_row_count(tmp_path) -> None:
    output_path = tmp_path / "dataset.parquet"

    result_path = build_dataset(
        [
            "--n-rows",
            "1000",
            "--anomaly-rate",
            "0.05",
            "--seed",
            "1",
            "--output-path",
            str(output_path),
        ]
    )

    assert result_path == output_path
    df = pd.read_parquet(output_path)
    assert len(df) == 1000


def test_build_dataset_uses_config_defaults_when_no_args_given(tmp_path) -> None:
    output_path = tmp_path / "default_dataset.parquet"

    build_dataset(["--n-rows", "500", "--output-path", str(output_path)])

    assert output_path.exists()
