"""Unit tests for the CLI entry points."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from microtwin.cli import build_dataset, run_build_dataset_cli


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


def test_console_entrypoint_returns_none_so_the_process_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Regression test: a console-script's target is called as `sys.exit(fn())`.

    `build_dataset()` returns a `Path` (useful when called directly, e.g. in
    the tests above) — if that were the function registered in
    `pyproject.toml`'s `[project.scripts]`, `sys.exit(Path(...))` would print
    the path and exit with status 1 on every successful run, breaking e.g.
    `microtwin-build-dataset && microtwin-train` shell chaining. This is
    exactly what happened when building the Docker image (Step 5).
    """
    output_path = tmp_path / "cli_entrypoint_dataset.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        ["microtwin-build-dataset", "--n-rows", "100", "--output-path", str(output_path)],
    )

    result = run_build_dataset_cli()

    assert result is None
    assert output_path.exists()
