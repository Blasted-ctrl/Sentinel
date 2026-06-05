"""Tests for the Typer CLI (pipeline + DB patched out)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from sentinel import cli
from sentinel.ingest.pipeline import IngestSummary

runner = CliRunner()


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_engine", lambda: object())

    @contextmanager
    def fake_scope(engine: Any) -> Iterator[Any]:
        yield MagicMock()

    monkeypatch.setattr(cli, "session_scope", fake_scope)


def test_ingest_invokes_pipeline(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    fake_pipe = MagicMock()
    fake_pipe.run.return_value = IngestSummary(
        "sierra", 1, weather_readings=2, fire_events=3, satellite_tiles=1
    )
    monkeypatch.setattr(
        cli, "build_pipeline", lambda session, settings=None, sources=None: fake_pipe
    )

    result = runner.invoke(
        cli.app,
        [
            "ingest",
            "--region",
            "-120.5,38.5,-120.0,39.0",
            "--start",
            "2024-08-01",
            "--end",
            "2024-08-02",
            "--sources",
            "weather",
            "--name",
            "sierra",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"weather_readings": 2' in result.stdout
    fake_pipe.run.assert_called_once()


def test_ingest_rejects_bad_bbox(fake_engine: None) -> None:
    result = runner.invoke(
        cli.app,
        ["ingest", "--region", "not-a-bbox", "--start", "2024-08-01", "--end", "2024-08-02"],
    )
    assert result.exit_code != 0


def test_ingest_rejects_reversed_dates(fake_engine: None) -> None:
    result = runner.invoke(
        cli.app,
        [
            "ingest",
            "--region",
            "-120.5,38.5,-120.0,39.0",
            "--start",
            "2024-08-05",
            "--end",
            "2024-08-01",
        ],
    )
    assert result.exit_code != 0


def test_init_db_command(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {}
    monkeypatch.setattr(cli, "get_engine", lambda: object())
    monkeypatch.setattr(cli, "init_db", lambda engine: called.setdefault("ran", True))

    result = runner.invoke(cli.app, ["init-db"])
    assert result.exit_code == 0, result.output
    assert called.get("ran") is True
