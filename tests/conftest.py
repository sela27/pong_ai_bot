"""Fixtures for every test."""

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _runs_in_a_temporary_directory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """No test ever writes to the real Runs' directory, ~/pong_ai_bot_runs: every
    Run a test creates goes to a temporary one, unless the test sets its own."""
    runs = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PONG_AI_BOT_RUNS_DIR", str(runs))
        yield runs
