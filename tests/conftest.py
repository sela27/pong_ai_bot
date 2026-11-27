"""Fixtures for every test."""

import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

from pong_ai_bot.run_directory import RUNS_DIR_VARIABLE

_test_numbers = itertools.count()
"""A number for each test, to name its Runs' directory by."""


@pytest.fixture(scope="session")
def _temporary_runs_directories(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The temporary directory every test's Runs' directory is in."""
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(autouse=True)
def runs(_temporary_runs_directories: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The test's own Runs' directory, a temporary one, so no test ever writes to
    the real one, ~/pong_ai_bot_runs. It is empty when the test starts: it doesn't
    exist until the test creates its first Run."""
    path = _temporary_runs_directories / f"test{next(_test_numbers)}"
    monkeypatch.setenv(RUNS_DIR_VARIABLE, str(path))
    return path


@pytest.fixture(autouse=True, scope="session")
def _no_game_on_the_confirmation_s_set() -> Iterator[None]:
    """No test ever plays a Game on the Confirmation's set, whatever the player
    and environment: only the end of a real Run plays it, once (issue #16). An
    Evaluation the package's modules play on any of its seeds fails the test
    instead, before its first Game. A Run's Confirmation in a test is a fake."""
    from pong_ai_bot import evaluation, learning_players, scripted_players

    confirmation = set(evaluation.SeedSet.CONFIRMATION.seeds)
    real_evaluate = evaluation.evaluate

    def guarded(
        player: Any, make_env: Any, seeds: Iterable[int], **options: Any
    ) -> Any:
        seeds = list(seeds)
        played = sorted(confirmation.intersection(seeds))
        assert not played, f"a test played the Confirmation's seeds {played}"
        return real_evaluate(player, make_env, seeds, **options)

    with pytest.MonkeyPatch.context() as patch:
        for module in (evaluation, learning_players, scripted_players):
            patch.setattr(module, "evaluate", guarded)
        yield
