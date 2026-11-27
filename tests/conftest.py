"""Fixtures for every test."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

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
