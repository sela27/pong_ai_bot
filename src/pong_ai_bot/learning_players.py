"""The learning players `pong train` trains: each with the environment it plays,
wrapped as it sees it, and its network (#17).

So far they are the test environments' tiny players, so that an agent can run tiny
Runs in a temporary directory. The Position learner and the Agent come later.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from pong_ai_bot.test_environments import (
    ConstantRewardEnv,
    RewardByActionEnv,
    RewardIsInputEnv,
    TwoStepEnv,
)

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True, slots=True, kw_only=True)
class LearningPlayer:
    """A player that learns in the Agent's place."""

    make_env: Callable[[], gym.Env[Any, Any]]
    """Makes the environment it plays, wrapped as it sees it."""
    make_network: Callable[[], nn.Module]
    """Makes its network, with first weights from PyTorch's random generator."""
    actions: int
    """How many Actions it chooses from."""
    stack_length: int
    """How many Observations each of its inputs stacks."""


def _test_environment_player(
    make_env: Callable[[], gym.Env[Any, Any]], *, actions: int
) -> LearningPlayer:
    """The player of a test environment: its input is one number, a stack of one
    Observation, and its network is tiny."""

    def make_network() -> nn.Module:
        # Only here: PyTorch takes seconds to import, and `pong`'s other commands
        # don't need it.
        from torch import nn

        return nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, actions))

    return LearningPlayer(
        make_env=make_env, make_network=make_network, actions=actions, stack_length=1
    )


LEARNING_PLAYERS: dict[str, LearningPlayer] = {
    "test-env-1": _test_environment_player(ConstantRewardEnv, actions=1),
    "test-env-2": _test_environment_player(RewardIsInputEnv, actions=1),
    "test-env-3": _test_environment_player(TwoStepEnv, actions=1),
    "test-env-4": _test_environment_player(RewardByActionEnv, actions=2),
}
"""The learning players, by the name `pong train --player` takes."""
