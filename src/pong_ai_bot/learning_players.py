"""The learning players `pong train` trains: each with the environment it plays,
wrapped as it sees it, and its network (#17), and the Evaluation of a learning
player's Milestones.

So far they are the test environments' tiny players, so that an agent can run tiny
Runs in a temporary directory. The Position learner and the Agent come later.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from pong_ai_bot.evaluation import EnvMaker, Evaluation, SeedSet, evaluate
from pong_ai_bot.test_environments import (
    ConstantRewardEnv,
    RewardByActionEnv,
    RewardIsInputEnv,
    TwoStepEnv,
)

if TYPE_CHECKING:
    from torch import nn

    from pong_ai_bot.run_directory import SavedFile
    from pong_ai_bot.training import MilestoneEvaluation


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
    evaluated: bool
    """Whether its Runs evaluate their Milestones. An Evaluation counts each Game
    by the score in its environment's `info`, which only Pong's has: a test
    environment's Run has no Evaluation (#17)."""


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
        make_env=make_env,
        make_network=make_network,
        actions=actions,
        stack_length=1,
        evaluated=False,
    )


LEARNING_PLAYERS: dict[str, LearningPlayer] = {
    "test-env-1": _test_environment_player(ConstantRewardEnv, actions=1),
    "test-env-2": _test_environment_player(RewardIsInputEnv, actions=1),
    "test-env-3": _test_environment_player(TwoStepEnv, actions=1),
    "test-env-4": _test_environment_player(RewardByActionEnv, actions=2),
}
"""The learning players, by the name `pong train --player` takes."""


def milestone_evaluation(network: nn.Module, make_env: EnvMaker) -> MilestoneEvaluation:
    """The Evaluation of a learning player's Milestones, as its Run plays it: each
    Milestone's weights loaded into `network`, played by the greedy player, with
    one pass through the network a Step for all the Games still going, on the set
    the Run asks for, in environments from `make_env` (#13). Only the end of a
    Run asks for the Confirmation's set, once (issue #16).

    `network` is the Evaluation's own, of the player's kind, on the device the
    Games are played on: each Milestone loaded changes its weights.
    """
    # Only here: PyTorch takes seconds to import, and `pong`'s other commands
    # don't need it.
    import torch

    from pong_ai_bot.learner import GreedyPlayer

    device = next(network.parameters()).device

    def evaluate_milestone(
        milestone: SavedFile, *, seed_set: SeedSet, stop: Callable[[], bool]
    ) -> Evaluation:
        network.load_state_dict(torch.load(milestone.path, map_location=device))
        return evaluate(GreedyPlayer(network), make_env, seed_set.seeds, stop=stop)

    return evaluate_milestone
