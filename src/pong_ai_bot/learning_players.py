"""The learning players `pong train` trains: each with the environment it plays,
wrapped as it sees it, and its network (#17), and the Evaluation of a learning
player's Milestones.

They are the Position learner, and the test environments' tiny players, with which
an agent can run tiny Runs in a temporary directory. The Agent comes later.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from pong_ai_bot.environment import PongEnv
from pong_ai_bot.evaluation import EnvMaker, Evaluation, SeedSet, evaluate
from pong_ai_bot.positions import Positions
from pong_ai_bot.stack import STACK_LENGTH, ObservationStack
from pong_ai_bot.test_environments import (
    ConstantRewardEnv,
    RewardByActionEnv,
    RewardIsInputEnv,
    TwoStepEnv,
)

if TYPE_CHECKING:
    from torch import nn

    from pong_ai_bot.learner import GreedyPlayer
    from pong_ai_bot.run_directory import RunDirectory, SavedFile
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


def _position_learner() -> LearningPlayer:
    """The Position learner: it plays Pong in the Agent's place, with the Agent's
    Actions, Steps and stack of four, but decides from the positions of the ball
    and the Paddles instead of from the Observation (issue #9).

    Its network is small (#17): the stack's 16 numbers, in a row, two hidden layers
    of 256 with ReLU, and a linear output of a Q for each Action. The positions are
    in Court units already, so they go in as they are.
    """

    def make_env() -> gym.Env[Any, Any]:
        return ObservationStack(Positions(PongEnv()), STACK_LENGTH)

    def make_network() -> nn.Module:
        # Only here: PyTorch takes seconds to import, and `pong`'s other commands
        # don't need it.
        from torch import nn

        # Each stack of four Steps' positions, four numbers each, in a row of 16.
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    return LearningPlayer(
        make_env=make_env,
        make_network=make_network,
        actions=3,
        stack_length=STACK_LENGTH,
        evaluated=True,
    )


LEARNING_PLAYERS: dict[str, LearningPlayer] = {
    "position-learner": _position_learner(),
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

    def evaluate_in_the_run(
        milestone: SavedFile, *, seed_set: SeedSet, stop: Callable[[], bool]
    ) -> Evaluation:
        player = _greedy_player(network, milestone)
        return evaluate(player, make_env, seed_set.seeds, stop=stop)

    return evaluate_in_the_run


def evaluate_milestone(run: RunDirectory, milestone: SavedFile) -> Evaluation:
    """The Evaluation of `milestone`, one of the Run `run`'s Milestones, after the
    Run, as `pong evaluate` plays it: its greedy player, on the CPU, in the
    environment of the Run's player, on the regular set.

    Never on the Confirmation's set, which only the end of a Run plays, once. It
    records nothing in the Run either, so it never changes the Milestone the Run
    chooses (issue #16).
    """
    make_env = LEARNING_PLAYERS[run.player].make_env
    return evaluate(milestone_player(run, milestone), make_env, SeedSet.REGULAR.seeds)


def milestone_player(run: RunDirectory, milestone: SavedFile) -> GreedyPlayer:
    """The greedy player of `milestone`, one of the Run `run`'s Milestones: a new
    network of the kind of the Run's player, as its settings file records it, on
    the CPU, with the weights the Milestone holds."""
    network = LEARNING_PLAYERS[run.player].make_network()
    return _greedy_player(network, milestone)


def _greedy_player(network: nn.Module, milestone: SavedFile) -> GreedyPlayer:
    """The greedy player of `network`, with the weights `milestone` holds loaded
    into it, on the device it is on."""
    # Only here: PyTorch takes seconds to import, and `pong`'s other commands
    # don't need it.
    import torch

    from pong_ai_bot.learner import GreedyPlayer

    device = next(network.parameters()).device
    network.load_state_dict(torch.load(milestone.path, map_location=device))
    return GreedyPlayer(network)
