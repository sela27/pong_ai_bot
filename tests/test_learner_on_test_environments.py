"""The learner on the test environments, through the real training loop (issue #9).

Each slow test trains a test environment's player, as `pong train --player`
makes it: its tiny network, from scratch, on its environment, wrapped as it sees it,
on the CPU, with fixed seeds. It checks the values the network learned against the
values known in advance.

They are slow for the training, and for the learner's optimizer: making one loads
a part of PyTorch that takes about 5 seconds.
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pong_ai_bot.learning_players import LEARNING_PLAYERS
from pong_ai_bot.test_environments import (
    ConstantRewardEnv,
    RewardByActionEnv,
    RewardIsInputEnv,
    TwoStepEnv,
)
from pong_ai_bot.training import EpsilonSchedule, train

if TYPE_CHECKING:
    from torch import nn

    from pong_ai_bot.learner import Learner

SEED = 0
"""The seed of the network's first weights, of the learner and of the
environment."""

TOLERANCE = 0.01
"""How far from the value known in advance a learned value may be."""

DISCOUNT_TOLERANCE = 0.001
"""How far from the value known in advance a learned value may be, where the
discount is what is checked: TOLERANCE would let 1, the value without the discount
of 0.99, pass for 0.99."""


def trained(name: str) -> tuple[Learner, nn.Module]:
    """The network of the test environment's player `name`, trained on its
    environment through the real loop, with its learner."""
    import torch

    from pong_ai_bot.learner import Learner, LearnerSettings
    from pong_ai_bot.run_settings import RunSettings

    player = LEARNING_PLAYERS[name]
    torch.manual_seed(SEED)
    network = player.make_network()
    settings = LearnerSettings.of_run(
        RunSettings(
            batch_size=8,
            learning_rate=1e-3,
            update_every=1,
            replay_buffer_size=100,
            fill_steps=8,
            target_copy_every=10,
            discount=0.99,
        )
    )
    learner = Learner(
        network,
        actions=player.actions,
        stack_length=player.stack_length,
        settings=settings,
        seed=SEED,
    )
    epsilon = EpsilonSchedule(start=1.0, end=0.01, decay_steps=100)

    train(player.make_env(), learner, steps=1_000, epsilon=epsilon, seed=SEED)

    return learner, network


def q_values(network: nn.Module, inputs: list[float]) -> list[list[float]]:
    """`network`'s Q of each Action, for each of `inputs`, each a stack of one
    Observation."""
    import torch

    with torch.no_grad():
        values = network(torch.tensor([[[value]] for value in inputs]))
    return values.tolist()


def learned_values(name: str, inputs: list[float]) -> list[float]:
    """The values the network of the test environment's player `name` learns,
    through the real loop, for each of `inputs`: the Q of its only Action."""
    _, network = trained(name)
    return [value for [value] in q_values(network, inputs)]


@pytest.mark.parametrize(
    "make_env", [ConstantRewardEnv, RewardIsInputEnv, TwoStepEnv, RewardByActionEnv]
)
def test_a_test_environment_passes_gymnasiums_check_env_with_no_warnings(
    make_env: Callable[[], gym.Env[Any, Any]],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # A test environment draws nothing, so there is no render to check.
        check_env(make_env(), skip_render_check=True)


@pytest.mark.parametrize(
    ("name", "actions"),
    [("test-env-1", 1), ("test-env-2", 1), ("test-env-3", 1), ("test-env-4", 2)],
)
def test_a_test_environment_s_player_decides_from_a_stack_of_one_observation(
    name: str, actions: int
) -> None:
    # Torch only here, for its seconds of import; no optimizer, which takes more.
    import torch

    player = LEARNING_PLAYERS[name]

    stack, _ = player.make_env().reset(seed=SEED)
    q = player.make_network()(torch.as_tensor(np.stack([stack, stack])))

    # As the Position learner's stack of four, by the same wrapper: one
    # Observation of one number, along the stack's first axis.
    assert (player.stack_length, stack.shape, stack.dtype) == (1, (1, 1), np.float32)
    assert (player.actions, q.shape) == (actions, (2, actions))


@pytest.mark.slow
def test_the_value_learned_in_test_environment_1_is_1() -> None:
    [value] = learned_values("test-env-1", [ConstantRewardEnv.INPUT])

    assert value == pytest.approx(1.0, abs=TOLERANCE), (
        f"the value learned is {value:+.4f}"
    )


@pytest.mark.slow
def test_the_value_learned_in_test_environment_2_is_the_input() -> None:
    values = learned_values("test-env-2", [-1.0, 1.0])

    assert values == pytest.approx([-1.0, 1.0], abs=TOLERANCE), (
        f"the values learned for -1 and +1 are {values[0]:+.4f} and {values[1]:+.4f}"
    )


@pytest.mark.slow
def test_the_value_learned_in_test_environment_3_is_the_discount_at_the_first_step() -> (
    None
):
    values = learned_values(
        "test-env-3", [TwoStepEnv.FIRST_INPUT, TwoStepEnv.SECOND_INPUT]
    )

    assert values == pytest.approx([0.99, 1.0], abs=DISCOUNT_TOLERANCE), (
        f"the values learned at the first and second Steps are {values[0]:+.4f} "
        f"and {values[1]:+.4f}"
    )


@pytest.mark.slow
def test_the_loop_chooses_the_action_rewarded_1_in_test_environment_4() -> None:
    learner, network = trained("test-env-4")

    # A stack of one Observation.
    observation = np.array([[RewardByActionEnv.INPUT]], np.float32)
    action = learner.choose_action(observation, epsilon=0.0)

    assert action == RewardByActionEnv.RIGHT_ACTION, (
        f"the Action chosen is {action}, and the values learned for Actions 0 and 1"
        f" are {q_values(network, [RewardByActionEnv.INPUT])[0]}"
    )
