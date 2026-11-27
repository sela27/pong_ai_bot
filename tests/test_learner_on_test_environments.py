"""The learner on the test environments, through the real training loop (issue #9).

Each slow test trains a tiny network from scratch, on the CPU, with fixed seeds,
and checks the values it learned against the values known in advance.

They are slow for the training, and for the learner's optimizer: making one loads
a part of PyTorch that takes about 5 seconds.
"""

import warnings
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

from pong_ai_bot.test_environments import (
    ConstantRewardEnv,
    RewardIsInputEnv,
    TwoStepEnv,
)
from pong_ai_bot.training import EpsilonSchedule, train

SEED = 0
"""The seed of the network's first weights, of the learner and of the
environment."""

TOLERANCE = 0.01
"""How far from the value known in advance a learned value may be."""

DISCOUNT_TOLERANCE = 0.001
"""How far from the value known in advance a learned value may be, where the
discount is what is checked: TOLERANCE would let 1, the value without the discount
of 0.99, pass for 0.99."""


def learned_values(env: gym.Env[Any, Any], inputs: list[float]) -> list[float]:
    """The values a tiny network learns on `env`, through the real loop, for each
    of `inputs`: the Q of its only Action."""
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner, LearnerSettings

    torch.manual_seed(SEED)
    network = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    settings = LearnerSettings(
        batch_size=8,
        learning_rate=1e-3,
        update_every=1,
        replay_buffer_size=100,
        fill_steps=8,
        target_copy_every=10,
        discount=0.99,
        double_dqn=True,
    )
    learner = Learner(network, actions=1, settings=settings, seed=SEED)
    epsilon = EpsilonSchedule(start=1.0, end=0.01, decay_steps=100)

    train(env, learner, steps=1_000, epsilon=epsilon, seed=SEED)

    with torch.no_grad():
        values = network(torch.tensor([[value] for value in inputs]))
    return [float(value) for value in values[:, 0]]


@pytest.mark.parametrize("make_env", [ConstantRewardEnv, RewardIsInputEnv, TwoStepEnv])
def test_a_test_environment_passes_gymnasiums_check_env_with_no_warnings(
    make_env: Callable[[], gym.Env[Any, Any]],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # A test environment draws nothing, so there is no render to check.
        check_env(make_env(), skip_render_check=True)


@pytest.mark.slow
def test_the_value_learned_in_test_environment_1_is_1() -> None:
    [value] = learned_values(ConstantRewardEnv(), [ConstantRewardEnv.INPUT])

    assert value == pytest.approx(1.0, abs=TOLERANCE), (
        f"the value learned is {value:+.4f}"
    )


@pytest.mark.slow
def test_the_value_learned_in_test_environment_2_is_the_input() -> None:
    values = learned_values(RewardIsInputEnv(), [-1.0, 1.0])

    assert values == pytest.approx([-1.0, 1.0], abs=TOLERANCE), (
        f"the values learned for -1 and +1 are {values[0]:+.4f} and {values[1]:+.4f}"
    )


@pytest.mark.slow
def test_the_value_learned_in_test_environment_3_is_the_discount_at_the_first_step() -> (
    None
):
    values = learned_values(
        TwoStepEnv(), [TwoStepEnv.FIRST_INPUT, TwoStepEnv.SECOND_INPUT]
    )

    assert values == pytest.approx([0.99, 1.0], abs=DISCOUNT_TOLERANCE), (
        f"the values learned at the first and second Steps are {values[0]:+.4f} "
        f"and {values[1]:+.4f}"
    )
