"""The learner on the test environments, through the real training loop (issue #9).

Each slow test trains a tiny network from scratch, on the CPU, with fixed seeds,
and checks the values it learned against the values known in advance.

PyTorch is imported inside them, not at the top: importing it, and the part of it
the learner's optimizer loads, takes about 9 seconds, and pytest imports every test
module, slow or not, before it leaves the slow tests out, so the fast tests would
pay for it too.
"""

import warnings
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

from pong_ai_bot.test_environments import ConstantRewardEnv, RewardIsInputEnv
from pong_ai_bot.training import EpsilonSchedule, train

SEED = 0
"""The seed of the network's first weights, of the learner and of the
environment."""

TOLERANCE = 0.01
"""How far from the value known in advance a learned value may be."""


def learned_values(env: gym.Env[Any, Any], inputs: list[float]) -> list[float]:
    """The values a tiny network learns on `env`, through the real loop, for each
    of `inputs`: the Q of its only Action."""
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner, LearnerSettings

    torch.manual_seed(SEED)
    network = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    settings = LearnerSettings(
        batch_size=8, learning_rate=1e-3, update_every=1, replay_size=100, fill=8
    )
    learner = Learner(network, actions=1, settings=settings, seed=SEED)
    epsilon = EpsilonSchedule(start=1.0, end=0.01, decay_steps=100)

    train(env, learner, steps=1_000, epsilon=epsilon, seed=SEED)

    with torch.no_grad():
        values = network(torch.tensor([[value] for value in inputs]))
    return [float(value) for value in values[:, 0]]


@pytest.mark.parametrize("make_env", [ConstantRewardEnv, RewardIsInputEnv])
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
