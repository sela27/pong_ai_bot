"""The test environments: tiny Gymnasium environments, in each of which what must
be learned is known in advance (issue #9).

They run through the real training loop, with tiny settings of their own, to show
that the loop learns before it trains on Pong. Each was written before the part of
the loop it checks. Every Game is one Step long, but in test environment 3, and the
input is one number.
"""

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

Input = npt.NDArray[np.float32]
"""A test environment's input: one number, from -1 to +1."""


def _input(value: float) -> Input:
    return np.array([value], np.float32)


class ConstantRewardEnv(gym.Env[Input, np.int64]):
    """Test environment 1: a constant input, one Action, a Reward of +1, and the
    Game ends. The value learned is 1."""

    INPUT = 0.0

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = spaces.Discrete(1)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Input, dict[str, Any]]:
        super().reset(seed=seed)
        return _input(self.INPUT), {}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[Input, SupportsFloat, bool, bool, dict[str, Any]]:
        return _input(self.INPUT), 1.0, True, False, {}


class RewardIsInputEnv(gym.Env[Input, np.int64]):
    """Test environment 2: an input of -1 or +1, at random, one Action, and a
    Reward equal to the input. The value learned is the input, so the network must
    use it."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = spaces.Discrete(1)
        self._value = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Input, dict[str, Any]]:
        super().reset(seed=seed)
        self._value = float(self.np_random.choice([-1.0, 1.0]))
        return _input(self._value), {}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[Input, SupportsFloat, bool, bool, dict[str, Any]]:
        return _input(self._value), self._value, True, False, {}


class TwoStepEnv(gym.Env[Input, np.int64]):
    """Test environment 3: two Steps, one Action, and a Reward only at the end: 0
    after the first Step, and +1 after the second, when the Game ends. The value
    learned at the second Step is 1, and at the first 0.99, the discount, since the
    Reward comes a Step later.

    Each Step has an input of its own, so that the network can tell them apart.
    """

    FIRST_INPUT = -1.0
    SECOND_INPUT = 1.0

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = spaces.Discrete(1)
        self._at_first = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Input, dict[str, Any]]:
        super().reset(seed=seed)
        self._at_first = True
        return _input(self.FIRST_INPUT), {}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[Input, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._at_first:
            self._at_first = False
            return _input(self.SECOND_INPUT), 0.0, False, False, {}
        return _input(self.SECOND_INPUT), 1.0, True, False, {}
