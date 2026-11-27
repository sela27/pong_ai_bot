"""The stack of a Game's last Observations, which the learning players decide from.

The environment returns one Observation at a time, and the stack is built around it,
in a wrapper (issue #11). The rule is set in issue #6: the stack holds the Game's
last Observations, oldest first, and at the start of a Game copies of its first
Observation fill it. A Point doesn't empty it.

The rule is `stack_indices`, in one place: the wrapper builds every stack by it,
and the replay buffer, which keeps each Observation once, rebuilds the same stacks
by it (issue #12).
"""

from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.vector.utils import batch_space

STACK_LENGTH = 4
"""How many of a Game's last Observations the Agent decides from."""


def stack_indices(steps: int, length: int = STACK_LENGTH) -> list[int]:
    """The stacking rule: which of a Game's Observations make up its stack after
    `steps` Steps, oldest first.

    Each Observation is numbered by how many Steps of the Game came before it, so
    the one `reset` returns is 0. The stack holds the Game's last `length`
    Observations, and while the Game has fewer, copies of its first fill the front:
    with 4, after 2 Steps it is [0, 0, 1, 2], and after 5 it is [2, 3, 4, 5]. No
    number is below 0, so a stack never reaches back into the Game before.
    """
    return [max(steps - length + 1 + place, 0) for place in range(length)]


class ObservationStack(gym.Wrapper[npt.NDArray[Any], Any, npt.NDArray[Any], Any]):
    """An environment that gives the stack of its Game's last Observations, oldest
    first along a new first axis, instead of the last one alone."""

    _last: deque[npt.NDArray[Any]]
    """The Game's last Observations, as many as a stack holds."""
    _steps: int
    """How many Steps the Game has played."""

    def __init__(
        self, env: gym.Env[npt.NDArray[Any], Any], length: int = STACK_LENGTH
    ) -> None:
        """Wrap `env`, with `length` Observations in each stack: the Agent's 4 by
        default, and fewer for the test environments."""
        super().__init__(env)
        self._length = length
        self.observation_space = batch_space(env.observation_space, length)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._last = deque([observation], maxlen=self._length)
        self._steps = 0
        return self._stack(), info

    def step(
        self, action: Any
    ) -> tuple[npt.NDArray[Any], SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._last.append(observation)
        self._steps += 1
        return self._stack(), reward, terminated, truncated, info

    def _stack(self) -> npt.NDArray[Any]:
        """The stack after the Game's Steps so far, by the stacking rule."""
        # The number, by the stacking rule's count, of the oldest Observation kept.
        oldest = self._steps + 1 - len(self._last)
        indices = stack_indices(self._steps, self._length)
        return np.stack([self._last[index - oldest] for index in indices])
