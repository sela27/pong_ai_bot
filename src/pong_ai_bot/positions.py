"""The positions of the ball and the Paddles, which the Position learner decides from.

The Position learner plays in the Agent's place, with everything the same but its
input (issue #9). Instead of the Observation, a wrapper gives it four numbers each
Step, read from the physics state in `info`: the ball's x and y, and the height of
each Paddle, in Court units and without velocities. The stack of the last four
Steps makes them 16. So the training loop learns from what the environment returns,
and never reads `info` itself.
"""

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from pong_ai_bot.game import BALL_SIZE


class Positions(gym.Wrapper[npt.NDArray[np.float32], Any, npt.NDArray[Any], Any]):
    """An environment that gives the positions of the ball and the Paddles, from
    its `info`, instead of the Observation.

    Its `info` is the environment's own, so an Evaluation still reads the score
    from it.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        # Everything stays within the Court but the ball's x: the ball's center is
        # up to half its size past an end before the whole ball is out, and a
        # Point won.
        edge = BALL_SIZE / 2
        self.observation_space = spaces.Box(
            np.array([-edge, 0.0, 0.0, 0.0], np.float32),
            np.array([1.0 + edge, 1.0, 1.0, 1.0], np.float32),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        return _positions(info), info

    def step(
        self, action: Any
    ) -> tuple[npt.NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self.env.step(action)
        return _positions(info), reward, terminated, truncated, info


def _positions(info: dict[str, Any]) -> npt.NDArray[np.float32]:
    """The ball's x and y, then the left Paddle's height, then the right's."""
    ball = info["ball"]
    return np.array(
        [ball.x, ball.y, info["left_paddle"], info["right_paddle"]], np.float32
    )
