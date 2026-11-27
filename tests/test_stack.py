"""The stack of a Game's last Observations, built around the environment.

Most tests wrap a fake environment whose Observation is two numbers, the Game's
number and how many Steps it has played, so the expected stacks are written by hand.
The others wrap the real environment, whose Observation is an image.
"""

import warnings
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from pong_ai_bot.environment import Action, PongEnv
from pong_ai_bot.game import Ball, PhysicsState, Score
from pong_ai_bot.stack import ObservationStack


class CountingEnv(gym.Env[npt.NDArray[np.int64], int]):
    """A fake environment whose Observation is (Game, Step): the Game's number,
    counting from 0 at the first `reset`, and how many Steps the Game has played."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(0, 1_000, (2,), np.int64)
        self._game = -1

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
        super().reset(seed=seed)
        self._game += 1
        self._steps = 0
        return self._observation(), {}

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.int64], SupportsFloat, bool, bool, dict[str, Any]]:
        self._steps += 1
        return self._observation(), 0.0, False, False, {}

    def _observation(self) -> npt.NDArray[np.int64]:
        return np.array([self._game, self._steps], dtype=np.int64)


def test_reset_fills_the_stack_with_4_copies_of_the_games_first_observation() -> None:
    env = ObservationStack(CountingEnv())

    stack, _ = env.reset(seed=0)

    assert stack.tolist() == [[0, 0], [0, 0], [0, 0], [0, 0]]


def test_each_step_adds_its_observation_last_and_drops_the_oldest() -> None:
    env = ObservationStack(CountingEnv())
    env.reset(seed=0)

    stacks = [env.step(0)[0].tolist() for _ in range(6)]

    # Until the Game has 4 Observations, copies of its first fill the front.
    assert stacks == [
        [[0, 0], [0, 0], [0, 0], [0, 1]],
        [[0, 0], [0, 0], [0, 1], [0, 2]],
        [[0, 0], [0, 1], [0, 2], [0, 3]],
        [[0, 1], [0, 2], [0, 3], [0, 4]],
        [[0, 2], [0, 3], [0, 4], [0, 5]],
        [[0, 3], [0, 4], [0, 5], [0, 6]],
    ]


def test_a_new_game_starts_a_new_stack_with_none_of_the_last_games() -> None:
    env = ObservationStack(CountingEnv())
    env.reset(seed=0)
    for _ in range(3):
        env.step(0)

    first, _ = env.reset()
    second = env.step(0)[0]

    assert first.tolist() == [[1, 0], [1, 0], [1, 0], [1, 0]]
    assert second.tolist() == [[1, 0], [1, 0], [1, 0], [1, 1]]


def test_a_point_does_not_empty_the_stack() -> None:
    # The ball is leaving the Court past the left end, so the player on the right
    # wins a Point in the first Step. The same environment, unwrapped, gives the
    # Observations the stack should hold.
    start = PhysicsState(
        ball=Ball(x=0.0, y=0.5, vx=-0.01, vy=0.0),
        left_paddle=0.5,
        right_paddle=0.5,
        score=Score(left=0, right=0),
    )
    env, alone = ObservationStack(PongEnv(start=start)), PongEnv(start=start)
    env.reset(seed=0)
    observations = [alone.reset(seed=0)[0]]

    rewards = []
    for _ in range(2):
        stack, reward, _, _, info = env.step(Action.UP)
        rewards.append(reward)
        observations.append(alone.step(Action.UP)[0])

    assert (rewards, info["score"]) == ([1.0, 0.0], Score(left=0, right=1))
    first, after_the_point, last = observations
    assert stack.shape == (4, 84, 84)
    assert np.array_equal(stack, np.stack([first, first, after_the_point, last]))


def test_the_stack_can_hold_another_number_of_observations() -> None:
    two, one = ObservationStack(CountingEnv(), 2), ObservationStack(CountingEnv(), 1)

    stacks = []
    for env in (two, one):
        first, _ = env.reset(seed=0)
        stacks.append([first.tolist()] + [env.step(0)[0].tolist() for _ in range(2)])

    assert stacks == [
        [[[0, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [0, 2]]],
        [[[0, 0]], [[0, 1]], [[0, 2]]],
    ]


def test_the_stacked_environment_passes_gymnasiums_check_env() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The checker warns about any wrapper, so only that warning is let through.
        warnings.filterwarnings("ignore", ".* is different from the unwrapped")
        check_env(ObservationStack(PongEnv()), skip_render_check=True)
