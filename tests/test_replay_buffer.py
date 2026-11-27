"""The replay buffer, through adding Steps and sampling them (issue #12).

Most tests play a fake environment whose Observation is two numbers, the Game's
number and how many Steps it has played, through the stack's wrapper, as the
training loop does, so a stack shows which Observations it holds.
"""

from collections import Counter
from collections.abc import Collection, Sequence
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium import spaces

from pong_ai_bot.environment import Action, PongEnv
from pong_ai_bot.game import Ball, PhysicsState, Score
from pong_ai_bot.replay_buffer import ReplayBuffer
from pong_ai_bot.stack import ObservationStack
from pong_ai_bot.training import Transition


class FakeGames(gym.Env[npt.NDArray[np.int64], int]):
    """A fake environment whose Games last as many Steps as `lengths` says, one
    after another. Its Observation is (Game, Step): the Game's number, counting
    from 0 at the first `reset`, and how many Steps the Game has played.

    The second Step of each Game wins a Point, a Reward of +1, and the Game goes
    on. A Game ends terminated at its length, but the Games `cut_off` names, which
    are truncated.
    """

    def __init__(self, lengths: Sequence[int], cut_off: Collection[int] = ()) -> None:
        self.observation_space = spaces.Box(0, 1_000, (2,), np.int64)
        self._lengths = lengths
        self._cut_off = cut_off
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
        reward = 1.0 if self._steps == 2 else 0.0
        ended = self._steps == self._lengths[self._game]
        cut_off = self._game in self._cut_off
        return self._observation(), reward, ended and not cut_off, ended and cut_off, {}

    def _observation(self) -> npt.NDArray[np.int64]:
        return np.array([self._game, self._steps], dtype=np.int64)


def play(
    env: gym.Env[npt.NDArray[Any], int], buffer: ReplayBuffer, steps: int
) -> list[Transition]:
    """Play `steps` Steps of `env` as the training loop does, a Game after another,
    and give each to `buffer`. The Steps played, with the stacks the player was
    given. The Actions go 0, 1, 2, 0, and so on, over all the Steps."""
    played = []
    observation, _ = env.reset(seed=0)
    for step in range(steps):
        action = step % 3
        next_observation, reward, terminated, truncated, _ = env.step(action)
        transition = Transition(
            observation, action, float(reward), next_observation, terminated, truncated
        )
        buffer.add(transition)
        played.append(transition)
        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation
    return played


Plain = tuple[str, int, float, str, bool, bool]
"""A transition as plain values, which a set can hold: each stack as the text of
its list."""


def plain(transition: Transition) -> Plain:
    observation, action, reward, next_observation, terminated, truncated = transition
    return (
        str(np.asarray(observation).tolist()),
        action,
        reward,
        str(np.asarray(next_observation).tolist()),
        terminated,
        truncated,
    )


def sampled(buffer: ReplayBuffer) -> set[Plain]:
    """Every transition drawn in a large sample from `buffer`: enough for each of
    a few dozen Steps to be drawn."""
    rng = np.random.default_rng(0)
    return {plain(transition) for transition in buffer.sample(1_000, rng)}


def test_a_sampled_step_has_the_stacks_the_player_was_given_before_and_after_it() -> (
    None
):
    buffer = ReplayBuffer(size=100, stack_length=2)

    play(ObservationStack(FakeGames([3]), 2), buffer, steps=3)

    # The second Step wins a Point.
    assert sampled(buffer) == {
        ("[[0, 0], [0, 0]]", 0, 0.0, "[[0, 0], [0, 1]]", False, False),
        ("[[0, 0], [0, 1]]", 1, 1.0, "[[0, 1], [0, 2]]", False, False),
        ("[[0, 1], [0, 2]]", 2, 0.0, "[[0, 2], [0, 3]]", True, False),
    }


def test_the_stacks_are_the_wrappers_from_the_start_of_a_game_and_after_a_point() -> (
    None
):
    buffer = ReplayBuffer(size=100, stack_length=4)

    played = play(ObservationStack(FakeGames([7])), buffer, steps=7)

    assert sampled(buffer) == {plain(transition) for transition in played}


def test_a_stack_never_holds_observations_of_the_game_before() -> None:
    # The second Game is cut off, and the others end.
    buffer = ReplayBuffer(size=100, stack_length=4)

    played = play(ObservationStack(FakeGames([2, 3, 5], cut_off={1})), buffer, 10)

    assert sampled(buffer) == {plain(transition) for transition in played}


def test_terminated_and_truncated_are_kept_apart_with_the_observation_after_each() -> (
    None
):
    buffer = ReplayBuffer(size=100, stack_length=2)

    play(ObservationStack(FakeGames([2, 3, 1], cut_off={1}), 2), buffer, steps=6)

    # The last Steps of the three Games. The second Game is cut off, and the
    # Observation after its last Step is its own, not the next Game's first.
    last_steps = {transition for transition in sampled(buffer) if any(transition[4:])}
    assert last_steps == {
        ("[[0, 0], [0, 1]]", 1, 1.0, "[[0, 1], [0, 2]]", True, False),
        ("[[1, 1], [1, 2]]", 1, 0.0, "[[1, 2], [1, 3]]", False, True),
        ("[[2, 0], [2, 0]]", 2, 0.0, "[[2, 0], [2, 1]]", True, False),
    }


@pytest.mark.parametrize(
    "lengths",
    [
        # The kept Steps' stacks reach back to Steps the buffer no longer keeps.
        [20],
        # The kept Steps' stacks hold copies of their Game's first Observation,
        # and the Game before is gone.
        [4, 6],
    ],
)
def test_the_buffer_keeps_only_its_size_of_the_latest_steps(lengths: list[int]) -> None:
    buffer = ReplayBuffer(size=3, stack_length=4)

    played = play(ObservationStack(FakeGames(lengths)), buffer, steps=9)

    assert sampled(buffer) == {plain(transition) for transition in played[-3:]}


def test_its_length_is_how_many_steps_it_holds_up_to_its_size() -> None:
    lengths = []
    for steps in range(6):
        buffer = ReplayBuffer(size=3, stack_length=4)
        play(ObservationStack(FakeGames([20])), buffer, steps)
        lengths.append(len(buffer))

    assert lengths == [0, 1, 2, 3, 3, 3]


def test_a_sample_draws_each_step_it_holds_as_often() -> None:
    buffer = ReplayBuffer(size=3, stack_length=4)
    play(ObservationStack(FakeGames([20])), buffer, steps=5)

    drawn = buffer.sample(3_000, np.random.default_rng(0))

    # It holds Steps 3, 4 and 5, so each should be drawn about 1,000 times.
    counts = Counter(int(transition.next_observation[-1][1]) for transition in drawn)
    assert sorted(counts) == [3, 4, 5]
    assert all(abs(count - 1_000) < 100 for count in counts.values()), (
        f"the Steps are drawn {dict(counts)} times"
    )


def test_a_buffer_of_one_step_gives_back_the_latest_with_pongs_own_images() -> None:
    # The ball is leaving the Court past the left end, so the player on the right
    # wins a Point in the first Step.
    start = PhysicsState(
        ball=Ball(x=0.0, y=0.5, vx=-0.01, vy=0.0),
        left_paddle=0.5,
        right_paddle=0.5,
        score=Score(left=0, right=0),
    )
    env = ObservationStack(PongEnv(start=start))
    buffer = ReplayBuffer(size=1, stack_length=4)
    rng = np.random.default_rng(0)

    observation, _ = env.reset(seed=0)
    rewards, differences = [], []
    for step in range(1, 7):
        next_observation, reward, terminated, truncated, _ = env.step(Action.UP)
        latest = Transition(
            observation,
            Action.UP,
            float(reward),
            next_observation,
            terminated,
            truncated,
        )
        buffer.add(latest)
        [kept] = buffer.sample(1, rng)
        rewards.append(reward)
        if not all(map(np.array_equal, kept, latest, strict=True)):
            differences.append(step)
        observation = next_observation

    assert rewards == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert differences == [], f"the Steps given back otherwise are {differences}"
