"""The positions of the ball and the Paddles, which the Position learner decides from.

The tests start the Game from an exact physics state, with a fake player on the
left that keeps its Paddle still, so the expected positions are worked out by hand.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pong_ai_bot.environment import Action, PongEnv
from pong_ai_bot.game import Ball, PhysicsState, Score
from pong_ai_bot.positions import Positions
from pong_ai_bot.stack import ObservationStack

START = PhysicsState(
    ball=Ball(x=0.3, y=0.6, vx=0.01, vy=0.005),
    left_paddle=0.4,
    right_paddle=0.7,
    score=Score(left=0, right=0),
)
"""The ball in play, moving right and up, far from the Paddles and the walls."""


class Still:
    """A fake player for the left side, that keeps its Paddle still."""

    def move(self, state: PhysicsState) -> float:
        return 0.0


def positions_env() -> Positions:
    return Positions(PongEnv(left_player=lambda rng: Still(), start=START))


def test_reset_gives_the_ball_and_both_paddles_and_no_velocities() -> None:
    env = positions_env()

    positions, _ = env.reset(seed=0)

    # The ball's x and y, then the left Paddle's height, then the right's.
    assert positions.dtype == np.float32
    assert positions.tolist() == pytest.approx([0.3, 0.6, 0.4, 0.7])


def test_a_step_gives_the_positions_after_it_and_the_rest_as_the_env_gives_it() -> None:
    env, alone = positions_env(), PongEnv(left_player=lambda rng: Still(), start=START)
    env.reset(seed=0)
    alone.reset(seed=0)

    positions, *rest = env.step(Action.UP)

    # In the Step's 4 Frames the ball goes 0.04 right and 0.02 up, and the right
    # Paddle 0.08 up.
    assert positions.dtype == np.float32
    assert positions.tolist() == pytest.approx([0.34, 0.62, 0.4, 0.78])
    assert rest == list(alone.step(Action.UP)[1:])


def test_the_stack_of_four_steps_is_16_numbers() -> None:
    env = ObservationStack(positions_env())
    env.reset(seed=0)

    env.step(Action.UP)
    stack, *_ = env.step(Action.UP)

    assert env.observation_space.shape == stack.shape == (4, 4)
    # Oldest first, with copies of the Game's first positions in front.
    assert stack.tolist() == [
        pytest.approx([0.3, 0.6, 0.4, 0.7]),
        pytest.approx([0.3, 0.6, 0.4, 0.7]),
        pytest.approx([0.34, 0.62, 0.4, 0.78]),
        pytest.approx([0.38, 0.64, 0.4, 0.86]),
    ]


def test_the_balls_x_can_be_past_an_end_before_the_point_is_won() -> None:
    # The ball is already past the left Paddle's face, leaving the Court.
    start = replace(START, ball=Ball(x=0.03, y=0.5, vx=-0.01, vy=0.0))
    env = Positions(PongEnv(left_player=lambda rng: Still(), start=start))
    env.reset(seed=0)

    positions, reward, *_ = env.step(Action.STAY)

    # Its center is 0.01 past the end, but its right edge, 0.015 from the center,
    # is still in the Court, so no Point yet.
    assert (positions[0], reward) == (pytest.approx(-0.01), 0.0)
    assert env.observation_space.contains(positions)


def test_the_positions_environment_passes_gymnasiums_check_env() -> None:
    for env in (Positions(PongEnv()), ObservationStack(Positions(PongEnv()))):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # The checker warns about any wrapper, so only that warning is let
            # through.
            warnings.filterwarnings("ignore", ".* is different from the unwrapped")
            check_env(env, skip_render_check=True)
