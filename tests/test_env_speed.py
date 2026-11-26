"""The measurement of the environment's speed, the logic around `pong measure-env`.

The speed itself is measured by the command, not tested (issue #11). Expected numbers
are worked out by hand.
"""

import itertools
from collections.abc import Callable

import pytest

from pong_ai_bot.env_speed import EnvSpeed, measure_env_speed
from pong_ai_bot.environment import PongEnv
from pong_ai_bot.game import Ball, PhysicsState, Score


def from_state(
    ball: Ball,
    score: Score,
    on_frame: Callable[[PhysicsState], None] | None = None,
) -> PongEnv:
    """An environment whose every Game starts with the given ball and score, and
    both Paddles in the center."""
    return PongEnv(
        on_frame=on_frame,
        start=PhysicsState(ball=ball, left_paddle=0.5, right_paddle=0.5, score=score),
    )


# The ball only goes up and down between the walls, so no Point ever comes.
NO_POINT_EVER = Ball(x=0.5, y=0.5, vx=0.0, vy=0.01)

# A ball already past the left Paddle's face, moving 0.01 a Frame from the Court's
# left end. Its last part leaves the Court in the 2nd Frame, 0.02 past the end: the
# ball is 0.03 square, so half of it is 0.015.
LEAVING_PAST_THE_LEFT = Ball(x=0.0, y=0.5, vx=-0.01, vy=0.0)


def test_the_speed_is_the_steps_played_over_the_seconds_they_took() -> None:
    # The clock reads 10.0 seconds as the measurement starts and 12.5 as it ends.
    clock = iter([10.0, 12.5]).__next__

    speed = measure_env_speed(
        from_state(NO_POINT_EVER, Score(left=0, right=0)),
        steps=1_000,
        seed=0,
        clock=clock,
    )

    # 1,000 Steps in 2.5 seconds.
    assert speed.steps_per_second == 400.0


def test_the_next_game_starts_whenever_one_ends() -> None:
    # Every Game starts at 20 Points for the right side, with the ball leaving past
    # the left Paddle, so the right side wins its 21st Point in the 2nd Frame of the
    # Game's first Step, and the Game ends there.
    seen: list[PhysicsState] = []
    env = from_state(LEAVING_PAST_THE_LEFT, Score(left=0, right=20), seen.append)

    speed = measure_env_speed(env, steps=3, seed=0)

    assert speed.steps == 3
    assert [(state.frame, state.score) for state in seen] == [
        (1, Score(left=0, right=20)),
        (2, Score(left=0, right=21)),
    ] * 3


def test_the_actions_are_random_up_down_and_staying_all_come_up() -> None:
    seen: list[PhysicsState] = []
    env = from_state(NO_POINT_EVER, Score(left=0, right=0), seen.append)

    measure_env_speed(env, steps=300, seed=0)

    # The right Paddle starts in the center, and after each Step's 4 Frames it is
    # 0.08 higher (up), 0.08 lower (down), or where it was (staying). Near a wall a
    # move is cut short, so only Steps that start away from the walls count.
    heights = [0.5] + [state.right_paddle for state in seen[3::4]]
    moves = {
        round(after - before, 6)
        for before, after in itertools.pairwise(heights)
        if 0.2 < before < 0.8
    }
    assert moves == {0.08, -0.08, 0.0}


@pytest.mark.parametrize(
    ("seconds", "passes"),
    [
        # 50,000 Steps in 10 seconds is exactly 5,000 a second.
        (10.0, True),
        (8.0, True),
        # About 4,995 a second.
        (10.01, False),
    ],
)
def test_the_gate_is_at_least_5000_steps_a_second(seconds: float, passes: bool) -> None:
    assert EnvSpeed(steps=50_000, seconds=seconds).passes_the_gate is passes
