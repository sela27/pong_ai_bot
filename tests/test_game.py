"""The Game core, Frame by Frame, from exact physics states.

Positions are in Court units: x from the left end (0) to the right end (1), y from
the bottom wall (0) to the top wall (1). Expected numbers are worked out by hand.
"""

import dataclasses

import pytest

from pong_ai_bot.game import Ball, Game, PhysicsState, Score, Side

NO_POINTS_YET = Score(left=0, right=0)


def a_game(
    ball: Ball,
    left_paddle: float = 0.5,
    right_paddle: float = 0.5,
    score: Score = NO_POINTS_YET,
) -> Game:
    return Game(
        PhysicsState(
            ball=ball, left_paddle=left_paddle, right_paddle=right_paddle, score=score
        )
    )


def test_a_game_starts_from_the_exact_state_it_is_given() -> None:
    state = PhysicsState(
        ball=Ball(x=0.4, y=0.7, vx=-0.0135, vy=0.006),
        left_paddle=0.3,
        right_paddle=0.6,
        score=Score(left=4, right=9),
    )

    game = Game(state)

    assert game.state == PhysicsState(
        ball=Ball(x=0.4, y=0.7, vx=-0.0135, vy=0.006),
        left_paddle=0.3,
        right_paddle=0.6,
        score=Score(left=4, right=9),
    )


def test_the_state_can_only_be_read() -> None:
    game = a_game(Ball(x=0.5, y=0.5, vx=0.0135, vy=0.0))

    with pytest.raises(AttributeError):
        game.state = game.state  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        game.state.ball.x = 0.9  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        game.state.right_paddle = 0.9  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        game.state.score.left = 20  # type: ignore[misc]


def test_the_ball_moves_by_its_velocity_every_frame() -> None:
    game = a_game(Ball(x=0.5, y=0.5, vx=0.01, vy=-0.005))

    for _ in range(3):
        game.play_frame(left=0.0, right=0.0)

    ball = game.state.ball
    assert (ball.x, ball.y) == (pytest.approx(0.53), pytest.approx(0.485))
    assert (ball.vx, ball.vy) == (0.01, -0.005)


def test_the_ball_bounces_off_the_top_wall_like_a_mirror() -> None:
    # The ball is 0.03 square, so its center can rise to 0.985. From 0.975 it
    # would reach 0.995, 0.01 past that, and the mirror puts it at 0.975 instead.
    game = a_game(Ball(x=0.5, y=0.975, vx=0.01, vy=0.02))

    game.play_frame(left=0.0, right=0.0)
    after_bounce = game.state.ball
    game.play_frame(left=0.0, right=0.0)
    a_frame_later = game.state.ball

    assert (after_bounce.x, after_bounce.y) == (
        pytest.approx(0.51),
        pytest.approx(0.975),
    )
    assert (after_bounce.vx, after_bounce.vy) == (0.01, -0.02)
    assert (a_frame_later.x, a_frame_later.y) == (
        pytest.approx(0.52),
        pytest.approx(0.955),
    )


def test_the_ball_bounces_off_the_bottom_wall_like_a_mirror() -> None:
    # The ball's center can fall to 0.015. From 0.03 it would reach 0.005, 0.01
    # past that, and the mirror puts it at 0.025 instead.
    game = a_game(Ball(x=0.5, y=0.03, vx=-0.01, vy=-0.025))

    game.play_frame(left=0.0, right=0.0)
    after_bounce = game.state.ball
    game.play_frame(left=0.0, right=0.0)
    a_frame_later = game.state.ball

    assert (after_bounce.x, after_bounce.y) == (
        pytest.approx(0.49),
        pytest.approx(0.025),
    )
    assert (after_bounce.vx, after_bounce.vy) == (-0.01, 0.025)
    assert (a_frame_later.x, a_frame_later.y) == (
        pytest.approx(0.48),
        pytest.approx(0.05),
    )


def test_each_paddle_moves_up_or_down_by_its_own_move_every_frame() -> None:
    game = a_game(
        Ball(x=0.5, y=0.5, vx=0.0135, vy=0.0), left_paddle=0.3, right_paddle=0.6
    )

    for _ in range(3):
        game.play_frame(left=0.02, right=-0.01)

    assert game.state.left_paddle == pytest.approx(0.36)
    assert game.state.right_paddle == pytest.approx(0.57)


def test_a_paddle_stops_at_the_edge_of_the_court() -> None:
    # A Paddle is 0.15 tall, so its center stays between 0.075 and 0.925.
    game = a_game(
        Ball(x=0.5, y=0.5, vx=0.0135, vy=0.0), left_paddle=0.91, right_paddle=0.09
    )

    for _ in range(2):
        game.play_frame(left=0.02, right=-0.02)

    assert game.state.left_paddle == pytest.approx(0.925)
    assert game.state.right_paddle == pytest.approx(0.075)


@pytest.mark.parametrize(("left", "right"), [(0.021, 0.0), (0.0, -0.03)])
def test_a_paddle_cannot_move_more_than_0_02_in_a_frame(
    left: float, right: float
) -> None:
    game = a_game(Ball(x=0.5, y=0.5, vx=0.0135, vy=0.0))

    with pytest.raises(ValueError, match="0.02"):
        game.play_frame(left=left, right=right)


def test_a_ball_that_leaves_past_the_right_paddle_is_a_point_for_the_left() -> None:
    # The right Paddle is low, away from the ball's path.
    game = a_game(
        Ball(x=1.0, y=0.5, vx=0.02, vy=0.0),
        right_paddle=0.1,
        score=Score(left=3, right=5),
    )

    result = game.play_frame(left=0.0, right=0.0)

    assert result.point_won_by is Side.LEFT
    assert game.state.score == Score(left=4, right=5)


def test_a_ball_that_leaves_past_the_left_paddle_is_a_point_for_the_right() -> None:
    # The left Paddle is high, away from the ball's path.
    game = a_game(
        Ball(x=0.0, y=0.3, vx=-0.02, vy=0.0),
        left_paddle=0.9,
        score=Score(left=3, right=5),
    )

    result = game.play_frame(left=0.0, right=0.0)

    assert result.point_won_by is Side.RIGHT
    assert game.state.score == Score(left=3, right=6)


def test_the_point_comes_only_once_the_whole_ball_has_left_the_court() -> None:
    # The ball's center goes 0.99, 1.01, 1.03: its inner edge is 0.975, then
    # 0.995 (still inside), then 1.015 (out).
    game = a_game(
        Ball(x=0.97, y=0.5, vx=0.02, vy=0.0),
        right_paddle=0.1,
        score=Score(left=0, right=0),
    )

    winners = [game.play_frame(left=0.0, right=0.0).point_won_by for _ in range(3)]

    assert winners == [None, None, Side.LEFT]
    assert game.state.score == Score(left=1, right=0)


def test_a_point_is_scored_only_once() -> None:
    game = a_game(
        Ball(x=1.0, y=0.5, vx=0.02, vy=0.0),
        right_paddle=0.1,
        score=Score(left=0, right=0),
    )
    game.play_frame(left=0.0, right=0.0)

    winners = [game.play_frame(left=0.0, right=0.0).point_won_by for _ in range(10)]

    assert winners == [None] * 10
    assert game.state.score == Score(left=1, right=0)
