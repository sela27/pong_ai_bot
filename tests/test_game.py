"""The Game core, Frame by Frame, from exact physics states and from seeds.

Positions are in Court units: x from the left end (0) to the right end (1), y from
the bottom wall (0) to the top wall (1). Expected numbers are worked out by hand.
"""

import dataclasses
import itertools
import math
import random

import pytest

from pong_ai_bot.game import Ball, Game, PhysicsState, Score, Serve, Side

NO_POINTS_YET = Score(left=0, right=0)


def a_game(
    ball: Ball,
    left_paddle: float = 0.5,
    right_paddle: float = 0.5,
    score: Score = NO_POINTS_YET,
    seed: int = 0,
) -> Game:
    return Game(
        PhysicsState(
            ball=ball, left_paddle=left_paddle, right_paddle=right_paddle, score=score
        ),
        seed=seed,
    )


def test_a_game_starts_from_the_exact_state_it_is_given() -> None:
    state = PhysicsState(
        ball=Ball(x=0.4, y=0.7, vx=-0.0135, vy=0.006),
        left_paddle=0.3,
        right_paddle=0.6,
        score=Score(left=4, right=9),
    )

    game = Game(state, seed=0)

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


# The right Paddle spans x from 0.92 to 0.95, so its face is at 0.92, and the ball
# (0.03 square) touches it when the ball's center is at 0.905. The left Paddle's face
# is at 0.08, touched with the center at 0.095: the two are 0.81 apart, 60 Frames at
# the base speed of 0.0135 a Frame. At that speed the ball covers 0.00675 in half a
# Frame, so from 0.89825 it touches the right face half way through the next Frame,
# or it left that face half a Frame ago.
HALF_A_FRAME_FROM_THE_RIGHT_FACE = 0.89825
HALF_A_FRAME_FROM_THE_LEFT_FACE = 0.10175
BASE_SPEED = 0.0135
TOP_SPEED = 0.027
"""0.81 in 30 Frames, the shortest crossing."""


def test_a_ball_that_hits_the_center_of_the_right_paddle_comes_straight_back() -> None:
    game = a_game(
        Ball(x=HALF_A_FRAME_FROM_THE_RIGHT_FACE, y=0.4, vx=BASE_SPEED, vy=0.0),
        right_paddle=0.4,
    )

    game.play_frame(left=0.0, right=0.0)

    ball = game.state.ball
    assert ball.vx < 0, f"vx={ball.vx}"
    assert ball.vy == 0.0, f"vy={ball.vy}"
    assert ball.x + 0.015 <= 0.92, f"x={ball.x}"


def test_a_ball_that_hits_the_center_of_the_left_paddle_comes_straight_back() -> None:
    game = a_game(
        Ball(x=HALF_A_FRAME_FROM_THE_LEFT_FACE, y=0.7, vx=-BASE_SPEED, vy=0.0),
        left_paddle=0.7,
    )

    game.play_frame(left=0.0, right=0.0)

    ball = game.state.ball
    assert ball.vx > 0, f"vx={ball.vx}"
    assert ball.vy == 0.0, f"vy={ball.vy}"
    assert ball.x - 0.015 >= 0.08, f"x={ball.x}"


def a_game_with_a_hit_half_way_through_the_frame(
    side: Side, *, paddle: float, hit_at: float, vy: float = 0.0
) -> Game:
    """The ball touches the face of the Paddle on `side`, at height `hit_at`, half way
    through the next Frame. The Paddle on the other side is out of the way."""
    y = hit_at - vy / 2
    if side is Side.RIGHT:
        ball = Ball(x=HALF_A_FRAME_FROM_THE_RIGHT_FACE, y=y, vx=BASE_SPEED, vy=vy)
        return a_game(ball, right_paddle=paddle)
    ball = Ball(x=HALF_A_FRAME_FROM_THE_LEFT_FACE, y=y, vx=-BASE_SPEED, vy=vy)
    return a_game(ball, left_paddle=paddle)


@pytest.mark.parametrize("side", [Side.RIGHT, Side.LEFT])
@pytest.mark.parametrize(
    ("above_the_center", "direction"), [(0.075, "up"), (-0.075, "down")]
)
def test_a_hit_on_the_edge_of_a_paddle_leaves_at_45_degrees_away_from_its_center(
    side: Side, above_the_center: float, direction: str
) -> None:
    # A Paddle is 0.15 tall, so its edges are 0.075 from its height.
    game = a_game_with_a_hit_half_way_through_the_frame(
        side, paddle=0.5, hit_at=0.5 + above_the_center
    )

    game.play_frame(left=0.0, right=0.0)

    # At 45 degrees the ball moves as much vertically as horizontally.
    ball = game.state.ball
    expected_vy = abs(ball.vx) if direction == "up" else -abs(ball.vx)
    assert ball.vy == pytest.approx(expected_vy), f"vx={ball.vx}, vy={ball.vy}"


@pytest.mark.parametrize(
    ("above_the_center", "tan_of_the_angle"),
    [(0.0375, 0.41421356), (-0.0375, -0.41421356)],
)
def test_the_angle_grows_linearly_from_the_center_to_the_edge(
    above_the_center: float, tan_of_the_angle: float
) -> None:
    # Half way from the center (0 degrees) to the edge (45) is 22.5 degrees, and
    # tan(22.5 degrees) = sqrt(2) - 1 = 0.41421356.
    game = a_game_with_a_hit_half_way_through_the_frame(
        Side.RIGHT, paddle=0.3, hit_at=0.3 + above_the_center
    )

    game.play_frame(left=0.0, right=0.0)

    ball = game.state.ball
    assert ball.vy == pytest.approx(tan_of_the_angle * abs(ball.vx)), (
        f"vx={ball.vx}, vy={ball.vy}"
    )


@pytest.mark.parametrize(
    ("above_the_center", "direction"), [(0.085, "up"), (-0.085, "down")]
)
def test_a_hit_on_the_corner_of_a_paddle_counts_as_a_hit_on_its_edge(
    above_the_center: float, direction: str
) -> None:
    # The ball's center is 0.01 past the Paddle's edge (0.075), but the ball is
    # 0.03 square, so it still overlaps the Paddle's corner by 0.005.
    game = a_game_with_a_hit_half_way_through_the_frame(
        Side.LEFT, paddle=0.6, hit_at=0.6 + above_the_center
    )

    game.play_frame(left=0.0, right=0.0)

    ball = game.state.ball
    assert ball.vx > 0, f"vx={ball.vx}"
    expected_vy = abs(ball.vx) if direction == "up" else -abs(ball.vx)
    assert ball.vy == pytest.approx(expected_vy), f"vx={ball.vx}, vy={ball.vy}"


@pytest.mark.parametrize("above_the_center", [0.095, -0.095])
def test_a_ball_that_passes_just_beyond_the_corner_misses_the_paddle(
    above_the_center: float,
) -> None:
    # The ball's center is 0.02 past the Paddle's edge, so its near side is 0.005
    # past the Paddle.
    game = a_game_with_a_hit_half_way_through_the_frame(
        Side.LEFT, paddle=0.6, hit_at=0.6 + above_the_center
    )

    game.play_frame(left=0.0, right=0.0)

    ball = game.state.ball
    assert (ball.vx, ball.vy) == (-BASE_SPEED, 0.0)


@pytest.mark.parametrize("incoming_vy", [0.012, -0.009])
def test_the_angle_does_not_depend_on_the_angle_the_ball_came_in_at(
    incoming_vy: float,
) -> None:
    # Hit half way between the center and the top edge, so at 22.5 degrees.
    game = a_game_with_a_hit_half_way_through_the_frame(
        Side.RIGHT, paddle=0.5, hit_at=0.5375, vy=incoming_vy
    )

    game.play_frame(left=0.0, right=0.0)

    ball = game.state.ball
    assert ball.vy == pytest.approx(0.41421356 * abs(ball.vx)), (
        f"vx={ball.vx}, vy={ball.vy}"
    )


def test_the_angle_changes_only_the_vertical_speed() -> None:
    # Hits at the center, half way to the edge, at the edge and on the corner.
    leaving_vx = set()
    for above_the_center in [0.0, 0.0375, -0.075, 0.085]:
        game = a_game_with_a_hit_half_way_through_the_frame(
            Side.RIGHT, paddle=0.5, hit_at=0.5 + above_the_center
        )
        game.play_frame(left=0.0, right=0.0)
        leaving_vx.add(game.state.ball.vx)

    assert len(leaving_vx) == 1, f"vx={leaving_vx}"


def test_a_hit_just_after_a_wall_bounce_is_where_the_ball_touches_the_paddle() -> None:
    # Half way through the Frame the ball would rise from 0.98 to 0.99, but its
    # center can't go past 0.985, so it bounces off the top wall back to 0.98 and
    # touches the face there. The Paddle is as high as it goes (0.925), so that is
    # 0.055 above its center: 45 * 0.055 / 0.075 = 33 degrees, and
    # tan(33 degrees) = 0.64940759.
    game = a_game(
        Ball(x=HALF_A_FRAME_FROM_THE_RIGHT_FACE, y=0.98, vx=BASE_SPEED, vy=0.02),
        right_paddle=0.925,
    )

    game.play_frame(left=0.0, right=0.0)

    ball = game.state.ball
    assert ball.vy == pytest.approx(0.64940759 * abs(ball.vx)), (
        f"vx={ball.vx}, vy={ball.vy}"
    )


def test_a_paddle_move_counts_in_the_frame_it_is_made() -> None:
    # The ball touches the face 0.1 above the Paddle's height, out of its reach
    # (0.075 + 0.015 = 0.09). Moving up by 0.02 in the same Frame brings the
    # Paddle's corner to the ball.
    game = a_game_with_a_hit_half_way_through_the_frame(
        Side.RIGHT, paddle=0.5, hit_at=0.6
    )

    game.play_frame(left=0.0, right=0.02)

    assert game.state.ball.vx < 0, f"vx={game.state.ball.vx}"


def test_a_ball_already_past_the_face_of_a_paddle_does_not_bounce_off_it() -> None:
    # The ball's near side is at 0.925, past the right face (0.92). The Paddle's top
    # edge is at 0.575, below the ball's bottom (0.585), and it moves up into the
    # ball: by the end of the Frame its top edge is at 0.595.
    game = a_game(Ball(x=0.91, y=0.6, vx=BASE_SPEED, vy=0.0), right_paddle=0.5)

    game.play_frame(left=0.0, right=0.02)
    after_the_paddle_moved = game.state.ball
    winners = [game.play_frame(left=0.0, right=0.0).point_won_by for _ in range(10)]

    assert (after_the_paddle_moved.vx, after_the_paddle_moved.vy) == (BASE_SPEED, 0.0)
    assert Side.LEFT in winners


def frames_of_paddle_hits(game: Game, *, hits: int) -> list[int]:
    """Play still Paddles until `hits` hits, and return the Frame of each."""
    frames: list[int] = []
    frame = 0
    while len(frames) < hits:
        vx_before = game.state.ball.vx
        game.play_frame(left=0.0, right=0.0)
        frame += 1
        if (game.state.ball.vx > 0) != (vx_before > 0):
            frames.append(frame)
        assert frame < 10_000, f"only {len(frames)} hits in {frame} Frames"
    return frames


def test_at_the_base_speed_the_ball_goes_from_paddle_to_paddle_in_60_frames() -> None:
    # The ball left the right face half a Frame ago, so it touches the left face
    # 59.5 Frames from now, during the 60th Frame.
    game = a_game(
        Ball(x=HALF_A_FRAME_FROM_THE_RIGHT_FACE, y=0.5, vx=-BASE_SPEED, vy=0.0)
    )

    assert frames_of_paddle_hits(game, hits=1) == [60]


def test_each_hit_on_either_paddle_shortens_the_crossing_by_2_frames_down_to_30() -> (
    None
):
    # The ball goes straight between the centers of the two Paddles, hitting the
    # right one first.
    game = a_game(
        Ball(x=HALF_A_FRAME_FROM_THE_RIGHT_FACE, y=0.5, vx=BASE_SPEED, vy=0.0)
    )

    frames = frames_of_paddle_hits(game, hits=20)

    crossings = [later - earlier for earlier, later in itertools.pairwise(frames)]
    assert crossings == [
        *[58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32],
        *[30, 30, 30, 30, 30],
    ]


@pytest.mark.parametrize("start", [0.85, 0.87, 0.878, 0.885, 0.89, 0.895, 0.9, 0.9049])
@pytest.mark.parametrize("hit_at", [0.5, 0.55, 0.585])
def test_even_at_the_top_speed_the_ball_does_not_skip_through_a_paddle(
    start: float, hit_at: float
) -> None:
    # The ball starts from a range of places just short of the right face's touching
    # line (0.905), so it reaches that line at many parts of a Frame.
    game = a_game(Ball(x=start, y=hit_at, vx=TOP_SPEED, vy=0.0), right_paddle=0.5)

    for _ in range(5):
        result = game.play_frame(left=0.0, right=0.0)
        assert result.point_won_by is None
        assert game.state.ball.x <= 0.905 + 1e-9, f"x={game.state.ball.x}"

    assert game.state.ball.vx == pytest.approx(-TOP_SPEED)


def test_a_new_game_starts_with_everything_in_the_center_and_a_serve_to_come() -> None:
    game = Game.new(seed=1)

    assert game.state == PhysicsState(
        ball=Ball(x=0.5, y=0.5, vx=0.0, vy=0.0),
        left_paddle=0.5,
        right_paddle=0.5,
        score=Score(left=0, right=0),
        frame=0,
        serve=Serve(frames_left=30, toward=None),
    )


def test_the_game_counts_its_frames() -> None:
    game = Game(
        PhysicsState(
            ball=Ball(x=0.5, y=0.5, vx=0.0135, vy=0.0),
            left_paddle=0.5,
            right_paddle=0.5,
            score=NO_POINTS_YET,
            frame=41,
        ),
        seed=0,
    )

    for _ in range(3):
        game.play_frame(left=0.0, right=0.0)

    assert game.state.frame == 44


def test_the_ball_waits_in_the_center_for_30_frames_while_the_paddles_move() -> None:
    game = Game.new(seed=1)

    for frame in range(1, 31):
        game.play_frame(left=0.01, right=-0.01)
        ball = game.state.ball
        assert (ball.x, ball.y) == (0.5, 0.5), f"Frame {frame}: {ball}"
    paddles_after_the_wait = (game.state.left_paddle, game.state.right_paddle)
    game.play_frame(left=0.0, right=0.0)

    assert paddles_after_the_wait == (pytest.approx(0.8), pytest.approx(0.2))
    assert game.state.ball.x != 0.5, f"{game.state.ball}"


def served_ball(game: Game) -> Ball:
    """Play still Paddles through the 30 Frames of a serve, and return the ball."""
    for _ in range(30):
        game.play_frame(left=0.0, right=0.0)
    return game.state.ball


def test_the_first_serve_of_a_game_goes_toward_a_random_side() -> None:
    # A fair coin over 200 Games lands 100 times either way, give or take about 7,
    # so the bounds below are more than 4 of those away.
    serves_to_the_right = sum(
        served_ball(Game.new(seed=seed)).vx > 0 for seed in range(200)
    )

    assert 70 <= serves_to_the_right <= 130, f"{serves_to_the_right} of 200"


def test_the_serve_angle_is_uniform_between_minus_30_and_30_degrees() -> None:
    # 1000 serves in six bins of 10 degrees: about 167 in each, give or take about
    # 12, so the bounds below are more than 4 of those away.
    angles = []
    for seed in range(1000):
        ball = served_ball(Game.new(seed=seed))
        angles.append(math.degrees(math.atan2(ball.vy, abs(ball.vx))))
    bins = [sum(low <= a < low + 10 for a in angles) for low in range(-30, 30, 10)]

    assert -30 <= min(angles) and max(angles) <= 30, f"{min(angles)}, {max(angles)}"
    assert all(117 <= count <= 217 for count in bins), f"bins of 10 degrees: {bins}"


# Each ball leaves the Court in the next Frame, past a Paddle that is out of its way.
LEAVING_PAST_THE_RIGHT_PADDLE = Ball(x=1.0, y=0.5, vx=0.02, vy=0.0)
LEAVING_PAST_THE_LEFT_PADDLE = Ball(x=0.0, y=0.5, vx=-0.02, vy=0.0)


@pytest.mark.parametrize(
    ("ball", "lost_by"),
    [
        pytest.param(LEAVING_PAST_THE_RIGHT_PADDLE, Side.RIGHT, id="right"),
        pytest.param(LEAVING_PAST_THE_LEFT_PADDLE, Side.LEFT, id="left"),
    ],
)
def test_after_a_point_the_ball_is_served_toward_the_side_that_lost_it(
    ball: Ball, lost_by: Side
) -> None:
    # The Paddles stay where they are: only a new Game puts them in the center.
    served_toward: set[Side | None] = set()
    for seed in range(20):
        game = a_game(ball, left_paddle=0.7, right_paddle=0.2, seed=seed)
        game.play_frame(left=0.0, right=0.0)
        after_the_point = game.state
        served = served_ball(game)
        served_toward.add(
            Side.RIGHT if served.vx > 0 else Side.LEFT if served.vx < 0 else None
        )

        assert after_the_point.ball == Ball(x=0.5, y=0.5, vx=0.0, vy=0.0)
        assert (game.state.left_paddle, game.state.right_paddle) == (0.7, 0.2)
        assert (served.x, served.y) == (0.5, 0.5)

    assert served_toward == {lost_by}


def test_a_serve_brings_the_crossing_back_to_60_frames() -> None:
    # The ball leaves past the left Paddle at the top speed, a 30-Frame crossing.
    game = a_game(Ball(x=0.0, y=0.5, vx=-TOP_SPEED, vy=0.0), left_paddle=0.9)
    game.play_frame(left=0.0, right=0.0)

    served = served_ball(game)

    assert served.vx == pytest.approx(-BASE_SPEED), f"vx={served.vx}"


@pytest.mark.parametrize(
    ("ball", "score", "final_score"),
    [
        pytest.param(
            LEAVING_PAST_THE_RIGHT_PADDLE,
            Score(left=20, right=13),
            Score(left=21, right=13),
            id="left",
        ),
        pytest.param(
            LEAVING_PAST_THE_LEFT_PADDLE,
            Score(left=17, right=20),
            Score(left=17, right=21),
            id="right",
        ),
    ],
)
def test_a_game_ends_when_a_side_reaches_21_points(
    ball: Ball, score: Score, final_score: Score
) -> None:
    game = a_game(ball, left_paddle=0.2, right_paddle=0.2, score=score)

    result = game.play_frame(left=0.0, right=0.0)

    assert game.state.score == final_score
    assert (result.terminated, result.truncated) == (True, False)


def test_a_point_short_of_21_does_not_end_the_game() -> None:
    game = a_game(
        LEAVING_PAST_THE_LEFT_PADDLE,
        left_paddle=0.2,
        score=Score(left=20, right=19),
    )

    result = game.play_frame(left=0.0, right=0.0)

    assert game.state.score == Score(left=20, right=20)
    assert (result.terminated, result.truncated) == (False, False)


def test_a_game_is_cut_off_after_108000_frames_with_no_extra_point() -> None:
    game = Game(
        PhysicsState(
            ball=Ball(x=0.5, y=0.5, vx=BASE_SPEED, vy=0.0),
            left_paddle=0.5,
            right_paddle=0.5,
            score=Score(left=12, right=9),
            frame=107_998,
        ),
        seed=0,
    )

    results = [game.play_frame(left=0.0, right=0.0) for _ in range(2)]

    assert [(r.terminated, r.truncated) for r in results] == [
        (False, False),
        (False, True),
    ]
    assert results[-1].point_won_by is None
    assert game.state.score == Score(left=12, right=9)


def test_a_game_won_in_its_last_frame_ends_rather_than_being_cut_off() -> None:
    game = Game(
        PhysicsState(
            ball=LEAVING_PAST_THE_LEFT_PADDLE,
            left_paddle=0.2,
            right_paddle=0.5,
            score=Score(left=4, right=20),
            frame=107_999,
        ),
        seed=0,
    )

    result = game.play_frame(left=0.0, right=0.0)

    assert game.state.score == Score(left=4, right=21)
    assert (result.terminated, result.truncated) == (True, False)


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(
            PhysicsState(
                ball=Ball(x=0.5, y=0.5, vx=0.0, vy=0.0),
                left_paddle=0.5,
                right_paddle=0.5,
                score=Score(left=21, right=18),
                serve=Serve(frames_left=30, toward=Side.LEFT),
            ),
            id="won",
        ),
        pytest.param(
            PhysicsState(
                ball=Ball(x=0.5, y=0.5, vx=BASE_SPEED, vy=0.0),
                left_paddle=0.5,
                right_paddle=0.5,
                score=Score(left=12, right=9),
                frame=108_000,
            ),
            id="cut off",
        ),
    ],
)
def test_a_game_that_is_over_plays_no_more_frames(state: PhysicsState) -> None:
    game = Game(state, seed=0)

    with pytest.raises(RuntimeError, match="over"):
        game.play_frame(left=0.0, right=0.0)
    assert game.state == state


def test_a_game_hands_out_one_generator_seeded_by_its_seed() -> None:
    # The Opponent draws from it, so a Game's randomness all comes from its seed.
    first_draws = [Game.new(seed=seed).rng.random() for seed in (3, 3, 4)]

    assert first_draws[0] == first_draws[1] != first_draws[2], f"{first_draws}"


def test_the_handed_out_generator_is_the_one_the_game_serves_with() -> None:
    untouched = Game.new(seed=3)
    drawn_from = Game.new(seed=3)
    drawn_from.rng.random()

    assert served_ball(untouched) != served_ball(drawn_from)


def frame_by_frame(seed: int, moves: list[tuple[float, float]]) -> list[PhysicsState]:
    """The state after each Frame of a new Game, with the given Paddle moves, until
    the moves run out or the Game ends."""
    game = Game.new(seed=seed)
    states = []
    for left, right in moves:
        result = game.play_frame(left=left, right=right)
        states.append(game.state)
        if result.terminated or result.truncated:
            break
    return states


def test_the_same_seed_and_the_same_moves_give_the_same_game_frame_by_frame() -> None:
    # Paddles moving at random let a Point go by every 80 Frames or so, serve
    # included, so this takes in many serves, and maybe the end of the Game.
    moves_from = random.Random(99)
    moves = [
        (moves_from.uniform(-0.02, 0.02), moves_from.uniform(-0.02, 0.02))
        for _ in range(3000)
    ]

    game = frame_by_frame(7, moves)
    same_seed = frame_by_frame(7, moves)
    other_seed = frame_by_frame(8, moves)

    assert game[-1].score.left + game[-1].score.right >= 10, f"{game[-1].score}"
    assert same_seed == game
    assert other_seed != game
