"""The Opponent, from given physics states, and playing in a Game.

Positions are in Court units, y from the bottom wall (0) to the top wall (1). A
Paddle is 0.15 tall, so half of it is 0.075, and the offset width is a fraction of
that. Expected numbers are worked out by hand.
"""

import dataclasses
import itertools

import numpy as np
import pytest

from pong_ai_bot.game import Ball, Game, PhysicsState, Score, Serve, Side
from pong_ai_bot.opponent import Opponent

BASE_SPEED = 0.0135


def _on_court(ball: Ball, serve: Serve | None = None) -> PhysicsState:
    return PhysicsState(
        ball=ball,
        left_paddle=0.2,
        right_paddle=0.2,
        score=Score(left=0, right=0),
        serve=serve,
    )


def coming_toward(side: Side, y: float, vy: float = 0.0) -> PhysicsState:
    """The ball in the middle of the Court, moving toward the Paddle on `side`."""
    vx = -BASE_SPEED if side is Side.LEFT else BASE_SPEED
    return _on_court(Ball(x=0.5, y=y, vx=vx, vy=vy))


def going_away_from(side: Side, y: float, vy: float = 0.0) -> PhysicsState:
    """The ball in the middle of the Court, moving away from the Paddle on `side`."""
    vx = BASE_SPEED if side is Side.LEFT else -BASE_SPEED
    return _on_court(Ball(x=0.5, y=y, vx=vx, vy=vy))


def waiting_to_be_served(toward: Side | None) -> PhysicsState:
    """The ball at rest in the center of the Court, during a serve's wait."""
    return _on_court(
        Ball(x=0.5, y=0.5, vx=0.0, vy=0.0), Serve(frames_left=15, toward=toward)
    )


def paddle_heights(
    opponent: Opponent, side: Side, states: list[PhysicsState], paddle: float
) -> list[float]:
    """The height of the Opponent's Paddle after each Frame, from the given state at
    the start of each Frame. Its Paddle starts at `paddle` and moves by each move
    the Opponent makes."""
    heights = []
    for state in states:
        if side is Side.LEFT:
            state = dataclasses.replace(state, left_paddle=paddle)
        else:
            state = dataclasses.replace(state, right_paddle=paddle)
        paddle += opponent.move(state)
        heights.append(paddle)
    return heights


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
@pytest.mark.parametrize(
    ("paddle", "ball_y", "heights"),
    [
        # 0.11 below the ball: five Frames of 0.02, then the last 0.01.
        (0.5, 0.61, [0.52, 0.54, 0.56, 0.58, 0.6, 0.61, 0.61, 0.61]),
        # 0.05 above the ball: two Frames of 0.02, then the last 0.01.
        (0.4, 0.35, [0.38, 0.36, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35]),
    ],
)
def test_the_opponent_moves_toward_the_ball_at_most_0_02_a_frame_and_stops_on_it(
    side: Side, paddle: float, ball_y: float, heights: list[float]
) -> None:
    # With no offset width the Opponent aims its Paddle's center at the ball.
    opponent = Opponent(np.random.default_rng(0), width=0.0, side=side)

    moved_to = paddle_heights(opponent, side, [coming_toward(side, ball_y)] * 8, paddle)

    assert moved_to == pytest.approx(heights)


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
@pytest.mark.parametrize(("width", "reach"), [(1.0, 0.075), (0.5, 0.0375)])
def test_a_ball_that_comes_gets_an_offset_uniform_within_the_width(
    side: Side, width: float, reach: float
) -> None:
    # The offset is up to `reach` from the ball, `width` of half the Paddle, so the
    # Paddle gets there within 4 Frames. 1000 offsets in five equal bins across the
    # width: about 200 in each, give or take about 13, so the bounds below are more
    # than 4 of those away.
    offsets = []
    for seed in range(1000):
        opponent = Opponent(np.random.default_rng(seed), width=width, side=side)
        states = [coming_toward(side, 0.5)] * 5
        offsets.append(paddle_heights(opponent, side, states, paddle=0.5)[-1] - 0.5)
    edges = np.linspace(-reach, reach, 6)
    bins = [
        sum(low <= o < high for o in offsets) for low, high in itertools.pairwise(edges)
    ]

    assert -reach - 1e-12 <= min(offsets) and max(offsets) <= reach + 1e-12, (
        f"{min(offsets)}, {max(offsets)}"
    )
    assert all(140 <= count <= 260 for count in bins), f"bins: {bins}"


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
def test_the_offset_is_drawn_once_for_each_ball_that_comes(side: Side) -> None:
    # The ball rises 0.005 a Frame as it comes, and the Paddle catches up with the
    # ball and its offset (at most 0.075) by the 5th Frame. From then on the Paddle
    # stays the same distance from the ball.
    offsets = set()
    for seed in range(5):
        opponent = Opponent(np.random.default_rng(seed), width=1.0, side=side)
        states = [coming_toward(side, 0.3 + 0.005 * f, vy=0.005) for f in range(40)]
        heights = paddle_heights(opponent, side, states, paddle=0.3)
        gaps = [h - s.ball.y for h, s in zip(heights, states, strict=True)][5:]

        assert gaps == pytest.approx([gaps[0]] * len(gaps)), f"seed {seed}: {gaps}"
        offsets.add(gaps[0])
    assert len(offsets) == 5, f"{offsets}"


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
def test_as_the_ball_goes_away_and_waits_to_be_served_the_offset_stays(
    side: Side,
) -> None:
    # The ball comes level at 0.5, goes away falling 0.004 a Frame, then waits in
    # the center through a serve. The Paddle moves 0.02 a Frame, so it keeps up
    # with the falling ball, and gets back to it in the center within 4 Frames.
    for seed in range(5):
        opponent = Opponent(np.random.default_rng(seed), width=1.0, side=side)
        coming = [coming_toward(side, 0.5)] * 5
        away = [going_away_from(side, 0.5 - 0.004 * f, vy=-0.004) for f in range(20)]
        waiting = [waiting_to_be_served(toward=None)] * 30
        heights = paddle_heights(opponent, side, coming + away + waiting, paddle=0.5)
        offset = heights[4] - 0.5

        gaps_going_away = [
            h - s.ball.y for h, s in zip(heights[5:25], away, strict=True)
        ]
        gaps_waiting = [h - 0.5 for h in heights[29:]]
        assert gaps_going_away == pytest.approx([offset] * 20), f"seed {seed}"
        assert gaps_waiting == pytest.approx([offset] * 26), f"seed {seed}"


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
@pytest.mark.parametrize("comes_after", ["a return", "a serve toward it"])
def test_each_ball_that_comes_gets_a_new_offset(side: Side, comes_after: str) -> None:
    # The ball comes level at 0.5 twice. In between, the other side returns it, or
    # the Opponent misses it and it waits in the center to be served back.
    if comes_after == "a return":
        in_between = [going_away_from(side, 0.5)] * 5
    else:
        in_between = [waiting_to_be_served(toward=side)] * 30
    for seed in range(5):
        opponent = Opponent(np.random.default_rng(seed), width=1.0, side=side)
        coming = [coming_toward(side, 0.5)] * 5
        heights = paddle_heights(
            opponent, side, coming + in_between + coming, paddle=0.5
        )
        first, second = heights[4] - 0.5, heights[-1] - 0.5

        assert first != pytest.approx(second), f"seed {seed}: {first}, {second}"
        assert abs(second) <= 0.075 + 1e-12, f"seed {seed}: {second}"


def test_at_the_start_of_a_game_the_offset_is_0() -> None:
    # The ball waits in the center through the first serve, so the Paddle stays
    # there with it. When the serve goes away to the right, the Paddle goes on
    # following the ball itself: a serve's vertical speed is at most
    # 0.0135 * tan(30 degrees) = 0.0078 a Frame, and the Paddle keeps up with that.
    served_to_the_right = 0
    for seed in range(20):
        game = Game.new(seed=seed)
        opponent = Opponent(game.rng, width=1.0)
        for frame in range(1, 31):
            game.play_frame(left=opponent.move(game.state), right=0.0)
            assert game.state.left_paddle == 0.5, f"seed {seed}, Frame {frame}"
        if game.state.ball.vx < 0:
            continue
        served_to_the_right += 1
        for frame in range(31, 51):
            ball_y = game.state.ball.y
            game.play_frame(left=opponent.move(game.state), right=0.0)
            assert game.state.left_paddle == pytest.approx(ball_y), (
                f"seed {seed}, Frame {frame}"
            )

    assert served_to_the_right >= 5, f"{served_to_the_right} of 20"


def opponent_game(seed: int, frames: int) -> list[PhysicsState]:
    """The state after each Frame of a new Game, with the Opponent on the left, for
    `frames` Frames or until the Game ends.

    On the right, a player with the Opponent's rule but no offset returns the ball
    to it again and again, moving by the state alone."""
    game = Game.new(seed=seed)
    opponent = Opponent(game.rng)
    right = Opponent(np.random.default_rng(0), width=0.0, side=Side.RIGHT)
    states = []
    for _ in range(frames):
        result = game.play_frame(
            left=opponent.move(game.state), right=right.move(game.state)
        )
        states.append(game.state)
        if result.terminated or result.truncated:
            break
    return states


def test_the_same_seed_and_the_same_moves_give_the_same_game_with_the_opponent() -> (
    None
):
    game = opponent_game(7, 3000)
    same_seed = opponent_game(7, 3000)
    other_seed = opponent_game(8, 3000)

    balls_that_came = sum(
        before.ball.vx >= 0 > after.ball.vx
        for before, after in itertools.pairwise(game)
    )
    assert balls_that_came >= 20, f"{balls_that_came} balls came to the Opponent"
    assert same_seed == game
    assert other_seed != game
