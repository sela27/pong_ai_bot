"""The Game core: one Game, Frame by Frame, in Court units.

It knows nothing about pixels, Gymnasium, Steps, or who moves the Paddles.

A Game starts with a serve and serves again after each Point. It ends when a side
reaches WINNING_POINTS, or is cut off after GAME_FRAMES. Its randomness, in its
serves and in the Opponent, all comes from one generator made from its seed, so the
same seed and the same Paddle moves give the same Game.

Positions are in Court units. The Court is 1 by 1: x grows from the left end (0) to
the right end (1), and y grows from the bottom wall (0) to the top wall (1). The
ball's position is its center, and a Paddle's height is the y of its center.
Velocities are in Court units per Frame.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Self

import numpy as np

BALL_SIZE = 0.03
"""The side of the square ball."""

PADDLE_HEIGHT = 0.15
"""The height of each Paddle."""

PADDLE_THICKNESS = 0.03
"""The thickness of each Paddle."""

PADDLE_GAP = 0.05
"""The gap between each end of the Court and the back of the Paddle there."""

MAX_PADDLE_MOVE = 0.02
"""The most a Paddle can move up or down in one Frame."""

BASE_CROSSING_FRAMES = 60
"""How many Frames the ball takes from one Paddle to the other after a serve."""

CROSSING_SHORTENING_FRAMES = 2
"""How many Frames each Paddle hit takes off the ball's crossing."""

SHORTEST_CROSSING_FRAMES = 30
"""The fewest Frames the ball can take from one Paddle to the other."""

SERVE_FRAMES = 30
"""How many Frames the ball waits in the center before each serve."""

MAX_SERVE_ANGLE = 30.0
"""The serve's angle is uniform from minus this to this many degrees."""

WINNING_POINTS = 21
"""The Points that end a Game."""

GAME_FRAMES = 108_000
"""The Frames after which a Game is cut off, if no side has won it by then."""

_BALL_LOWEST = BALL_SIZE / 2
_BALL_HIGHEST = 1.0 - BALL_SIZE / 2
_PADDLE_LOWEST = PADDLE_HEIGHT / 2
_PADDLE_HIGHEST = 1.0 - PADDLE_HEIGHT / 2

_RIGHT_CONTACT = 1.0 - PADDLE_GAP - PADDLE_THICKNESS - BALL_SIZE / 2
"""The ball's x when it touches the right Paddle's face."""
_LEFT_CONTACT = PADDLE_GAP + PADDLE_THICKNESS + BALL_SIZE / 2
"""The ball's x when it touches the left Paddle's face."""
_CROSSING = _RIGHT_CONTACT - _LEFT_CONTACT
"""How far the ball's center goes from one Paddle's face to the other's."""
_PADDLE_REACH = PADDLE_HEIGHT / 2 + BALL_SIZE / 2
"""How far the ball's y can be from a Paddle's height and still touch it."""


class Side(Enum):
    """A side of the Court. The player in the Agent's place is on the right."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class Ball:
    """The ball's center and its velocity per Frame."""

    x: float
    y: float
    vx: float
    vy: float


_WAITING_BALL = Ball(x=0.5, y=0.5, vx=0.0, vy=0.0)
"""The ball at rest in the center of the Court, waiting to be served."""


@dataclass(frozen=True, slots=True)
class Score:
    """The Points of each side."""

    left: int
    right: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Serve:
    """A serve to come: the ball waits in the center, then leaves."""

    frames_left: int
    """How many more Frames the ball waits in the center."""
    toward: Side | None
    """The side the ball leaves toward, or None for a random side."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicsState:
    """Everything the Game knows. A Paddle's only state is its height."""

    ball: Ball
    left_paddle: float
    """The height of the left Paddle, the Opponent's side."""
    right_paddle: float
    """The height of the right Paddle, the side of the player in the Agent's place."""
    score: Score
    frame: int = 0
    """How many Frames of the Game have been played."""
    serve: Serve | None = None
    """The serve to come, or None while the ball is in play."""


@dataclass(frozen=True, slots=True)
class FrameResult:
    """What happened in one Frame."""

    point_won_by: Side | None
    """The side that won a Point in this Frame, if any."""
    terminated: bool
    """Whether the Game ended in this Frame, with a side reaching WINNING_POINTS."""
    truncated: bool
    """Whether the Game was cut off in this Frame, its GAME_FRAMES-th, unended."""


class Game:
    """One Game, Frame by Frame, with one random generator made from its seed."""

    def __init__(self, state: PhysicsState, *, seed: int) -> None:
        """A Game that goes on from an exact physics state, to test exact cases."""
        self._state = state
        self._rng = np.random.default_rng(seed)

    @classmethod
    def new(cls, *, seed: int) -> Self:
        """A new Game: the ball and both Paddles in the center, and a serve to come.

        Only a new Game puts the Paddles in the center; after a Point they stay
        where they are.
        """
        return cls(
            PhysicsState(
                ball=_WAITING_BALL,
                left_paddle=0.5,
                right_paddle=0.5,
                score=Score(left=0, right=0),
                serve=Serve(frames_left=SERVE_FRAMES, toward=None),
            ),
            seed=seed,
        )

    @property
    def state(self) -> PhysicsState:
        return self._state

    @property
    def rng(self) -> np.random.Generator:
        """The Game's one random generator, made from its seed.

        The Game draws its serves from it, and hands it to the Opponent to draw
        from too, so the same seed and the same moves give the same Game.
        """
        return self._rng

    def play_frame(self, *, left: float, right: float) -> FrameResult:
        """Play one Frame, moving each Paddle vertically by the given amount.

        A positive move is up. Each move is at most MAX_PADDLE_MOVE either way.
        The Paddles move first and then the ball, so a move counts against the
        ball in the Frame it is made.
        """
        if abs(left) > MAX_PADDLE_MOVE or abs(right) > MAX_PADDLE_MOVE:
            raise ValueError(
                f"a Paddle moves at most {MAX_PADDLE_MOVE} in a Frame, "
                f"got left={left}, right={right}"
            )
        state = self._state
        if _is_won(state) or _is_cut_off(state):
            raise RuntimeError(
                f"the Game is over, at {state.score} after {state.frame} Frames"
            )
        left_paddle = _move_paddle(state.left_paddle, left)
        right_paddle = _move_paddle(state.right_paddle, right)
        point_won_by = None
        score = state.score
        if state.serve is not None:
            ball, serve = self._wait_to_serve(state.ball, state.serve)
        else:
            ball, serve = _move_ball(state.ball, left_paddle, right_paddle), None
            point_won_by = _point_won_by(ball)
            if point_won_by is not None:
                score = _with_a_point_for(point_won_by, score)
                # The whole ball has left the Court. It waits in the center, to be
                # served toward the side that lost the Point.
                ball = _WAITING_BALL
                serve = Serve(frames_left=SERVE_FRAMES, toward=_other(point_won_by))

        self._state = replace(
            state,
            ball=ball,
            left_paddle=left_paddle,
            right_paddle=right_paddle,
            score=score,
            frame=state.frame + 1,
            serve=serve,
        )
        terminated = _is_won(self._state)
        return FrameResult(
            point_won_by=point_won_by,
            terminated=terminated,
            truncated=not terminated and _is_cut_off(self._state),
        )

    def _wait_to_serve(self, ball: Ball, serve: Serve) -> tuple[Ball, Serve | None]:
        """The ball and the serve after one more Frame of the ball's wait.

        At the end of the wait's last Frame the ball is served: it gets its speed
        where it waits, and starts moving in the next Frame.
        """
        frames_left = serve.frames_left - 1
        if frames_left > 0:
            return ball, replace(serve, frames_left=frames_left)
        toward = serve.toward
        if toward is None:
            toward = Side.LEFT if self._rng.random() < 0.5 else Side.RIGHT
        angle = self._rng.uniform(-MAX_SERVE_ANGLE, MAX_SERVE_ANGLE)
        speed = _CROSSING / BASE_CROSSING_FRAMES
        vx = speed if toward is Side.RIGHT else -speed
        return replace(ball, vx=vx, vy=speed * math.tan(math.radians(angle))), None


def _point_won_by(ball: Ball) -> Side | None:
    """The side that wins a Point once the whole ball is past the other's end."""
    if ball.x - BALL_SIZE / 2 >= 1.0:
        return Side.LEFT
    if ball.x + BALL_SIZE / 2 <= 0.0:
        return Side.RIGHT
    return None


def _with_a_point_for(side: Side, score: Score) -> Score:
    if side is Side.LEFT:
        return replace(score, left=score.left + 1)
    return replace(score, right=score.right + 1)


def _other(side: Side) -> Side:
    return Side.RIGHT if side is Side.LEFT else Side.LEFT


def _is_won(state: PhysicsState) -> bool:
    return max(state.score.left, state.score.right) >= WINNING_POINTS


def _is_cut_off(state: PhysicsState) -> bool:
    return state.frame >= GAME_FRAMES


def _move_ball(ball: Ball, left_paddle: float, right_paddle: float) -> Ball:
    """Move the ball through one Frame, off the walls and the Paddles' faces."""
    x = ball.x + ball.vx
    if ball.x <= _RIGHT_CONTACT < x:
        contact, paddle = _RIGHT_CONTACT, right_paddle
    elif x < _LEFT_CONTACT <= ball.x:
        contact, paddle = _LEFT_CONTACT, left_paddle
    else:
        # The ball doesn't reach a face in this Frame, or it is already past one
        # and so never touches it.
        y, vy = _move_vertically(ball.y, ball.vy, 1.0)
        return Ball(x=x, y=y, vx=ball.vx, vy=vy)

    # The ball reaches the face's line part way through the Frame, and comes back
    # from it for the rest of the Frame.
    before = (contact - ball.x) / ball.vx
    hit_at, vy = _move_vertically(ball.y, ball.vy, before)
    if abs(hit_at - paddle) > _PADDLE_REACH:
        y, vy = _move_vertically(ball.y, ball.vy, 1.0)
        return Ball(x=x, y=y, vx=ball.vx, vy=vy)
    vx = _faster_back(ball.vx)
    vy = _vertical_speed_off(paddle, hit_at, vx)
    y, vy = _move_vertically(hit_at, vy, 1.0 - before)
    return Ball(x=contact + vx * (1.0 - before), y=y, vx=vx, vy=vy)


def _move_vertically(y: float, vy: float, frames: float) -> tuple[float, float]:
    """Move for part of a Frame, bouncing off the walls like a mirror."""
    y += vy * frames
    if y > _BALL_HIGHEST:
        return 2 * _BALL_HIGHEST - y, -vy
    if y < _BALL_LOWEST:
        return 2 * _BALL_LOWEST - y, -vy
    return y, vy


def _faster_back(vx: float) -> float:
    """The ball's vx after a Paddle hit: back the other way, on a shorter crossing."""
    frames = _CROSSING / abs(vx) - CROSSING_SHORTENING_FRAMES
    frames = max(frames, SHORTEST_CROSSING_FRAMES)
    return -math.copysign(_CROSSING / frames, vx)


def _vertical_speed_off(paddle: float, hit_at: float, vx: float) -> float:
    """The ball's vy after it touches a Paddle at height `hit_at`, leaving with `vx`.

    Only where it touches sets the angle: 0 at the Paddle's center, 45 degrees at
    its edges, and linear between, up from the top half and down from the bottom.
    """
    # From -1 at the bottom edge to 1 at the top edge. A corner counts as an edge.
    place = min(max((hit_at - paddle) / (PADDLE_HEIGHT / 2), -1.0), 1.0)
    return abs(vx) * math.tan(math.radians(45.0 * place))


def _move_paddle(height: float, move: float) -> float:
    return min(max(height + move, _PADDLE_LOWEST), _PADDLE_HIGHEST)
