"""The Game core: one Game, Frame by Frame, in Court units.

It knows nothing about pixels, Gymnasium, Steps, or who moves the Paddles.

Positions are in Court units. The Court is 1 by 1: x grows from the left end (0) to
the right end (1), and y grows from the bottom wall (0) to the top wall (1). The
ball's position is its center, and a Paddle's height is the y of its center.
Velocities are in Court units per Frame.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum

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

CROSSING_SHORTENING_FRAMES = 2
"""How many Frames each Paddle hit takes off the ball's crossing."""

SHORTEST_CROSSING_FRAMES = 30
"""The fewest Frames the ball can take from one Paddle to the other."""

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


@dataclass(frozen=True, slots=True)
class Score:
    """The Points of each side."""

    left: int
    right: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicsState:
    """Everything the Game knows. A Paddle's only state is its height."""

    ball: Ball
    left_paddle: float
    """The height of the left Paddle, the Opponent's side."""
    right_paddle: float
    """The height of the right Paddle, the side of the player in the Agent's place."""
    score: Score


@dataclass(frozen=True, slots=True)
class FrameResult:
    """What happened in one Frame."""

    point_won_by: Side | None
    """The side that won a Point in this Frame, if any."""


class Game:
    """One Game, created from an exact physics state."""

    def __init__(self, state: PhysicsState) -> None:
        self._state = state

    @property
    def state(self) -> PhysicsState:
        return self._state

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
        left_paddle = _move_paddle(state.left_paddle, left)
        right_paddle = _move_paddle(state.right_paddle, right)
        new_ball = _move_ball(state.ball, left_paddle, right_paddle)

        point_won_by = None
        score = state.score
        if new_ball.x - BALL_SIZE / 2 >= 1.0:
            point_won_by = Side.LEFT
            score = Score(left=score.left + 1, right=score.right)
        elif new_ball.x + BALL_SIZE / 2 <= 0.0:
            point_won_by = Side.RIGHT
            score = Score(left=score.left, right=score.right + 1)
        if point_won_by is not None:
            # The whole ball has left the Court. It waits in the center.
            new_ball = Ball(x=0.5, y=0.5, vx=0.0, vy=0.0)

        self._state = replace(
            state,
            ball=new_ball,
            left_paddle=left_paddle,
            right_paddle=right_paddle,
            score=score,
        )
        return FrameResult(point_won_by=point_won_by)


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
