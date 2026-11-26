"""The Game core: one Game, Frame by Frame, in Court units.

It knows nothing about pixels, Gymnasium, Steps, or who moves the Paddles.

Positions are in Court units. The Court is 1 by 1: x grows from the left end (0) to
the right end (1), and y grows from the bottom wall (0) to the top wall (1). The
ball's position is its center, and a Paddle's height is the y of its center.
Velocities are in Court units per Frame.
"""

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

_BALL_LOWEST = BALL_SIZE / 2
_BALL_HIGHEST = 1.0 - BALL_SIZE / 2
_PADDLE_LOWEST = PADDLE_HEIGHT / 2
_PADDLE_HIGHEST = 1.0 - PADDLE_HEIGHT / 2


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
        """
        if abs(left) > MAX_PADDLE_MOVE or abs(right) > MAX_PADDLE_MOVE:
            raise ValueError(
                f"a Paddle moves at most {MAX_PADDLE_MOVE} in a Frame, "
                f"got left={left}, right={right}"
            )
        state = self._state
        ball = state.ball
        x = ball.x + ball.vx
        y, vy = ball.y + ball.vy, ball.vy
        if y > _BALL_HIGHEST:
            y, vy = 2 * _BALL_HIGHEST - y, -vy
        elif y < _BALL_LOWEST:
            y, vy = 2 * _BALL_LOWEST - y, -vy

        point_won_by = None
        score = state.score
        new_ball = Ball(x=x, y=y, vx=ball.vx, vy=vy)
        if x - BALL_SIZE / 2 >= 1.0:
            point_won_by = Side.LEFT
            score = Score(left=score.left + 1, right=score.right)
        elif x + BALL_SIZE / 2 <= 0.0:
            point_won_by = Side.RIGHT
            score = Score(left=score.left, right=score.right + 1)
        if point_won_by is not None:
            # The whole ball has left the Court. It waits in the center.
            new_ball = Ball(x=0.5, y=0.5, vx=0.0, vy=0.0)

        self._state = replace(
            state,
            ball=new_ball,
            left_paddle=_move_paddle(state.left_paddle, left),
            right_paddle=_move_paddle(state.right_paddle, right),
            score=score,
        )
        return FrameResult(point_won_by=point_won_by)


def _move_paddle(height: float, move: float) -> float:
    return min(max(height + move, _PADDLE_LOWEST), _PADDLE_HIGHEST)
