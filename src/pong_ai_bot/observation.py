"""Drawing the Observation: the image of the Court that the Agent sees.

The environment and the window both draw through draw_observation, so the window
shows exactly what the Agent sees (issue #6).

The image is the whole Court, OBSERVATION_SIZE pixels on each side. Rows count down
from the top wall and columns right from the left end. The background is black and
the ball and both Paddles are white, with no walls, center line or score, so the
same positions on the Court always give the same image. Each object has a fixed
size in pixels, drawn around the pixel its center falls in, which is the nearest
pixel, so the ball never vanishes or changes size. A center exactly on the line
between two pixels, like the ball's at a serve, goes to the pixel below it or to
its right. The part of the ball past an end of the Court is cut off. Collisions
stay in the continuous physics; this only shows them.
"""

import math

import numpy as np
import numpy.typing as npt

from pong_ai_bot.game import PhysicsState

OBSERVATION_SIZE = 84
"""The side of the square image, in pixels."""

BLACK = 0
WHITE = 255

_BALL_SIZE = 3
_PADDLE_HEIGHT = 13
_PADDLE_THICKNESS = 3
_PADDLE_GAP = 4
"""The empty pixels between each end of the image and the back of the Paddle there."""

_LEFT_PADDLE_COLUMN = _PADDLE_GAP + _PADDLE_THICKNESS // 2
_RIGHT_PADDLE_COLUMN = OBSERVATION_SIZE - 1 - _PADDLE_GAP - _PADDLE_THICKNESS // 2


def draw_observation(state: PhysicsState) -> npt.NDArray[np.uint8]:
    """Draw the Observation of a physics state as an 84 by 84 image of 0s and 255s."""
    image = np.full((OBSERVATION_SIZE, OBSERVATION_SIZE), BLACK, dtype=np.uint8)
    ball = state.ball
    _draw(image, _row(ball.y), _column(ball.x), _BALL_SIZE, _BALL_SIZE)
    for column, height in (
        (_LEFT_PADDLE_COLUMN, state.left_paddle),
        (_RIGHT_PADDLE_COLUMN, state.right_paddle),
    ):
        _draw(image, _row(height), column, _PADDLE_HEIGHT, _PADDLE_THICKNESS)
    return image


def _row(y: float) -> int:
    """The row that a height on the Court falls in."""
    return math.floor((1.0 - y) * OBSERVATION_SIZE)


def _column(x: float) -> int:
    """The column that a position across the Court falls in."""
    return math.floor(x * OBSERVATION_SIZE)


def _draw(
    image: npt.NDArray[np.uint8], row: int, column: int, height: int, width: int
) -> None:
    """Paint a white block of odd height and width, centered on (row, column).

    The part of the block past an edge of the image is cut off.
    """
    top = row - height // 2
    left = column - width // 2
    rows = slice(_clip(top), _clip(top + height))
    columns = slice(_clip(left), _clip(left + width))
    image[rows, columns] = WHITE


def _clip(index: int) -> int:
    """Clip a block's edge to the image, so a negative one doesn't wrap around."""
    return min(max(index, 0), OBSERVATION_SIZE)
