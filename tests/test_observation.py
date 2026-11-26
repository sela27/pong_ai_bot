"""Drawing the Observation from a physics state.

The image is the Court, 84 by 84 pixels, so a pixel is 1/84 of the Court's side.
Rows count down from the top wall and columns right from the left end, while the
physics y grows upward. A ball or Paddle is drawn around the pixel its center falls
in. Expected pixels are worked out by hand.
"""

import numpy as np
import numpy.typing as npt
import pytest

from pong_ai_bot.game import Ball, PhysicsState, Score
from pong_ai_bot.observation import draw_observation

Pixel = tuple[int, int]
"""A (row, column) of the image."""


def a_state(
    ball_x: float, ball_y: float, left_paddle: float = 0.5, right_paddle: float = 0.5
) -> PhysicsState:
    return PhysicsState(
        ball=Ball(x=ball_x, y=ball_y, vx=0.0135, vy=0.0),
        left_paddle=left_paddle,
        right_paddle=right_paddle,
        score=Score(left=0, right=0),
    )


def block(rows: range, columns: range) -> set[Pixel]:
    return {(row, column) for row in rows for column in columns}


def white_pixels(image: npt.NDArray[np.uint8]) -> set[Pixel]:
    """The white pixels, after checking the image is 84 by 84 and only black or white."""
    assert image.shape == (84, 84)
    assert image.dtype == np.uint8
    assert set(np.unique(image).tolist()) <= {0, 255}
    rows, columns = np.nonzero(image)
    return set(zip(rows.tolist(), columns.tolist(), strict=True))


# A Paddle at height 0.5 has its center 42 pixels below the top wall, on the line
# between rows 41 and 42, so it falls in row 42: 13 rows from 36 to 48. The left
# Paddle leaves columns 0-3 empty and takes 4-6; the right one takes 77-79 and
# leaves 80-83 empty.
LEFT_PADDLE_AT_CENTER = block(range(36, 49), range(4, 7))
RIGHT_PADDLE_AT_CENTER = block(range(36, 49), range(77, 80))


def test_the_ball_and_both_paddles_at_the_center() -> None:
    # The ball's center is 42 pixels from the left end and from the top wall,
    # so it falls in column 42 and row 42: a 3 by 3 block from 41 to 43.
    image = draw_observation(a_state(ball_x=0.5, ball_y=0.5))

    assert white_pixels(image) == (
        block(range(41, 44), range(41, 44))
        | LEFT_PADDLE_AT_CENTER
        | RIGHT_PADDLE_AT_CENTER
    )


def test_each_object_is_drawn_around_the_nearest_pixel_with_up_at_the_top() -> None:
    # Ball: 0.26 is 21.84 pixels from the left end, nearest to the center of
    # column 21 (21.5), so columns 20-22. 0.8 is 0.2 below the top wall, 16.8
    # pixels, nearest to row 16, so rows 15-17.
    # Left Paddle: 0.2 is 0.8 below the top, 67.2 pixels: row 67, rows 61-73.
    # Right Paddle: 0.7 is 0.3 below the top, 25.2 pixels: row 25, rows 19-31.
    image = draw_observation(
        a_state(ball_x=0.26, ball_y=0.8, left_paddle=0.2, right_paddle=0.7)
    )

    assert white_pixels(image) == (
        block(range(15, 18), range(20, 23))
        | block(range(61, 74), range(4, 7))
        | block(range(19, 32), range(77, 80))
    )


@pytest.mark.parametrize(
    ("ball_x", "columns"),
    [
        # 83.58 pixels: column 83, so 82-84, and 84 is past the edge.
        pytest.param(0.995, range(82, 84), id="partly past the right end"),
        # 84.84 pixels: column 84, so 83-85, and only 83 is in the image.
        pytest.param(1.01, range(83, 84), id="mostly past the right end"),
        # 85.176 pixels: column 85, so 84-86, all past the edge.
        pytest.param(1.014, range(0), id="drawn past the right end"),
        # 0.42 pixels: column 0, so -1 to 1, and -1 is past the edge.
        pytest.param(0.005, range(2), id="partly past the left end"),
        # -0.42 pixels: column -1, so -2 to 0, and only 0 is in the image.
        pytest.param(-0.005, range(1), id="mostly past the left end"),
        # -1.176 pixels: column -2, so -3 to -1, all past the edge.
        pytest.param(-0.014, range(0), id="drawn past the left end"),
    ],
)
def test_a_ball_leaving_the_court_is_cut_at_the_edge_of_the_image(
    ball_x: float, columns: range
) -> None:
    # These are all still in play: a Point is won only once the ball's whole
    # 0.03 is out, past 1.015 or -0.015.
    image = draw_observation(a_state(ball_x=ball_x, ball_y=0.5))

    assert white_pixels(image) == (
        block(range(41, 44), columns) | LEFT_PADDLE_AT_CENTER | RIGHT_PADDLE_AT_CENTER
    )


def test_the_same_positions_always_give_the_same_image() -> None:
    state = a_state(ball_x=0.26, ball_y=0.8, left_paddle=0.2, right_paddle=0.7)
    first = draw_observation(state)
    expected = first.copy()
    first[:, :] = 255  # whoever holds an Observation may write on it

    again = draw_observation(state)
    # The score and the ball's velocity are not part of the image.
    other_score_and_velocity = draw_observation(
        PhysicsState(
            ball=Ball(x=0.26, y=0.8, vx=-0.027, vy=0.01),
            left_paddle=0.2,
            right_paddle=0.7,
            score=Score(left=20, right=7),
        )
    )

    np.testing.assert_array_equal(again, expected)
    np.testing.assert_array_equal(other_score_and_velocity, expected)
