"""Human vs the Opponent, Frame by Frame, from the keys the Human holds.

Positions are in Court units, y from the bottom wall (0) to the top wall (1). Expected
numbers are worked out by hand.
"""

import pytest

from pong_ai_bot.game import Game
from pong_ai_bot.opponent import Opponent
from pong_ai_bot.play import HumanVsOpponent


@pytest.mark.parametrize(
    ("up", "down", "heights"),
    [
        (True, False, [0.52, 0.54, 0.56]),
        (False, True, [0.48, 0.46, 0.44]),
        (False, False, [0.5, 0.5, 0.5]),
        # Both keys at once cancel out.
        (True, True, [0.5, 0.5, 0.5]),
    ],
)
def test_the_humans_paddle_on_the_right_moves_0_02_a_frame_by_the_keys_held(
    up: bool, down: bool, heights: list[float]
) -> None:
    # A new Game starts with both Paddles in the center, at 0.5.
    play = HumanVsOpponent(seed=0)

    moved_to = []
    for _ in range(3):
        play.play_frame(up=up, down=down)
        moved_to.append(play.state.right_paddle)

    assert moved_to == pytest.approx(heights)


def test_the_opponent_plays_the_left_paddle_as_in_training_to_the_end_of_the_game() -> (
    None
):
    # The Game core with the Opponent on the left, drawing from the Game's generator,
    # as the Opponent plays everywhere else. The Human holds no key, so the
    # Opponent wins most Points, and the Game ends at 21 within a few thousand
    # Frames.
    game = Game.new(seed=3)
    opponent = Opponent(game.rng)
    play = HumanVsOpponent(seed=3)

    for frame in range(1, 20_001):
        expected = game.play_frame(left=opponent.move(game.state), right=0.0)
        result = play.play_frame(up=False, down=False)

        assert (result, play.state) == (expected, game.state), f"Frame {frame}"
        if result.terminated:
            break
    assert 21 in (play.state.score.left, play.state.score.right), (
        f"{play.state.score} after {play.state.frame} Frames"
    )
