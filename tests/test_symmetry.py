"""The symmetry check: the Opponent's rule on both sides of the Court (issue #62).

It checks that the Game itself is the same for both sides: the physics, the serve
and the scoring. It isn't a Scripted player. It plays at the Game's level, every
Frame, not through the environment's Steps, so it measures nothing of what can be
won against the Opponent.
"""

import numpy as np
import pytest

from pong_ai_bot.evaluation import SeedSet
from pong_ai_bot.game import Game, Side
from pong_ai_bot.opponent import Opponent

SYMMETRY_WIDTH = 1.0
"""The offset width on both sides, fixed at 100% rather than the Opponent's locked
width: at the narrower widths two exact trackers never miss, and every Game is cut
off at 0:0 (issue #62)."""

RIGHT_SIDE_SEED = 1
"""With a Game's seed, the seed of the generator the rule on the right draws its
offsets from in that Game. The Game's own generator, from its seed alone, gives
other numbers."""


def right_side_score_margin(seed: int) -> tuple[int, bool]:
    """The right side's Score margin at the end of the Game of `seed`, and whether
    the Game was cut off."""
    game = Game.new(seed=seed)
    left = Opponent(game.rng, width=SYMMETRY_WIDTH)
    right = Opponent(
        np.random.default_rng([RIGHT_SIDE_SEED, seed]),
        width=SYMMETRY_WIDTH,
        side=Side.RIGHT,
    )
    while True:
        state = game.state
        result = game.play_frame(left=left.move(state), right=right.move(state))
        if result.terminated or result.truncated:
            score = game.state.score
            return score.right - score.left, result.truncated


@pytest.mark.slow
def test_the_opponents_rule_scores_as_well_on_the_right_as_on_the_left() -> None:
    results = [right_side_score_margin(seed) for seed in SeedSet.REGULAR.seeds]

    mean = sum(margin for margin, _ in results) / len(results)
    cut_off = sum(cut for _, cut in results)
    assert -2 <= mean <= 2, (
        f"the right side's mean Score margin is {mean:+.2f} over {len(results)} "
        f"Games, {cut_off} of them cut off"
    )
