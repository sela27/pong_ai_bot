"""Choosing the Opponent's offset width, from fake Evaluations whose Score margins
are set by hand, so each mean is worked out by hand too: (21, 20) is +20.50.

The real Evaluations, 15 of 100 Games each, are a script, `pong choose-width`, and
not a test.
"""

from collections.abc import Mapping, Sequence

import pytest

from pong_ai_bot.evaluation import Evaluation, GameResult
from pong_ai_bot.width_choice import choose_width

RANDOM, COPY, PREDICTOR = "random", "opponent-copy", "edge-predictor"


def evaluation_of(margins: Sequence[int]) -> Evaluation:
    """An Evaluation whose Games ended with these Score margins."""
    return Evaluation(
        games=tuple(
            GameResult(seed=seed, score_margin=margin, cut_off=False)
            for seed, margin in enumerate(margins)
        )
    )


class FakeEvaluations:
    """Plays no Games: gives, at each width, each Scripted player an Evaluation with
    the Score margins set for it, and keeps which it was asked for."""

    def __init__(self, margins: Mapping[float, Mapping[str, Sequence[int]]]) -> None:
        self._margins = margins
        self.played: list[tuple[str, float]] = []

    def __call__(self, player: str, width: float) -> Evaluation:
        self.played.append((player, width))
        return evaluation_of(self._margins[width][player])


SOME_PASS = {
    0.0: {RANDOM: [-21], COPY: [0], PREDICTOR: [21]},
    # Passes, with the predictor at +20.50, the lowest of the widths that do.
    0.25: {RANDOM: [-21, -20], COPY: [1, 2], PREDICTOR: [21, 20]},
    # The predictor's +19.50 is lower, but below +20.
    0.5: {RANDOM: [-21], COPY: [0], PREDICTOR: [19, 20]},
    0.75: {RANDOM: [-21], COPY: [-1], PREDICTOR: [21]},
    # The predictor's +20.00 is lower, but the copy's +3.00 is above +2.
    1.0: {RANDOM: [-21], COPY: [3], PREDICTOR: [20]},
}
"""Fake Evaluations at which three of the five widths pass."""


def test_it_chooses_the_width_the_predictor_does_worst_at_of_those_that_pass() -> None:
    evaluations = FakeEvaluations(SOME_PASS)

    choice = choose_width(evaluations)

    assert choice.chosen == 0.25
    assert sorted(evaluations.played) == sorted(
        (player, width)
        for player in (RANDOM, COPY, PREDICTOR)
        for width in (0.0, 0.25, 0.5, 0.75, 1.0)
    )


@pytest.mark.parametrize(
    ("player", "margins", "passes"),
    [
        # At least +20.
        (PREDICTOR, [20], True),
        (PREDICTOR, [20, 19], False),
        # Between -2 and +2, both included.
        (COPY, [-2], True),
        (COPY, [2], True),
        (COPY, [-2, -3], False),
        (COPY, [2, 3], False),
        # Below -19.
        (RANDOM, [-19, -20], True),
        (RANDOM, [-19], False),
    ],
)
def test_the_rule_s_bounds(player: str, margins: list[int], passes: bool) -> None:
    # The same at every width: each Scripted player well inside its bound, but one.
    at_each_width = {RANDOM: [-21], COPY: [0], PREDICTOR: [21], player: margins}
    evaluations = FakeEvaluations(
        {width: at_each_width for width in (0.0, 0.25, 0.5, 0.75, 1.0)}
    )

    choice = choose_width(evaluations)

    assert (choice.chosen is not None) == passes, choice.chosen


def test_the_report_is_a_table_of_the_means_at_each_width_and_the_width_chosen() -> (
    None
):
    choice = choose_width(FakeEvaluations(SOME_PASS))

    assert choice.report() == (
        "width  random  opponent-copy  edge-predictor\n"
        "   0%  -21.00          +0.00          +21.00  passes\n"
        "  25%  -20.50          +1.50          +20.50  passes\n"
        "  50%  -21.00          +0.00          +19.50  fails: edge-predictor below "
        "+20\n"
        "  75%  -21.00          -1.00          +21.00  passes\n"
        " 100%  -21.00          +3.00          +20.00  fails: opponent-copy outside "
        "-2 to +2\n"
        "\n"
        "Chosen: 25% of half the Paddle's height, the width that passes with the "
        "lowest edge-predictor mean, +20.50."
    )


def test_of_widths_that_tie_it_chooses_the_narrowest_and_says_they_tie() -> None:
    # The rule doesn't say which, so the report shows the tie.
    choice = choose_width(
        FakeEvaluations(
            {
                0.0: {RANDOM: [-21], COPY: [3], PREDICTOR: [21]},
                0.25: {RANDOM: [-21], COPY: [0], PREDICTOR: [21]},
                0.5: {RANDOM: [-21], COPY: [0], PREDICTOR: [21, 20]},
                0.75: {RANDOM: [-21], COPY: [0], PREDICTOR: [20, 21]},
                1.0: {RANDOM: [-21], COPY: [0], PREDICTOR: [21]},
            }
        )
    )

    assert choice.chosen == 0.5
    assert choice.report().splitlines()[-1] == (
        "Chosen: 50% of half the Paddle's height, the width that passes with the "
        "lowest edge-predictor mean, +20.50. It ties with 75%, and of the widths "
        "that tie, the narrowest is chosen."
    )


def test_when_no_width_passes_it_chooses_none_and_says_what_fails_at_each() -> None:
    choice = choose_width(
        FakeEvaluations(
            {
                0.0: {RANDOM: [-21], COPY: [-3], PREDICTOR: [21]},
                0.25: {RANDOM: [-19], COPY: [0], PREDICTOR: [21]},
                0.5: {RANDOM: [-21], COPY: [5, 6], PREDICTOR: [21]},
                0.75: {RANDOM: [-21], COPY: [3], PREDICTOR: [19]},
                1.0: {RANDOM: [-18], COPY: [4], PREDICTOR: [17, 18]},
            }
        )
    )

    assert choice.chosen is None
    assert choice.report() == (
        "width  random  opponent-copy  edge-predictor\n"
        "   0%  -21.00          -3.00          +21.00  fails: opponent-copy outside "
        "-2 to +2\n"
        "  25%  -19.00          +0.00          +21.00  fails: random not below -19\n"
        "  50%  -21.00          +5.50          +21.00  fails: opponent-copy outside "
        "-2 to +2\n"
        "  75%  -21.00          +3.00          +19.00  fails: opponent-copy outside "
        "-2 to +2, edge-predictor below +20\n"
        " 100%  -18.00          +4.00          +17.50  fails: random not below -19, "
        "opponent-copy outside -2 to +2, edge-predictor below +20\n"
        "\n"
        "No width passes, so none is chosen."
    )
