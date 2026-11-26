"""Choosing the Opponent's offset width by a written rule, not by feel (issue #7).

Each of three Scripted players plays an Evaluation on the regular set against an
Opponent of each of five offset widths. A width passes when the edge predictor's
mean Score margin is at least +20, the copy of the Opponent's is between -2 and +2,
and the random player's is below -19. Of the widths that pass, the hardest is
chosen: the one the edge predictor's mean is lowest at. If none passes, none is.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pong_ai_bot.evaluation import Evaluation

OFFSET_WIDTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
"""The offset widths tried, as fractions of half the Paddle's height."""

EvaluateAt = Callable[[str, float], Evaluation]
"""Plays the Evaluation of the Scripted player of this name, by the name `pong
evaluate` knows it by, against an Opponent of this offset width."""


@dataclass(frozen=True, slots=True)
class _Condition:
    """What one Scripted player's mean must be for a width to pass."""

    player: str
    passes: Callable[[float], bool]
    failure: str
    """What fails, when the mean doesn't pass."""


_CONDITIONS = (
    _Condition("random", lambda mean: mean < -19, "random not below -19"),
    _Condition(
        "opponent-copy",
        lambda mean: -2 <= mean <= 2,
        "opponent-copy outside -2 to +2",
    ),
    _Condition("edge-predictor", lambda mean: mean >= 20, "edge-predictor below +20"),
)
_HARDEST_BY = "edge-predictor"
"""The Scripted player whose mean tells how hard a width is: the lower, the
harder."""


@dataclass(frozen=True, slots=True)
class WidthResult:
    """How the Scripted players did against an Opponent of one offset width."""

    width: float
    means: Mapping[str, float]
    """Each Scripted player's mean Score margin, by its name."""

    @property
    def failed(self) -> tuple[str, ...]:
        """What fails at this width, if anything."""
        return tuple(
            condition.failure
            for condition in _CONDITIONS
            if not condition.passes(self.means[condition.player])
        )


@dataclass(frozen=True, slots=True)
class WidthChoice:
    """The Scripted players' means at every offset width tried, and the width
    chosen from them."""

    results: tuple[WidthResult, ...]
    """Each width's results, from the narrowest width to the widest."""

    @property
    def chosen(self) -> float | None:
        """The width chosen, or None if no width passes."""
        chosen = self._chosen()
        return None if chosen is None else chosen.width

    def report(self) -> str:
        """A table of each Scripted player's mean at each width, with what fails
        there, and the width chosen."""
        players = [condition.player for condition in _CONDITIONS]
        lines = ["  ".join([f"{'width':>5}", *(f"{name:>6}" for name in players)])]
        for result in self.results:
            means = (
                f"{result.means[name]:>+{max(len(name), 6)}.2f}" for name in players
            )
            verdict = (
                "fails: " + ", ".join(result.failed) if result.failed else "passes"
            )
            lines.append("  ".join([f"{result.width:>5.0%}", *means, verdict]))
        lines.append("")
        chosen = self._chosen()
        if chosen is None:
            lines.append("No width passes, so none is chosen.")
            return "\n".join(lines)
        lowest = chosen.means[_HARDEST_BY]
        verdict = (
            f"Chosen: {chosen.width:.0%} of half the Paddle's height, the width that "
            f"passes with the lowest {_HARDEST_BY} mean, {lowest:+.2f}."
        )
        ties = [
            result
            for result in self._passing()
            if result is not chosen and result.means[_HARDEST_BY] == lowest
        ]
        if ties:
            verdict += (
                " It ties with "
                + ", ".join(f"{result.width:.0%}" for result in ties)
                + ", and of the widths that tie, the narrowest is chosen."
            )
        lines.append(verdict)
        return "\n".join(lines)

    def _passing(self) -> list[WidthResult]:
        return [result for result in self.results if not result.failed]

    def _chosen(self) -> WidthResult | None:
        # Of widths that tie, the first, the narrowest: the rule doesn't say which.
        passing = self._passing()
        if not passing:
            return None
        return min(passing, key=lambda result: result.means[_HARDEST_BY])


def choose_width(evaluate_at: EvaluateAt) -> WidthChoice:
    """Play each Scripted player of the rule at each width, through
    `evaluate_at`, and choose the width by the rule."""
    return WidthChoice(
        results=tuple(
            WidthResult(
                width=width,
                means={
                    condition.player: evaluate_at(condition.player, width).summary.mean
                    for condition in _CONDITIONS
                },
            )
            for width in OFFSET_WIDTHS
        )
    )
