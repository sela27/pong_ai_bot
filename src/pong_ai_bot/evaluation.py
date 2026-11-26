"""The Evaluation: how well a player in the Agent's place plays (issue #13).

Every player in the Agent's place, the Agent, the Position learner and the Scripted
players, is measured the same way: one Game against the Opponent for each seed, all
of them played together, Step by Step, with one call to the player's `act` for all
the Games still going. A Game that ends leaves the batch. Each player comes with
the function that builds its environment, wrapped as it needs.

A Game counts by its Score margin at its end, read from the score in the last
Step's `info`, even if it was cut off before either side reached 21: 12 to 9 for
the player is +3.

The seeds are a parameter, so a fast test can play a few Games, but there are two
fixed sets of them: the regular set, and the Confirmation's.
"""

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from pong_ai_bot.game import Score


class Player(Protocol):
    """A player in the Agent's place, choosing an Action for each of a batch of
    Games."""

    def act(
        self, observations: npt.NDArray[Any], infos: Sequence[dict[str, Any]]
    ) -> npt.ArrayLike:
        """The Actions for a batch of Games, one for each, in the batch's order.

        `observations` holds each Game's Observation from its environment, stacked
        along a new first axis, and `infos` each Game's `info`. The Agent reads only
        the Observations, and the Scripted players only the `info`s. The Actions
        may be a list or a NumPy array.
        """
        ...


EnvMaker = Callable[[], gym.Env[Any, Any]]
"""A function that builds a player its environment, for one Game at a time.

Whatever it wraps around the environment, the `info` it gives holds the score, as
the environment's own does.
"""


class SeedSet(StrEnum):
    """The two fixed sets of seeds, one seed for each Game. Every seed of a Run is
    below them, under 1,000,000."""

    REGULAR = "regular"
    """The set every Evaluation of a Milestone, or of a Scripted player, plays."""
    CONFIRMATION = "confirmation"
    """The set only the Confirmation plays, once, at the end of a Run (issue #16)."""

    @property
    def seeds(self) -> range:
        """The set's 100 seeds."""
        if self is SeedSet.REGULAR:
            return range(1_000_000, 1_000_100)
        return range(1_000_100, 1_000_200)


@dataclass(frozen=True, slots=True)
class GameResult:
    """How one Game of an Evaluation ended."""

    seed: int
    score_margin: int
    """The player's Points minus the Opponent's, at the end of the Game."""
    cut_off: bool
    """Whether the Game was cut off, before either side reached 21."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Summary:
    """An Evaluation's Games in a few numbers. There is no maximum: it is nearly
    always 21."""

    mean: float
    """The mean Score margin."""
    std: float
    """The standard deviation of the Score margins, of these Games themselves
    (NumPy's default, ddof=0)."""
    min: int
    """The lowest Score margin."""
    wins: int
    """How many Games the player won: ended with a positive Score margin, cut off
    or not."""
    cut_off: int
    """How many Games were cut off."""


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The result of an Evaluation."""

    games: tuple[GameResult, ...]
    """Each Game's result, in the order of the seeds."""

    @property
    def summary(self) -> Summary:
        margins = np.array([game.score_margin for game in self.games])
        return Summary(
            mean=float(margins.mean()),
            std=float(margins.std()),
            min=int(margins.min()),
            wins=int((margins > 0).sum()),
            cut_off=sum(game.cut_off for game in self.games),
        )

    def json_line(self, *, seed_set: SeedSet, step: int | None = None) -> str:
        """The Evaluation as one line of JSON, as `evaluations.jsonl` keeps it.

        `step` is the Run's Step at the Milestone evaluated, or None for a player
        without a Run.
        """
        return json.dumps(
            {
                "step": step,
                "set": seed_set.value,
                "summary": asdict(self.summary),
                "games": [asdict(game) for game in self.games],
            }
        )

    def summary_line(self, *, seed_set: SeedSet, step: int | None = None) -> str:
        """The Evaluation's summary, in one line for the terminal."""
        where = f"the {seed_set} set" + ("" if step is None else f" at Step {step:,}")
        summary = self.summary
        return (
            f"Evaluation on {where}: mean {summary.mean:+.2f}, std {summary.std:.2f}, "
            f"min {summary.min:+d}, {summary.wins} of {len(self.games)} Games won, "
            f"{summary.cut_off} cut off"
        )


def evaluate(player: Player, make_env: EnvMaker, seeds: Iterable[int]) -> Evaluation:
    """Play `player` one Game for each seed, all together, in environments from
    `make_env` reset with that seed."""
    seeds = list(seeds)
    envs = [make_env() for _ in seeds]
    observations: list[Any] = []
    infos: list[dict[str, Any]] = []
    for env, seed in zip(envs, seeds, strict=True):
        observation, info = env.reset(seed=seed)
        observations.append(observation)
        infos.append(info)
    results: dict[int, GameResult] = {}
    going = list(range(len(seeds)))
    while going:
        actions = player.act(
            np.stack([observations[game] for game in going]),
            [infos[game] for game in going],
        )
        still_going = []
        for game, action in zip(going, np.asarray(actions), strict=True):
            observation, _, terminated, truncated, info = envs[game].step(action)
            observations[game], infos[game] = observation, info
            if terminated or truncated:
                score: Score = info["score"]
                results[game] = GameResult(
                    seed=seeds[game],
                    score_margin=score.right - score.left,
                    cut_off=not terminated,
                )
            else:
                still_going.append(game)
        going = still_going
    return Evaluation(games=tuple(results[game] for game in range(len(seeds))))
