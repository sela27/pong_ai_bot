"""The Evaluation: players in the Agent's place, measured over a batch of Games.

Most tests play fake environments whose Game with each seed ends as set, so the
expected numbers are worked out by hand. One plays the real environment.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest

from pong_ai_bot.environment import Action, PongEnv
from pong_ai_bot.evaluation import (
    EvaluationStopped,
    GameResult,
    SeedSet,
    Summary,
    evaluate,
)
from pong_ai_bot.game import Score

NO_POINTS_YET = Score(left=0, right=0)


@dataclass(frozen=True)
class Ending:
    """How the fake environment's Game with one seed ends."""

    steps: int
    """The Step the Game ends in."""
    score: Score
    """The score it ends at."""
    cut_off: bool = False
    """Whether it is cut off then, rather than ended by a side reaching 21."""


class EndingEnv(gym.Env[np.int64, int]):
    """A fake environment whose Game with each seed ends as set.

    Its Observation is the Game's seed. Its score stays at 0 to 0 until the Game's
    last Step, and is the set score from then on.
    """

    def __init__(self, endings: Mapping[int, Ending]) -> None:
        self._endings = endings

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.int64, dict[str, Any]]:
        super().reset(seed=seed)
        assert seed is not None
        self._seed, self._ending, self._steps = seed, self._endings[seed], 0
        return np.int64(seed), {"score": NO_POINTS_YET}

    def step(
        self, action: int
    ) -> tuple[np.int64, SupportsFloat, bool, bool, dict[str, Any]]:
        self._steps += 1
        ending = self._ending
        over = self._steps == ending.steps
        score = ending.score if over else NO_POINTS_YET
        terminated, truncated = over and not ending.cut_off, over and ending.cut_off
        return np.int64(self._seed), 0.0, terminated, truncated, {"score": score}


def ending_as(endings: Mapping[int, Ending]) -> Callable[[], EndingEnv]:
    """A function that builds the fake environment with these endings."""
    return lambda: EndingEnv(endings)


class Staying:
    """A fake player that keeps still in every Game."""

    def act(
        self,
        observations: npt.NDArray[Any],
        infos: Sequence[dict[str, Any]],
    ) -> list[int]:
        return [Action.STAY] * len(observations)


class Watching(Staying):
    """A fake player that keeps still, and keeps the Observations of each batch it
    is given: in the fake environment, the seeds of the Games in it."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def act(
        self,
        observations: npt.NDArray[Any],
        infos: Sequence[dict[str, Any]],
    ) -> list[int]:
        self.batches.append(observations.tolist())
        return super().act(observations, infos)


def test_all_the_games_still_going_are_played_together_until_each_ends() -> None:
    player = Watching()
    make_env = ending_as(
        {
            1: Ending(3, Score(left=21, right=0)),
            2: Ending(1, Score(left=21, right=0)),
            3: Ending(2, Score(left=21, right=0)),
        }
    )

    evaluate(player, make_env, [1, 2, 3])

    # One call for each Step: the Game of seed 2 ends in the 1st, and that of 3 in
    # the 2nd.
    assert player.batches == [[1, 2, 3], [1, 3], [1]]


def test_an_evaluation_stops_with_no_result_before_the_step_stop_says() -> None:
    # As Ctrl+C abandons a Run's Evaluation (#17): `stop` says yes before the
    # 3rd Step, and the Games of seed 1 and 3 have 3 and 4 Steps.
    player = Watching()
    make_env = ending_as(
        {
            1: Ending(3, Score(left=21, right=0)),
            2: Ending(1, Score(left=21, right=0)),
            3: Ending(4, Score(left=21, right=0)),
        }
    )
    asked = 0

    def stop() -> bool:
        nonlocal asked
        asked += 1
        return asked == 3

    with pytest.raises(EvaluationStopped):
        evaluate(player, make_env, [1, 2, 3], stop=stop)

    # Two Steps played, and the third never asked for.
    assert player.batches == [[1, 2, 3], [1, 3]]


class InfoReading(Staying):
    """A fake player that keeps still, and keeps the `info`s of each batch it is
    given."""

    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []

    def act(
        self,
        observations: npt.NDArray[Any],
        infos: Sequence[dict[str, Any]],
    ) -> list[int]:
        self.batches.append(list(infos))
        return super().act(observations, infos)


def test_each_games_info_comes_as_its_environment_gives_it() -> None:
    # The fake environment's `info` holds only the score, 0 to 0 until the Game's
    # last Step. The Game's seed is in its result, not its `info`.
    player = InfoReading()
    make_env = ending_as(
        {
            7: Ending(2, Score(left=21, right=0)),
            8: Ending(1, Score(left=21, right=0)),
            9: Ending(3, Score(left=21, right=0)),
        }
    )

    evaluation = evaluate(player, make_env, [7, 8, 9])

    no_points_yet = {"score": NO_POINTS_YET}
    assert player.batches == [[no_points_yet] * 3, [no_points_yet] * 2, [no_points_yet]]
    assert [game.seed for game in evaluation.games] == [7, 8, 9]


class TallyEnv(gym.Env[np.int64, int]):
    """A fake environment whose Game with seed n ends after n Steps, at a score
    that tallies the Actions it was given: a Point on the right for each up, and one
    on the left for each down. Its Observation is the Game's seed."""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.int64, dict[str, Any]]:
        super().reset(seed=seed)
        assert seed is not None
        self._seed, self._score = seed, NO_POINTS_YET
        return np.int64(seed), {"score": self._score}

    def step(
        self, action: int
    ) -> tuple[np.int64, SupportsFloat, bool, bool, dict[str, Any]]:
        score = self._score
        self._score = Score(
            left=score.left + (action == Action.DOWN),
            right=score.right + (action == Action.UP),
        )
        over = self._score.left + self._score.right == self._seed
        return np.int64(self._seed), 0.0, over, False, {"score": self._score}


class UpWhenOdd:
    """A fake player that moves up in the Games of odd seeds and down in the others,
    by the seed in each Game's Observation."""

    def act(
        self,
        observations: npt.NDArray[Any],
        infos: Sequence[dict[str, Any]],
    ) -> list[int]:
        return [Action.UP if seed % 2 else Action.DOWN for seed in observations]


def test_each_game_is_given_the_action_chosen_for_it() -> None:
    evaluation = evaluate(UpWhenOdd(), TallyEnv, [4, 1, 2, 3])

    # Games of 4, 1, 2 and 3 Steps, all down, up, down and up.
    assert [game.score_margin for game in evaluation.games] == [-4, 1, -2, 3]


def test_a_game_cut_off_counts_by_its_score_margin_at_the_cut() -> None:
    # 12 to 9 for the player in the Agent's place, on the right, is +3.
    make_env = ending_as({7: Ending(2, Score(left=9, right=12), cut_off=True)})

    evaluation = evaluate(Staying(), make_env, [7])

    assert evaluation.games == (GameResult(seed=7, score_margin=3, cut_off=True),)


def test_a_game_cut_off_at_a_level_score_is_not_a_win() -> None:
    make_env = ending_as(
        {
            1: Ending(1, Score(left=7, right=7), cut_off=True),
            2: Ending(1, Score(left=12, right=9), cut_off=True),
        }
    )

    summary = evaluate(Staying(), make_env, [1, 2]).summary

    # Score margins of 0 and -3: a mean of -1.5, each 1.5 away from it.
    assert summary == Summary(mean=-1.5, std=1.5, min=-3, wins=0, cut_off=2)


FOUR_GAMES = {
    1: Ending(3, Score(left=0, right=21)),
    2: Ending(1, Score(left=18, right=21)),
    3: Ending(4, Score(left=21, right=18)),
    4: Ending(2, Score(left=9, right=12), cut_off=True),
}
"""Four Games: won 21 to 0 and 21 to 18, lost 18 to 21, and cut off at 12 to 9."""


def test_the_summary_matches_the_result_of_each_seed() -> None:
    evaluation = evaluate(Staying(), ending_as(FOUR_GAMES), [1, 2, 3, 4])

    assert evaluation.games == (
        GameResult(seed=1, score_margin=21, cut_off=False),
        GameResult(seed=2, score_margin=3, cut_off=False),
        GameResult(seed=3, score_margin=-3, cut_off=False),
        GameResult(seed=4, score_margin=3, cut_off=True),
    )
    # The mean of 21, 3, -3 and 3 is 24 / 4 = 6. Their distances from it are 15, 3,
    # 9 and 3, whose squares average (225 + 9 + 81 + 9) / 4 = 81: the standard
    # deviation of these 4 Games is 9. 3 Games have a positive Score margin.
    assert evaluation.summary == Summary(mean=6.0, std=9.0, min=-3, wins=3, cut_off=1)


def test_the_json_line_holds_the_milestone_the_set_the_summary_and_each_seed() -> None:
    evaluation = evaluate(Staying(), ending_as(FOUR_GAMES), [1, 2, 3, 4])

    line = evaluation.json_line(seed_set=SeedSet.REGULAR, step=500_000)

    assert "\n" not in line
    assert json.loads(line) == {
        "step": 500_000,
        "set": "regular",
        "summary": {"mean": 6.0, "std": 9.0, "min": -3, "wins": 3, "cut_off": 1},
        "games": [
            {"seed": 1, "score_margin": 21, "cut_off": False},
            {"seed": 2, "score_margin": 3, "cut_off": False},
            {"seed": 3, "score_margin": -3, "cut_off": False},
            {"seed": 4, "score_margin": 3, "cut_off": True},
        ],
    }


def test_the_json_line_of_a_player_without_a_run_has_no_step() -> None:
    evaluation = evaluate(Staying(), ending_as(FOUR_GAMES), [1, 2, 3, 4])

    line = json.loads(evaluation.json_line(seed_set=SeedSet.REGULAR))

    assert line["step"] is None


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (
            500_000,
            (
                "Evaluation on the regular set at Step 500,000: mean +6.00, "
                "std 9.00, min -3, 3 of 4 Games won, 1 cut off"
            ),
        ),
        (
            None,
            (
                "Evaluation on the regular set: mean +6.00, std 9.00, min -3, "
                "3 of 4 Games won, 1 cut off"
            ),
        ),
    ],
)
def test_the_summary_line_for_the_terminal(step: int | None, expected: str) -> None:
    evaluation = evaluate(Staying(), ending_as(FOUR_GAMES), [1, 2, 3, 4])

    assert evaluation.summary_line(seed_set=SeedSet.REGULAR, step=step) == expected


class Late:
    """A simple player that goes for the ball only once it is past 0.6 of the
    Court, in each Game by that Game's own `info`. It loses most Points, not all."""

    def act(
        self,
        observations: npt.NDArray[Any],
        infos: Sequence[dict[str, Any]],
    ) -> list[int]:
        actions: list[int] = []
        for info in infos:
            ball, paddle = info["ball"], info["right_paddle"]
            if ball.x < 0.6:
                actions.append(Action.STAY)
            else:
                actions.append(Action.UP if ball.y > paddle else Action.DOWN)
        return actions


def test_the_same_seed_gives_the_same_result_alone_or_among_other_games() -> None:
    seeds = [1_000_000, 1_000_001, 1_000_002]

    together = evaluate(Late(), PongEnv, seeds).games
    alone = [evaluate(Late(), PongEnv, [seed]).games[0] for seed in reversed(seeds)]

    assert together == tuple(reversed(alone))


@pytest.mark.parametrize(
    ("seed_set", "first", "last"),
    [
        (SeedSet.REGULAR, 1_000_000, 1_000_099),
        (SeedSet.CONFIRMATION, 1_000_100, 1_000_199),
    ],
)
def test_each_set_of_seeds_is_100_seeds_in_a_row(
    seed_set: SeedSet, first: int, last: int
) -> None:
    assert list(seed_set.seeds) == list(range(first, last + 1))
