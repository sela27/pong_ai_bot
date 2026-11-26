"""The Scripted players, through `act`, on batches of real Games' Observations and
`info`s."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pong_ai_bot.environment import Action, PongEnv
from pong_ai_bot.game import Ball, Score
from pong_ai_bot.scripted_players import OpponentCopy, RandomPlayer


def new_games(seeds: list[int]) -> tuple[npt.NDArray[Any], list[dict[str, Any]]]:
    """The first Observations and `info`s of the Games of these seeds, as a batch."""
    starts = [PongEnv().reset(seed=seed) for seed in seeds]
    observations = np.stack([observation for observation, _ in starts])
    return observations, [info for _, info in starts]


def test_the_random_player_chooses_one_action_for_each_game_in_the_batch() -> None:
    observations, infos = new_games([1, 2, 3, 4, 5])

    actions = np.asarray(RandomPlayer().act(observations, infos))

    assert actions.shape == (5,)
    assert set(actions.tolist()) <= {Action.UP, Action.DOWN, Action.STAY}


def test_each_action_comes_up_about_a_third_of_the_time() -> None:
    player = RandomPlayer()
    observations, infos = new_games(list(range(10)))

    # 300 Steps of 10 Games: 3,000 Actions, 1,000 of each if they came up evenly.
    actions = np.concatenate(
        [np.asarray(player.act(observations, infos)) for _ in range(300)]
    )

    counts = {action: int((actions == action).sum()) for action in Action}
    assert all(900 <= count <= 1_100 for count in counts.values()), counts


def test_every_new_random_player_chooses_the_same_actions() -> None:
    # Its seed is fixed, so its Evaluation gives the same results every time.
    first, second = RandomPlayer(), RandomPlayer()
    observations, infos = new_games(list(range(10)))

    for _ in range(20):
        first_actions = np.asarray(first.act(observations, infos)).tolist()
        second_actions = np.asarray(second.act(observations, infos)).tolist()
        assert first_actions == second_actions


# The copy of the Opponent, from known physics states in `info`. Positions are in
# Court units, y from the bottom wall (0) to the top wall (1). Holding an Action for
# a Step's 4 Frames moves the Paddle 4 x 0.02 = 0.08, and the Opponent's rule would
# move it toward its target by at most that much, stopping right on it.

BASE_SPEED = 0.0135


def info_of(
    seed: int, *, ball_y: float, right_paddle: float, vx: float = BASE_SPEED
) -> dict[str, Any]:
    """The `info` of the Game of `seed`, as the Evaluation hands it: the ball in the
    middle of the Court at `ball_y`, coming toward the right Paddle unless `vx` is
    negative."""
    return {
        "ball": Ball(x=0.5, y=ball_y, vx=vx, vy=0.0),
        "left_paddle": 0.5,
        "right_paddle": right_paddle,
        "score": Score(left=0, right=0),
        "seed": seed,
    }


def actions_of(player: OpponentCopy, infos: list[dict[str, Any]]) -> list[int]:
    """The Actions the player chooses for a batch of Games, from their `info`s."""
    observations = np.zeros((len(infos), 84, 84), dtype=np.uint8)
    return np.asarray(player.act(observations, infos)).tolist()


@pytest.mark.parametrize(
    ("ball_y", "action"),
    [
        # The rule would end at 0.58: up ends there, staying 0.08 short.
        (0.6, Action.UP),
        # The rule would end at 0.53: staying ends 0.03 short, up 0.05 past.
        (0.53, Action.STAY),
        (0.5, Action.STAY),
        # The rule would end at 0.47: staying ends 0.03 short, down 0.05 past.
        (0.47, Action.STAY),
        # The rule would end at 0.45: down ends 0.03 past, staying 0.05 short.
        (0.45, Action.DOWN),
        # The rule would end at 0.42: down ends there.
        (0.2, Action.DOWN),
    ],
)
def test_the_copy_chooses_the_action_that_ends_the_step_nearest_where_the_rule_would(
    ball_y: float, action: Action
) -> None:
    # With no offset width, the rule aims its Paddle's center at the ball.
    player = OpponentCopy(width=0.0)

    actions = actions_of(player, [info_of(1, ball_y=ball_y, right_paddle=0.5)])

    assert actions == [action]


def level_and_coming(seeds: list[int], vx: float = BASE_SPEED) -> list[dict[str, Any]]:
    """The `info`s of the Games of these seeds, each with its Paddle level with the
    ball, at 0.5, and the ball coming toward it unless `vx` is negative."""
    return [info_of(seed, ball_y=0.5, right_paddle=0.5, vx=vx) for seed in seeds]


@pytest.mark.parametrize(
    # The fewest and the most Games that move up, move down and stay.
    ("width", "fewest_moving", "most_moving"),
    [
        # Offsets up to 0.075 from the ball. It moves up for an offset over 0.04,
        # 0.035 of the 0.15 they spread over: 7 in 30, about 233 of 1000 Games, give
        # or take about 13. As many move down, and 16 in 30 stay, about 533 give or
        # take about 16. The bounds below are 4 of those away.
        (1.0, (180, 180, 470), (287, 287, 597)),
        # Offsets up to 0.0375, never over 0.04: it never moves.
        (0.5, (0, 0, 1000), (0, 0, 1000)),
    ],
)
def test_level_with_a_ball_that_comes_the_copy_moves_by_its_offset_within_the_width(
    width: float,
    fewest_moving: tuple[int, int, int],
    most_moving: tuple[int, int, int],
) -> None:
    player = OpponentCopy(width=width)

    actions = actions_of(player, level_and_coming(list(range(1000))))

    counts = tuple(actions.count(action) for action in Action)
    assert all(
        low <= count <= high
        for low, count, high in zip(fewest_moving, counts, most_moving, strict=True)
    ), f"up, down and stay: {counts}"


def test_every_new_copy_chooses_the_same_for_each_game_alone_or_among_others() -> None:
    # Its offsets are drawn for each Game by the Game's seed, so its Evaluation
    # gives the same results every time. About half the Games move.
    seeds = list(range(200))

    together = actions_of(OpponentCopy(width=1.0), level_and_coming(seeds))
    alone = [
        actions_of(OpponentCopy(width=1.0), level_and_coming([seed]))[0]
        for seed in seeds
    ]
    reversed_batch = actions_of(OpponentCopy(width=1.0), level_and_coming(seeds[::-1]))

    assert 50 <= sum(action != Action.STAY for action in together) <= 150, together
    assert alone == together
    assert reversed_batch[::-1] == together


def test_each_game_keeps_its_offset_as_the_ball_comes_and_goes_and_games_leave() -> (
    None
):
    # Games leave the batch as they end; the rest keep their order. The ball stays
    # level with each Paddle, so each Game chooses the same as long as its offset
    # stays the same.
    player = OpponentCopy(width=1.0)
    seeds = list(range(200))
    first = dict(zip(seeds, actions_of(player, level_and_coming(seeds)), strict=True))

    for step in range(2, 8):
        going = [seed for seed in seeds if seed % step]
        vx = BASE_SPEED if step < 5 else -BASE_SPEED
        actions = actions_of(player, level_and_coming(going, vx=vx))

        assert actions == [first[seed] for seed in going], f"Step {step}"


@pytest.mark.parametrize("comes_after", ["a return", "a serve toward it"])
def test_each_ball_that_comes_gets_a_new_offset(comes_after: str) -> None:
    # Between two balls that come level, the other side returns the first, or it
    # misses it and the ball waits in the center, at 0.5, to be served back. Each
    # time an offset over 0.04 from the ball moves it up, one under -0.04 down: 7 in
    # 30 each, and 16 in 30 it stays. The second Action differs from the first in
    # 1 - (7/30)^2 - (7/30)^2 - (16/30)^2 = 546/900 of the Games, about 121 of 200,
    # give or take about 7. The bounds below are about 6 of those away.
    player = OpponentCopy(width=1.0)
    seeds = list(range(200))
    first = actions_of(player, level_and_coming(seeds))
    actions_of(player, level_and_coming(seeds, vx=-BASE_SPEED))
    if comes_after == "a serve toward it":
        actions_of(player, level_and_coming(seeds, vx=0.0))

    second = actions_of(player, level_and_coming(seeds))

    changed = sum(a != b for a, b in zip(first, second, strict=True))
    assert 80 <= changed <= 160, f"{changed} of 200 changed"
