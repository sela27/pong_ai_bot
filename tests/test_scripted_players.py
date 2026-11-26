"""The Scripted players, through `act`, on batches of real Games' Observations and
`info`s."""

from typing import Any

import numpy as np
import numpy.typing as npt

from pong_ai_bot.environment import Action, PongEnv
from pong_ai_bot.scripted_players import RandomPlayer


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
