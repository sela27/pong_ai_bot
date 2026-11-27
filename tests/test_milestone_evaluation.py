"""The Evaluation of a learning player's Milestones, as a Run plays it and as
`pong evaluate` plays it after the Run: the greedy player, with the weights a
Milestone holds, on the regular set (#17).

The environment is a fake of Pong whose Games are a Step long, and the network a
single layer whose Q is its bias, so the expected numbers are worked out by hand.
"""

from datetime import datetime
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import torch
from gymnasium import spaces
from helpers import A_COMMIT
from torch import nn

from pong_ai_bot.environment import Action
from pong_ai_bot.evaluation import EvaluationStopped, SeedSet, Summary
from pong_ai_bot.game import Score
from pong_ai_bot.learning_players import (
    LEARNING_PLAYERS,
    milestone_evaluation,
    milestone_player,
)
from pong_ai_bot.run_directory import RunDirectory, SavedFile


class OneStepGames(gym.Env[npt.NDArray[np.float32], np.int64]):
    """A fake of Pong whose every Game ends in its first Step: won 21 to 0 if the
    player in the Agent's place moves up, and lost 0 to 21 otherwise. Its
    Observation is always 1."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        return np.ones(1, np.float32), {"score": Score(left=0, right=0)}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[npt.NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        won = action == Action.UP
        score = Score(left=0, right=21) if won else Score(left=21, right=0)
        return np.ones(1, np.float32), 0.0, True, False, {"score": score}


@pytest.fixture
def run() -> RunDirectory:
    return RunDirectory.create(
        player="agent",
        name="baseline",
        seed=1,
        commit=A_COMMIT,
        now=datetime(2026, 11, 27, 14, 30),
    )


def milestone_preferring(run: RunDirectory, action: Action, *, step: int) -> SavedFile:
    """The Run's Milestone at `step`, of a network whose highest Q is always
    `action`'s: no weights, and a bias of 1 for `action` and 0 for the others."""
    weights = {"weight": torch.zeros(3, 1), "bias": torch.zeros(3)}
    weights["bias"][action] = 1.0
    return run.save_milestone(step, lambda file: torch.save(weights, file))


def test_a_milestone_is_played_greedily_by_its_weights_on_the_regular_set(
    run: RunDirectory,
) -> None:
    up = milestone_preferring(run, Action.UP, step=10)
    down = milestone_preferring(run, Action.DOWN, step=20)
    evaluate_milestone = milestone_evaluation(nn.Linear(1, 3), OneStepGames)

    moving_up = evaluate_milestone(up, seed_set=SeedSet.REGULAR, stop=lambda: False)
    moving_down = evaluate_milestone(down, seed_set=SeedSet.REGULAR, stop=lambda: False)

    # A Game for each seed of the regular set, every one won by moving up, and
    # lost by moving down.
    assert [game.seed for game in moving_up.games] == list(range(1_000_000, 1_000_100))
    assert moving_up.summary == Summary(mean=21.0, std=0.0, min=21, wins=100, cut_off=0)
    assert moving_down.summary == Summary(
        mean=-21.0, std=0.0, min=-21, wins=0, cut_off=0
    )


def test_a_milestone_loads_into_a_greedy_player_with_the_saved_network_s_q_values() -> (
    None
):
    # A Position learner's Run, its player as its settings file records it, and a
    # Milestone of a network of its kind.
    created = RunDirectory.create(
        player="position-learner", name="baseline", seed=1, commit=A_COMMIT
    )
    torch.manual_seed(0)
    saved = LEARNING_PLAYERS["position-learner"].make_network()
    milestone = created.save_milestone(
        500_000, lambda file: torch.save(saved.state_dict(), file)
    )
    run = RunDirectory.open(created.path, current_commit=A_COMMIT)

    player = milestone_player(run, milestone)

    # A batch of the Position learner's stacks of four Steps' positions, four
    # numbers each.
    stacks = np.random.default_rng(0).uniform(size=(5, 4, 4)).astype(np.float32)
    with torch.no_grad():
        expected = saved(torch.from_numpy(stacks)).numpy()
    np.testing.assert_array_equal(player.q_values(stacks), expected)


def test_a_milestone_s_evaluation_is_abandoned_when_stop_says(
    run: RunDirectory,
) -> None:
    up = milestone_preferring(run, Action.UP, step=10)
    evaluate_milestone = milestone_evaluation(nn.Linear(1, 3), OneStepGames)

    with pytest.raises(EvaluationStopped):
        evaluate_milestone(up, seed_set=SeedSet.REGULAR, stop=lambda: True)
