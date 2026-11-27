"""How a learning player chooses its Actions: by ε in training (issue #12), and as
the greedy player in an Evaluation and in watching (issue #13).

Each test gives a network whose Q-values are known in advance, with none of the
learner's optimizer: creating one takes seconds, too long for a fast test.
"""

import numpy as np
import pytest
import torch
from torch import nn

from pong_ai_bot.learner import GreedyPlayer, epsilon_greedy_action

SEED = 0
"""The seed of the learner's random choices."""
OBSERVATIONS = np.array([[1.0], [-1.0], [0.0]], np.float32)
"""A batch of three Games' Observations, of one number each."""
Q_VALUES = [[1.0, -1.0, 0.5], [-1.0, 1.0, 0.5], [0.0, 0.0, 0.5]]
"""The network's Q of each of its three Actions, for each of OBSERVATIONS."""
BEST_ACTIONS = [0, 1, 2]
"""The Action of the highest Q, for each of OBSERVATIONS."""
INFOS: list[dict[str, object]] = [{}, {}, {}]
"""The three Games' `info`s, which a learning player never reads."""


def network() -> nn.Module:
    """A network that gives Q_VALUES for OBSERVATIONS: its Q of the first Action is
    the input, of the second minus the input, and of the third 0.5."""
    network = nn.Linear(1, 3)
    network.load_state_dict(
        {
            "weight": torch.tensor([[1.0], [-1.0], [0.0]]),
            "bias": torch.tensor([0.0, 0.0, 0.5]),
        }
    )
    return network


def test_the_greedy_player_chooses_the_action_of_the_highest_q_every_time() -> None:
    player = GreedyPlayer(network())

    actions = [player.act(OBSERVATIONS, INFOS).tolist() for _ in range(100)]

    wrong = [chosen for chosen in actions if chosen != BEST_ACTIONS]
    assert not wrong, f"the Actions chosen are {wrong[0]} in {len(wrong)} of 100 tries"


def test_the_greedy_player_shows_the_q_of_each_action_in_each_game() -> None:
    player = GreedyPlayer(network())

    q_values = player.q_values(OBSERVATIONS)

    assert q_values == pytest.approx(np.array(Q_VALUES)), (
        f"the Q-values shown are {q_values.tolist()}"
    )


def test_at_epsilon_0_the_learner_chooses_the_action_of_the_highest_q() -> None:
    player = GreedyPlayer(network())
    rng = np.random.default_rng(SEED)

    actions = [
        [
            epsilon_greedy_action(player, observation, 0.0, actions=3, rng=rng)
            for observation in OBSERVATIONS
        ]
        for _ in range(100)
    ]

    wrong = [chosen for chosen in actions if chosen != BEST_ACTIONS]
    assert not wrong, f"the Actions chosen are {wrong[0]} in {len(wrong)} of 100 tries"


def test_at_epsilon_1_the_learner_chooses_each_action_a_third_of_the_time() -> None:
    player = GreedyPlayer(network())
    rng = np.random.default_rng(SEED)
    # Its Q of the first Action is the highest, so the greedy player chooses it.
    observation = OBSERVATIONS[0]

    actions = [
        epsilon_greedy_action(player, observation, 1.0, actions=3, rng=rng)
        for _ in range(3_000)
    ]

    # 1,000 each, give or take about 4 standard deviations of a binomial count of
    # 3,000 choices of a third: the square root of 3,000 * 1/3 * 2/3, about 26.
    counts = [actions.count(action) for action in range(3)]
    assert all(900 <= count <= 1_100 for count in counts), (
        f"the three Actions are chosen {counts} times"
    )
