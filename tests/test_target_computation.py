"""The target computation, on small batches whose targets are worked out by hand.

Each network here is a table: its Q of each Action for the Observation
`observation(i)` is the table's row `i`. The discount is 0.5, so every target is
exact.

They make networks only, never a learner: a learner makes an optimizer, which loads
a part of PyTorch that takes seconds, too long for the fast tests.
"""

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from pong_ai_bot.learner import targets
from pong_ai_bot.transition import Transition

DISCOUNT = 0.5


def observation(i: int) -> npt.NDArray[np.float32]:
    """The Observation of a table's row `i`: 1 at `i`, and 0 elsewhere."""
    one_hot = np.zeros(2, np.float32)
    one_hot[i] = 1.0
    return one_hot


def table(*rows: list[float]) -> nn.Module:
    """A network whose Q of each Action for `observation(i)` is `rows[i]`."""
    network = nn.Linear(len(rows), len(rows[0]), bias=False)
    with torch.no_grad():
        network.weight.copy_(torch.tensor(rows).T)
    return network


NETWORK = table([1.0, 3.0, 2.0], [5.0, 0.0, 4.0])
"""The network: its highest Q is Action 1's after `observation(0)`, and Action 0's
after `observation(1)`."""
TARGET_NETWORK = table([10.0, 20.0, 40.0], [8.0, 30.0, 6.0])
"""The target network, whose highest Q is another Action's after each."""


def step_to(
    next_observation: int,
    reward: float,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> Transition:
    """A transition with `reward`, to `observation(next_observation)`. Its own
    Observation and Action play no part in its target."""
    return Transition(
        observation(0),
        0,
        reward,
        observation(next_observation),
        terminated,
        truncated,
    )


def test_double_dqns_target_is_the_networks_next_action_valued_by_the_target_network() -> (
    None
):
    batch = [step_to(0, reward=1.0), step_to(1, reward=-1.0)]

    values = targets(
        batch, NETWORK, TARGET_NETWORK, discount=DISCOUNT, double_dqn=True
    ).tolist()

    # 1 + 0.5 * 20, by Action 1, and -1 + 0.5 * 8, by Action 0.
    assert values == pytest.approx([11.0, 3.0])


def test_without_double_dqn_the_target_is_the_target_networks_highest_q() -> None:
    batch = [step_to(0, reward=1.0), step_to(1, reward=-1.0)]

    values = targets(
        batch, NETWORK, TARGET_NETWORK, discount=DISCOUNT, double_dqn=False
    ).tolist()

    # 1 + 0.5 * 40, and -1 + 0.5 * 30.
    assert values == pytest.approx([21.0, 14.0])


@pytest.mark.parametrize("double_dqn", [True, False])
def test_after_a_terminated_step_there_is_no_future(double_dqn: bool) -> None:
    batch = [step_to(0, reward=1.0, terminated=True)]

    values = targets(
        batch, NETWORK, TARGET_NETWORK, discount=DISCOUNT, double_dqn=double_dqn
    ).tolist()

    assert values == pytest.approx([1.0])


@pytest.mark.parametrize(("double_dqn", "expected"), [(True, 11.0), (False, 21.0)])
def test_after_a_truncated_step_there_is_a_future(
    double_dqn: bool, expected: float
) -> None:
    batch = [step_to(0, reward=1.0, truncated=True)]

    values = targets(
        batch, NETWORK, TARGET_NETWORK, discount=DISCOUNT, double_dqn=double_dqn
    ).tolist()

    # The Game was cut off, though it had a future: 1 + 0.5 * 20 by the network's
    # next Action, or 1 + 0.5 * 40 by the target network's highest Q.
    assert values == pytest.approx([expected])
