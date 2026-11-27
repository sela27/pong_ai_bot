"""The learner's state, as a `dict`, for a checkpoint.

The test is slow only for PyTorch: importing it, and the part of it the learner's
optimizer loads, takes about 9 seconds, and pytest imports every test module, slow
or not, before it leaves the slow tests out. So PyTorch is imported inside the
test, not at the top.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pong_ai_bot.training import Transition

if TYPE_CHECKING:
    from pong_ai_bot.learner import Learner

INPUT = np.array([1.0], np.float32)


def play(learner: Learner, steps: int) -> None:
    """`steps` one-Step Games of two Actions, the first rewarded +1 and the other
    -1, with the learner choosing half of its Actions at random."""
    for _ in range(steps):
        action = learner.choose_action(INPUT, epsilon=0.5)
        reward = 1.0 if action == 0 else -1.0
        learner.learn(Transition(INPUT, action, reward, INPUT, True, False))


@pytest.mark.slow
def test_a_learner_given_another_learners_state_learns_on_as_the_other_does() -> None:
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner, LearnerSettings

    # Each update learns from the latest transition alone, so both learners learn
    # from the same ones, though only the first kept those before.
    settings = LearnerSettings(
        batch_size=1, learning_rate=0.1, update_every=1, replay_size=1, fill=1
    )
    torch.manual_seed(1)
    network = nn.Linear(1, 2)
    learner = Learner(network, actions=2, settings=settings, seed=1)
    play(learner, steps=10)
    torch.manual_seed(2)
    other_network = nn.Linear(1, 2)
    other = Learner(other_network, actions=2, settings=settings, seed=2)

    other.load_state_dict(learner.state_dict())
    play(learner, steps=10)
    play(other, steps=10)

    with torch.no_grad():
        values = network(torch.tensor(INPUT)).tolist()
        other_values = other_network(torch.tensor(INPUT)).tolist()
    assert other_values == values, (
        f"the Q-values learned are {other_values}, and the other's {values}"
    )
