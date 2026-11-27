"""The learner's state, as a `dict`, for a checkpoint, and its target network.

The tests are slow only for the learner's optimizer: making one loads a part of
PyTorch that takes about 5 seconds. The fast tests that need PyTorch make networks
only, never a learner.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pong_ai_bot.run_settings import Loss
from pong_ai_bot.training import Transition

if TYPE_CHECKING:
    from pong_ai_bot.learner import Learner

INPUT = np.array([1.0], np.float32)
"""The input of every Step: a stack of one Observation, the number 1."""


def play(learner: Learner, steps: int) -> None:
    """`steps` Steps of a Game that never ends, always with the same input, and two
    Actions, the first rewarded +1 and the other -1, with the learner choosing half
    of its Actions at random."""
    for _ in range(steps):
        action = learner.choose_action(INPUT, epsilon=0.5)
        reward = 1.0 if action == 0 else -1.0
        learner.learn(Transition(INPUT, action, reward, INPUT, False, False))


@pytest.mark.slow
def test_a_learner_given_another_learners_state_learns_on_as_the_other_does() -> None:
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner, LearnerSettings

    # Each update learns from the latest transition alone, so both learners learn
    # from the same ones, though only the first kept those before. The state
    # changes hands between two copies to the target network, so the other must
    # take on the target network, and when the next copy comes, too.
    settings = LearnerSettings(
        batch_size=1,
        learning_rate=0.1,
        adam_eps=1e-8,
        loss=Loss.MSE,
        max_gradient_norm=None,
        update_every=1,
        replay_buffer_size=1,
        fill_steps=1,
        target_copy_every=4,
        discount=0.99,
        double_dqn=True,
    )
    torch.manual_seed(1)
    network = nn.Linear(1, 2)
    learner = Learner(network, actions=2, stack_length=1, settings=settings, seed=1)
    play(learner, steps=10)
    torch.manual_seed(2)
    other_network = nn.Linear(1, 2)
    other = Learner(other_network, actions=2, stack_length=1, settings=settings, seed=2)

    other.load_state_dict(learner.state_dict())
    play(learner, steps=10)
    play(other, steps=10)

    with torch.no_grad():
        values = network(torch.tensor(INPUT)).tolist()
        other_values = other_network(torch.tensor(INPUT)).tolist()
    assert other_values == values, (
        f"the Q-values learned are {other_values}, and the other's {values}"
    )


@pytest.mark.slow
def test_the_network_is_copied_whole_to_the_target_network_every_so_many_steps() -> (
    None
):
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner, LearnerSettings

    # The network is updated every Step, so it is the target network only right
    # after a copy.
    settings = LearnerSettings(
        batch_size=1,
        learning_rate=0.1,
        adam_eps=1e-8,
        loss=Loss.MSE,
        max_gradient_norm=None,
        update_every=1,
        replay_buffer_size=1,
        fill_steps=1,
        target_copy_every=3,
        discount=0.99,
        double_dqn=True,
    )
    torch.manual_seed(1)
    learner = Learner(
        nn.Linear(1, 2), actions=2, stack_length=1, settings=settings, seed=1
    )

    copies = []
    for step in range(1, 8):
        play(learner, steps=1)
        state = learner.state_dict()
        network, target_network = state["network"], state["target_network"]
        if all(torch.equal(network[name], target_network[name]) for name in network):
            copies.append(step)

    assert copies == [3, 6], f"the target network is the network after Steps {copies}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("adam_eps", "max_gradient_norm", "weight_after"),
    [
        # Adam's first step is the learning rate times g / (|g| + eps), for the
        # gradient g of the loss. Here g is -2: the MSE loss is (Q - 1)², with Q
        # the weight, 0, times the input, 1.
        (1e-8, None, 0.1),
        # 0.1 * 2 / (2 + 2): Adam's eps is the setting's.
        (2.0, None, 0.05),
        # The gradient clipped to a norm of 0.5 first: 0.1 * 0.5 / (0.5 + 2).
        (2.0, 0.5, 0.02),
    ],
)
def test_an_update_moves_the_weight_by_the_mse_loss_adam_s_eps_and_the_clipping(
    adam_eps: float, max_gradient_norm: float | None, weight_after: float
) -> None:
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner, LearnerSettings

    settings = LearnerSettings(
        batch_size=1,
        learning_rate=0.1,
        adam_eps=adam_eps,
        loss=Loss.MSE,
        max_gradient_norm=max_gradient_norm,
        update_every=1,
        replay_buffer_size=1,
        fill_steps=1,
        target_copy_every=1_000,
        discount=0.99,
        double_dqn=True,
    )
    # One Action, whose Q is the weight times the input.
    network = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        network.weight.zero_()
    learner = Learner(network, actions=1, stack_length=1, settings=settings, seed=1)

    # A Reward of +1, and the Game ends: the target is 1.
    learner.learn(Transition(INPUT, 0, 1.0, INPUT, True, False))

    weight = network.weight.item()
    assert weight == pytest.approx(weight_after, abs=1e-6), (
        f"the weight after the update is {weight:.6f}"
    )
