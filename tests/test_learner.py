"""The learner's state, as a `dict`, for a checkpoint, and its target network.

The tests are slow only for the learner's optimizer: making one loads a part of
PyTorch that takes about 5 seconds. The fast tests that need PyTorch make networks
only, never a learner.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pong_ai_bot.run_settings import RunSettings
from pong_ai_bot.transition import Transition

if TYPE_CHECKING:
    from pong_ai_bot.learner import Learner, LearnerSettings

INPUT = np.array([1.0], np.float32)
"""The input of every Step: a stack of one Observation, the number 1."""


def one_transition_a_batch(**changes: Any) -> LearnerSettings:
    """The learner's settings in these tests, with `changes`: an update every
    Step, from a batch of one transition, at a learning rate of 0.1, from the
    first transition in the replay buffer, of one, on. The rest are a Run's
    defaults: Adam's eps of 1e-8, the MSE loss, a copy to the target network
    every 1,000 Steps, a discount of 0.99, and Double DQN's target."""
    from pong_ai_bot.learner import LearnerSettings

    settings = {
        "batch_size": 1,
        "learning_rate": 0.1,
        "update_every": 1,
        "replay_buffer_size": 1,
        "fill_steps": 1,
        **changes,
    }
    return LearnerSettings.of_run(RunSettings(**settings))


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

    from pong_ai_bot.learner import Learner

    # Each update learns from the latest transition alone, so both learners learn
    # from the same ones, though only the first kept those before. The state
    # changes hands between two copies to the target network, so the other must
    # take on the target network, and when the next copy comes, too.
    settings = one_transition_a_batch(target_copy_every=4)
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

    from pong_ai_bot.learner import Learner

    # The network is updated every Step, so it is the target network only right
    # after a copy.
    settings = one_transition_a_batch(target_copy_every=3)
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
    ("adam_eps", "weight_after"),
    [
        # Adam's first step is the learning rate times g / (|g| + eps), for the
        # gradient g of the loss. Here g is -2: the MSE loss is (Q - 1)², with Q
        # the weight, 0, times the input, 1.
        (1e-8, 0.1),
        # 0.1 * 2 / (2 + 2): Adam's eps is the setting's, and the gradient is
        # not clipped.
        (2.0, 0.05),
    ],
)
def test_an_update_moves_the_weight_by_the_mse_loss_and_adam_s_eps(
    adam_eps: float, weight_after: float
) -> None:
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner

    settings = one_transition_a_batch(adam_eps=adam_eps)
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


@pytest.mark.slow
def test_learning_gives_the_loss_of_the_update_made_and_none_before_the_first() -> None:
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner

    settings = one_transition_a_batch(replay_buffer_size=2, fill_steps=2)
    # One Action, whose Q is the weight times the input.
    network = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        network.weight.zero_()
    learner = Learner(network, actions=1, stack_length=1, settings=settings, seed=1)
    # A Reward of +1, and the Game ends: the target is 1.
    transition = Transition(INPUT, 0, 1.0, INPUT, True, False)

    # The replay buffer isn't filled yet: no update.
    assert learner.learn(transition) is None
    # The first update: the MSE loss of a Q of 0, the weight times the input,
    # against the target of 1, (0 - 1)².
    assert learner.learn(transition) == pytest.approx(1.0)


@pytest.mark.slow
def test_the_learner_gives_its_network_s_q_of_each_action_for_a_batch_of_inputs() -> (
    None
):
    import torch
    from torch import nn

    from pong_ai_bot.learner import Learner, LearnerSettings

    # Two Actions, whose Q are 2 and -3 times the input.
    network = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        network.weight.copy_(torch.tensor([[2.0], [-3.0]]))
    learner = Learner(
        network,
        actions=2,
        stack_length=1,
        settings=LearnerSettings.of_run(RunSettings()),
        seed=1,
    )

    q = learner.q_values(np.array([[1.0], [0.5]], np.float32))

    assert q.tolist() == [[2.0, -3.0], [1.0, -1.5]]
