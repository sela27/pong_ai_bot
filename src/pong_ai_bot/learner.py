"""The learner: DQN, learning from the Steps of any Gymnasium environment
(issue #3).

The network comes from outside: from a batch of inputs, it gives each input's Q of
each Action. The learner chooses Actions by ε, keeps the latest transitions in its
replay buffer, and updates the network from random batches of them, with Adam and
the MSE loss. It knows nothing of Pong, of files or of TensorBoard. In an
Evaluation and in watching, the network plays as the greedy player instead, with
no randomness.

Each update moves the Q of a transition's Action toward its target: its Reward, and
the discounted value of the Observation after it, unless the Game ended in it. That
value comes from the target network, a copy of the network that changes only when
the whole network is copied to it again, every so many Steps. With Double DQN's
target, the network chooses the next Action, and the target network gives its
value.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from pong_ai_bot.replay_buffer import ReplayBuffer
from pong_ai_bot.training import Transition


@dataclass(frozen=True, slots=True, kw_only=True)
class LearnerSettings:
    """How the learner learns. Each has the name of the Run's setting it comes
    from, and its default there, in RunSettings."""

    batch_size: int
    """How many transitions each update of the network learns from."""
    learning_rate: float
    """Adam's learning rate, the same throughout. Adam's other settings are
    PyTorch's defaults."""
    update_every: int
    """How many Steps from one update of the network to the next."""
    replay_buffer_size: int
    """How many of the latest transitions the replay buffer keeps."""
    fill_steps: int
    """How many transitions the learner keeps before its first update."""
    target_copy_every: int
    """How many Steps from one full copy of the network to the target network to
    the next."""
    discount: float
    """How much a Reward one Step later is worth."""
    double_dqn: bool
    """Whether the target is Double DQN's: the network chooses the next Action,
    and the target network gives its value."""


class Learner:
    """Learns the Q of each Action with `network`, from the transitions it is
    given."""

    def __init__(
        self,
        network: nn.Module,
        *,
        actions: int,
        stack_length: int,
        settings: LearnerSettings,
        seed: int,
    ) -> None:
        """A learner for `network`, whose every input has `actions` Actions, and is
        a stack of `stack_length` Observations, as the stack's wrapper gives them.
        It draws its random choices from `seed`. Its target network starts as a
        copy of `network`."""
        self._network = network
        self._target_network = copy.deepcopy(network)
        self._actions = actions
        self._player = GreedyPlayer(network)
        self._settings = settings
        self._optimizer = torch.optim.Adam(
            network.parameters(), lr=settings.learning_rate
        )
        self._rng = np.random.default_rng(seed)
        self._replay = ReplayBuffer(
            size=settings.replay_buffer_size, stack_length=stack_length
        )
        self._steps_given = 0

    def choose_action(self, observation: Any, epsilon: float) -> int:
        """The Action for `observation`, by `epsilon`, as `epsilon_greedy_action`
        chooses it."""
        return epsilon_greedy_action(
            self._player, observation, epsilon, actions=self._actions, rng=self._rng
        )

    def learn(self, transition: Transition) -> None:
        """Keep `transition`, update the network if its settings say so, and then
        copy it to the target network if they say so."""
        self._replay.add(transition)
        self._steps_given += 1
        settings = self._settings
        if (
            len(self._replay) >= settings.fill_steps
            and self._steps_given % settings.update_every == 0
        ):
            self._update()
        if self._steps_given % settings.target_copy_every == 0:
            self._target_network.load_state_dict(self._network.state_dict())

    def state_dict(self) -> dict[str, Any]:
        """The learner's state now: the network's weights, the target network's,
        Adam's state, the random generator's, and how many Steps it was given, by
        which it updates and copies. Not its replay buffer: it is never saved.

        It is a copy, which the learner's learning on never changes. PyTorch's own
        state dicts hold the learner's own tensors, and another learner given them
        would share Adam's state with this one.
        """
        return copy.deepcopy(
            {
                "network": self._network.state_dict(),
                "target_network": self._target_network.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "rng": self._rng.bit_generator.state,
                "steps_given": self._steps_given,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Take on the state another learner gave, in `state_dict`."""
        self._network.load_state_dict(state["network"])
        self._target_network.load_state_dict(state["target_network"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._rng.bit_generator.state = state["rng"]
        self._steps_given = state["steps_given"]

    def _update(self) -> None:
        batch = self._replay.sample(self._settings.batch_size, self._rng)
        batch_targets = targets(
            batch,
            self._network,
            self._target_network,
            discount=self._settings.discount,
            double_dqn=self._settings.double_dqn,
        )
        actions = torch.tensor([transition.action for transition in batch])
        q = self._network(_inputs([transition.observation for transition in batch]))
        loss = nn.functional.mse_loss(
            q.gather(1, actions[:, None])[:, 0], batch_targets
        )
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


def targets(
    batch: Sequence[Transition],
    network: nn.Module,
    target_network: nn.Module,
    *,
    discount: float,
    double_dqn: bool,
) -> torch.Tensor:
    """The target of each of `batch`'s transitions, in the batch's order: its
    Reward, and `discount` times the value of its next Observation, unless the Game
    ended in it (`terminated`). A Game cut off in it (`truncated`) still had a
    future, so the value counts then.

    With `double_dqn`, Double DQN's value: `network` chooses the next Action, the
    one of its highest Q, and `target_network` gives that Action's Q. Without it,
    the target network's highest Q.
    """
    rewards = torch.tensor(
        [transition.reward for transition in batch], dtype=torch.float32
    )
    ended = torch.tensor([transition.terminated for transition in batch])
    next_inputs = _inputs([transition.next_observation for transition in batch])
    with torch.no_grad():
        next_q = target_network(next_inputs)
        if double_dqn:
            next_actions = network(next_inputs).argmax(dim=1)
            value = next_q.gather(1, next_actions[:, None])[:, 0]
        else:
            value = next_q.max(dim=1).values
    return rewards + discount * torch.where(ended, 0.0, value)


class GreedyPlayer:
    """A learning player in the Agent's place, as an Evaluation plays it and as
    watching shows it: every Step, the Action of the highest Q by its network, with
    no randomness (issue #13)."""

    def __init__(self, network: nn.Module) -> None:
        self._network = network

    def act(
        self, observations: npt.NDArray[Any], infos: Sequence[dict[str, Any]]
    ) -> npt.NDArray[np.int64]:
        """The Action of the highest Q for each of a batch of Games' Observations.
        It never reads the `info`s."""
        return self.q_values(observations).argmax(axis=1)

    def q_values(self, observations: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """The Q of each Action for each of a batch of Games' Observations, one row
        for each Game, in the batch's order: what `act` chooses from."""
        with torch.no_grad():
            return self._network(_inputs(observations)).numpy()


def epsilon_greedy_action(
    player: GreedyPlayer,
    observation: Any,
    epsilon: float,
    *,
    actions: int,
    rng: np.random.Generator,
) -> int:
    """The Action for one Observation by ε, as the learner chooses it in training
    (issue #12): with probability `epsilon`, one of the `actions` Actions at random,
    each as likely, drawn from `rng`, and otherwise `player`'s, the Action of the
    highest Q."""
    if rng.random() < epsilon:
        return int(rng.integers(actions))
    return int(player.q_values([observation])[0].argmax())


def _inputs(observations: npt.ArrayLike) -> torch.Tensor:
    """The network's input from a batch of Observations: a list of them, or an
    array with one along each place of its first axis."""
    return torch.as_tensor(np.asarray(observations), dtype=torch.float32)
