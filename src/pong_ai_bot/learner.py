"""The learner: DQN, learning from the Steps of any Gymnasium environment
(issue #3).

The network comes from outside: from a batch of inputs, it gives each input's Q of
each Action. The learner chooses Actions by ε, keeps the latest transitions, and
updates the network from random batches of them, with Adam and the MSE loss. It
knows nothing of Pong, of files or of TensorBoard. In an Evaluation and in
watching, the network plays as the greedy player instead, with no randomness.

For now each transition is valued by its Reward alone, as if every Game ended in
one Step: the future, through the discount and the target network, comes with test
environment 3 (#36).
"""

import copy
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

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
    """How many of the latest transitions the learner keeps."""
    fill_steps: int
    """How many transitions the learner keeps before its first update."""


class Learner:
    """Learns the Q of each Action with `network`, from the transitions it is
    given."""

    def __init__(
        self,
        network: nn.Module,
        *,
        actions: int,
        settings: LearnerSettings,
        seed: int,
    ) -> None:
        """A learner for `network`, whose every input has `actions` Actions. It
        draws its random choices from `seed`."""
        self._network = network
        self._actions = actions
        self._player = GreedyPlayer(network)
        self._settings = settings
        self._optimizer = torch.optim.Adam(
            network.parameters(), lr=settings.learning_rate
        )
        self._rng = np.random.default_rng(seed)
        self._replay: deque[Transition] = deque(maxlen=settings.replay_buffer_size)
        self._steps_given = 0

    def choose_action(self, observation: Any, epsilon: float) -> int:
        """The Action for `observation`, by `epsilon`, as `epsilon_greedy_action`
        chooses it."""
        return epsilon_greedy_action(
            self._player, observation, epsilon, actions=self._actions, rng=self._rng
        )

    def learn(self, transition: Transition) -> None:
        """Keep `transition`, and update the network if its settings say so."""
        self._replay.append(transition)
        self._steps_given += 1
        settings = self._settings
        if (
            len(self._replay) >= settings.fill_steps
            and self._steps_given % settings.update_every == 0
        ):
            self._update()

    def state_dict(self) -> dict[str, Any]:
        """The learner's state now: the network's weights, Adam's state and the
        random generator's. Not the transitions it keeps: they are never saved.

        It is a copy, which the learner's learning on never changes. PyTorch's own
        state dicts hold the learner's own tensors, and another learner given them
        would share Adam's state with this one.
        """
        return copy.deepcopy(
            {
                "network": self._network.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "rng": self._rng.bit_generator.state,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Take on the state another learner gave, in `state_dict`."""
        self._network.load_state_dict(state["network"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._rng.bit_generator.state = state["rng"]

    def _update(self) -> None:
        picks = self._rng.integers(len(self._replay), size=self._settings.batch_size)
        batch = [self._replay[pick] for pick in picks]
        actions = torch.tensor([transition.action for transition in batch])
        rewards = torch.tensor(
            [transition.reward for transition in batch], dtype=torch.float32
        )
        q = self._network(_inputs([transition.observation for transition in batch]))
        loss = nn.functional.mse_loss(q.gather(1, actions[:, None])[:, 0], rewards)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


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
