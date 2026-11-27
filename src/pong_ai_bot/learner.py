"""The learner: DQN, learning from the Steps of any Gymnasium environment
(issue #3).

The network comes from outside: from a batch of inputs, it gives each input's Q of
each Action. The learner chooses Actions by ε, keeps the latest transitions, and
updates the network from random batches of them, with Adam and the MSE loss. It
knows nothing of Pong, of files or of TensorBoard.

For now each transition is valued by its Reward alone, as if every Game ended in
one Step: the future, through the discount and the target network, comes with test
environment 3 (#36).
"""

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
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
        self._settings = settings
        self._optimizer = torch.optim.Adam(
            network.parameters(), lr=settings.learning_rate
        )
        self._rng = np.random.default_rng(seed)
        self._replay: deque[Transition] = deque(maxlen=settings.replay_buffer_size)
        self._steps_given = 0

    def choose_action(self, observation: Any, epsilon: float) -> int:
        """A random Action with probability `epsilon`, and otherwise the Action of
        the highest Q."""
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self._actions))
        with torch.no_grad():
            q = self._network(_inputs([observation]))
        return int(q[0].argmax())

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


def _inputs(observations: list[Any]) -> torch.Tensor:
    """The network's input from a batch of Observations."""
    return torch.as_tensor(np.stack(observations), dtype=torch.float32)
