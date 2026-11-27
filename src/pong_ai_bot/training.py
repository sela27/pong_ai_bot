"""The training loop: Steps against any Gymnasium environment (issue #10).

Each Step, the learner chooses an Action, with ε by the Step counter alone, and
learns from the Step's transition. The loop never reads the `info`. It knows
nothing of Pong, of files or of TensorBoard.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import gymnasium as gym

if TYPE_CHECKING:
    # Only for its type: the learner imports PyTorch, which takes seconds to
    # import, and the ε schedule doesn't need it.
    from pong_ai_bot.learner import Learner


@dataclass(frozen=True, slots=True, kw_only=True)
class EpsilonSchedule:
    """How likely the learner is to choose a random Action, by the Run's Step
    counter alone (issue #12): in a straight line from `start` to `end` over the
    first `decay_steps` Steps, and `end` from then on."""

    start: float = 1.0
    end: float = 0.01
    decay_steps: int = 1_000_000

    def at(self, step: int) -> float:
        """ε at the Run's Step `step`, counting from 0."""
        done = min(step / self.decay_steps, 1.0)
        return self.start + (self.end - self.start) * done


class Transition(NamedTuple):
    """One Step, as the learner learns from it."""

    observation: Any
    """What the learner chose its Action from."""
    action: int
    reward: float
    next_observation: Any
    """What the environment gave after the Step, even if the Game ended in it."""
    terminated: bool
    """Whether the Game ended in the Step, so that it has no future."""
    truncated: bool
    """Whether the Game was cut off in the Step, though it had a future."""


def train(
    env: gym.Env[Any, Any],
    learner: Learner,
    *,
    steps: int,
    epsilon: EpsilonSchedule = EpsilonSchedule(),
    seed: int,
) -> None:
    """Play `steps` Steps against `env`, a Game after another, the first reset with
    `seed`, with `learner` choosing each Action and learning from each Step."""
    observation, _ = env.reset(seed=seed)
    for step in range(steps):
        action = learner.choose_action(observation, epsilon.at(step))
        next_observation, reward, terminated, truncated, _ = env.step(action)
        learner.learn(
            Transition(
                observation,
                action,
                float(reward),
                next_observation,
                terminated,
                truncated,
            )
        )
        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation
