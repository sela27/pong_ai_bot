"""A Run's settings, each with its default.

Every hyperparameter is a setting of the Run rather than a constant in the code, so
the tests can run with tiny numbers and a real Run with the chosen values (issues
#10, #12, #14). The defaults are the spec's table (#17), for both learning players.
A Run's settings are written to its settings file when it is created and never
change after: other settings are another Run.
"""

from dataclasses import dataclass
from enum import StrEnum


class Loss(StrEnum):
    """The loss a learning player's network is fitted with."""

    MSE = "mse"
    """The mean squared error of the Q-values against their targets."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSettings:
    """A Run's settings. Change one by naming it; the rest keep their defaults."""

    budget: int = 10_000_000
    """The most Steps the Run may take, the filling's included."""
    replay_buffer_size: int = 1_000_000
    """How many Steps the replay buffer holds."""
    fill_steps: int = 50_000
    """How many Steps fill the replay buffer before learning starts, at the start
    of the Run and at every resume."""

    epsilon_start: float = 1.0
    """ε at Step 0."""
    epsilon_end: float = 0.01
    """ε from the end of its decay on."""
    epsilon_decay_steps: int = 1_000_000
    """Over how many Steps ε goes down in a straight line, from its start to its
    end. It depends only on the Run's Step count."""

    learning_rate: float = 1e-4
    """Adam's learning rate, the same all through the Run."""
    adam_eps: float = 1e-8
    """Adam's eps, PyTorch's own default."""
    batch_size: int = 32
    """How many transitions each update learns from."""
    update_every: int = 4
    """How many Steps between updates of the network."""
    target_copy_every: int = 1_000
    """How many Steps between full copies of the network to the target network."""
    loss: Loss = Loss.MSE
    """The loss the network is fitted with, with no clipping of the gradients."""
    discount: float = 0.99
    """How much a Reward one Step later is worth."""
    double_dqn: bool = True
    """Whether the target is Double DQN's: the network chooses the next Action,
    and the target network gives its value."""

    checkpoint_every: int = 50_000
    """How many Steps between full checkpoints."""
    checkpoints_kept: int = 2
    """How many of the latest full checkpoints are kept."""
    milestone_every: int = 500_000
    """How many Steps between Milestones, each kept for good."""

    stop_mean: float = 19.0
    """The Run stops at the first Milestone whose mean Score margin on the
    regular set is at least this."""
    confirmation_gate: float = 18.0
    """The Run reaches its goal only if its Confirmation's mean Score margin is at
    least this."""
