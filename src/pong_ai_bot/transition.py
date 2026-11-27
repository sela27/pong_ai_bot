"""A transition: one Step, as the learner learns from it and its replay buffer keeps
it (issue #12).

It is a module of its own so that the learner and the replay buffer can name it
without importing the training loop, which plays the Steps that make them.
"""

from typing import Any, NamedTuple


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
