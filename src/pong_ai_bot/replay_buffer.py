"""The replay buffer: the latest Steps the learner learns from (issue #12).

The player decides from stacks of its Game's last Observations, but the buffer
keeps each Observation once, and rebuilds the stacks when it is sampled, by the
same rule as the stack's wrapper, `stack_indices`. Keeping each Step's two stacks
would take eight times the memory with the Agent's stacks of four, and a million
Steps wouldn't fit. The buffer is never written to disk.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from pong_ai_bot.stack import stack_indices
from pong_ai_bot.transition import Transition


class ReplayBuffer:
    """The latest Steps of a player's Games, each with the stacks the player was
    given before and after it.

    Each Step keeps the Observation after it, and each Game the Observation its
    `reset` returned, beside its first Step. A stack is rebuilt from the Steps
    before in its Game, so it never reaches into the Game before. Nothing depends
    on the Observation's shape: an image or numbers.
    """

    _firsts: dict[int, npt.NDArray[Any]]
    """The Observation each Game's `reset` returned, by the slot of its first
    Step."""
    _game_steps: int
    """How many Steps the Game being played has had so far: 0 when the next Step
    starts a new one."""

    def __init__(self, *, size: int, stack_length: int) -> None:
        """A buffer of the latest `size` Steps, whose stacks each hold
        `stack_length` Observations, as the stack's wrapper gives them."""
        self._size = size
        self._stack_length = stack_length
        # The oldest Step's stacks reach back as many Steps as a stack holds, so
        # the buffer keeps those Steps' Observations too.
        self._capacity = size + stack_length
        self._added = 0
        self._game_steps = 0
        self._firsts = {}

    def __len__(self) -> int:
        """How many Steps the buffer holds."""
        return min(self._added, self._size)

    def add(self, transition: Transition) -> None:
        """Keep `transition`, a Step the player played: its stacks are the
        wrapper's, oldest first along their first axis.

        The Steps come in the order they were played, each Game to its end, so a
        new Game starts with the Step after one that ended it, terminated or
        truncated. The oldest Step is let go once the buffer holds its size.
        """
        after = np.asarray(transition.next_observation)[-1]
        if self._added == 0:
            self._allocate(after)
        slot = self._added % self._capacity
        if self._game_steps == 0:
            self._firsts[slot] = np.array(np.asarray(transition.observation)[-1])
        else:
            # The Game that started in this slot is gone.
            self._firsts.pop(slot, None)
        self._game_steps += 1
        self._after[slot] = after
        self._numbers[slot] = self._game_steps
        self._actions[slot] = transition.action
        self._rewards[slot] = transition.reward
        self._terminated[slot] = transition.terminated
        self._truncated[slot] = transition.truncated
        self._added += 1
        if transition.terminated or transition.truncated:
            self._game_steps = 0

    def sample(self, count: int, rng: np.random.Generator) -> list[Transition]:
        """`count` of the Steps the buffer holds, each drawn from `rng` at random,
        each as likely, with the stacks the player was given before and after
        it."""
        oldest = self._added - len(self)
        picks = rng.integers(len(self), size=count)
        return [self._transition(oldest + int(pick)) for pick in picks]

    def _allocate(self, observation: npt.NDArray[Any]) -> None:
        """Make room for as many Steps as the buffer keeps, each with the
        Observation after it, of the shape and type of `observation`, and its
        number in its Game, counting from 1."""
        capacity = self._capacity
        self._after = np.empty((capacity, *observation.shape), observation.dtype)
        self._numbers = np.empty(capacity, np.int64)
        self._actions = np.empty(capacity, np.int64)
        self._rewards = np.empty(capacity, np.float64)
        self._terminated = np.empty(capacity, np.bool_)
        self._truncated = np.empty(capacity, np.bool_)

    def _transition(self, step: int) -> Transition:
        """The buffer's `step`th Step, counting from 0 at the first ever added."""
        slot = step % self._capacity
        number = int(self._numbers[slot])
        first = step - number + 1
        return Transition(
            self._stack(first, number - 1),
            int(self._actions[slot]),
            float(self._rewards[slot]),
            self._stack(first, number),
            bool(self._terminated[slot]),
            bool(self._truncated[slot]),
        )

    def _stack(self, first: int, steps: int) -> npt.NDArray[Any]:
        """The stack after `steps` Steps of the Game whose first Step is the
        buffer's `first`th."""
        return np.stack(
            [
                self._observation(first, index)
                for index in stack_indices(steps, self._stack_length)
            ]
        )

    def _observation(self, first: int, index: int) -> npt.NDArray[Any]:
        """The Observation numbered `index`, by the stacking rule's count, of the
        Game whose first Step is the buffer's `first`th: the one `reset` returned
        if 0, and otherwise the one after the Game's Step `index`."""
        if index == 0:
            return self._firsts[first % self._capacity]
        return self._after[(first + index - 1) % self._capacity]
