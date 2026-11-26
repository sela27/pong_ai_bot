"""The Scripted players: non-learning players in the Agent's place, by hand-written
rules (issue #7).

They play the environment as it is, PongEnv, and decide every Step, as the Agent
does, with the same three Actions. They read only each Game's `info`, never its
Observation.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pong_ai_bot.environment import Action
from pong_ai_bot.evaluation import Player

RANDOM_PLAYER_SEED = 0
"""The random player's seed, the same for every new one, so that its Evaluation
gives the same results every time."""


class RandomPlayer:
    """A random Action every Step, in every Game. It shows that without really
    playing, nothing is won."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(RANDOM_PLAYER_SEED)

    def act(
        self, observations: npt.NDArray[Any], infos: Sequence[dict[str, Any]]
    ) -> npt.NDArray[np.int64]:
        return self._rng.integers(len(Action), size=len(infos))


SCRIPTED_PLAYERS: dict[str, Callable[[], Player]] = {
    "random": RandomPlayer,
}
"""Each Scripted player, by the name `pong evaluate` knows it by, with the function
that makes a new one for an Evaluation. A new Scripted player is one more line."""
