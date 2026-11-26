"""The Scripted players: non-learning players in the Agent's place, by hand-written
rules (issue #7).

They play the environment as it is, PongEnv, and decide every Step, as the Agent
does, with the same three Actions. They read only each Game's `info`, never its
Observation.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pong_ai_bot.environment import FRAMES_PER_STEP, Action
from pong_ai_bot.evaluation import Player
from pong_ai_bot.game import MAX_PADDLE_MOVE, PhysicsState, Side
from pong_ai_bot.opponent import DEFAULT_OFFSET_WIDTH, Opponent

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


OPPONENT_COPY_SEED = 1
"""With a Game's seed, the seed of the generator the copy of the Opponent draws its
offsets from in that Game. The Game's own generator, from its seed alone, gives
other numbers."""

_STEP_MOVE = FRAMES_PER_STEP * MAX_PADDLE_MOVE
"""How far holding up or down for a Step moves the Paddle."""


class OpponentCopy:
    """The Opponent's tracking rule on the right, deciding every Step: it shows
    whether the Game is the same for both sides.

    It keeps the rule, with its offset, for each Game, knowing the Game by the seed
    in its `info`, and draws that Game's offsets from a generator of its own. Each
    Step, it asks the rule where it would take the Paddle in the Step's Frames, were
    the ball to stay where it is, and chooses the Action that ends the Step nearest
    there: up or down when that is more than half a Step's move away, and otherwise
    staying.
    """

    def __init__(self, *, width: float = DEFAULT_OFFSET_WIDTH) -> None:
        """A copy with the offset width `width`, as the Opponent's: how far from its
        Paddle's center its offset can be, as a fraction of half the Paddle's
        height."""
        self._width = width
        self._rules: dict[int, Opponent] = {}

    def act(
        self, observations: npt.NDArray[Any], infos: Sequence[dict[str, Any]]
    ) -> list[Action]:
        return [self._decide(info) for info in infos]

    def _decide(self, info: dict[str, Any]) -> Action:
        seed = info["seed"]
        if seed not in self._rules:
            rng = np.random.default_rng([OPPONENT_COPY_SEED, seed])
            self._rules[seed] = Opponent(rng, width=self._width, side=Side.RIGHT)
        rule = self._rules[seed]
        state = PhysicsState(
            ball=info["ball"],
            left_paddle=info["left_paddle"],
            right_paddle=info["right_paddle"],
            score=info["score"],
        )
        paddle = state.right_paddle
        for _ in range(FRAMES_PER_STEP):
            paddle += rule.move(dataclasses.replace(state, right_paddle=paddle))
        wanted = paddle - state.right_paddle
        if wanted > _STEP_MOVE / 2:
            return Action.UP
        if wanted < -_STEP_MOVE / 2:
            return Action.DOWN
        return Action.STAY


SCRIPTED_PLAYERS: dict[str, Callable[[], Player]] = {
    "random": RandomPlayer,
    "opponent-copy": OpponentCopy,
}
"""Each Scripted player, by the name `pong evaluate` knows it by, with the function
that makes a new one for an Evaluation. A new Scripted player is one more line."""
