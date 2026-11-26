"""The environment: the Game, Step by Step, through Gymnasium's API (issue #8).

The player in the Agent's place is on the right. Each Step it chooses an Action,
which is held for FRAMES_PER_STEP Frames, and then it gets the Observation drawn
after the last of them and the Reward: +1 for each Point it won in them and -1 for
each it lost. A Point doesn't stop a Step, but reaching WINNING_POINTS does, at once
(`terminated`). A Game is cut off after GAME_FRAMES Frames, its 27,000th Step
(`truncated`).

The player on the left moves every Frame, by the physics state. It is a parameter:
the Opponent by default, or a Human at the keyboard. An `on_frame` hook, if given,
gets the physics state after every Frame, so a window can draw it, read the keyboard
and keep to its pace. Without it nothing ever waits.

`reset` and `step` return one Observation, not a stack, and in `info` a copy of the
physics state: the ball, both Paddles and the score, under the names PhysicsState
gives them. Changing the copy never changes the Game.
"""

from collections.abc import Callable
from enum import IntEnum
from typing import Any, Protocol, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from pong_ai_bot.game import MAX_PADDLE_MOVE, Ball, Game, PhysicsState, Score, Side
from pong_ai_bot.observation import BLACK, OBSERVATION_SIZE, WHITE, draw_observation
from pong_ai_bot.opponent import Opponent

FRAMES_PER_STEP = 4
"""How many Frames each Step holds its Action for."""


class Action(IntEnum):
    """The three choices of the player in the Agent's place, each Step."""

    UP = 0
    DOWN = 1
    STAY = 2


_MOVES: dict[int, float] = {
    Action.UP: MAX_PADDLE_MOVE,
    Action.DOWN: -MAX_PADDLE_MOVE,
    Action.STAY: 0.0,
}
"""How far each Action moves the Paddle up in a Frame. Staying stops it at once."""


class LeftPlayer(Protocol):
    """The player on the left side: the Opponent, or a Human at the keyboard."""

    def move(self, state: PhysicsState) -> float:
        """How far to move its Paddle up in the Frame played from `state`."""
        ...


class PongEnv(gym.Env[npt.NDArray[np.uint8], np.int64]):
    """One Game at a time, for the player in the Agent's place, on the right."""

    _game: Game
    _left: LeftPlayer

    def __init__(
        self,
        *,
        left_player: Callable[[np.random.Generator], LeftPlayer] = Opponent,
        on_frame: Callable[[PhysicsState], None] | None = None,
        start: PhysicsState | None = None,
    ) -> None:
        """An environment whose every Game has `left_player(rng)` on the left.

        `left_player` is called at the start of each Game with the Game's random
        generator, so the Opponent draws from it and the same seed gives the same
        Game. `on_frame` gets the physics state after every Frame. Each Game starts
        from `start` instead of a new Game's state, if it is given, to test exact
        cases.
        """
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            BLACK, WHITE, (OBSERVATION_SIZE, OBSERVATION_SIZE), np.uint8
        )
        self._new_left_player = left_player
        self._on_frame = on_frame
        self._start = start

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.uint8], dict[str, Any]]:
        """Start a Game: the Game of `seed`, or without one, the next Game of the
        last seed given."""
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2**63))
        if self._start is None:
            self._game = Game.new(seed=seed)
        else:
            self._game = Game(self._start, seed=seed)
        self._left = self._new_left_player(self._game.rng)
        state = self._game.state
        return draw_observation(state), _info(state)

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[npt.NDArray[np.uint8], SupportsFloat, bool, bool, dict[str, Any]]:
        try:
            right = _MOVES[int(action)]
        except KeyError:
            raise ValueError(f"an Action is 0, 1 or 2, got {action}") from None
        game, left, on_frame = self._game, self._left, self._on_frame
        reward = 0
        for _ in range(FRAMES_PER_STEP):
            result = game.play_frame(left=left.move(game.state), right=right)
            if result.point_won_by is Side.RIGHT:
                reward += 1
            elif result.point_won_by is Side.LEFT:
                reward -= 1
            if on_frame is not None:
                on_frame(game.state)
            if result.terminated:
                break
        state = game.state
        return (
            draw_observation(state),
            float(reward),
            result.terminated,
            result.truncated,
            _info(state),
        )


def _info(state: PhysicsState) -> dict[str, Any]:
    """A copy of the physics state that the players in the Agent's place may read."""
    ball, score = state.ball, state.score
    return {
        "ball": Ball(ball.x, ball.y, ball.vx, ball.vy),
        "left_paddle": state.left_paddle,
        "right_paddle": state.right_paddle,
        "score": Score(score.left, score.right),
    }
