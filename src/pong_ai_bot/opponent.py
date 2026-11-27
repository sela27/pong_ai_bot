"""The Opponent: the non-learning player on the left, the same in every Game.

It reads the physics state every Frame, not the Observation, and tracks the ball
(issue #7). It moves its Paddle toward the ball's height plus an offset, at most
MAX_PADDLE_MOVE a Frame, and stops right on that target. Each time the ball starts
coming toward it, from the other side's return or from a serve toward it, it draws
a new offset, uniform within its width. While the ball goes away, through a serve's
wait too, it goes on tracking with its last offset. At the start of a Game the
offset is 0.

It draws from the Game's generator, handed to it from outside, so the same seed and
the same moves give the same Game. The same rule can play the right side too.
"""

import numpy as np

from pong_ai_bot.game import MAX_PADDLE_MOVE, PADDLE_HEIGHT, PhysicsState, Side

DEFAULT_OFFSET_WIDTH = 1.0
"""The offset width, locked: 100% of half the Paddle's height, as `pong choose-width`
chose it (#31), by the rule of issue #62."""


class Opponent:
    """Tracks the ball with the Paddle on its side, by the physics state."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        width: float = DEFAULT_OFFSET_WIDTH,
        side: Side = Side.LEFT,
    ) -> None:
        """An Opponent at the start of a Game, drawing its offsets from `rng`.

        `width` is the Opponent's one difficulty knob: how far from its Paddle's
        center its offset can be, as a fraction of half the Paddle's height.
        """
        self._rng = rng
        self._reach = width * PADDLE_HEIGHT / 2
        self._side = side
        self._offset = 0.0
        self._ball_was_coming = False

    def move(self, state: PhysicsState) -> float:
        """How far to move its Paddle up in the Frame played from `state`.

        A negative move is down. Give it each state in turn: it notices the ball
        start coming toward it from one call to the next.
        """
        ball = state.ball
        ball_is_coming = ball.vx < 0 if self._side is Side.LEFT else ball.vx > 0
        if ball_is_coming and not self._ball_was_coming:
            self._offset = self._rng.uniform(-self._reach, self._reach)
        self._ball_was_coming = ball_is_coming
        paddle = state.left_paddle if self._side is Side.LEFT else state.right_paddle
        target = ball.y + self._offset
        return min(max(target - paddle, -MAX_PADDLE_MOVE), MAX_PADDLE_MOVE)
