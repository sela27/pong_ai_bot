"""Human vs the Opponent, the mode of `pong play`, Frame by Frame.

The Human plays in the Agent's place, on the right, with the same Paddle at the same
speed, and decides every Frame (issue #5). On the left is the same Opponent as in
training. The mode drives the Game core directly, not through the environment. This
is the mode without the window, so that the window only shows it and reads the
keyboard.
"""

from pong_ai_bot.game import MAX_PADDLE_MOVE, FrameResult, Game, PhysicsState
from pong_ai_bot.opponent import Opponent


class HumanVsOpponent:
    """One Game of a Human on the right against the Opponent on the left."""

    def __init__(self, *, seed: int) -> None:
        """A new Game, from `seed`."""
        self._game = Game.new(seed=seed)
        self._opponent = Opponent(self._game.rng)

    @property
    def state(self) -> PhysicsState:
        return self._game.state

    def play_frame(self, *, up: bool, down: bool) -> FrameResult:
        """Play one Frame, with the Human holding the up key, the down key, or not.

        Each key held moves the Human's Paddle MAX_PADDLE_MOVE that way in this
        Frame, so holding both keys, like holding neither, keeps it still.
        """
        right = MAX_PADDLE_MOVE * (int(up) - int(down))
        left = self._opponent.move(self._game.state)
        return self._game.play_frame(left=left, right=right)
