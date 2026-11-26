"""The Scripted players: non-learning players in the Agent's place, by hand-written
rules (issue #7).

They play the environment as it is, PongEnv, and decide every Step, as the Agent
does, with the same three Actions. They read only each Game's `info`, never its
Observation.
"""

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pong_ai_bot.environment import FRAMES_PER_STEP, Action
from pong_ai_bot.evaluation import Player
from pong_ai_bot.game import (
    BALL_SIZE,
    MAX_PADDLE_MOVE,
    PADDLE_GAP,
    PADDLE_HEIGHT,
    PADDLE_THICKNESS,
    Ball,
    PhysicsState,
    Side,
)
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


_CONTACT_X = 1.0 - PADDLE_GAP - PADDLE_THICKNESS - BALL_SIZE / 2
"""The ball's x when it touches the right Paddle's face, as the Game works it out."""
_EDGE = PADDLE_HEIGHT / 2
"""How far from its Paddle's center the ball must touch it to leave at 45 degrees."""
_REACH = PADDLE_HEIGHT / 2 + BALL_SIZE / 2
"""How far from its Paddle's center the ball can be and still touch it."""
_AIM = (_EDGE + _REACH) / 2
"""How far from its Paddle's center it aims the ball: the middle of the edge."""
_MARGIN = 1e-9
"""Room for the rounding between its own arithmetic and the Game's."""
_BALL_LOWEST = BALL_SIZE / 2
_BALL_HIGHEST = 1.0 - BALL_SIZE / 2
_PADDLE_LOWEST = PADDLE_HEIGHT / 2
_PADDLE_HIGHEST = 1.0 - PADDLE_HEIGHT / 2
_MOVES: dict[Action, float] = {
    Action.STAY: 0.0,
    Action.UP: MAX_PADDLE_MOVE,
    Action.DOWN: -MAX_PADDLE_MOVE,
}
"""How far each Action moves the Paddle up in a Frame, staying first, so that a tie
goes to staying."""


class EdgePredictor:
    """Works out where the ball will reach its Paddle, off the walls too, and
    returns it off the Paddle's edge, at 45 degrees (issue #7). It shows that 18+
    can be won deciding only every Step.

    It aims for the edge that sends the ball toward the half of the Court away from
    the Opponent's Paddle. If it can't get that edge to the ball in time, it takes
    the other edge, and if neither, the hit nearest an edge. While the ball goes
    away, or waits to be served, it goes back to the center.

    It decides every Step, as the Agent does, so it plans its Steps to the Frame
    the ball arrives in: some Steps of moving, and the arrival's own Step, which may
    go on after it.
    """

    def act(
        self, observations: npt.NDArray[Any], infos: Sequence[dict[str, Any]]
    ) -> npt.NDArray[np.int64]:
        return np.array([_edge_predictor_action(info) for info in infos], np.int64)


def _edge_predictor_action(info: dict[str, Any]) -> Action:
    ball: Ball = info["ball"]
    paddle: float = info["right_paddle"]
    if ball.vx <= 0 or ball.x > _CONTACT_X:
        # The ball goes away, waits to be served, or is past the face already.
        return min(_MOVES, key=lambda a: abs(_after_a_step(paddle, a) - 0.5))
    frames = _frames_to_reach_the_face(ball)
    arrival = _off_the_walls(ball.y + ball.vy * (_CONTACT_X - ball.x) / ball.vx)
    # Up, off the top edge, if the Opponent's Paddle is in the bottom half of the
    # Court, and down, off the bottom edge, if it is in the top half.
    aim = _AIM if info["left_paddle"] <= 0.5 else -_AIM
    whole_steps, last_frames = divmod(frames - 1, FRAMES_PER_STEP)
    last_frames += 1
    best_cost, best_action = None, Action.STAY
    for height, first in _heights_after(paddle, whole_steps):
        for last, move in _MOVES.items():
            at_arrival = _clamp_paddle(height + move * last_frames)
            cost = _cost(arrival - at_arrival, aim)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_action = first if whole_steps > 0 else last
    return best_action


def _frames_to_reach_the_face(ball: Ball) -> int:
    """In which Frame from now the ball reaches its Paddle's face, moving in x as
    the Game moves it, one Frame at a time."""
    x, frames = ball.x, 0
    while x <= _CONTACT_X:
        x += ball.vx
        frames += 1
    return frames


def _off_the_walls(y: float) -> float:
    """Where the ball's center is when it would be at `y` without the walls, after
    bouncing off them like a mirror, as many times as it takes."""
    span = _BALL_HIGHEST - _BALL_LOWEST
    y = (y - _BALL_LOWEST) % (2 * span)
    return _BALL_LOWEST + (2 * span - y if y > span else y)


def _heights_after(paddle: float, steps: int) -> Iterator[tuple[float, Action]]:
    """Each height its Paddle can get to from `paddle` in `steps` Steps, with the
    Action to choose first on the way there: some Steps of moving one way, or all
    the way to a wall, where it stops."""
    yield paddle, Action.STAY
    for n in range(1, steps + 1):
        for action, sign in ((Action.UP, 1), (Action.DOWN, -1)):
            height = paddle + sign * n * _STEP_MOVE
            if _PADDLE_LOWEST <= height <= _PADDLE_HIGHEST:
                yield height, action
    for wall, action in ((_PADDLE_HIGHEST, Action.UP), (_PADDLE_LOWEST, Action.DOWN)):
        if abs(wall - paddle) <= steps * _STEP_MOVE:
            yield wall, action


def _cost(offset: float, aim: float) -> tuple[bool, float, bool, float]:
    """How bad it is for the ball to touch its Paddle `offset` above the center,
    aiming at `aim` above it. Lower is better: a hit first, then at 45 degrees, then
    off the edge aimed at, then near that edge's middle."""
    misses = abs(offset) > _REACH - _MARGIN
    short_of_the_edge = max(_EDGE - _MARGIN - abs(offset), 0.0)
    other_edge = (offset > 0) != (aim > 0)
    return misses, short_of_the_edge, other_edge, abs(offset - aim)


def _after_a_step(paddle: float, action: Action) -> float:
    return _clamp_paddle(paddle + _MOVES[action] * FRAMES_PER_STEP)


def _clamp_paddle(height: float) -> float:
    return min(max(height, _PADDLE_LOWEST), _PADDLE_HIGHEST)


SCRIPTED_PLAYERS: dict[str, Callable[[], Player]] = {
    "random": RandomPlayer,
    "opponent-copy": OpponentCopy,
    "edge-predictor": EdgePredictor,
}
"""Each Scripted player, by the name `pong evaluate` knows it by, with the function
that makes a new one for an Evaluation. A new Scripted player is one more line."""
