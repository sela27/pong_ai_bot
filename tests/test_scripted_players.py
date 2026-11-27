"""The Scripted players, through `act`, on batches of real Games' Observations and
`info`s, or of `info`s of given physics states.

Positions are in Court units, y from the bottom wall (0) to the top wall (1).
Expected numbers are worked out by hand:

- The ball's center touches the right Paddle's face at x = 1 - 0.05 (the gap) -
  0.03 (the Paddle) - 0.015 (half the ball) = 0.905.
- The ball's center stays between 0.015 and 0.985, half the ball off each wall.
- A Paddle is 0.15 tall. It touches the ball up to 0.075 + 0.015 = 0.09 from its
  center, and from 0.075 out, its edge, the ball leaves at 45 degrees. Its center
  stays between 0.075 and 0.925.
- A Step moves the Paddle 4 Frames of 0.02: 0.08.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pong_ai_bot.environment import Action, PongEnv
from pong_ai_bot.game import Ball, PhysicsState, Score
from pong_ai_bot.observation import OBSERVATION_SIZE
from pong_ai_bot.scripted_players import EdgePredictor, RandomPlayer


def new_games(seeds: list[int]) -> tuple[npt.NDArray[Any], list[dict[str, Any]]]:
    """The first Observations and `info`s of the Games of these seeds, as a batch."""
    starts = [PongEnv().reset(seed=seed) for seed in seeds]
    observations = np.stack([observation for observation, _ in starts])
    return observations, [info for _, info in starts]


def test_the_random_player_chooses_one_action_for_each_game_in_the_batch() -> None:
    observations, infos = new_games([1, 2, 3, 4, 5])

    actions = np.asarray(RandomPlayer().act(observations, infos))

    assert actions.shape == (5,)
    assert set(actions.tolist()) <= {Action.UP, Action.DOWN, Action.STAY}


def test_each_action_comes_up_about_a_third_of_the_time() -> None:
    player = RandomPlayer()
    observations, infos = new_games(list(range(10)))

    # 300 Steps of 10 Games: 3,000 Actions, 1,000 of each if they came up evenly.
    actions = np.concatenate(
        [np.asarray(player.act(observations, infos)) for _ in range(300)]
    )

    counts = {action: int((actions == action).sum()) for action in Action}
    assert all(900 <= count <= 1_100 for count in counts.values()), counts


def test_every_new_random_player_chooses_the_same_actions() -> None:
    # Its seed is fixed, so its Evaluation gives the same results every time.
    first, second = RandomPlayer(), RandomPlayer()
    observations, infos = new_games(list(range(10)))

    for _ in range(20):
        first_actions = np.asarray(first.act(observations, infos)).tolist()
        second_actions = np.asarray(second.act(observations, infos)).tolist()
        assert first_actions == second_actions


# The edge predictor, from known physics states in `info`, with the numbers at the
# top of this file.


def predictor_info(
    ball: Ball, *, paddle: float, opponent_paddle: float
) -> dict[str, Any]:
    """The `info` of a Game in this state, as the environment gives it."""
    return {
        "ball": ball,
        "left_paddle": opponent_paddle,
        "right_paddle": paddle,
        "score": Score(left=0, right=0),
    }


def predictor_actions(infos: list[dict[str, Any]]) -> list[int]:
    """The edge predictor's Actions for a batch of these `info`s. It never looks at
    the Observations, so they are all black."""
    observations = np.zeros((len(infos), OBSERVATION_SIZE, OBSERVATION_SIZE), np.uint8)
    return np.asarray(EdgePredictor().act(observations, infos)).tolist()


def predictor_action(ball: Ball, *, paddle: float, opponent_paddle: float) -> int:
    """The edge predictor's Action in one Game in this state."""
    info = predictor_info(ball, paddle=paddle, opponent_paddle=opponent_paddle)
    [action] = predictor_actions([info])
    return action


@pytest.mark.parametrize(
    ("paddle", "action"),
    [(0.5, Action.DOWN), (0.34, Action.UP), (0.42, Action.STAY)],
)
def test_the_predictor_moves_its_paddle_so_that_the_ball_arrives_at_its_edge(
    paddle: float, action: Action
) -> None:
    # The ball comes level at 0.5, from x 0.5 at 0.01 a Frame, so it reaches the
    # face at 0.905 in its 41st Frame, 10 Steps away. The Opponent's Paddle is low,
    # so it sends the ball up, off the top edge: its Paddle's center 0.075 to 0.09
    # below the ball, between 0.41 and 0.425. From 0.5, one Step down gets there,
    # to 0.42, and from 0.34 one Step up. At 0.42 it is there already.
    ball = Ball(x=0.5, y=0.5, vx=0.01, vy=0.0)

    assert predictor_action(ball, paddle=paddle, opponent_paddle=0.3) == action


@pytest.mark.parametrize(
    ("paddle", "action"),
    [(0.5, Action.UP), (0.66, Action.DOWN), (0.58, Action.STAY)],
)
def test_the_predictor_sends_the_ball_toward_the_side_away_from_the_opponent(
    paddle: float, action: Action
) -> None:
    # The same ball, but the Opponent's Paddle is high, so it sends the ball down,
    # off the bottom edge: its Paddle's center 0.075 to 0.09 above the ball, between
    # 0.575 and 0.59. One Step up from 0.5 gets there, to 0.58.
    ball = Ball(x=0.5, y=0.5, vx=0.01, vy=0.0)

    assert predictor_action(ball, paddle=paddle, opponent_paddle=0.7) == action


@pytest.mark.parametrize(
    ("ball", "paddle", "opponent_paddle", "action"),
    [
        # Rising 0.01 a Frame from 0.8, the ball would get to 0.8 + 40.5 * 0.01 =
        # 1.205 when it reaches the face, but it bounces off the top wall at 0.985
        # and gets to 0.985 - 0.22 = 0.765. Sent up, its Paddle's center belongs
        # between 0.675 and 0.69: one Step down from 0.76, to 0.68.
        (Ball(x=0.5, y=0.8, vx=0.01, vy=0.01), 0.76, 0.3, Action.DOWN),
        # Falling 0.01 a Frame from 0.2, it would get to -0.205, but it bounces off
        # the bottom wall at 0.015 and gets to 0.015 + 0.22 = 0.235. Sent down, its
        # Paddle's center belongs between 0.31 and 0.325: one Step up from 0.24.
        (Ball(x=0.5, y=0.2, vx=0.01, vy=-0.01), 0.24, 0.7, Action.UP),
    ],
)
def test_the_predictor_works_out_where_the_ball_arrives_off_a_wall(
    ball: Ball, paddle: float, opponent_paddle: float, action: Action
) -> None:
    assert (
        predictor_action(ball, paddle=paddle, opponent_paddle=opponent_paddle) == action
    )


def test_the_predictor_counts_only_the_frames_before_the_ball_arrives() -> None:
    # From x 0.89 at 0.01 a Frame, the ball reaches the face in the 2nd Frame of
    # this Step, level at 0.5. Sent up, its Paddle's center belongs between 0.41
    # and 0.425. Going down from 0.455, it is at 0.415 by then, though by the end of
    # the Step it would be at 0.375, too low to touch the ball at all.
    ball = Ball(x=0.89, y=0.5, vx=0.01, vy=0.0)

    assert predictor_action(ball, paddle=0.455, opponent_paddle=0.3) == Action.DOWN


def test_the_predictor_can_stop_its_paddle_on_a_wall_on_the_way() -> None:
    # From x 0.85 at 0.01 a Frame, the ball reaches the face in the 2nd Frame of
    # the next Step, low, at 0.03. Its Paddle's center can't get below 0.075, so it
    # can't send the ball up. To send it down, off the bottom edge, its center
    # belongs between 0.105 and 0.12. Going down from 0.13 this Step, it stops on the
    # wall at 0.075, and 2 Frames up the next Step get it to 0.115. Staying instead,
    # it gets no nearer than 0.09, 2 Frames down: off its bottom half, not its edge.
    ball = Ball(x=0.85, y=0.03, vx=0.01, vy=0.0)

    assert predictor_action(ball, paddle=0.13, opponent_paddle=0.3) == Action.DOWN


@pytest.mark.parametrize(
    ("ball", "paddle", "action"),
    [
        (Ball(x=0.5, y=0.8, vx=-0.01, vy=0.01), 0.7, Action.DOWN),
        (Ball(x=0.5, y=0.2, vx=-0.01, vy=-0.01), 0.3, Action.UP),
        # 0.03 from the center: a Step down would leave it 0.05 below it.
        (Ball(x=0.5, y=0.8, vx=-0.01, vy=0.01), 0.53, Action.STAY),
        # The ball waiting in the center to be served.
        (Ball(x=0.5, y=0.5, vx=0.0, vy=0.0), 0.34, Action.UP),
    ],
)
def test_as_the_ball_goes_away_the_predictor_goes_back_to_the_center(
    ball: Ball, paddle: float, action: Action
) -> None:
    assert predictor_action(ball, paddle=paddle, opponent_paddle=0.5) == action


def test_the_predictor_chooses_an_action_for_each_game_in_the_batch() -> None:
    # Games from the tests above, each with its own Action, in the batch's order.
    infos = [
        predictor_info(Ball(0.5, 0.8, -0.01, 0.01), paddle=0.7, opponent_paddle=0.5),
        predictor_info(Ball(0.5, 0.5, 0.01, 0.0), paddle=0.34, opponent_paddle=0.3),
        predictor_info(Ball(0.5, 0.5, 0.01, 0.0), paddle=0.58, opponent_paddle=0.7),
        predictor_info(Ball(0.5, 0.2, 0.01, -0.01), paddle=0.24, opponent_paddle=0.7),
    ]

    assert predictor_actions(infos) == [
        Action.DOWN,
        Action.UP,
        Action.STAY,
        Action.UP,
    ]


@pytest.mark.parametrize(
    ("ball", "paddle", "opponent_paddle", "sent"),
    [
        # Off the top wall to 0.765. The Opponent's Paddle tracks the ball in the
        # top half of the Court, so the predictor sends it down.
        (Ball(x=0.5, y=0.8, vx=0.01, vy=0.01), 0.76, 0.8, "down"),
        # Off the bottom wall to 0.235, with the Opponent in the bottom half.
        (Ball(x=0.5, y=0.2, vx=0.01, vy=-0.01), 0.24, 0.2, "up"),
    ],
)
def test_in_a_game_the_predictor_returns_the_ball_at_45_degrees(
    ball: Ball, paddle: float, opponent_paddle: float, sent: str
) -> None:
    # The ball crossed in 0.81 / 0.01 = 81 Frames, so it comes back across in 79,
    # at 0.81 / 79 a Frame, and at 45 degrees just as fast up or down.
    env = PongEnv(
        start=PhysicsState(
            ball=ball,
            left_paddle=opponent_paddle,
            right_paddle=paddle,
            score=Score(left=0, right=0),
        )
    )
    observation, info = env.reset(seed=0)
    predictor = EdgePredictor()
    for _ in range(12):
        [action] = np.asarray(predictor.act(observation[np.newaxis], [info]))
        observation, reward, _, _, info = env.step(action)
        assert reward == 0, "the predictor missed the ball"
        if info["ball"].vx < 0:
            break

    speed = 0.81 / 79
    vy = -speed if sent == "down" else speed
    assert (info["ball"].vx, info["ball"].vy) == pytest.approx((-speed, vy))
