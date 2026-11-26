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
from pong_ai_bot.scripted_players import EdgePredictor, OpponentCopy, RandomPlayer


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


# The copy of the Opponent, from known physics states in `info`. Positions are in
# Court units, y from the bottom wall (0) to the top wall (1). Holding an Action for
# a Step's 4 Frames moves the Paddle 4 x 0.02 = 0.08, and the Opponent's rule would
# move it toward its target by at most that much, stopping right on it.

BASE_SPEED = 0.0135


def info_of(
    seed: int, *, ball_y: float, right_paddle: float, vx: float = BASE_SPEED
) -> dict[str, Any]:
    """The `info` of the Game of `seed`, as the Evaluation hands it: the ball in the
    middle of the Court at `ball_y`, coming toward the right Paddle unless `vx` is
    negative."""
    return {
        "ball": Ball(x=0.5, y=ball_y, vx=vx, vy=0.0),
        "left_paddle": 0.5,
        "right_paddle": right_paddle,
        "score": Score(left=0, right=0),
        "seed": seed,
    }


def actions_of(player: OpponentCopy, infos: list[dict[str, Any]]) -> list[int]:
    """The Actions the player chooses for a batch of Games, from their `info`s."""
    observations = np.zeros((len(infos), 84, 84), dtype=np.uint8)
    return np.asarray(player.act(observations, infos)).tolist()


@pytest.mark.parametrize(
    ("ball_y", "action"),
    [
        # The rule would end at 0.58: up ends there, staying 0.08 short.
        (0.6, Action.UP),
        # The rule would end at 0.53: staying ends 0.03 short, up 0.05 past.
        (0.53, Action.STAY),
        (0.5, Action.STAY),
        # The rule would end at 0.47: staying ends 0.03 short, down 0.05 past.
        (0.47, Action.STAY),
        # The rule would end at 0.45: down ends 0.03 past, staying 0.05 short.
        (0.45, Action.DOWN),
        # The rule would end at 0.42: down ends there.
        (0.2, Action.DOWN),
    ],
)
def test_the_copy_chooses_the_action_that_ends_the_step_nearest_where_the_rule_would(
    ball_y: float, action: Action
) -> None:
    # With no offset width, the rule aims its Paddle's center at the ball.
    player = OpponentCopy(width=0.0)

    actions = actions_of(player, [info_of(1, ball_y=ball_y, right_paddle=0.5)])

    assert actions == [action]


def level_and_coming(seeds: list[int], vx: float = BASE_SPEED) -> list[dict[str, Any]]:
    """The `info`s of the Games of these seeds, each with its Paddle level with the
    ball, at 0.5, and the ball coming toward it unless `vx` is negative."""
    return [info_of(seed, ball_y=0.5, right_paddle=0.5, vx=vx) for seed in seeds]


@pytest.mark.parametrize(
    # The fewest and the most Games that move up, move down and stay.
    ("width", "fewest_moving", "most_moving"),
    [
        # Offsets up to 0.075 from the ball. It moves up for an offset over 0.04,
        # 0.035 of the 0.15 they spread over: 7 in 30, about 233 of 1000 Games, give
        # or take about 13. As many move down, and 16 in 30 stay, about 533 give or
        # take about 16. The bounds below are 4 of those away.
        (1.0, (180, 180, 470), (287, 287, 597)),
        # Offsets up to 0.0375, never over 0.04: it never moves.
        (0.5, (0, 0, 1000), (0, 0, 1000)),
    ],
)
def test_level_with_a_ball_that_comes_the_copy_moves_by_its_offset_within_the_width(
    width: float,
    fewest_moving: tuple[int, int, int],
    most_moving: tuple[int, int, int],
) -> None:
    player = OpponentCopy(width=width)

    actions = actions_of(player, level_and_coming(list(range(1000))))

    counts = tuple(actions.count(action) for action in Action)
    assert all(
        low <= count <= high
        for low, count, high in zip(fewest_moving, counts, most_moving, strict=True)
    ), f"up, down and stay: {counts}"


def test_every_new_copy_chooses_the_same_for_each_game_alone_or_among_others() -> None:
    # Its offsets are drawn for each Game by the Game's seed, so its Evaluation
    # gives the same results every time. About half the Games move.
    seeds = list(range(200))

    together = actions_of(OpponentCopy(width=1.0), level_and_coming(seeds))
    alone = [
        actions_of(OpponentCopy(width=1.0), level_and_coming([seed]))[0]
        for seed in seeds
    ]
    reversed_batch = actions_of(OpponentCopy(width=1.0), level_and_coming(seeds[::-1]))

    assert 50 <= sum(action != Action.STAY for action in together) <= 150, together
    assert alone == together
    assert reversed_batch[::-1] == together


def test_each_game_keeps_its_offset_as_the_ball_comes_and_goes_and_games_leave() -> (
    None
):
    # Games leave the batch as they end; the rest keep their order. The ball stays
    # level with each Paddle, so each Game chooses the same as long as its offset
    # stays the same.
    player = OpponentCopy(width=1.0)
    seeds = list(range(200))
    first = dict(zip(seeds, actions_of(player, level_and_coming(seeds)), strict=True))

    for step in range(2, 8):
        going = [seed for seed in seeds if seed % step]
        vx = BASE_SPEED if step < 5 else -BASE_SPEED
        actions = actions_of(player, level_and_coming(going, vx=vx))

        assert actions == [first[seed] for seed in going], f"Step {step}"


@pytest.mark.parametrize("comes_after", ["a return", "a serve toward it"])
def test_each_ball_that_comes_gets_a_new_offset(comes_after: str) -> None:
    # Between two balls that come level, the other side returns the first, or it
    # misses it and the ball waits in the center, at 0.5, to be served back. Each
    # time an offset over 0.04 from the ball moves it up, one under -0.04 down: 7 in
    # 30 each, and 16 in 30 it stays. The second Action differs from the first in
    # 1 - (7/30)^2 - (7/30)^2 - (16/30)^2 = 546/900 of the Games, about 121 of 200,
    # give or take about 7. The bounds below are about 6 of those away.
    player = OpponentCopy(width=1.0)
    seeds = list(range(200))
    first = actions_of(player, level_and_coming(seeds))
    actions_of(player, level_and_coming(seeds, vx=-BASE_SPEED))
    if comes_after == "a serve toward it":
        actions_of(player, level_and_coming(seeds, vx=0.0))

    second = actions_of(player, level_and_coming(seeds))

    changed = sum(a != b for a, b in zip(first, second, strict=True))
    assert 80 <= changed <= 160, f"{changed} of 200 changed"
