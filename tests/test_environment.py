"""The environment, Step by Step, through Gymnasium's API.

Positions are in Court units, y from the bottom wall (0) to the top wall (1). The
image is 84 by 84 pixels, rows counting down from the top wall. Expected numbers are
worked out by hand.
"""

import dataclasses
import itertools
import time
import warnings

import numpy as np
import numpy.typing as npt
import pytest
from gymnasium.utils.env_checker import check_env

from pong_ai_bot.environment import Action, PongEnv
from pong_ai_bot.game import Ball, Game, PhysicsState, Score
from pong_ai_bot.opponent import Opponent

NO_POINTS_YET = Score(left=0, right=0)

Pixel = tuple[int, int]
"""A (row, column) of the image."""


def block(rows: range, columns: range) -> set[Pixel]:
    return {(row, column) for row in rows for column in columns}


def white_pixels(image: npt.NDArray[np.uint8]) -> set[Pixel]:
    rows, columns = np.nonzero(image)
    return set(zip(rows.tolist(), columns.tolist(), strict=True))


def test_the_environment_passes_gymnasiums_check_env_with_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The environment draws nothing itself, so there is no render to check:
        # the window draws every Frame through the on_frame hook.
        check_env(PongEnv(), skip_render_check=True)


def test_reset_gives_the_observation_and_the_physics_state_of_a_new_game() -> None:
    env = PongEnv()

    observation, info = env.reset(seed=0)

    # The ball waits in the center of the Court, whose center pixel is (42, 42), and
    # both Paddles are in the center too: 13 pixels tall around row 42, and 3 wide
    # around column 5 on the left and column 78 on the right.
    assert observation.shape == (84, 84)
    assert observation.dtype == np.uint8
    assert set(np.unique(observation).tolist()) == {0, 255}
    assert white_pixels(observation) == (
        block(range(41, 44), range(41, 44))
        | block(range(36, 49), range(4, 7))
        | block(range(36, 49), range(77, 80))
    )
    assert info == {
        "ball": Ball(x=0.5, y=0.5, vx=0.0, vy=0.0),
        "left_paddle": 0.5,
        "right_paddle": 0.5,
        "score": Score(left=0, right=0),
    }


def serve_of(env: PongEnv) -> tuple[float, float]:
    """The velocity the ball is served with, by Frame 32 of a Game just reset."""
    ball = [env.step(Action.STAY)[4] for _ in range(8)][-1]["ball"]
    return ball.vx, ball.vy


def test_reset_without_a_seed_starts_another_game_set_by_the_last_seed() -> None:
    serves = []
    for env in (PongEnv(), PongEnv()):
        env.reset(seed=3)
        serves.append([serve_of(env)])
        for _ in range(2):
            env.reset()
            serves[-1].append(serve_of(env))

    assert serves[0] == serves[1]
    assert len(set(serves[0])) == 3, f"{serves[0]}"


def test_changing_the_info_does_not_change_the_game() -> None:
    env, untouched = PongEnv(), PongEnv()
    _, info = env.reset(seed=4)
    untouched.reset(seed=4)

    # 60 Steps go through the serve and on into play.
    for step in range(60):
        with pytest.raises(dataclasses.FrozenInstanceError):
            info["ball"].x = 0.0
        # Even forced past its frozen fields, a change stays in the copy.
        object.__setattr__(info["ball"], "y", 0.0)
        object.__setattr__(info["score"], "left", 20)
        info["left_paddle"] = info["right_paddle"] = 0.9
        info = env.step(Action.UP)[4]

        assert info == untouched.step(Action.UP)[4], f"Step {step}"


@pytest.mark.parametrize(
    ("actions", "heights"),
    [
        # A Step holds its Action for 4 Frames of 0.02 each.
        ([Action.UP, Action.UP, Action.UP], [0.58, 0.66, 0.74]),
        ([Action.DOWN, Action.DOWN, Action.DOWN], [0.42, 0.34, 0.26]),
        ([Action.STAY, Action.STAY, Action.STAY], [0.5, 0.5, 0.5]),
        # Staying stops the Paddle at once, with no slowing down.
        ([Action.UP, Action.STAY, Action.DOWN], [0.58, 0.58, 0.5]),
    ],
)
def test_each_step_moves_the_right_paddle_by_its_action_for_4_frames(
    actions: list[Action], heights: list[float]
) -> None:
    env = PongEnv()
    env.reset(seed=0)

    moved_to = [env.step(action)[4]["right_paddle"] for action in actions]

    assert moved_to == pytest.approx(heights)


def test_the_observation_is_drawn_after_the_last_frame_of_the_step() -> None:
    env = PongEnv()
    env.reset(seed=0)

    observation = env.step(Action.UP)[0]

    # After 4 Frames up the right Paddle is at 0.58, so its center falls in row
    # (1 - 0.58) * 84 = 35.28, row 35. The ball still waits in the center, and the
    # left Paddle with it.
    assert white_pixels(observation) == (
        block(range(41, 44), range(41, 44))
        | block(range(36, 49), range(4, 7))
        | block(range(29, 42), range(77, 80))
    )


def from_state(ball: Ball, score: Score = NO_POINTS_YET) -> PongEnv:
    """An environment whose Game starts with the given ball and score, and both
    Paddles in the center."""
    env = PongEnv(
        start=PhysicsState(ball=ball, left_paddle=0.5, right_paddle=0.5, score=score)
    )
    env.reset(seed=0)
    return env


# A ball already past the face of a Paddle leaves the Court past it. Moving 0.01 a
# Frame, from an end of the Court, its last part leaves in the Step's 2nd Frame,
# 0.02 past the end: the ball is 0.03 square, so half of it is 0.015.
LEAVING_PAST_THE_LEFT = Ball(x=0.0, y=0.5, vx=-0.01, vy=0.0)
LEAVING_PAST_THE_RIGHT = Ball(x=1.0, y=0.5, vx=0.01, vy=0.0)


@pytest.mark.parametrize(
    ("ball", "reward", "score"),
    [
        (LEAVING_PAST_THE_LEFT, 1.0, Score(left=0, right=1)),
        (LEAVING_PAST_THE_RIGHT, -1.0, Score(left=1, right=0)),
        (Ball(x=0.5, y=0.5, vx=0.01, vy=0.0), 0.0, Score(left=0, right=0)),
    ],
)
def test_the_reward_is_1_for_a_point_won_and_minus_1_for_a_point_lost(
    ball: Ball, reward: float, score: Score
) -> None:
    env = from_state(ball)

    _, got, terminated, truncated, info = env.step(Action.STAY)

    assert (got, info["score"]) == (reward, score)
    assert (terminated, truncated) == (False, False)


@pytest.mark.parametrize("ball", [LEAVING_PAST_THE_LEFT, LEAVING_PAST_THE_RIGHT])
def test_a_point_in_the_middle_of_a_step_does_not_stop_it(ball: Ball) -> None:
    env = from_state(ball)

    info = env.step(Action.UP)[4]

    # The Paddle went on moving up through all 4 Frames, and after the Point the
    # ball waits in the center to be served.
    assert info["right_paddle"] == pytest.approx(0.58)
    assert info["ball"] == Ball(x=0.5, y=0.5, vx=0.0, vy=0.0)


@pytest.mark.parametrize(
    ("ball", "score", "reward", "final_score"),
    [
        (LEAVING_PAST_THE_LEFT, Score(left=0, right=20), 1.0, Score(left=0, right=21)),
        (
            LEAVING_PAST_THE_RIGHT,
            Score(left=20, right=0),
            -1.0,
            Score(left=21, right=0),
        ),
    ],
)
def test_reaching_21_in_the_middle_of_a_step_ends_the_game_at_once(
    ball: Ball, score: Score, reward: float, final_score: Score
) -> None:
    env = from_state(ball, score)

    _, got, terminated, truncated, info = env.step(Action.UP)

    # The 21st Point comes in the Step's 2nd Frame, so the Paddle moved up only twice.
    assert (got, terminated, truncated) == (reward, True, False)
    assert info["score"] == final_score
    assert info["right_paddle"] == pytest.approx(0.54)


def test_a_game_is_cut_off_after_27000_steps() -> None:
    # The ball only goes up and down between the walls, so no Point ever comes.
    env = from_state(Ball(x=0.5, y=0.5, vx=0.0, vy=0.01), Score(left=3, right=5))

    ends = [env.step(Action.STAY)[2:4] for _ in range(27_000)]

    assert ends[:-1] == [(False, False)] * 26_999
    assert ends[-1] == (False, True)


UP_A_FRAME = {Action.UP: 0.02, Action.DOWN: -0.02, Action.STAY: 0.0}


def test_by_default_the_opponent_plays_on_the_left_with_the_games_generator() -> None:
    # The same Game, played Frame by Frame on the Game core itself: a new Game from
    # the same seed, with the Opponent drawing from its generator. 300 Steps are
    # 1,200 Frames, too few for 21 Points of at least 69 Frames each.
    actions = [Action.UP] * 3 + [Action.STAY] * 2 + [Action.DOWN] * 4 + [Action.STAY]
    env = PongEnv()
    env.reset(seed=7)
    game = Game.new(seed=7)
    opponent = Opponent(game.rng)
    balls_that_came = 0

    for step, action in enumerate(itertools.islice(itertools.cycle(actions), 300)):
        info = env.step(action)[4]
        for _ in range(4):
            vx = game.state.ball.vx
            game.play_frame(left=opponent.move(game.state), right=UP_A_FRAME[action])
            balls_that_came += vx >= 0 > game.state.ball.vx
        state = game.state

        assert (
            info["ball"],
            info["left_paddle"],
            info["right_paddle"],
            info["score"],
        ) == (state.ball, state.left_paddle, state.right_paddle, state.score), (
            f"Step {step}"
        )
    # Each ball that came to the Opponent had it draw an offset from the generator.
    assert balls_that_came >= 2, f"{balls_that_came} balls came to the Opponent"


class AlwaysUp:
    """A fake player for the left side, that always moves its Paddle up 0.02."""

    def move(self, state: PhysicsState) -> float:
        return 0.02


class Watching:
    """A fake player for the left side, that keeps still and keeps each state it
    is given."""

    def __init__(self) -> None:
        self.states: list[PhysicsState] = []

    def move(self, state: PhysicsState) -> float:
        self.states.append(state)
        return 0.0


def test_a_fake_player_can_take_the_opponents_place_on_the_left() -> None:
    env = PongEnv(left_player=lambda rng: AlwaysUp())
    env.reset(seed=0)

    heights = [env.step(Action.STAY)[4]["left_paddle"] for _ in range(3)]

    # The Opponent would keep its Paddle in the center with the waiting ball.
    assert heights == pytest.approx([0.58, 0.66, 0.74])


def test_the_player_on_the_left_moves_every_frame_by_the_state_at_its_start() -> None:
    player = Watching()
    env = PongEnv(left_player=lambda rng: player)
    env.reset(seed=0)

    env.step(Action.UP)
    env.step(Action.UP)

    assert [state.frame for state in player.states] == list(range(8))
    assert [state.right_paddle for state in player.states] == pytest.approx(
        [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64]
    )


def test_the_on_frame_hook_gets_the_state_after_every_frame() -> None:
    seen: list[PhysicsState] = []
    env = PongEnv(on_frame=seen.append)
    env.reset(seed=0)

    env.step(Action.UP)
    env.step(Action.DOWN)

    assert [state.frame for state in seen] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [state.right_paddle for state in seen] == pytest.approx(
        [0.52, 0.54, 0.56, 0.58, 0.56, 0.54, 0.52, 0.5]
    )


def test_the_on_frame_hook_sees_the_last_frame_of_a_game_and_no_more() -> None:
    seen: list[PhysicsState] = []
    env = PongEnv(
        on_frame=seen.append,
        start=PhysicsState(
            ball=LEAVING_PAST_THE_LEFT,
            left_paddle=0.5,
            right_paddle=0.5,
            score=Score(left=0, right=20),
        ),
    )
    env.reset(seed=0)

    env.step(Action.STAY)

    assert [(state.frame, state.score) for state in seen] == [
        (1, Score(left=0, right=20)),
        (2, Score(left=0, right=21)),
    ]


def test_without_an_on_frame_hook_the_environment_never_waits() -> None:
    env = PongEnv()
    env.reset(seed=0)
    started = time.perf_counter()

    for _ in range(300):
        env.step(Action.STAY)

    # 300 Steps are 1,200 Frames: 20 seconds at the window's 60 Frames a second.
    assert time.perf_counter() - started < 2.0
