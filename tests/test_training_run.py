"""A Run's training loop, in a temporary directory: the files it saves at its Steps,
the fixed inputs of its Q graph, its stop at the Budget, and its request that
Windows stays awake (issue #10).

The learner is a fake, with no network and no optimizer: creating the real one's
optimizer takes seconds, too long for a fast test. So are the request to Windows,
and TensorBoard's writer, which takes seconds to import: the graphs are checked by
eye in TensorBoard, on a test environment's Run (#17). The environment is test
environment 1, whose every Game is one Step long, unless a test says otherwise.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import torch
from gymnasium import spaces

from pong_ai_bot.run_directory import Commit, RunDirectory
from pong_ai_bot.run_settings import RunSettings
from pong_ai_bot.test_environments import ConstantRewardEnv, TwoStepEnv
from pong_ai_bot.training import Transition, train_run

A_COMMIT = Commit(
    sha="0123456789abcdef0123456789abcdef01234567", uncommitted_changes=False
)
"""The commit the tests' Runs are created at, rather than asking git for it."""


@pytest.fixture(autouse=True)
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The Runs' directory for one test, a temporary one."""
    root = tmp_path / "runs"
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(root))
    return root


def new_run(settings: RunSettings) -> RunDirectory:
    return RunDirectory.create(
        player="test-env-1",
        name="tiny",
        seed=1,
        settings=settings,
        commit=A_COMMIT,
        now=datetime(2026, 11, 27, 14, 30),
    )


class Awake:
    """A fake of the request that keeps Windows awake, which records whether it is
    held."""

    def __init__(self) -> None:
        self.held = False
        self.requests = 0

    @contextmanager
    def __call__(self) -> Iterator[None]:
        self.requests += 1
        self.held = True
        try:
            yield
        finally:
            self.held = False


class NoGraphs:
    """A fake of TensorBoard's writer, which writes nothing."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        pass


class CountingLearner:
    """A fake learner: it always chooses Action 0, and its state is how many
    transitions it was given."""

    def __init__(self) -> None:
        self.given = 0

    def choose_action(self, observation: Any, epsilon: float) -> int:
        return 0

    def learn(self, transition: Transition) -> None:
        self.given += 1

    def q_values(self, observations: Any) -> npt.NDArray[np.float32]:
        return np.zeros((len(observations), 1), np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"given": self.given}

    def network_state_dict(self) -> dict[str, Any]:
        return {"weights after": self.given}


def test_a_run_saves_its_files_at_step_0_every_so_many_steps_and_its_budget() -> None:
    # Checkpoints at every multiple of 5 up to 23, and Milestones at 0, at every
    # multiple of 10, and at the Run's end, 23.
    run = new_run(
        RunSettings(
            budget=23, checkpoint_every=5, checkpoints_kept=10, milestone_every=10
        )
    )
    learner = CountingLearner()

    train_run(run, ConstantRewardEnv(), learner, stay_awake=Awake(), writer=NoGraphs())

    assert [saved.step for saved in run.checkpoints()] == [5, 10, 15, 20]
    assert [saved.step for saved in run.milestones()] == [0, 10, 20, 23]
    # The Run stops at its Budget: 23 Steps, each learned from.
    assert learner.given == 23


def test_a_checkpoint_holds_the_learner_s_state_its_step_and_no_resumes() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
    )

    first, second = run.checkpoints()
    assert torch.load(first.path) == {"learner": {"given": 5}, "step": 5, "resumes": 0}
    assert torch.load(second.path) == {
        "learner": {"given": 10},
        "step": 10,
        "resumes": 0,
    }


def test_a_milestone_holds_the_network_s_weights_after_its_step() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
    )

    # The first before any Step, and the last at the end of the Run.
    assert [torch.load(saved.path) for saved in run.milestones()] == [
        {"weights after": 0},
        {"weights after": 10},
        {"weights after": 12},
    ]


def test_a_budget_on_a_milestone_s_step_saves_that_milestone_once() -> None:
    # The end of the Run falls on the Milestone of Step 20.
    run = new_run(RunSettings(budget=20, checkpoint_every=5, milestone_every=10))

    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
    )

    assert [saved.step for saved in run.milestones()] == [0, 10, 20]
    assert torch.load(run.milestones()[-1].path) == {"weights after": 20}


class AwakeCheckingLearner(CountingLearner):
    """A fake learner that records, at each Step, whether Windows is asked to stay
    awake, and stops the Run as Ctrl+C would at its Step `interrupted_at`, if
    given."""

    def __init__(self, awake: Awake, *, interrupted_at: int | None = None) -> None:
        super().__init__()
        self._awake = awake
        self._interrupted_at = interrupted_at
        self.held_at_each_step: list[bool] = []

    def learn(self, transition: Transition) -> None:
        super().learn(transition)
        self.held_at_each_step.append(self._awake.held)
        if self.given == self._interrupted_at:
            raise KeyboardInterrupt


def test_windows_is_asked_to_stay_awake_through_the_run_and_released_at_its_end() -> (
    None
):
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    awake = Awake()
    learner = AwakeCheckingLearner(awake)

    train_run(run, ConstantRewardEnv(), learner, stay_awake=awake, writer=NoGraphs())

    assert learner.held_at_each_step == [True] * 12
    assert awake.requests == 1
    assert not awake.held


def test_the_request_to_stay_awake_is_released_when_the_run_is_stopped() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    awake = Awake()
    learner = AwakeCheckingLearner(awake, interrupted_at=7)

    with pytest.raises(KeyboardInterrupt):
        train_run(
            run, ConstantRewardEnv(), learner, stay_awake=awake, writer=NoGraphs()
        )

    assert learner.held_at_each_step == [True] * 7
    assert not awake.held


def test_a_new_run_saves_1000_inputs_for_its_q_graph_as_its_games_give_them() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    train_run(
        run, TwoStepEnv(), CountingLearner(), stay_awake=Awake(), writer=NoGraphs()
    )

    # Each Game of test environment 3 gives its first input, -1, at its reset,
    # and its second, +1, after its first Step: the two inputs a player chooses
    # from, a Game after another.
    expected = np.array([[-1.0], [1.0]] * 500, np.float32)
    assert np.array_equal(run.q_graph_inputs(), expected)


class LastActionEnv(gym.Env[npt.NDArray[np.float32], np.int64]):
    """A fake environment of three Actions whose Games never end, and whose input is
    the last Action played, or -1 before the first."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 2.0, (1,), np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        return np.array([-1.0], np.float32), {}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[npt.NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        return np.array([float(action)], np.float32), 0.0, False, False, {}


def test_the_q_graph_s_inputs_come_from_a_game_of_random_actions() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    # The learner always chooses Action 0, and the Game isn't its own.
    train_run(
        run, LastActionEnv(), CountingLearner(), stay_awake=Awake(), writer=NoGraphs()
    )

    # After the input of the reset, the 999 Actions played: each about a third
    # of them, 333.
    inputs = run.q_graph_inputs()
    assert inputs[0, 0] == -1.0
    counts = [int(np.sum(inputs[1:, 0] == action)) for action in range(3)]
    assert sum(counts) == 999
    assert min(counts) > 250, f"each Action's count of the 999: {counts}"
