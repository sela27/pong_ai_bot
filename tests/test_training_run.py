"""A Run's training loop, in a temporary directory: the files it saves at its Steps,
its stop at the Budget or by Ctrl+C, and its request that Windows stays awake
(issue #10).

The learner is a fake, with no network and no optimizer: creating the real one's
optimizer takes seconds, too long for a fast test. So is the request to Windows.
The environment is test environment 1, whose every Game is one Step long. Ctrl+C
is pressed without a real signal: Python calls the handler of SIGINT when Ctrl+C is
pressed, and the tests call it themselves, while the Run plays a chosen Step.
"""

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch

from pong_ai_bot.run_directory import Commit, RunDirectory
from pong_ai_bot.run_settings import RunSettings
from pong_ai_bot.test_environments import ConstantRewardEnv
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


class CountingLearner:
    """A fake learner: it always chooses Action 0, and its state is how many
    transitions it was given."""

    def __init__(self) -> None:
        self.given = 0

    def choose_action(self, observation: Any, epsilon: float) -> int:
        return 0

    def learn(self, transition: Transition) -> None:
        self.given += 1

    def state_dict(self) -> dict[str, Any]:
        return {"given": self.given}

    def network_state_dict(self) -> dict[str, Any]:
        return {"weights after": self.given}


def press_ctrl_c() -> None:
    """Press Ctrl+C, as Python sees it: it calls the handler of SIGINT."""
    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler)
    handler(signal.SIGINT, None)


class CtrlCLearner(CountingLearner):
    """A fake learner that presses Ctrl+C, `presses` times, while it learns from
    its Step `at`, if given: as the developer would, while the Run plays that
    Step."""

    def __init__(self, *, at: int | None, presses: int = 1) -> None:
        super().__init__()
        self._at = at
        self._presses = presses

    def learn(self, transition: Transition) -> None:
        super().learn(transition)
        if self.given == self._at:
            for _ in range(self._presses):
                press_ctrl_c()


def test_a_run_saves_its_files_at_step_0_every_so_many_steps_and_its_budget() -> None:
    # Checkpoints at every multiple of 5 up to 23, and Milestones at 0, at every
    # multiple of 10, and at the Run's end, 23.
    run = new_run(
        RunSettings(
            budget=23, checkpoint_every=5, checkpoints_kept=10, milestone_every=10
        )
    )
    learner = CountingLearner()

    train_run(run, ConstantRewardEnv(), learner, stay_awake=Awake())

    assert [saved.step for saved in run.checkpoints()] == [5, 10, 15, 20]
    assert [saved.step for saved in run.milestones()] == [0, 10, 20, 23]
    # The Run stops at its Budget: 23 Steps, each learned from.
    assert learner.given == 23


def test_a_checkpoint_holds_the_learner_s_state_its_step_and_no_resumes() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    train_run(run, ConstantRewardEnv(), CountingLearner(), stay_awake=Awake())

    first, second = run.checkpoints()
    assert torch.load(first.path) == {"learner": {"given": 5}, "step": 5, "resumes": 0}
    assert torch.load(second.path) == {
        "learner": {"given": 10},
        "step": 10,
        "resumes": 0,
    }


def test_a_milestone_holds_the_network_s_weights_after_its_step() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    train_run(run, ConstantRewardEnv(), CountingLearner(), stay_awake=Awake())

    # The first before any Step, and the last at the end of the Run.
    assert [torch.load(saved.path) for saved in run.milestones()] == [
        {"weights after": 0},
        {"weights after": 10},
        {"weights after": 12},
    ]


def test_a_budget_on_a_milestone_s_step_saves_that_milestone_once() -> None:
    # The end of the Run falls on the Milestone of Step 20.
    run = new_run(RunSettings(budget=20, checkpoint_every=5, milestone_every=10))

    train_run(run, ConstantRewardEnv(), CountingLearner(), stay_awake=Awake())

    assert [saved.step for saved in run.milestones()] == [0, 10, 20]
    assert torch.load(run.milestones()[-1].path) == {"weights after": 20}


class AwakeCheckingLearner(CtrlCLearner):
    """A fake learner that records, at each Step, whether Windows is asked to stay
    awake, and presses Ctrl+C as CtrlCLearner does, if `at` is given."""

    def __init__(
        self, awake: Awake, *, at: int | None = None, presses: int = 1
    ) -> None:
        super().__init__(at=at, presses=presses)
        self._awake = awake
        self.held_at_each_step: list[bool] = []

    def learn(self, transition: Transition) -> None:
        self.held_at_each_step.append(self._awake.held)
        super().learn(transition)


def test_windows_is_asked_to_stay_awake_through_the_run_and_released_at_its_end() -> (
    None
):
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    awake = Awake()
    learner = AwakeCheckingLearner(awake)

    train_run(run, ConstantRewardEnv(), learner, stay_awake=awake)

    assert learner.held_at_each_step == [True] * 12
    assert awake.requests == 1
    assert not awake.held


@pytest.mark.parametrize("presses", [1, 2])
def test_the_request_to_stay_awake_is_released_when_ctrl_c_stops_the_run(
    presses: int,
) -> None:
    # Once the Run's Step is over, or at once.
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    awake = Awake()
    learner = AwakeCheckingLearner(awake, at=7, presses=presses)

    try:
        train_run(run, ConstantRewardEnv(), learner, stay_awake=awake)
    except KeyboardInterrupt:
        pass

    assert learner.held_at_each_step == [True] * 7
    assert not awake.held


def test_ctrl_c_stops_the_run_after_its_step_with_a_full_checkpoint_there() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    learner = CtrlCLearner(at=7)
    handler_before = signal.getsignal(signal.SIGINT)

    stopped_at = train_run(run, ConstantRewardEnv(), learner, stay_awake=Awake())

    # Step 7 is played to its end and learned from, and the Run stops there.
    assert stopped_at == 7
    assert learner.given == 7
    assert [saved.step for saved in run.checkpoints()] == [5, 7]
    assert torch.load(run.checkpoints()[-1].path) == {
        "learner": {"given": 7},
        "step": 7,
        "resumes": 0,
    }
    # A stop is not the Run's end, so it saves no Milestone.
    assert [saved.step for saved in run.milestones()] == [0]
    # Ctrl+C is Python's again once the Run is over.
    assert signal.getsignal(signal.SIGINT) is handler_before


def test_ctrl_c_in_a_step_with_its_checkpoint_saved_already_saves_it_once() -> None:
    # Step 10 has a checkpoint every 5 Steps, and a Milestone every 10.
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    stopped_at = train_run(
        run, ConstantRewardEnv(), CtrlCLearner(at=10), stay_awake=Awake()
    )

    assert stopped_at == 10
    assert [saved.step for saved in run.checkpoints()] == [5, 10]
    assert torch.load(run.checkpoints()[-1].path) == {
        "learner": {"given": 10},
        "step": 10,
        "resumes": 0,
    }
    assert [saved.step for saved in run.milestones()] == [0, 10]


def test_a_second_ctrl_c_stops_the_run_at_once_with_no_checkpoint() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    handler_before = signal.getsignal(signal.SIGINT)

    # Both in Step 7, before its end.
    with pytest.raises(KeyboardInterrupt):
        train_run(
            run, ConstantRewardEnv(), CtrlCLearner(at=7, presses=2), stay_awake=Awake()
        )

    assert [saved.step for saved in run.checkpoints()] == [5]
    assert signal.getsignal(signal.SIGINT) is handler_before
