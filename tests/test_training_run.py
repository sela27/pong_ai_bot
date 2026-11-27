"""A Run's training loop, in a temporary directory: the files it saves at its Steps,
the fixed inputs of its Q graph, its stop at the Budget, at its first Milestone of
19 or more, or by Ctrl+C, its one Confirmation, and its request that Windows stays
awake (issues #10, #13, #14).

The learner is a fake, with no network and no optimizer: creating the real one's
optimizer takes seconds, too long for a fast test. So are the request to Windows,
and TensorBoard's writer, which takes seconds to import: the graphs are checked by
eye in TensorBoard, on a test environment's Run (#17). So is the Evaluation, which
plays no Game: no test ever plays the Confirmation's set (issue #16). The
environment is test
environment 1, whose every Game is one Step long, unless a test says otherwise.
Ctrl+C is pressed without a real signal: Python calls the handler of SIGINT when
Ctrl+C is pressed, and the tests call it themselves, while the Run plays a chosen
Step.
"""

import math
import shutil
import signal
from collections.abc import Callable, Iterator
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
from helpers import A_COMMIT, press_ctrl_c

from pong_ai_bot.evaluation import Evaluation, EvaluationStopped, GameResult, SeedSet
from pong_ai_bot.run_directory import RunDirectory, SavedFile
from pong_ai_bot.run_settings import RunSettings
from pong_ai_bot.test_environments import ConstantRewardEnv, TwoStepEnv
from pong_ai_bot.training import ConfirmedRunError, RunStop, resume_run, train_run
from pong_ai_bot.transition import Transition


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

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.given = state["given"]

    def network_state_dict(self) -> dict[str, Any]:
        return {"weights after": self.given}


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


class FakeEvaluation:
    """A fake of the Evaluation, which plays no Game: it reads the network's
    weights from the Milestone it is given, a CountingLearner's, and gives the
    Milestone of the weights after n Steps two Games, of Score margins n - 21 and
    n - 19, the second cut off.

    It presses Ctrl+C while it evaluates the Milestone of the weights after
    `ctrl_c_at` Steps, if given, and, as the real one does, it asks `stop` before
    each of its two Steps whether to abandon the Evaluation. It records how many
    transitions `learner`, if given, had been given when it evaluated each.

    On the Confirmation's set, it records the Milestone's Step alone, and gives a
    fixed Game, of Score margin 0.
    """

    def __init__(
        self,
        *,
        ctrl_c_at: int | None = None,
        learner: CountingLearner | None = None,
    ) -> None:
        self._ctrl_c_at = ctrl_c_at
        self._learner = learner
        self.evaluated: list[int] = []
        """The Steps of the weights of each Milestone it was given on the regular
        set, in order."""
        self.given_then: list[int] = []
        """How many transitions `learner` had been given at each Evaluation on the
        regular set."""
        self.confirmed: list[int] = []
        """The Step of each Milestone it was given on the Confirmation's set."""

    def __call__(
        self, milestone: SavedFile, *, seed_set: SeedSet, stop: Callable[[], bool]
    ) -> Evaluation:
        if seed_set is SeedSet.CONFIRMATION:
            self.confirmed.append(milestone.step)
            return Evaluation(
                games=(GameResult(seed=0, score_margin=0, cut_off=False),)
            )
        steps = torch.load(milestone.path)["weights after"]
        self.evaluated.append(steps)
        if self._learner is not None:
            self.given_then.append(self._learner.given)
        for _ in range(2):
            if stop():
                raise EvaluationStopped
            if steps == self._ctrl_c_at:
                press_ctrl_c()
        return Evaluation(
            games=(
                GameResult(seed=1_000_000, score_margin=steps - 21, cut_off=False),
                GameResult(seed=1_000_001, score_margin=steps - 19, cut_off=True),
            )
        )


def test_a_run_evaluates_each_milestone_once_as_it_is_saved() -> None:
    # Milestones at Step 0, at every multiple of 10, and at the Run's end, 23.
    run = new_run(
        RunSettings(
            budget=23, checkpoint_every=5, checkpoints_kept=10, milestone_every=10
        )
    )
    learner = CountingLearner()
    evaluation = FakeEvaluation()

    train_run(
        run,
        ConstantRewardEnv(),
        learner,
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=evaluation,
    )

    # Each Milestone of the weights it holds, after its Step.
    assert evaluation.evaluated == [0, 10, 20, 23]
    # Its mean is its Step minus 20, and its line in evaluations.jsonl is kept by
    # its Step.
    means = {step: result.summary.mean for step, result in run.evaluations().items()}
    assert means == {0: -20.0, 10: -10.0, 20: 0.0, 23: 3.0}
    # The Run learns from its own 23 Steps alone, with its files at their Steps.
    assert learner.given == 23
    assert [saved.step for saved in run.checkpoints()] == [5, 10, 15, 20]
    assert [saved.step for saved in run.milestones()] == [0, 10, 20, 23]


def test_each_evaluation_prints_its_summary_line_with_its_milestone_s_step(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=FakeEvaluation(),
    )

    # Score margins of -21 and -19 at Step 0, -11 and -9 at Step 10, and -9 and
    # -7 at Step 12: each 1 away from their mean, and the second cut off. Then the
    # Confirmation of the best, Step 12's, of one Game of Score margin 0.
    assert capsys.readouterr().out.splitlines() == [
        (
            "Evaluation on the regular set at Step 0: mean -20.00, std 1.00, "
            "min -21, 0 of 2 Games won, 1 cut off"
        ),
        (
            "Evaluation on the regular set at Step 10: mean -10.00, std 1.00, "
            "min -11, 0 of 2 Games won, 1 cut off"
        ),
        (
            "Evaluation on the regular set at Step 12: mean -8.00, std 1.00, "
            "min -9, 0 of 2 Games won, 1 cut off"
        ),
        (
            "Evaluation on the confirmation set at Step 12: mean +0.00, std 0.00, "
            "min +0, 0 of 1 Games won, 0 cut off"
        ),
        (
            "The Confirmation's mean, +0.00, is below 18: the Run didn't reach its "
            "goal. It never goes on, and no other Milestone gets a Confirmation."
        ),
    ]


def test_the_evaluations_leave_epsilon_as_the_run_s_step_count_alone_sets_it() -> None:
    # ε falls from 1 by 0.1 a Step, to 0 at Step 10, where a Milestone is
    # evaluated.
    run = new_run(
        RunSettings(
            budget=12,
            checkpoint_every=5,
            milestone_every=10,
            epsilon_start=1.0,
            epsilon_end=0.0,
            epsilon_decay_steps=10,
        )
    )
    learner = EpsilonRecordingLearner()

    train_run(
        run,
        ConstantRewardEnv(),
        learner,
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=FakeEvaluation(),
    )

    # An Action for each of the Run's 12 Steps alone, each by its Step.
    assert learner.epsilons == pytest.approx(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]
    )


class RecordingWriter:
    """A fake of TensorBoard's writer, which keeps the points of the graphs whose
    names start with `prefix`, as (name, Step, value)."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self.points: list[tuple[str, int, float]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if tag.startswith(self._prefix):
            self.points.append((tag, global_step, scalar_value))


def test_each_evaluation_graphs_its_mean_min_and_cut_off_at_its_milestone_s_step() -> (
    None
):
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    writer = RecordingWriter("eval/")

    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=writer,
        evaluation=FakeEvaluation(),
    )

    assert writer.points == [
        ("eval/mean", 0, -20.0),
        ("eval/min", 0, -21),
        ("eval/cut_off", 0, 1),
        ("eval/mean", 10, -10.0),
        ("eval/min", 10, -11),
        ("eval/cut_off", 10, 1),
        ("eval/mean", 12, -8.0),
        ("eval/min", 12, -9),
        ("eval/cut_off", 12, 1),
    ]


@pytest.mark.parametrize(
    ("at", "checkpoints", "milestones"),
    [
        # At the Run's start, its end, and in between.
        (0, [0], [0]),
        (10, [4, 8, 10], [0, 10]),
        (13, [4, 8, 12, 13], [0, 10, 13]),
    ],
)
def test_ctrl_c_in_an_evaluation_abandons_it_and_stops_the_run_with_a_checkpoint(
    at: int,
    checkpoints: list[int],
    milestones: list[int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Ctrl+C in the Evaluation of the Milestone at Step `at`. Checkpoints every 4
    # Steps, and Milestones every 10, and at the Budget, 13.
    run = new_run(
        RunSettings(
            budget=13, checkpoint_every=4, checkpoints_kept=10, milestone_every=10
        )
    )
    learner = CountingLearner()
    writer = RecordingWriter("eval/")

    stopped = train_run(
        run,
        ConstantRewardEnv(),
        learner,
        stay_awake=Awake(),
        writer=writer,
        evaluation=FakeEvaluation(ctrl_c_at=at),
    )

    assert stopped == RunStop(step=at, by_ctrl_c=True)
    assert learner.given == at
    # The Evaluation abandoned isn't recorded, graphed or printed: only those
    # before it are.
    measured = [0, 10, 13][: [0, 10, 13].index(at)]
    assert list(run.evaluations()) == measured
    assert sorted({step for _, step, _ in writer.points}) == measured
    out = capsys.readouterr().out.splitlines()
    summaries = [line for line in out if line.startswith("Evaluation on")]
    assert len(summaries) == len(measured)
    assert out[-1] == (
        f"Ctrl+C abandoned the Evaluation of the Milestone at Step {at}: "
        "resuming the Run measures it first."
    )
    # The Run resumes from a full checkpoint at the Step.
    assert [saved.step for saved in run.checkpoints()] == checkpoints
    assert torch.load(run.checkpoints()[-1].path) == {
        "learner": {"given": at},
        "step": at,
        "resumes": 0,
    }
    assert [saved.step for saved in run.milestones()] == milestones


class FixedMeans:
    """A fake of the Evaluation, which plays no Game and reads no file: it records
    the Step of each Milestone it is asked for, with the set, and gives it a fixed
    mean Score margin, of two Games one apart at most. On the regular set, it is
    the one `means` holds for the Milestone's Step; on the Confirmation's, it is
    `confirmation`.

    Ctrl+C is pressed in the Confirmation, if `ctrl_c_in_confirmation`, and as the
    real one does, it asks `stop` there whether to abandon it. If
    `crash_in_confirmation`, the Confirmation stops the Run at once instead, as a
    crash would.
    """

    def __init__(
        self,
        means: dict[int, float],
        *,
        confirmation: float = 0.0,
        ctrl_c_in_confirmation: bool = False,
        crash_in_confirmation: bool = False,
    ) -> None:
        self._means = means
        self._confirmation = confirmation
        self._ctrl_c_in_confirmation = ctrl_c_in_confirmation
        self._crash_in_confirmation = crash_in_confirmation
        self.asked: list[tuple[int, SeedSet]] = []
        """The Step of each Milestone it was asked for, and the set, in order."""

    def __call__(
        self, milestone: SavedFile, *, seed_set: SeedSet, stop: Callable[[], bool]
    ) -> Evaluation:
        self.asked.append((milestone.step, seed_set))
        if seed_set is SeedSet.CONFIRMATION:
            if self._crash_in_confirmation:
                raise KeyboardInterrupt
            if self._ctrl_c_in_confirmation:
                press_ctrl_c()
            if stop():
                raise EvaluationStopped
            mean = self._confirmation
        else:
            mean = self._means[milestone.step]
        # Two Games whose mean is `mean`, a whole number or a half.
        low = math.floor(mean)
        return Evaluation(
            games=(
                GameResult(seed=0, score_margin=low, cut_off=False),
                GameResult(seed=1, score_margin=int(2 * mean) - low, cut_off=False),
            )
        )


REGULAR, CONFIRMATION = SeedSet.REGULAR, SeedSet.CONFIRMATION


@pytest.mark.parametrize(
    ("means", "confirmed"),
    [
        ({0: -21, 10: 5, 20: 3, 23: 4}, 10),
        # A tie: the earlier Milestone.
        ({0: -21, 10: 5, 20: 5, 23: 4}, 10),
    ],
    ids=["best", "tie"],
)
def test_a_run_at_its_budget_gives_its_best_milestone_one_confirmation(
    means: dict[int, float], confirmed: int
) -> None:
    # Milestones at Step 0, at every multiple of 10, and at the Budget, 23.
    run = new_run(RunSettings(budget=23, checkpoint_every=5, milestone_every=10))
    evaluation = FixedMeans(means, confirmation=2.5)

    stopped = train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=evaluation,
    )

    assert stopped == RunStop(step=23, by_ctrl_c=False)
    # Each Milestone on the regular set, as it is saved, and once the Run is over,
    # the best of them alone on the Confirmation's.
    assert evaluation.asked == [
        (0, REGULAR),
        (10, REGULAR),
        (20, REGULAR),
        (23, REGULAR),
        (confirmed, CONFIRMATION),
    ]
    confirmation = run.confirmation()
    assert confirmation is not None
    assert (confirmation.step, confirmation.evaluation.summary.mean) == (confirmed, 2.5)


@pytest.mark.parametrize(
    ("means", "end"),
    [
        ({0: -21, 10: 19, 20: 21, 23: 21}, 10),
        # 18.5 is below 19, and 19.5 isn't.
        ({0: -21, 10: 18.5, 20: 19.5, 23: 21}, 20),
    ],
    ids=["exactly-19", "above-19"],
)
def test_a_run_stops_at_its_first_milestone_with_a_mean_of_at_least_19(
    means: dict[int, float], end: int
) -> None:
    # Milestones at Step 0, at every multiple of 10, and at the Budget, 23.
    run = new_run(
        RunSettings(
            budget=23, checkpoint_every=5, checkpoints_kept=10, milestone_every=10
        )
    )
    learner = CountingLearner()
    evaluation = FixedMeans(means)

    stopped = train_run(
        run,
        ConstantRewardEnv(),
        learner,
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=evaluation,
    )

    # The Run is over at that Milestone's Step, with no Step after it, and that
    # Milestone, the best of the Run's, is the one confirmed.
    assert stopped == RunStop(step=end, by_ctrl_c=False)
    assert learner.given == end
    milestones = [step for step in (0, 10, 20) if step <= end]
    assert [saved.step for saved in run.milestones()] == milestones
    assert evaluation.asked == [(step, REGULAR) for step in milestones] + [
        (end, CONFIRMATION)
    ]


@pytest.mark.parametrize(
    ("mean", "summary", "verdict"),
    [
        (
            18.0,
            "mean +18.00, std 0.00, min +18, 2 of 2 Games won, 0 cut off",
            (
                "The Confirmation's mean, +18.00, is at least 18: the Run reached "
                "its goal."
            ),
        ),
        (
            17.5,
            "mean +17.50, std 0.50, min +17, 2 of 2 Games won, 0 cut off",
            (
                "The Confirmation's mean, +17.50, is below 18: the Run didn't reach "
                "its goal. It never goes on, and no other Milestone gets a "
                "Confirmation."
            ),
        ),
    ],
    ids=["passed", "failed"],
)
def test_the_confirmation_prints_its_summary_line_and_whether_it_passed_18(
    mean: float, summary: str, verdict: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The Run stops at its Milestone of Step 10.
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))

    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=FixedMeans({0: -21, 10: 19}, confirmation=mean),
    )

    assert capsys.readouterr().out.splitlines()[-2:] == [
        f"Evaluation on the confirmation set at Step 10: {summary}",
        verdict,
    ]


def test_the_confirmation_graphs_its_mean_once_at_its_milestone_s_step() -> None:
    # Milestones at Step 0, 10, and the Budget, 12: the best is Step 10's.
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    writer = RecordingWriter("confirmation/")

    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=writer,
        evaluation=FixedMeans({0: -21, 10: 5, 12: 3}, confirmation=2.5),
    )

    assert writer.points == [("confirmation/mean", 10, 2.5)]


def test_ctrl_c_in_the_confirmation_abandons_it_and_stops_the_run_with_a_checkpoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The Run stops at its Milestone of Step 10, which no checkpoint falls on.
    run = new_run(
        RunSettings(
            budget=23, checkpoint_every=4, checkpoints_kept=10, milestone_every=10
        )
    )
    writer = RecordingWriter("confirmation/")

    stopped = train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=writer,
        evaluation=FixedMeans({0: -21, 10: 19}, ctrl_c_in_confirmation=True),
    )

    assert stopped == RunStop(step=10, by_ctrl_c=True)
    # Not recorded, graphed or printed.
    assert run.confirmation() is None
    assert writer.points == []
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Ctrl+C abandoned the Confirmation of the Milestone at Step 10: resuming "
        "the Run gives it its Confirmation."
    )
    # The Run resumes from a full checkpoint at the Step.
    assert [saved.step for saved in run.checkpoints()] == [4, 8, 10]
    assert torch.load(run.checkpoints()[-1].path) == {
        "learner": {"given": 10},
        "step": 10,
        "resumes": 0,
    }


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

    train_run(run, ConstantRewardEnv(), learner, stay_awake=awake, writer=NoGraphs())

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
        train_run(
            run, ConstantRewardEnv(), learner, stay_awake=awake, writer=NoGraphs()
        )
    except KeyboardInterrupt:
        pass

    assert learner.held_at_each_step == [True] * 7
    assert not awake.held


def test_ctrl_c_stops_the_run_after_its_step_with_a_full_checkpoint_there() -> None:
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    learner = CtrlCLearner(at=7)
    handler_before = signal.getsignal(signal.SIGINT)

    stopped = train_run(
        run, ConstantRewardEnv(), learner, stay_awake=Awake(), writer=NoGraphs()
    )

    # Step 7 is played to its end and learned from, and the Run stops there.
    assert stopped == RunStop(step=7, by_ctrl_c=True)
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

    stopped = train_run(
        run,
        ConstantRewardEnv(),
        CtrlCLearner(at=10),
        stay_awake=Awake(),
        writer=NoGraphs(),
    )

    assert stopped == RunStop(step=10, by_ctrl_c=True)
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
            run,
            ConstantRewardEnv(),
            CtrlCLearner(at=7, presses=2),
            stay_awake=Awake(),
            writer=NoGraphs(),
        )

    assert [saved.step for saved in run.checkpoints()] == [5]
    assert signal.getsignal(signal.SIGINT) is handler_before


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


def stopped_by_ctrl_c(settings: RunSettings, *, at: int) -> RunDirectory:
    """A new Run with `settings`, stopped by Ctrl+C in its Step `at`, with a full
    checkpoint there, and reopened, as `pong resume` opens it."""
    run = new_run(settings)
    train_run(
        run,
        ConstantRewardEnv(),
        CtrlCLearner(at=at),
        stay_awake=Awake(),
        writer=NoGraphs(),
    )
    return RunDirectory.open(run.path, current_commit=A_COMMIT)


def test_a_resumed_run_plays_on_from_its_latest_checkpoint_to_its_budget() -> None:
    run = stopped_by_ctrl_c(
        RunSettings(budget=12, checkpoint_every=5, milestone_every=10), at=7
    )
    learner = CountingLearner()

    stopped = resume_run(
        run, ConstantRewardEnv(), learner, stay_awake=Awake(), writer=NoGraphs()
    )

    # The learner takes on its state of Step 7, and learns from Steps 8 to 12.
    assert stopped == RunStop(step=12, by_ctrl_c=False)
    assert learner.given == 12
    # The Step count goes on from 7: a checkpoint at Step 10, and Milestones at 10
    # and at the Budget.
    assert [saved.step for saved in run.checkpoints()] == [7, 10]
    assert [saved.step for saved in run.milestones()] == [0, 10, 12]


def test_each_resume_counts_one_more_in_the_run_s_checkpoints() -> None:
    run = stopped_by_ctrl_c(
        RunSettings(budget=12, checkpoint_every=5, milestone_every=10), at=7
    )

    # The first resume stopped by Ctrl+C in Step 9, and the second to the Budget.
    resume_run(
        run,
        ConstantRewardEnv(),
        CtrlCLearner(at=9),
        stay_awake=Awake(),
        writer=NoGraphs(),
    )
    assert torch.load(run.checkpoints()[-1].path) == {
        "learner": {"given": 9},
        "step": 9,
        "resumes": 1,
    }
    resume_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
    )

    assert torch.load(run.checkpoints()[-1].path) == {
        "learner": {"given": 10},
        "step": 10,
        "resumes": 2,
    }


class EpsilonRecordingLearner(CountingLearner):
    """A fake learner that records the ε of each Action it chooses."""

    def __init__(self) -> None:
        super().__init__()
        self.epsilons: list[float] = []

    def choose_action(self, observation: Any, epsilon: float) -> int:
        self.epsilons.append(epsilon)
        return super().choose_action(observation, epsilon)


def test_a_resumed_run_s_epsilon_goes_on_from_its_value_at_the_checkpoint() -> None:
    # ε falls from 1 by 0.1 a Step, to 0 at Step 10.
    run = stopped_by_ctrl_c(
        RunSettings(
            budget=12,
            checkpoint_every=5,
            milestone_every=10,
            epsilon_start=1.0,
            epsilon_end=0.0,
            epsilon_decay_steps=10,
        ),
        at=7,
    )
    learner = EpsilonRecordingLearner()

    resume_run(run, ConstantRewardEnv(), learner, stay_awake=Awake(), writer=NoGraphs())

    # Step 8 is chosen with ε after 7 Steps, 0.3, as it would have been with no
    # stop, and so on to the Budget.
    assert learner.epsilons == pytest.approx([0.3, 0.2, 0.1, 0.0, 0.0])


def test_windows_is_asked_to_stay_awake_through_a_resumed_run_and_released() -> None:
    run = stopped_by_ctrl_c(
        RunSettings(budget=12, checkpoint_every=5, milestone_every=10), at=7
    )
    awake = Awake()
    learner = AwakeCheckingLearner(awake)

    resume_run(run, ConstantRewardEnv(), learner, stay_awake=awake, writer=NoGraphs())

    # Steps 8 to 12.
    assert learner.held_at_each_step == [True] * 5
    assert awake.requests == 1
    assert not awake.held


class SeedRecordingEnv(ConstantRewardEnv):
    """Test environment 1, which records the seed of each reset, or None for a
    reset with none."""

    def __init__(self) -> None:
        super().__init__()
        self.seeds: list[int | None] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        self.seeds.append(seed)
        return super().reset(seed=seed, options=options)


def test_every_resume_from_a_checkpoint_plays_the_same_games_not_the_run_s_first(
    runs: Path,
) -> None:
    run = stopped_by_ctrl_c(
        RunSettings(budget=12, checkpoint_every=5, milestone_every=10), at=7
    )
    # The same Run, to resume twice from the same checkpoint.
    copy = RunDirectory.open(
        shutil.copytree(run.path, runs / "copy"), current_commit=A_COMMIT
    )

    first_seeds = []
    for resumed in (run, copy):
        env = SeedRecordingEnv()
        resume_run(
            resumed, env, CountingLearner(), stay_awake=Awake(), writer=NoGraphs()
        )
        first_seeds.append(env.seeds[0])

    # The Run's first Game, and the Q graph's, were reset with the Run's seed, 1.
    first, again = first_seeds
    assert first is not None and first != 1, f"the first Game's seed: {first}"
    assert again == first


def evaluation_abandoned_by_ctrl_c(at: int) -> RunDirectory:
    """A new Run with checkpoints every 4 Steps, Milestones every 10 and at its
    Budget, 13, and Evaluations, stopped by Ctrl+C in the Evaluation of its
    Milestone at Step `at`, and reopened, as `pong resume` opens it."""
    run = new_run(
        RunSettings(
            budget=13, checkpoint_every=4, checkpoints_kept=10, milestone_every=10
        )
    )
    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=FakeEvaluation(ctrl_c_at=at),
    )
    return RunDirectory.open(run.path, current_commit=A_COMMIT)


@pytest.mark.parametrize(
    ("at", "evaluated"),
    [
        # At the Run's start, its end, and in between.
        (0, [0, 10, 13]),
        (10, [10, 13]),
        (13, [13]),
    ],
)
def test_a_resume_measures_first_the_milestone_whose_evaluation_was_abandoned(
    at: int, evaluated: list[int]
) -> None:
    run = evaluation_abandoned_by_ctrl_c(at)
    learner = CountingLearner()
    evaluation = FakeEvaluation(learner=learner)

    stopped = resume_run(
        run,
        ConstantRewardEnv(),
        learner,
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=evaluation,
    )

    assert stopped == RunStop(step=13, by_ctrl_c=False)
    # The Milestone of Step `at` is measured before the Step after it is played,
    # and each Milestone after it once it is saved.
    assert evaluation.evaluated == evaluated
    assert evaluation.given_then == evaluated
    assert list(run.evaluations()) == [0, 10, 13]


@pytest.mark.parametrize(
    ("means", "end", "interruption"),
    [
        ({0: -21, 10: 19}, 10, "ctrl_c"),
        ({0: -21, 10: 19}, 10, "crash"),
        ({0: -21, 10: 5, 20: 3, 23: 4}, 23, "crash"),
    ],
    ids=["ctrl_c-stopped-at-19", "crash-stopped-at-19", "crash-at-the-budget"],
)
def test_a_run_stopped_before_its_confirmation_has_it_on_resume_and_no_step(
    means: dict[int, float], end: int, interruption: str
) -> None:
    # Checkpoints every 4 Steps, and Milestones every 10, and at the Budget, 23. A
    # crash leaves the latest checkpoint before the Run's end, and Ctrl+C one at
    # it. Either way, the Milestone of Step 10 is the best.
    run = new_run(
        RunSettings(
            budget=23, checkpoint_every=4, checkpoints_kept=10, milestone_every=10
        )
    )
    try:
        train_run(
            run,
            ConstantRewardEnv(),
            CountingLearner(),
            stay_awake=Awake(),
            writer=NoGraphs(),
            evaluation=FixedMeans(means, **{f"{interruption}_in_confirmation": True}),
        )
    except KeyboardInterrupt:
        pass
    run = RunDirectory.open(run.path, current_commit=A_COMMIT)
    milestones = run.milestones()
    learner = EpsilonRecordingLearner()
    evaluation = FixedMeans(means, confirmation=18.5)

    stopped = resume_run(
        run,
        ConstantRewardEnv(),
        learner,
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=evaluation,
    )

    assert stopped == RunStop(step=end, by_ctrl_c=False)
    # The Confirmation alone: no Step, no Milestone, and no Evaluation again.
    assert evaluation.asked == [(10, CONFIRMATION)]
    assert learner.epsilons == []
    assert run.milestones() == milestones
    confirmation = run.confirmation()
    assert confirmation is not None
    assert (confirmation.step, confirmation.evaluation.summary.mean) == (10, 18.5)


def test_a_run_that_had_its_confirmation_refuses_to_resume() -> None:
    # The Run stops at its Milestone of Step 10, before its Budget, and has its
    # Confirmation.
    run = new_run(
        RunSettings(
            budget=23, checkpoint_every=4, checkpoints_kept=10, milestone_every=10
        )
    )
    train_run(
        run,
        ConstantRewardEnv(),
        CountingLearner(),
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=FixedMeans({0: -21, 10: 19}, confirmation=17.0),
    )
    run = RunDirectory.open(run.path, current_commit=A_COMMIT)
    files = sorted(run.path.iterdir())
    awake = Awake()
    evaluation = FixedMeans({0: -21, 10: 19, 20: 21, 23: 21})

    with pytest.raises(ConfirmedRunError, match="Step 10"):
        resume_run(
            run,
            ConstantRewardEnv(),
            CountingLearner(),
            stay_awake=awake,
            writer=NoGraphs(),
            evaluation=evaluation,
        )

    # Nothing is played, and nothing written.
    assert evaluation.asked == []
    assert awake.requests == 0
    assert sorted(run.path.iterdir()) == files


class CrashingLearner(CountingLearner):
    """A fake learner that stops the Run at once, as a crash would, when it is
    asked for its network's weights after its Step `at`: for the Milestone of the
    Step, once the Step's full checkpoint is saved."""

    def __init__(self, *, at: int) -> None:
        super().__init__()
        self._at = at

    def network_state_dict(self) -> dict[str, Any]:
        if self.given == self._at:
            raise KeyboardInterrupt
        return super().network_state_dict()


def test_a_resume_saves_and_measures_first_a_milestone_a_crash_left_unsaved() -> None:
    # Step 10 has a checkpoint every 5 Steps, and a Milestone every 10.
    run = new_run(RunSettings(budget=12, checkpoint_every=5, milestone_every=10))
    with pytest.raises(KeyboardInterrupt):
        train_run(
            run,
            ConstantRewardEnv(),
            CrashingLearner(at=10),
            stay_awake=Awake(),
            writer=NoGraphs(),
            evaluation=FakeEvaluation(),
        )
    assert [saved.step for saved in run.checkpoints()] == [5, 10]
    assert [saved.step for saved in run.milestones()] == [0]
    run = RunDirectory.open(run.path, current_commit=A_COMMIT)
    learner = CountingLearner()
    evaluation = FakeEvaluation(learner=learner)

    resume_run(
        run,
        ConstantRewardEnv(),
        learner,
        stay_awake=Awake(),
        writer=NoGraphs(),
        evaluation=evaluation,
    )

    # The Milestone of Step 10 holds the network of the checkpoint there, and is
    # measured before Step 11 is played.
    assert [saved.step for saved in run.milestones()] == [0, 10, 12]
    assert torch.load(run.milestones()[1].path) == {"weights after": 10}
    assert evaluation.evaluated == [10, 12]
    assert evaluation.given_then == [10, 12]
    assert list(run.evaluations()) == [0, 10, 12]
