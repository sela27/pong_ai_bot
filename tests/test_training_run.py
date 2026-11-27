"""A Run's training loop, in a temporary directory: the files it saves at its Steps,
the fixed inputs of its Q graph, its stop at the Budget or by Ctrl+C, and its
request that Windows stays awake (issue #10).

The learner is a fake, with no network and no optimizer: creating the real one's
optimizer takes seconds, too long for a fast test. So are the request to Windows,
and TensorBoard's writer, which takes seconds to import: the graphs are checked by
eye in TensorBoard, on a test environment's Run (#17). The environment is test
environment 1, whose every Game is one Step long, unless a test says otherwise.
Ctrl+C is pressed without a real signal: Python calls the handler of SIGINT when
Ctrl+C is pressed, and the tests call it themselves, while the Run plays a chosen
Step.
"""

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

from pong_ai_bot.evaluation import Evaluation, EvaluationStopped, GameResult
from pong_ai_bot.run_directory import Commit, RunDirectory, SavedFile
from pong_ai_bot.run_settings import RunSettings
from pong_ai_bot.test_environments import ConstantRewardEnv, TwoStepEnv
from pong_ai_bot.training import RunStop, Transition, resume_run, train_run

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

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.given = state["given"]

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


class FakeEvaluation:
    """A fake of the Evaluation, which plays no Game: it reads the network's
    weights from the Milestone it is given, a CountingLearner's, and gives the
    Milestone of the weights after n Steps two Games, of Score margins n - 21 and
    n - 19, the second cut off.

    It presses Ctrl+C while it evaluates the Milestone of the weights after
    `ctrl_c_at` Steps, if given, and, as the real one does, it asks `stop` before
    each of its two Steps whether to abandon the Evaluation. It records how many
    transitions `learner`, if given, had been given when it evaluated each.
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
        """The Steps of the weights of each Milestone it was given, in order."""
        self.given_then: list[int] = []
        """How many transitions `learner` had been given at each Evaluation."""

    def __call__(self, milestone: SavedFile, *, stop: Callable[[], bool]) -> Evaluation:
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
    # -7 at Step 12: each 1 away from their mean, and the second cut off.
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
