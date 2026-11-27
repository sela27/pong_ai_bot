"""The installed `pong` command."""

import json
import re
import shutil
import signal
import subprocess
import sysconfig
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium import spaces
from helpers import A_COMMIT, press_ctrl_c

from pong_ai_bot.cli import main
from pong_ai_bot.evaluation import Evaluation, GameResult
from pong_ai_bot.game import Score
from pong_ai_bot.learning_players import LEARNING_PLAYERS, LearningPlayer
from pong_ai_bot.run_directory import RunDirectory, checked_out_commit
from pong_ai_bot.run_settings import Loss, RunSettings


def run_pong(*args: str) -> subprocess.CompletedProcess[str]:
    pong = shutil.which("pong", path=sysconfig.get_path("scripts"))
    assert pong is not None, "the `pong` command is not installed"
    return subprocess.run([pong, *args], capture_output=True, text=True, check=False)


def test_pong_help_runs() -> None:
    # The only test that runs the installed command itself, to show it works. The
    # others call its entry point in-process, to keep the hook fast.
    result = run_pong("--help")

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: pong"), result.stdout


def test_pong_has_a_measure_env_command(capsys: pytest.CaptureFixture[str]) -> None:
    # Only its help: the measurement itself is a script, not a test (issue #11).
    with pytest.raises(SystemExit) as exit_:
        main(["measure-env", "--help"])

    assert exit_.value.code == 0
    assert capsys.readouterr().out.startswith("usage: pong measure-env")


class OneStepGames(gym.Env[npt.NDArray[np.uint8], np.int64]):
    """A fake of the environment a Scripted player plays, whose every Game ends in
    its first Step: won 21 to 0 with an even seed, and lost 0 to 21 with an odd
    one."""

    def __init__(self, *, left_player: object) -> None:
        self._won = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.uint8], dict[str, Any]]:
        super().reset(seed=seed)
        assert seed is not None
        self._won = seed % 2 == 0
        return np.zeros(1, np.uint8), {"score": Score(left=0, right=0)}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[npt.NDArray[np.uint8], SupportsFloat, bool, bool, dict[str, Any]]:
        score = Score(left=0, right=21) if self._won else Score(left=21, right=0)
        return np.zeros(1, np.uint8), 0.0, True, False, {"score": score}


def test_pong_evaluate_prints_the_summary_line_and_adds_the_json_line_to_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # In-process, and with Games of one Step, to keep the hook fast: the random
    # player's 100 real Games take a couple of seconds.
    output = tmp_path / "random.jsonl"
    output.write_text('{"earlier": "line"}\n', encoding="utf-8")

    exit_code = main(
        ["evaluate", "--scripted", "random", "--output", str(output)],
        pong=OneStepGames,
    )

    assert exit_code == 0
    # Half of the 100 Games won 21 to 0, and the other half lost 0 to 21.
    summary = (
        "Evaluation on the regular set: mean +0.00, std 21.00, min -21, "
        "50 of 100 Games won, 0 cut off"
    )
    assert capsys.readouterr().out.splitlines() == [summary]
    # The line is added to the end of the file, after what was there.
    earlier, added = output.read_text(encoding="utf-8").splitlines()
    assert json.loads(earlier) == {"earlier": "line"}
    line = json.loads(added)
    # A Scripted player has no Run, so no Step, and plays the regular set.
    assert line["step"] is None
    assert line["set"] == "regular"
    assert [game["seed"] for game in line["games"]] == list(range(1_000_000, 1_000_100))


def test_pong_evaluate_offers_the_two_scripted_players(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Only its help, in-process: the edge predictor's 100 Games take far longer
    # than the hook allows.
    with pytest.raises(SystemExit) as exit_:
        main(["evaluate", "--help"])

    assert exit_.value.code == 0
    assert "--scripted {random,edge-predictor}" in capsys.readouterr().out


def test_pong_has_a_choose_width_command(capsys: pytest.CaptureFixture[str]) -> None:
    # Only its help, in-process: choosing is a script, not a test, and its 10
    # Evaluations take minutes.
    with pytest.raises(SystemExit) as exit_:
        main(["choose-width", "--help"])

    assert exit_.value.code == 0
    assert capsys.readouterr().out.startswith("usage: pong choose-width")


def test_pong_evaluate_offers_no_confirmation_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Only the end of a Run plays the Confirmation's set (issue #16).
    with pytest.raises(SystemExit) as exit_:
        main(["evaluate", "--help"])

    assert exit_.value.code == 0
    help_text = capsys.readouterr().out
    assert "confirmation" not in help_text.lower(), help_text


def test_pong_train_refuses_the_cpu_unless_asked_and_creates_no_run(
    runs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As with a PyTorch build that doesn't support the graphics card: it would
    # train 20 times slower, with no error (issue #11).
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    exit_code = main(
        ["train", "--player", "test-env-1", "--name", "tiny", "--seed", "1"]
    )

    assert exit_code == 1
    assert "--device cpu" in capsys.readouterr().err
    assert not runs.exists()


@pytest.mark.parametrize(
    ("option", "reason"),
    [
        # The Evaluation's seeds start there, so a Run never trains on a Game an
        # Evaluation plays (#13).
        (["--seed", "1000000"], "below 1,000,000"),
        # The name goes in the Run directory's name.
        (["--name", "a/b"], "letters, digits"),
    ],
    ids=["seed", "name"],
)
def test_pong_train_refuses_a_seed_or_name_a_run_can_t_have_at_once_and_creates_no_run(
    runs: Path, capsys: pytest.CaptureFixture[str], option: list[str], reason: str
) -> None:
    run = ["--player", "test-env-1", "--name", "tiny", "--seed", "1"]
    player = LEARNING_PLAYERS["test-env-1"]

    def make_network() -> Any:
        pytest.fail("the Run's setup began before the refusal")

    exit_code = main(
        ["train", *run, "--device", "cpu", *option],
        # At once: before the Run's setup, which takes seconds.
        learning_players={
            **LEARNING_PLAYERS,
            "test-env-1": replace(player, make_network=make_network),
        },
    )

    assert exit_code == 1
    assert reason in capsys.readouterr().err.splitlines()[-1]
    assert not runs.exists()


def test_pong_train_has_an_option_for_each_run_setting_with_its_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_:
        main(["train", "--help"])

    assert exit_.value.code == 0
    help_text = capsys.readouterr().out
    # One setting of each kind, with its default beside it.
    for option, default in [
        ("--budget N", "default 10_000_000"),
        ("--learning-rate X", "default 0.0001"),
        ("--loss {mse}", "default mse"),
        ("--double-dqn, --no-double-dqn", "default on"),
    ]:
        assert re.search(rf"{re.escape(option)}\s+{default}", help_text), help_text
    # And every setting, named after it.
    for field in fields(RunSettings):
        assert f"--{field.name.replace('_', '-')}" in help_text, help_text


@pytest.mark.slow
def test_pong_train_runs_a_test_environment_s_tiny_run_to_its_budget(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Slow for the learner's optimizer: making one loads a part of PyTorch that
    # takes about 5 seconds.
    import torch

    exit_code = main(
        [
            "train",
            "--player",
            "test-env-1",
            "--name",
            "tiny",
            "--seed",
            "1",
            "--device",
            "cpu",
            "--budget",
            "30",
            "--fill-steps",
            "10",
            "--replay-buffer-size",
            "100",
            "--batch-size",
            "4",
            "--update-every",
            "1",
            "--target-copy-every",
            "5",
            "--epsilon-decay-steps",
            "10",
            "--learning-rate",
            "1e-3",
            "--adam-eps",
            "1e-6",
            "--loss",
            "mse",
            "--no-double-dqn",
            "--checkpoint-every",
            "3",
            "--checkpoints-kept",
            "20",
            "--milestone-every",
            "20",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines()[0] == "Device: cpu"
    # A new Run, whose settings file records its changes from the defaults.
    [path] = runs.iterdir()
    run = RunDirectory.open(path)
    assert (run.player, run.name, run.seed) == ("test-env-1", "tiny", 1)
    assert run.settings == RunSettings(
        budget=30,
        fill_steps=10,
        replay_buffer_size=100,
        batch_size=4,
        update_every=1,
        target_copy_every=5,
        epsilon_decay_steps=10,
        learning_rate=1e-3,
        adam_eps=1e-6,
        loss=Loss.MSE,
        double_dqn=False,
        checkpoint_every=3,
        checkpoints_kept=20,
        milestone_every=20,
    )
    # A Milestone at Step 0, at every multiple of 20, and at the Budget, where the
    # Run stops; a checkpoint at every multiple of 3, and at each Milestone but
    # Step 0's.
    assert [saved.step for saved in run.milestones()] == [0, 20, 30]
    every_3 = list(range(3, 31, 3))
    assert [saved.step for saved in run.checkpoints()] == sorted([*every_3, 20])
    checkpoints = {saved.step: torch.load(saved.path) for saved in run.checkpoints()}
    last = checkpoints[30]
    assert (last["step"], last["resumes"]) == (30, 0)
    assert set(last["learner"]) == {
        "network",
        "target_network",
        "optimizer",
        "rng",
        "steps_given",
    }
    assert last["learner"]["steps_given"] == 30
    # The fill's 10 Steps count in the Run's Steps: the network learns first at
    # Step 10, so at Step 9 it is still the one of Step 0.
    first_weights = torch.load(run.milestones()[0].path)
    assert same_weights(checkpoints[9]["learner"]["network"], first_weights)
    assert not same_weights(checkpoints[12]["learner"]["network"], first_weights)
    assert same_weights(
        torch.load(run.milestones()[-1].path), last["learner"]["network"]
    )


@pytest.mark.slow
def test_ctrl_c_stops_pong_train_after_its_step_with_a_full_checkpoint(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Slow for the learner's optimizer, as above.
    import torch

    exit_code = main(
        [
            "train",
            "--player",
            "test-env-1",
            "--name",
            "tiny",
            "--seed",
            "1",
            "--device",
            "cpu",
            "--budget",
            "30",
            "--fill-steps",
            "10",
            "--replay-buffer-size",
            "100",
            "--batch-size",
            "4",
            "--checkpoint-every",
            "5",
        ],
        learning_players=pressing_ctrl_c_in_step(12, presses=1),
    )

    # As a program stopped by Ctrl+C: 128 and SIGINT's number, 2.
    assert exit_code == 130
    out, err = capsys.readouterr()
    assert out.splitlines()[-1] == (
        "Ctrl+C stopped the Run at Step 12, with a full checkpoint there."
    )
    assert err == ""
    # A second press, as `pong` ends, changes nothing: it is ignored.
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    [path] = runs.iterdir()
    run = RunDirectory.open(path)
    # The latest two checkpoints are kept, the one of Step 5 deleted.
    assert [saved.step for saved in run.checkpoints()] == [10, 12]
    last = torch.load(run.checkpoints()[-1].path)
    assert (last["step"], last["resumes"]) == (12, 0)
    assert last["learner"]["steps_given"] == 12
    assert [saved.step for saved in run.milestones()] == [0]


def test_pong_resume_takes_the_run_s_directory_and_no_setting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A Run's settings never change: they are read from its settings file.
    with pytest.raises(SystemExit) as exit_:
        main(["resume", str(tmp_path / "a-run"), "--budget", "20"])

    assert exit_.value.code == 2
    assert "unrecognized arguments: --budget 20" in capsys.readouterr().err


def test_pong_resume_refuses_the_cpu_unless_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As `pong train` does: a Run trains 20 times slower there.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    exit_code = main(["resume", str(tmp_path / "a-run")])

    assert exit_code == 1
    assert "--device cpu" in capsys.readouterr().err


@pytest.mark.parametrize("given", ["no directory", "a directory of no Run"])
def test_pong_resume_refuses_what_is_not_a_run_s_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], given: str
) -> None:
    path = tmp_path / "a-run"
    if given == "a directory of no Run":
        path.mkdir()

    exit_code = main(["resume", str(path), "--device", "cpu"])

    assert exit_code == 1
    assert "no Run" in capsys.readouterr().err.splitlines()[-1]


def test_pong_resume_warns_if_the_code_is_at_another_commit_than_the_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = RunDirectory.create(player="test-env-1", name="tiny", seed=1, commit=A_COMMIT)

    main(["resume", str(run.path), "--device", "cpu"])

    [warning] = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("Warning:")
    ]
    code_s_own = checked_out_commit().sha
    assert A_COMMIT.sha in warning and code_s_own in warning, warning


def test_pong_resume_refuses_a_run_with_no_full_checkpoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As a Run that crashed before its first: there is nothing to resume from.
    run = RunDirectory.create(player="test-env-1", name="tiny", seed=1, commit=A_COMMIT)

    exit_code = main(["resume", str(run.path), "--device", "cpu"])

    assert exit_code == 1
    assert "no full checkpoint" in capsys.readouterr().err


def test_pong_resume_of_a_run_that_reached_its_budget_plays_nothing_more(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The Run's end saved a full checkpoint and a Milestone at its Budget, Step
    # 23, off its checkpoints every 5 Steps.
    run = RunDirectory.create(
        player="test-env-1",
        name="tiny",
        seed=1,
        settings=RunSettings(budget=23, checkpoint_every=5),
        commit=A_COMMIT,
    )
    for save, step in [
        (run.save_checkpoint, 20),
        (run.save_checkpoint, 23),
        (run.save_milestone, 23),
    ]:
        save(step, lambda file: file.write(b"saved"))
    files = sorted(run.path.iterdir())

    exit_code = main(["resume", str(run.path), "--device", "cpu"])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "The Run reached its Budget, 23 Steps, already: there is nothing to resume."
    )
    assert sorted(run.path.iterdir()) == files


class CtrlCInStep(gym.Wrapper[Any, Any, Any, Any]):
    """An environment that presses Ctrl+C, `presses` times, while it plays the
    Run's Step `at`, as the developer would, without a real signal: Python calls
    the handler of SIGINT when Ctrl+C is pressed.

    Before Step 0, the Run plays a Game of random Actions for its Q graph's
    inputs, a Step for each.
    """

    def __init__(self, env: gym.Env[Any, Any], *, at: int, presses: int) -> None:
        from pong_ai_bot.training import Q_GRAPH_INPUTS

        super().__init__(env)
        self._at = at
        self._presses = presses
        self._steps = -Q_GRAPH_INPUTS

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self._steps += 1
        if self._steps == self._at:
            for _ in range(self._presses):
                press_ctrl_c()
        return self.env.step(action)


def pressing_ctrl_c_in_step(at: int, *, presses: int) -> dict[str, LearningPlayer]:
    """The learning players, with test environment 1's in an environment that
    presses Ctrl+C, `presses` times, while it plays the Run's Step `at`."""
    player = LEARNING_PLAYERS["test-env-1"]

    def make_env() -> gym.Env[Any, Any]:
        return CtrlCInStep(player.make_env(), at=at, presses=presses)

    return {**LEARNING_PLAYERS, "test-env-1": replace(player, make_env=make_env)}


@pytest.fixture
def pressing_ctrl_c_in_the_setup() -> dict[str, LearningPlayer]:
    """The learning players, with test environment 1's pressing Ctrl+C as its
    network is made, in the setup before its Run's first Step, as the developer
    would right after starting it: before its learner, and Adam with it, is made."""
    player = LEARNING_PLAYERS["test-env-1"]

    def make_network() -> Any:
        press_ctrl_c()
        return player.make_network()

    return {
        **LEARNING_PLAYERS,
        "test-env-1": replace(player, make_network=make_network),
    }


def test_ctrl_c_in_pong_train_s_setup_stops_it_with_one_line_and_no_run_directory(
    runs: Path,
    capsys: pytest.CaptureFixture[str],
    pressing_ctrl_c_in_the_setup: dict[str, LearningPlayer],
) -> None:
    run = ["--player", "test-env-1", "--name", "tiny", "--seed", "1"]

    exit_code = main(
        ["train", *run, "--device", "cpu"],
        learning_players=pressing_ctrl_c_in_the_setup,
    )

    assert exit_code == 130
    # One line, and no traceback.
    assert capsys.readouterr().err.splitlines() == [
        "Ctrl+C stopped pong train before the Run's first Step."
    ]
    # The Run's directory comes after the setup, so none is left to delete.
    assert not runs.exists()


def test_a_second_ctrl_c_after_pong_train_s_setup_stopped_is_ignored(
    pressing_ctrl_c_in_the_setup: dict[str, LearningPlayer],
) -> None:
    run = ["--player", "test-env-1", "--name", "tiny", "--seed", "1"]

    exit_code = main(
        ["train", *run, "--device", "cpu"],
        learning_players=pressing_ctrl_c_in_the_setup,
    )

    assert exit_code == 130
    # As the developer presses twice: the second comes as `pong` ends, which
    # takes most of a second once PyTorch is loaded. It is ignored to the
    # process's end, and not by Python's own handler: Python removes its handlers
    # as it ends, and then Windows's own would end the process with its own exit
    # code for Ctrl+C, 0xC000013A, instead of 130.
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def test_ctrl_c_is_as_it_was_after_pong_train_refuses_a_run() -> None:
    handler_before = signal.getsignal(signal.SIGINT)
    run = ["--player", "test-env-1", "--name", "a/b", "--seed", "1"]

    # Refused, before its Run.
    exit_code = main(["train", *run, "--device", "cpu"])

    assert exit_code == 1
    assert signal.getsignal(signal.SIGINT) is handler_before


def test_ctrl_c_in_pong_resume_s_setup_stops_it_with_one_line_and_the_run_as_it_was(
    capsys: pytest.CaptureFixture[str],
    pressing_ctrl_c_in_the_setup: dict[str, LearningPlayer],
) -> None:
    # Created at the code's own commit, so that resuming it gives no warning.
    run = RunDirectory.create(
        player="test-env-1",
        name="tiny",
        seed=1,
        settings=RunSettings(budget=30, checkpoint_every=5),
    )
    run.save_checkpoint(5, lambda file: file.write(b"saved"))
    files = {path: path.read_bytes() for path in run.path.iterdir()}

    exit_code = main(
        ["resume", str(run.path), "--device", "cpu"],
        learning_players=pressing_ctrl_c_in_the_setup,
    )

    assert exit_code == 130
    assert capsys.readouterr().err.splitlines() == [
        "Ctrl+C stopped pong resume before the Run went on."
    ]
    assert {path: path.read_bytes() for path in run.path.iterdir()} == files


def train_stopped_by_ctrl_c(settings: list[str], *, at: int, presses: int) -> Path:
    """The directory of a new Run of test environment 1 on the CPU, with the
    options `settings`, that Ctrl+C stopped while the environment played its Step
    `at`, pressed `presses` times: once, to stop after the Step, or twice, to stop
    at once, as a crash would."""
    from pong_ai_bot.run_directory import runs_root

    run = ["--player", "test-env-1", "--name", "tiny", "--seed", "1"]
    exit_code = main(
        ["train", *run, "--device", "cpu", *settings],
        learning_players=pressing_ctrl_c_in_step(at, presses=presses),
    )

    assert exit_code == 130
    [path] = runs_root().iterdir()
    return path


@pytest.mark.slow
def test_pong_resume_drops_the_graphs_points_written_after_the_checkpoint() -> None:
    # Slow for the learner's optimizer, as above, and for TensorBoard's reader.
    from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
        EventAccumulator,
    )

    # Stopped at once in Step 12, as by a crash: its latest checkpoint is at
    # Step 10, and the graphs have points of Step 11 too.
    path = train_stopped_by_ctrl_c(
        [
            "--budget",
            "20",
            "--fill-steps",
            "10",
            "--replay-buffer-size",
            "100",
            "--batch-size",
            "4",
            "--checkpoint-every",
            "5",
        ],
        at=12,
        presses=2,
    )

    exit_code = main(["resume", str(path), "--device", "cpu"])

    assert exit_code == 0
    graphs = EventAccumulator(str(path))
    graphs.Reload()
    # Every Game of test environment 1 is a Step long: one point at each Step,
    # those of the Steps before the checkpoint kept, and none twice.
    score_margins = [point.step for point in graphs.Scalars("train/score_margin")]
    assert score_margins == list(range(1, 21)), score_margins
    resumes = [(point.step, point.value) for point in graphs.Scalars("run/resumes")]
    assert resumes == [(0, 0.0), (10, 1.0)]


def same_weights(weights: dict[str, Any], other: dict[str, Any]) -> bool:
    import torch

    return weights.keys() == other.keys() and all(
        torch.equal(weights[name], other[name]) for name in weights
    )


class OneStepPong(gym.Env[npt.NDArray[np.float32], np.int64]):
    """A fake of Pong for a learning player, whose every Game ends in its first
    Step: won 21 to 0, with a Reward of +1, if the player moves up, and lost 0 to
    21, with -1, otherwise. Its input is always 0.

    It presses Ctrl+C, as the developer would, on its `ctrl_c_on`th reset with
    the regular set's first seed, if set: in that Evaluation of the Run's.
    """

    ctrl_c_on: int | None = None
    regular_resets = 0

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed == 1_000_000:
            OneStepPong.regular_resets += 1
            if OneStepPong.regular_resets == OneStepPong.ctrl_c_on:
                press_ctrl_c()
        return np.zeros(1, np.float32), {"score": Score(left=0, right=0)}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[npt.NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        won = action == 0
        score = Score(left=0, right=21) if won else Score(left=21, right=0)
        reward = 1.0 if won else -1.0
        return np.zeros(1, np.float32), reward, True, False, {"score": score}


class FakeConfirmation:
    """The Confirmation of a fake-pong Run, a fake that plays no Game, so that no
    test ever plays the Confirmation's set (issue #16): it gives the Milestone one
    Game of Score margin `MARGIN`, a fixed result.

    It presses Ctrl+C in it, as the developer would, if `ctrl_c`, and asks `stop`
    then whether to abandon it, as the real one does.
    """

    MARGIN = 17
    ctrl_c = False


PongCommand = Callable[[list[str]], int]
"""The `pong` command, run in-process with the command line it is given, giving
its exit code."""


@pytest.fixture
def pong_with_fake_pong(monkeypatch: pytest.MonkeyPatch) -> PongCommand:
    """`pong`, with one more learning player, of OneStepPong, whose Runs have
    Evaluations, as the Position learner's do and the Agent's will:
    `--player fake-pong`.

    Its Milestones' Evaluations on the regular set are the real ones, and its
    Runs' Confirmation is FakeConfirmation.
    """
    from torch import nn

    from pong_ai_bot.evaluation import EvaluationStopped, SeedSet
    from pong_ai_bot.learning_players import milestone_evaluation

    player = LearningPlayer(
        make_env=OneStepPong,
        make_network=lambda: nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 3)),
        actions=3,
        stack_length=1,
        evaluated=True,
    )
    monkeypatch.setattr(OneStepPong, "regular_resets", 0)
    monkeypatch.setattr(FakeConfirmation, "ctrl_c", False)

    def confirmation_faked(network: Any, make_env: Any) -> Any:
        evaluate_milestone = milestone_evaluation(network, make_env)

        def evaluation(milestone: Any, *, seed_set: SeedSet, stop: Any) -> Evaluation:
            if seed_set is not SeedSet.CONFIRMATION:
                return evaluate_milestone(milestone, seed_set=seed_set, stop=stop)
            if FakeConfirmation.ctrl_c:
                press_ctrl_c()
            if stop():
                raise EvaluationStopped
            margin = FakeConfirmation.MARGIN
            return Evaluation(
                games=(GameResult(seed=0, score_margin=margin, cut_off=False),)
            )

        return evaluation

    def pong(argv: list[str]) -> int:
        return main(
            argv,
            learning_players={**LEARNING_PLAYERS, "fake-pong": player},
            make_milestone_evaluation=confirmation_faked,
        )

    return pong


def test_pong_resume_refuses_a_run_that_had_its_confirmation(
    capsys: pytest.CaptureFixture[str],
    pong_with_fake_pong: PongCommand,
) -> None:
    # Stopped at its Milestone of Step 20, before its Budget, and confirmed: a
    # fixed result, of one Game, rather than one played.
    run = RunDirectory.create(
        player="fake-pong",
        name="tiny",
        seed=1,
        settings=RunSettings(budget=30, checkpoint_every=5, milestone_every=20),
        commit=A_COMMIT,
    )
    for save, step in [(run.save_checkpoint, 20), (run.save_milestone, 20)]:
        save(step, lambda file: file.write(b"saved"))
    run.record_confirmation(
        Evaluation(games=(GameResult(seed=0, score_margin=17, cut_off=False),)),
        step=20,
    )
    files = sorted(run.path.iterdir())

    exit_code = pong_with_fake_pong(["resume", str(run.path), "--device", "cpu"])

    assert exit_code == 1
    assert capsys.readouterr().err.splitlines()[-1] == (
        "The Run had its Confirmation, of the Milestone at Step 20: a Run never "
        "goes on after it."
    )
    assert sorted(run.path.iterdir()) == files


def milestone_moving_up(run: RunDirectory, *, step: int) -> Path:
    """The path of the fake-pong Run `run`'s Milestone at `step`, of a network whose
    highest Q is always the Action up's, which wins OneStepPong's Games: no
    weights, and a last bias of 1 for up and 0 for the others."""
    import torch

    from pong_ai_bot.environment import Action

    weights = {
        "0.weight": torch.zeros(8, 1),
        "0.bias": torch.zeros(8),
        "2.weight": torch.zeros(3, 8),
        "2.bias": torch.zeros(3),
    }
    weights["2.bias"][Action.UP] = 1.0
    return run.save_milestone(step, lambda file: torch.save(weights, file)).path


def test_pong_evaluate_plays_a_milestone_after_its_run_and_records_nothing_in_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pong_with_fake_pong: PongCommand,
) -> None:
    run = RunDirectory.create(player="fake-pong", name="tiny", seed=1, commit=A_COMMIT)
    milestone = milestone_moving_up(run, step=20)
    # The Run's own Evaluation of the Milestone, a fixed result of one Game.
    run.record_evaluation(
        Evaluation(games=(GameResult(seed=0, score_margin=-3, cut_off=True),)), step=20
    )
    recorded = (run.path / "evaluations.jsonl").read_text(encoding="utf-8")
    output = tmp_path / "milestone.jsonl"

    exit_code = pong_with_fake_pong(
        ["evaluate", "--milestone", str(milestone), "--output", str(output)]
    )

    assert exit_code == 0
    # Every one of the regular set's 100 Games won 21 to 0, by moving up.
    summary = (
        "Evaluation on the regular set at Step 20: mean +21.00, std 0.00, min +21, "
        "100 of 100 Games won, 0 cut off"
    )
    assert capsys.readouterr().out.splitlines() == [summary]
    [line] = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert (line["step"], line["set"]) == (20, "regular")
    assert [game["seed"] for game in line["games"]] == list(range(1_000_000, 1_000_100))
    # The Run's results are as they were, so the Milestone it chooses is too.
    assert (run.path / "evaluations.jsonl").read_text(encoding="utf-8") == recorded


@pytest.mark.parametrize(
    "evaluated",
    ["its own Run's Milestone", "another Run's Milestone", "a Scripted player"],
)
def test_pong_evaluate_refuses_to_add_its_line_to_any_run_s_results(
    capsys: pytest.CaptureFixture[str],
    pong_with_fake_pong: PongCommand,
    evaluated: str,
) -> None:
    # Only a Run itself records its Evaluations, so that one after it never
    # changes the Milestone it chooses (#17): a line of another player, or with
    # no Step, would also leave it unable to choose one for its Confirmation.
    run = RunDirectory.create(player="fake-pong", name="tiny", seed=1, commit=A_COMMIT)
    milestone = milestone_moving_up(run, step=20)
    other = RunDirectory.create(player="test-env-1", name="other", seed=2)
    results = run if evaluated == "its own Run's Milestone" else other
    files = {path: sorted(path.iterdir()) for path in (run.path, other.path)}
    output = ["--output", str(results.path / "evaluations.jsonl")]

    if evaluated == "a Scripted player":
        exit_code = main(
            ["evaluate", "--scripted", "random", *output], pong=OneStepGames
        )
    else:
        exit_code = pong_with_fake_pong(
            ["evaluate", "--milestone", str(milestone), *output]
        )

    assert exit_code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "evaluations.jsonl" in err.splitlines()[-1], err
    assert {path: sorted(path.iterdir()) for path in files} == files


@pytest.mark.parametrize(
    ("given", "reason"),
    [
        ("checkpoint", "not a Milestone"),
        ("file outside a run", "not a Milestone"),
        ("test environment's milestone", "no Evaluation"),
    ],
)
def test_pong_evaluate_refuses_what_is_not_a_milestone_it_can_evaluate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pong_with_fake_pong: PongCommand,
    given: str,
    reason: str,
) -> None:
    player = "test-env-1" if given == "test environment's milestone" else "fake-pong"
    run = RunDirectory.create(player=player, name="tiny", seed=1, commit=A_COMMIT)
    if given == "checkpoint":
        path = run.save_checkpoint(20, lambda file: file.write(b"saved")).path
    elif given == "file outside a run":
        path = tmp_path / "milestone_000000020.pt"
        path.write_bytes(b"saved")
    else:
        # A test environment's Games have no score to count them by (#17).
        path = run.save_milestone(20, lambda file: file.write(b"saved")).path

    exit_code = pong_with_fake_pong(["evaluate", "--milestone", str(path)])

    assert exit_code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert reason in err.splitlines()[-1], err


TINY_EVALUATED_RUN = [
    "--name",
    "tiny",
    "--seed",
    "1",
    "--device",
    "cpu",
    "--budget",
    "30",
    "--fill-steps",
    "10",
    "--replay-buffer-size",
    "100",
    "--batch-size",
    "4",
    "--checkpoint-every",
    "5",
    "--milestone-every",
    "20",
    "--stop-mean",
    "22",
]
"""A Run of 30 Steps, with Milestones at Step 0, 20 and 30, on the CPU. It never
stops before its Budget: no Evaluation's mean reaches 22, above a Game's highest
Score margin, though OneStepPong's greedy player wins all its Games from Step 0."""


@pytest.mark.slow
def test_pong_train_evaluates_each_milestone_on_the_regular_set(
    runs: Path,
    capsys: pytest.CaptureFixture[str],
    pong_with_fake_pong: PongCommand,
) -> None:
    # Slow for the learner's optimizer, as above.

    exit_code = pong_with_fake_pong(
        ["train", "--player", "fake-pong", *TINY_EVALUATED_RUN]
    )

    assert exit_code == 0
    [path] = runs.iterdir()
    lines = [
        json.loads(line)
        for line in (path / "evaluations.jsonl").read_text("utf-8").splitlines()
    ]
    regular, [confirmation] = lines[:3], lines[3:]
    assert [(line["step"], line["set"]) for line in regular] == [
        (0, "regular"),
        (20, "regular"),
        (30, "regular"),
    ]
    for line in regular:
        assert [game["seed"] for game in line["games"]] == list(
            range(1_000_000, 1_000_100)
        )
        # Every Game won or lost 21 to 0, as the greedy player moves up or not.
        assert {abs(game["score_margin"]) for game in line["games"]} == {21}
    # Then the fake Confirmation, of one of them.
    assert confirmation["set"] == "confirmation"
    assert confirmation["step"] in {0, 20, 30}
    summaries = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("Evaluation on")
    ]
    assert [summary.split(":")[0] for summary in summaries] == [
        "Evaluation on the regular set at Step 0",
        "Evaluation on the regular set at Step 20",
        "Evaluation on the regular set at Step 30",
        f"Evaluation on the confirmation set at Step {confirmation['step']}",
    ]


@pytest.mark.slow
def test_ctrl_c_in_the_run_s_last_evaluation_leaves_it_to_pong_resume(
    runs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pong_with_fake_pong: PongCommand,
) -> None:
    # Slow for the learner's optimizer, as above. Ctrl+C in the Evaluation at
    # the Budget, the Run's 3rd.
    monkeypatch.setattr(OneStepPong, "ctrl_c_on", 3)

    exit_code = pong_with_fake_pong(
        ["train", "--player", "fake-pong", *TINY_EVALUATED_RUN]
    )

    assert exit_code == 130
    out = capsys.readouterr().out.splitlines()
    assert out[-2:] == [
        (
            "Ctrl+C abandoned the Evaluation of the Milestone at Step 30: resuming "
            "the Run measures it first."
        ),
        "Ctrl+C stopped the Run at Step 30, with a full checkpoint there.",
    ]
    [path] = runs.iterdir()
    run = RunDirectory.open(path)
    assert list(run.evaluations()) == [0, 20]
    assert run.checkpoints()[-1].step == 30
    monkeypatch.setattr(OneStepPong, "ctrl_c_on", None)

    exit_code = pong_with_fake_pong(["resume", str(path), "--device", "cpu"])

    # The Milestone at the Budget is measured, and then the best is confirmed.
    assert exit_code == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-4].startswith("Evaluation on the regular set at Step 30: ")
    assert out[-3].startswith("Evaluation on the confirmation set at Step ")
    assert out[-2].startswith("The Confirmation's mean, +17.00, is below 18: ")
    assert out[-1] == "The Run reached its Budget, 30 Steps."
    assert list(run.evaluations()) == [0, 20, 30]
    assert run.confirmation() is not None


@pytest.mark.slow
def test_ctrl_c_in_the_run_s_confirmation_leaves_it_to_pong_resume_and_only_it(
    runs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pong_with_fake_pong: PongCommand,
) -> None:
    # Slow for the learner's optimizer, as above.
    monkeypatch.setattr(FakeConfirmation, "ctrl_c", True)

    exit_code = pong_with_fake_pong(
        ["train", "--player", "fake-pong", *TINY_EVALUATED_RUN]
    )

    assert exit_code == 130
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "Ctrl+C stopped the Run at Step 30, with a full checkpoint there."
    assert out[-2].startswith("Ctrl+C abandoned the Confirmation of the Milestone")
    [path] = runs.iterdir()
    run = RunDirectory.open(path)
    assert run.confirmation() is None
    monkeypatch.setattr(FakeConfirmation, "ctrl_c", False)

    exit_code = pong_with_fake_pong(["resume", str(path), "--device", "cpu"])

    # The Run at its Budget, measured, has only its Confirmation left.
    assert exit_code == 0
    out = capsys.readouterr().out.splitlines()
    assert not [line for line in out if line.startswith("Evaluation on the regular")]
    assert out[-3].startswith("Evaluation on the confirmation set at Step ")
    assert out[-1] == "The Run reached its Budget, 30 Steps."
    assert run.confirmation() is not None
    assert run.checkpoints()[-1].step == 30

    # And once it had it, it never goes on.
    assert pong_with_fake_pong(["resume", str(path), "--device", "cpu"]) == 1
