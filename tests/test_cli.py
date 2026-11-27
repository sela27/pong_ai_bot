"""The installed `pong` command."""

import json
import re
import shutil
import signal
import subprocess
import sysconfig
from dataclasses import fields
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest

from pong_ai_bot.cli import main
from pong_ai_bot.game import Score
from pong_ai_bot.run_directory import Commit, RunDirectory, checked_out_commit
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # In-process, and with Games of one Step, to keep the hook fast: the random
    # player's 100 real Games take a couple of seconds.
    monkeypatch.setattr("pong_ai_bot.scripted_players.PongEnv", OneStepGames)
    output = tmp_path / "random.jsonl"
    output.write_text('{"earlier": "line"}\n', encoding="utf-8")

    exit_code = main(["evaluate", "--scripted", "random", "--output", str(output)])

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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As with a PyTorch build that doesn't support the graphics card: it would
    # train 20 times slower, with no error (issue #11).
    runs = tmp_path / "runs"
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(runs))
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    exit_code = main(
        ["train", "--player", "test-env-1", "--name", "tiny", "--seed", "1"]
    )

    assert exit_code == 1
    assert "--device cpu" in capsys.readouterr().err
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
        ("--max-gradient-norm X", "default none"),
        ("--double-dqn, --no-double-dqn", "default on"),
    ]:
        assert re.search(rf"{re.escape(option)}\s+{default}", help_text), help_text
    # And every setting, named after it.
    for field in fields(RunSettings):
        assert f"--{field.name.replace('_', '-')}" in help_text, help_text


@pytest.mark.slow
def test_pong_train_runs_a_test_environment_s_tiny_run_to_its_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Slow for the learner's optimizer: making one loads a part of PyTorch that
    # takes about 5 seconds.
    import torch

    runs = tmp_path / "runs"
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(runs))

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
            "--max-gradient-norm",
            "10",
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
        max_gradient_norm=10.0,
        loss=Loss.MSE,
        double_dqn=False,
        checkpoint_every=3,
        checkpoints_kept=20,
        milestone_every=20,
    )
    # A Milestone at Step 0, at every multiple of 20, and at the Budget, where the
    # Run stops; a checkpoint at every multiple of 3.
    assert [saved.step for saved in run.milestones()] == [0, 20, 30]
    assert [saved.step for saved in run.checkpoints()] == list(range(3, 31, 3))
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Slow for the learner's optimizer, as above.
    import torch

    from pong_ai_bot.test_environments import ConstantRewardEnv
    from pong_ai_bot.training import Q_GRAPH_INPUTS

    runs = tmp_path / "runs"
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(runs))
    # Ctrl+C is pressed while the environment plays the Run's Step 12, without a
    # real signal: Python calls the handler of SIGINT when Ctrl+C is pressed.
    # Before Step 0, the Run plays a Game of random Actions for its Q graph's
    # inputs, a Step for each.
    play_step = ConstantRewardEnv.step
    steps_played = -Q_GRAPH_INPUTS

    def step_pressing_ctrl_c_in_step_12(
        env: ConstantRewardEnv, action: int
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        nonlocal steps_played
        steps_played += 1
        if steps_played == 12:
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            handler(signal.SIGINT, None)
        return play_step(env, action)

    monkeypatch.setattr(ConstantRewardEnv, "step", step_pressing_ctrl_c_in_step_12)

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
        ]
    )

    # As a program stopped by Ctrl+C: 128 and SIGINT's number, 2.
    assert exit_code == 130
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Ctrl+C stopped the Run at Step 12, with a full checkpoint there."
    )
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


A_COMMIT = Commit(
    sha="0123456789abcdef0123456789abcdef01234567", uncommitted_changes=False
)
"""A commit that is never the code's own."""


def test_pong_resume_warns_if_the_code_is_at_another_commit_than_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(tmp_path / "runs"))
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As a Run that crashed before its first: there is nothing to resume from.
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(tmp_path / "runs"))
    run = RunDirectory.create(player="test-env-1", name="tiny", seed=1, commit=A_COMMIT)

    exit_code = main(["resume", str(run.path), "--device", "cpu"])

    assert exit_code == 1
    assert "no full checkpoint" in capsys.readouterr().err


def test_pong_resume_of_a_run_that_reached_its_budget_plays_nothing_more(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The Run's end saved a Milestone at its Budget, Step 23, after its latest
    # checkpoint, at Step 20.
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(tmp_path / "runs"))
    run = RunDirectory.create(
        player="test-env-1",
        name="tiny",
        seed=1,
        settings=RunSettings(budget=23, checkpoint_every=5),
        commit=A_COMMIT,
    )
    for save, step in [(run.save_checkpoint, 20), (run.save_milestone, 23)]:
        save(step, lambda file: file.write(b"saved"))
    files = sorted(run.path.iterdir())

    exit_code = main(["resume", str(run.path), "--device", "cpu"])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "The Run reached its Budget, 23 Steps, already: there is nothing to resume."
    )
    assert sorted(run.path.iterdir()) == files


def train_stopped_by_ctrl_c(
    monkeypatch: pytest.MonkeyPatch, settings: list[str], *, at: int, presses: int
) -> Path:
    """The directory of a new Run of test environment 1 on the CPU, with the
    options `settings`, that Ctrl+C stopped while the environment played its Step
    `at`, pressed `presses` times: once, to stop after the Step, or twice, to stop
    at once, as a crash would."""
    from pong_ai_bot.run_directory import runs_root
    from pong_ai_bot.test_environments import ConstantRewardEnv
    from pong_ai_bot.training import Q_GRAPH_INPUTS

    # As above: before Step 0, the Run plays a Step for each of its Q graph's
    # inputs.
    play_step = ConstantRewardEnv.step
    steps_played = -Q_GRAPH_INPUTS

    def step_pressing_ctrl_c(
        env: ConstantRewardEnv, action: int
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        nonlocal steps_played
        steps_played += 1
        if steps_played == at:
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            for _ in range(presses):
                handler(signal.SIGINT, None)
        return play_step(env, action)

    monkeypatch.setattr(ConstantRewardEnv, "step", step_pressing_ctrl_c)
    run = ["--player", "test-env-1", "--name", "tiny", "--seed", "1"]
    exit_code = main(["train", *run, "--device", "cpu", *settings])
    monkeypatch.setattr(ConstantRewardEnv, "step", play_step)

    assert exit_code == 130
    [path] = runs_root().iterdir()
    return path


@pytest.mark.slow
def test_pong_resume_refills_the_buffer_before_learning_on_counting_its_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Slow for the learner's optimizer, as above.
    import torch

    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(tmp_path / "runs"))
    # An update every Step, once the buffer holds 10 transitions.
    path = train_stopped_by_ctrl_c(
        monkeypatch,
        [
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
            "--checkpoint-every",
            "3",
            "--checkpoints-kept",
            "20",
            "--milestone-every",
            "20",
        ],
        at=12,
        presses=1,
    )
    capsys.readouterr()

    exit_code = main(["resume", str(path), "--device", "cpu"])

    assert exit_code == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Device: cpu"
    assert out[-1] == "The Run reached its Budget, 30 Steps."
    run = RunDirectory.open(path)
    # The Step count goes on from 12 to the Budget, with its checkpoints and
    # Milestones, and those after the resume count it.
    assert [saved.step for saved in run.checkpoints()] == list(range(3, 31, 3))
    assert [saved.step for saved in run.milestones()] == [0, 20, 30]
    checkpoints = {saved.step: torch.load(saved.path) for saved in run.checkpoints()}
    assert {step: c["resumes"] for step, c in checkpoints.items()} == {
        step: 0 if step <= 12 else 1 for step in range(3, 31, 3)
    }
    assert checkpoints[30]["learner"]["steps_given"] == 30
    # The new buffer fills in Steps 13 to 22, so the network learns on only from
    # Step 22: at Step 21 it is still the one of the checkpoint.
    at_the_stop = checkpoints[12]["learner"]["network"]
    for step in (15, 18, 21):
        assert same_weights(checkpoints[step]["learner"]["network"], at_the_stop)
    assert not same_weights(checkpoints[24]["learner"]["network"], at_the_stop)


@pytest.mark.slow
def test_pong_resume_drops_the_graphs_points_written_after_the_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Slow for the learner's optimizer, as above, and for TensorBoard's reader.
    from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
        EventAccumulator,
    )

    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(tmp_path / "runs"))
    # Stopped at once in Step 12, as by a crash: its latest checkpoint is at
    # Step 10, and the graphs have points of Step 11 too.
    path = train_stopped_by_ctrl_c(
        monkeypatch,
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
