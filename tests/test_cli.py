"""The installed `pong` command."""

import json
import re
import shutil
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
from pong_ai_bot.run_directory import RunDirectory
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


def same_weights(weights: dict[str, Any], other: dict[str, Any]) -> bool:
    import torch

    return weights.keys() == other.keys() and all(
        torch.equal(weights[name], other[name]) for name in weights
    )
