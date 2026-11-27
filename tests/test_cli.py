"""The installed `pong` command."""

import json
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest

from pong_ai_bot.cli import main
from pong_ai_bot.game import Score


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
