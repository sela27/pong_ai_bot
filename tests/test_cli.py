"""The installed `pong` command."""

import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from pong_ai_bot.cli import main


def run_pong(*args: str) -> subprocess.CompletedProcess[str]:
    pong = shutil.which("pong", path=sysconfig.get_path("scripts"))
    assert pong is not None, "the `pong` command is not installed"
    return subprocess.run([pong, *args], capture_output=True, text=True, check=False)


def test_pong_help_runs() -> None:
    result = run_pong("--help")

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: pong"), result.stdout


def test_pong_has_a_measure_env_command() -> None:
    # Only its help: the measurement itself is a script, not a test (issue #11).
    result = run_pong("measure-env", "--help")

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: pong measure-env"), result.stdout


def test_pong_evaluate_prints_the_summary_line_and_adds_the_json_line_to_a_file(
    tmp_path: Path,
) -> None:
    # One test for both, to keep the hook fast: the random player loses nearly
    # every Point at once, but its 100 Games still take a couple of seconds.
    output = tmp_path / "random.jsonl"
    output.write_text('{"earlier": "line"}\n', encoding="utf-8")

    result = run_pong("evaluate", "--scripted", "random", "--output", str(output))

    assert result.returncode == 0, result.stderr
    printed = result.stdout.splitlines()
    assert len(printed) == 1, result.stdout
    assert printed[0].startswith("Evaluation on the regular set: mean "), printed[0]
    assert " of 100 Games won, " in printed[0], printed[0]
    # The line is added to the end of the file, after what was there.
    earlier, added = output.read_text(encoding="utf-8").splitlines()
    assert json.loads(earlier) == {"earlier": "line"}
    line = json.loads(added)
    # A Scripted player has no Run, so no Step, and plays the regular set.
    assert line["step"] is None
    assert line["set"] == "regular"
    assert [game["seed"] for game in line["games"]] == list(range(1_000_000, 1_000_100))


def test_pong_evaluate_offers_the_copy_of_the_opponent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Only its help, in-process: its 100 Games are evenly matched, and long.
    with pytest.raises(SystemExit) as exit_:
        main(["evaluate", "--help"])

    assert exit_.value.code == 0
    assert "opponent-copy" in capsys.readouterr().out


def test_pong_evaluate_offers_the_edge_predictor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Only its help, in-process: its 100 Games take far longer than the hook allows.
    with pytest.raises(SystemExit) as exit_:
        main(["evaluate", "--help"])

    assert exit_.value.code == 0
    assert "edge-predictor" in capsys.readouterr().out


def test_pong_evaluate_offers_no_confirmation_set() -> None:
    # Only the end of a Run plays the Confirmation's set (issue #16).
    result = run_pong("evaluate", "--help")

    assert result.returncode == 0, result.stderr
    assert "confirmation" not in result.stdout.lower(), result.stdout
