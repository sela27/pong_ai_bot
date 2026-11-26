"""The installed `pong` command."""

import shutil
import subprocess
import sysconfig


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
