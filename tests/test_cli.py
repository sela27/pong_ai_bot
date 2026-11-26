"""The installed `pong` command."""

import shutil
import subprocess
import sysconfig


def test_pong_help_runs() -> None:
    pong = shutil.which("pong", path=sysconfig.get_path("scripts"))
    assert pong is not None, "the `pong` command is not installed"

    result = subprocess.run(
        [pong, "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: pong"), result.stdout
