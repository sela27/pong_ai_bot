"""The Run directory, in a temporary directory: where a Run is kept, and the settings
file that lets it be reproduced."""

import os
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import pytest

import pong_ai_bot
from pong_ai_bot.run_directory import (
    Commit,
    CommitChangedWarning,
    RunDirectory,
    checked_out_commit,
    runs_root,
)
from pong_ai_bot.run_settings import RunSettings

A_COMMIT = Commit(
    sha="0123456789abcdef0123456789abcdef01234567", uncommitted_changes=False
)
"""The commit the tests' Runs are created at, rather than asking git for it."""

AT_14_30 = datetime(2026, 11, 27, 14, 30, 5)


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The Runs' directory for one test, a temporary one."""
    root = tmp_path / "runs"
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(root))
    return root


def test_the_runs_are_kept_in_the_directory_the_environment_variable_names(
    runs: Path,
) -> None:
    assert runs_root() == runs


def test_without_the_environment_variable_the_runs_are_kept_in_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A temporary home, so even a mistake never reaches the real one.
    monkeypatch.delenv("PONG_AI_BOT_RUNS_DIR")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    assert runs_root() == tmp_path / "pong_ai_bot_runs"


def test_a_new_run_s_directory_is_named_by_its_date_time_player_name_and_seed(
    runs: Path,
) -> None:
    run = RunDirectory.create(
        player="agent",
        name="baseline",
        seed=1,
        settings=RunSettings(),
        commit=A_COMMIT,
        now=AT_14_30,
    )

    assert run.path == runs / "2026-11-27_1430_agent_baseline_seed1"
    assert run.path.is_dir()


def test_reopening_a_run_reads_back_everything_its_settings_file_recorded(
    runs: Path,
) -> None:
    settings = RunSettings(budget=200, fill_steps=10, max_gradient_norm=10.0)
    at_a_changed_commit = Commit(
        sha="fedcba9876543210fedcba9876543210fedcba98", uncommitted_changes=True
    )
    created = RunDirectory.create(
        player="position-learner",
        name="tiny",
        seed=999_999,
        settings=settings,
        commit=at_a_changed_commit,
        now=AT_14_30,
    )

    run = RunDirectory.open(created.path, current_commit=at_a_changed_commit)

    assert run.path == created.path
    assert (run.player, run.name, run.seed) == ("position-learner", "tiny", 999_999)
    assert run.settings == settings
    # The locked width, 100% of half the Paddle's height (#31).
    assert run.opponent_offset_width == 1.0
    assert run.commit == at_a_changed_commit


def test_a_run_in_the_same_directory_fails_and_leaves_the_first_one_s_settings(
    runs: Path,
) -> None:
    # Same minute, player, name and seed: the same directory.
    first = RunDirectory.create(
        player="agent",
        name="baseline",
        seed=1,
        settings=RunSettings(budget=200),
        commit=A_COMMIT,
        now=datetime(2026, 11, 27, 14, 30, 5),
    )

    with pytest.raises(FileExistsError):
        RunDirectory.create(
            player="agent",
            name="baseline",
            seed=1,
            settings=RunSettings(budget=300),
            commit=A_COMMIT,
            now=datetime(2026, 11, 27, 14, 30, 50),
        )

    reopened = RunDirectory.open(first.path, current_commit=A_COMMIT)
    assert reopened.settings.budget == 200


def test_a_seed_of_a_million_or_more_is_refused_and_no_run_is_created(
    runs: Path,
) -> None:
    # A million and up are the Evaluation's seeds (#13).
    with pytest.raises(ValueError, match="1,000,000"):
        RunDirectory.create(
            player="agent",
            name="baseline",
            seed=1_000_000,
            settings=RunSettings(),
            commit=A_COMMIT,
            now=AT_14_30,
        )

    assert not runs.exists()


@pytest.mark.parametrize(
    ("player", "name"),
    [("agent", "../elsewhere"), ("agent", "a b"), ("agent", ""), ("a/b", "baseline")],
)
def test_a_player_or_name_that_is_not_a_plain_word_is_refused(
    runs: Path, player: str, name: str
) -> None:
    with pytest.raises(ValueError, match="letters, digits"):
        RunDirectory.create(
            player=player, name=name, seed=1, commit=A_COMMIT, now=AT_14_30
        )

    assert not runs.exists()


def test_reopening_a_run_with_other_settings_fails_and_names_what_differs(
    runs: Path,
) -> None:
    created = RunDirectory.create(
        player="agent",
        name="baseline",
        seed=1,
        settings=RunSettings(budget=200, batch_size=8),
        commit=A_COMMIT,
        now=AT_14_30,
    )
    # The same settings, given again, reopen it.
    RunDirectory.open(
        created.path,
        current_commit=A_COMMIT,
        settings=RunSettings(budget=200, batch_size=8),
    )

    with pytest.raises(ValueError, match="budget: 200 recorded, 300 given") as error:
        RunDirectory.open(
            created.path,
            current_commit=A_COMMIT,
            settings=RunSettings(budget=300, batch_size=8),
        )
    assert "batch_size" not in str(error.value)


def test_reopening_a_run_at_another_commit_warns_and_at_its_own_does_not(
    runs: Path,
) -> None:
    created = RunDirectory.create(
        player="agent",
        name="baseline",
        seed=1,
        settings=RunSettings(),
        commit=A_COMMIT,
        now=AT_14_30,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RunDirectory.open(created.path, current_commit=A_COMMIT)

    later = Commit(
        sha="89abcdef0123456789abcdef0123456789abcdef", uncommitted_changes=False
    )
    with pytest.warns(CommitChangedWarning) as warned:
        run = RunDirectory.open(created.path, current_commit=later)

    message = str(warned[0].message)
    assert A_COMMIT.sha in message and later.sha in message, message
    # It still opens, as the Run recorded it.
    assert run.commit == A_COMMIT


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A new git repository with one commit, and nothing changed since."""
    # Inside the pre-commit hook, git's own variables point at the outer
    # repository: clear them so git finds this one.
    for variable in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(variable)
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "--quiet")
    (repository / "tracked.txt").write_text("first\n", encoding="utf-8")
    git(repository, "add", "tracked.txt")
    git(repository, "commit", "--quiet", "--message", "First")
    return repository


def test_the_checked_out_commit_is_head_with_no_uncommitted_changes(
    repository: Path,
) -> None:
    assert checked_out_commit(repository) == Commit(
        sha=git(repository, "rev-parse", "HEAD"), uncommitted_changes=False
    )


@pytest.mark.parametrize("changed", ["tracked.txt", "untracked.txt"])
def test_a_changed_or_untracked_file_is_an_uncommitted_change(
    repository: Path, changed: str
) -> None:
    (repository / changed).write_text("second\n", encoding="utf-8")

    assert checked_out_commit(repository).uncommitted_changes


def test_a_run_created_with_no_commit_given_records_the_code_s_own(
    runs: Path,
) -> None:
    run = RunDirectory.create(player="agent", name="baseline", seed=1, now=AT_14_30)

    package = Path(pong_ai_bot.__file__).parent
    assert run.commit.sha == git(package, "rev-parse", "HEAD")
