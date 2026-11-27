"""The Run directory: where a Run is kept, outside the repository (issues #10, #11).

All the Runs are kept in one directory, ~/pong_ai_bot_runs unless the environment
variable PONG_AI_BOT_RUNS_DIR names another, so a Run is never deleted along with
a git worktree. Each Run has a directory of its own, named by the date and time it
was created, its player, its short name and its seed, such as
2026-11-27_1430_agent_baseline_seed1.

In it, a Run saves its full checkpoints, of which only the latest are kept, and its
Milestones, kept for good, each a new file named by its Step, such as
milestone_000500000.pt. What they hold is up to the training loop. The directory
also records the Run's one Confirmation (#16).
"""

import json
import os
import re
import subprocess
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

from pong_ai_bot.evaluation import Evaluation, GameResult, SeedSet
from pong_ai_bot.opponent import DEFAULT_OFFSET_WIDTH
from pong_ai_bot.run_settings import Loss, RunSettings

RUNS_DIR_VARIABLE = "PONG_AI_BOT_RUNS_DIR"
"""The environment variable that names the Runs' directory, if set (#17)."""

SETTINGS_FILE = "settings.json"
"""The Run's settings file, in its directory."""

CONFIRMATION_FILE = "confirmation.json"
"""The record of the Run's Confirmation, in its directory, once it has had one."""

SEED_LIMIT = min(seed_set.seeds.start for seed_set in SeedSet)
"""Every Run's seed is below this, 1,000,000: the Evaluation's seeds start here, so
a Run never trains on a Game an Evaluation plays (#13)."""


def runs_root() -> Path:
    """The directory all the Runs are kept in."""
    named = os.environ.get(RUNS_DIR_VARIABLE)
    if named:
        return Path(named)
    return Path.home() / "pong_ai_bot_runs"


@dataclass(frozen=True, slots=True, kw_only=True)
class Commit:
    """The code a Run was created at."""

    sha: str
    """The git commit checked out."""
    uncommitted_changes: bool
    """Whether there were changes not committed, untracked files included."""


PACKAGE_DIRECTORY = Path(__file__).resolve().parent
"""This package's own directory, inside the repository it is checked out from."""


def checked_out_commit(repository: Path = PACKAGE_DIRECTORY) -> Commit:
    """The commit checked out in the git repository `repository` is in, and whether
    it has changes not committed."""
    status = subprocess.run(
        ["git", "status", "--porcelain=v2", "--branch", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    # Headers start with "#", and every other line is a change.
    (sha,) = [line.split()[2] for line in status if line.startswith("# branch.oid ")]
    return Commit(
        sha=sha, uncommitted_changes=any(not line.startswith("#") for line in status)
    )


class CommitChangedWarning(UserWarning):
    """A Run reopened at another commit than the one it was created at."""


class Kind(StrEnum):
    """What a file a Run saves at a Step holds."""

    CHECKPOINT = "checkpoint"
    """A full checkpoint, all a Run needs to resume. Only the latest are kept."""
    MILESTONE = "milestone"
    """A Milestone, the network's weights alone, kept for good."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SavedFile:
    """A file a Run saved at one of its Steps."""

    kind: Kind
    step: int
    """The Run's Step the file was saved at."""
    path: Path


class SecondConfirmationError(RuntimeError):
    """A Run that has had its Confirmation given another (issue #16)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Confirmation:
    """A Run's Confirmation, as its directory recorded it."""

    step: int
    """The Step of the Milestone confirmed."""
    evaluation: Evaluation
    """The Milestone's Evaluation on the Confirmation's set."""


def _write_new(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a new file at `path`, with `write` writing its contents into the file
    it is given.

    It is written under a temporary name first, and renamed when complete, so no
    one ever reads it half written. If the writing fails, nothing is left. It never
    replaces a file, which a reader may have open: on Windows a rename never does,
    so if there is one at `path` already, it fails with a FileExistsError.
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as file:
            write(file)
            # On disk before the rename, so a crash of the whole machine can't
            # leave the name on a file that is only half there.
            file.flush()
            os.fsync(file.fileno())
        temporary.rename(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True, kw_only=True)
class RunDirectory:
    """One Run's directory, and what its settings file recorded when it was
    created."""

    path: Path
    player: str
    """The learning player the Run trains."""
    name: str
    """The Run's short name."""
    seed: int
    settings: RunSettings
    opponent_offset_width: float
    """The Opponent's offset width, as locked in the code when the Run was
    created."""
    commit: Commit
    """The code the Run was created at."""

    @classmethod
    def create(
        cls,
        *,
        player: str,
        name: str,
        seed: int,
        settings: RunSettings | None = None,
        commit: Commit | None = None,
        now: datetime | None = None,
    ) -> RunDirectory:
        """A new Run's directory, in the Runs' directory, with its settings file.

        `settings` are the defaults unless given. `commit` is the code's, the
        commit checked out, unless given; `now` is the time the Run is created at,
        the local time unless given. A Run's directory is never reused: if it
        exists already, it fails.
        """
        if seed >= SEED_LIMIT:
            raise ValueError(
                f"A Run's seed must be below {SEED_LIMIT:,}, the Evaluation's seeds; "
                f"got {seed:,}."
            )
        for part, value in (("player", player), ("name", name)):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError(
                    f"A Run's {part} must be letters, digits, '-' and '_' only, to "
                    f"name its directory; got {value!r}."
                )
        if settings is None:
            settings = RunSettings()
        if commit is None:
            commit = checked_out_commit()
        if now is None:
            now = datetime.now()
        path = runs_root() / f"{now:%Y-%m-%d_%H%M}_{player}_{name}_seed{seed}"
        path.mkdir(parents=True)
        run = cls(
            path=path,
            player=player,
            name=name,
            seed=seed,
            settings=settings,
            opponent_offset_width=DEFAULT_OFFSET_WIDTH,
            commit=commit,
        )
        record = {
            "player": player,
            "name": name,
            "seed": seed,
            "settings": asdict(settings),
            "opponent_offset_width": run.opponent_offset_width,
            "commit": commit.sha,
            "uncommitted_changes": commit.uncommitted_changes,
        }
        (path / SETTINGS_FILE).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        return run

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        settings: RunSettings | None = None,
        current_commit: Commit | None = None,
    ) -> RunDirectory:
        """An existing Run's directory, as its settings file recorded it.

        A Run's settings never change: given `settings` other than the recorded
        ones, it fails, naming each setting that differs. If the code is now at
        another commit than the Run was created at, it warns, with a
        CommitChangedWarning. `current_commit` is the commit checked out, unless
        given.
        """
        record = json.loads((path / SETTINGS_FILE).read_text(encoding="utf-8"))
        recorded = record["settings"]
        run = cls(
            path=path,
            player=record["player"],
            name=record["name"],
            seed=record["seed"],
            settings=RunSettings(**{**recorded, "loss": Loss(recorded["loss"])}),
            opponent_offset_width=record["opponent_offset_width"],
            commit=Commit(
                sha=record["commit"], uncommitted_changes=record["uncommitted_changes"]
            ),
        )
        if settings is not None and settings != run.settings:
            differences = "; ".join(
                f"{field.name}: {getattr(run.settings, field.name)} recorded, "
                f"{getattr(settings, field.name)} given"
                for field in fields(RunSettings)
                if getattr(run.settings, field.name) != getattr(settings, field.name)
            )
            raise ValueError(
                f"A Run's settings never change; other settings are a new Run. "
                f"{differences}."
            )
        if current_commit is None:
            current_commit = checked_out_commit()
        if current_commit.sha != run.commit.sha:
            warnings.warn(
                f"The Run was created at commit {run.commit.sha}, and the code is "
                f"now at commit {current_commit.sha}.",
                CommitChangedWarning,
                stacklevel=2,
            )
        return run

    def save_checkpoint(
        self, step: int, write: Callable[[BinaryIO], object]
    ) -> SavedFile:
        """Save a full checkpoint at `step`: `write` writes its contents into the
        file it is given, as torch.save does.

        No one sees the file before it is complete, and if `write` fails, nothing
        is left of it. It never replaces a file: if the Run has a checkpoint at
        `step` already, it fails with a FileExistsError. Only the latest
        checkpoints are kept, as many as the Run's settings say: the older ones are
        deleted.
        """
        saved = self._save(Kind.CHECKPOINT, step, write)
        for old in self.checkpoints()[: -self.settings.checkpoints_kept]:
            try:
                old.path.unlink()
            except PermissionError:
                # Windows can't delete a file a reader has open. An old
                # checkpoint is only disk space: a later save deletes it.
                pass
        return saved

    def save_milestone(
        self, step: int, write: Callable[[BinaryIO], object]
    ) -> SavedFile:
        """Save a Milestone at `step`, kept for good: `write` writes its contents
        into the file it is given, as torch.save does.

        As with a checkpoint, no one sees the file before it is complete, and if
        the Run has a Milestone at `step` already, it fails with a FileExistsError.
        """
        return self._save(Kind.MILESTONE, step, write)

    def checkpoints(self) -> list[SavedFile]:
        """The Run's full checkpoints, by Step."""
        return self._saved(Kind.CHECKPOINT)

    def milestones(self) -> list[SavedFile]:
        """The Run's Milestones, by Step."""
        return self._saved(Kind.MILESTONE)

    def newest(self) -> SavedFile | None:
        """The file saved at the Run's latest Step, a checkpoint or a Milestone, or
        None before the first.

        At a Step with both, it is the Milestone: it is never deleted, so it is
        still there when a reader opens it.
        """
        return max(
            self.checkpoints() + self.milestones(),
            key=lambda saved: (saved.step, saved.kind is Kind.MILESTONE),
            default=None,
        )

    def record_confirmation(self, evaluation: Evaluation, *, step: int) -> None:
        """Record the Run's Confirmation: `evaluation`, on the Confirmation's set,
        of the Milestone at `step`.

        A Run has one Confirmation, so its set never becomes a set to choose by: if
        it has had one, it fails with a SecondConfirmationError, and the first one
        stays recorded.
        """
        recorded = self.confirmation()
        if recorded is not None:
            raise SecondConfirmationError(
                f"The Run has had its Confirmation already, of the Milestone at "
                f"Step {recorded.step:,}; a Run has only one."
            )
        line = evaluation.json_line(seed_set=SeedSet.CONFIRMATION, step=step)
        _write_new(
            self.path / CONFIRMATION_FILE,
            lambda file: file.write(line.encode("utf-8") + b"\n"),
        )

    def confirmation(self) -> Confirmation | None:
        """The Run's Confirmation, as recorded, or None if it has had none."""
        path = self.path / CONFIRMATION_FILE
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return Confirmation(
            step=record["step"],
            evaluation=Evaluation(
                games=tuple(GameResult(**game) for game in record["games"])
            ),
        )

    def _save(
        self, kind: Kind, step: int, write: Callable[[BinaryIO], object]
    ) -> SavedFile:
        saved = SavedFile(
            kind=kind, step=step, path=self.path / f"{kind}_{step:09d}.pt"
        )
        _write_new(saved.path, write)
        return saved

    def _saved(self, kind: Kind) -> list[SavedFile]:
        saved = []
        for path in self.path.iterdir():
            match = re.fullmatch(rf"{kind}_(\d+)\.pt", path.name)
            if match:
                saved.append(SavedFile(kind=kind, step=int(match[1]), path=path))
        return sorted(saved, key=lambda file: file.step)
