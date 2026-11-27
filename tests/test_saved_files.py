"""The files a Run saves in its directory, in a temporary directory: its full
checkpoints, its Milestones, the fixed inputs of its Q graph, and the records of its
Milestones' Evaluations and of its Confirmation."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
import pytest

from pong_ai_bot.evaluation import Evaluation, GameResult, SeedSet
from pong_ai_bot.run_directory import (
    Commit,
    Confirmation,
    Kind,
    RunDirectory,
    SavedFile,
    SecondConfirmationError,
)


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunDirectory:
    """A new Run, with the default settings, in a temporary Runs' directory."""
    monkeypatch.setenv("PONG_AI_BOT_RUNS_DIR", str(tmp_path / "runs"))
    return RunDirectory.create(
        player="agent",
        name="baseline",
        seed=1,
        commit=Commit(
            sha="0123456789abcdef0123456789abcdef01234567", uncommitted_changes=False
        ),
        now=datetime(2026, 11, 27, 14, 30),
    )


def writing(payload: bytes) -> Callable[[BinaryIO], None]:
    """A function that writes `payload` into the file it is given, as torch.save
    would write a checkpoint's."""

    def write(file: BinaryIO) -> None:
        file.write(payload)

    return write


def test_a_saved_checkpoint_is_found_by_its_step_and_holds_what_was_written(
    run: RunDirectory,
) -> None:
    saved = run.save_checkpoint(50_000, writing(b"the payload"))

    assert saved == SavedFile(
        kind=Kind.CHECKPOINT,
        step=50_000,
        path=run.path / "checkpoint_000050000.pt",
    )
    assert run.checkpoints() == [saved]
    assert saved.path.read_bytes() == b"the payload"


def test_only_the_two_latest_checkpoints_are_kept(run: RunDirectory) -> None:
    for step in (50_000, 100_000, 150_000, 200_000):
        run.save_checkpoint(step, writing(f"at {step}".encode()))

    kept = run.checkpoints()
    assert [saved.step for saved in kept] == [150_000, 200_000]
    assert [saved.path.read_bytes() for saved in kept] == [b"at 150000", b"at 200000"]
    assert not (run.path / "checkpoint_000050000.pt").exists()
    assert not (run.path / "checkpoint_000100000.pt").exists()


def test_every_milestone_is_kept_while_old_checkpoints_are_deleted(
    run: RunDirectory,
) -> None:
    run.save_milestone(0, writing(b"weights at 0"))
    for step in (500_000, 1_000_000, 1_500_000):
        run.save_checkpoint(step, writing(f"checkpoint at {step}".encode()))
        run.save_milestone(step, writing(f"weights at {step}".encode()))

    milestones = run.milestones()
    assert [saved.step for saved in milestones] == [0, 500_000, 1_000_000, 1_500_000]
    assert milestones[1] == SavedFile(
        kind=Kind.MILESTONE,
        step=500_000,
        path=run.path / "milestone_000500000.pt",
    )
    assert milestones[1].path.read_bytes() == b"weights at 500000"
    assert [saved.step for saved in run.checkpoints()] == [1_000_000, 1_500_000]


def test_a_new_run_has_no_newest_file(run: RunDirectory) -> None:
    assert run.newest() is None


def test_the_newest_file_is_the_one_at_the_latest_step_checkpoint_or_milestone(
    run: RunDirectory,
) -> None:
    milestone = run.save_milestone(0, writing(b"weights at 0"))
    assert run.newest() == milestone

    checkpoint = run.save_checkpoint(50_000, writing(b"checkpoint at 50000"))
    assert run.newest() == checkpoint
    assert run.newest() == SavedFile(
        kind=Kind.CHECKPOINT, step=50_000, path=run.path / "checkpoint_000050000.pt"
    )


def test_at_a_step_with_both_the_newest_file_is_the_milestone(
    run: RunDirectory,
) -> None:
    # A Milestone is never deleted, so it is still there when a reader opens it,
    # however many checkpoints are saved meanwhile. Saved before the checkpoint
    # or after, it is the newest.
    first = run.save_milestone(500_000, writing(b"weights at 500000"))
    run.save_checkpoint(500_000, writing(b"checkpoint at 500000"))
    assert run.newest() == first

    run.save_checkpoint(1_000_000, writing(b"checkpoint at 1000000"))
    second = run.save_milestone(1_000_000, writing(b"weights at 1000000"))
    assert run.newest() == second


@pytest.mark.parametrize("kind", [Kind.CHECKPOINT, Kind.MILESTONE])
def test_a_file_being_written_is_not_seen_until_it_is_complete(
    run: RunDirectory, kind: Kind
) -> None:
    save = run.save_checkpoint if kind is Kind.CHECKPOINT else run.save_milestone
    before = save(50_000, writing(b"done"))
    seen_while_writing: list[SavedFile | None] = []

    def write(file: BinaryIO) -> None:
        file.write(b"half of it")
        file.flush()
        seen_while_writing.extend([run.newest(), *run.checkpoints(), *run.milestones()])
        file.write(b", and the rest")

    saved = save(100_000, write)

    # Halfway, only the file before it is there, newest or listed.
    assert seen_while_writing == [before, before]
    assert run.newest() == saved
    assert saved.path.read_bytes() == b"half of it, and the rest"


@pytest.mark.parametrize("kind", [Kind.CHECKPOINT, Kind.MILESTONE])
def test_a_write_that_fails_halfway_leaves_the_run_s_directory_as_it_was(
    run: RunDirectory, kind: Kind
) -> None:
    save = run.save_checkpoint if kind is Kind.CHECKPOINT else run.save_milestone
    before = save(50_000, writing(b"done"))
    names_before = sorted(path.name for path in run.path.iterdir())

    def write(file: BinaryIO) -> None:
        file.write(b"half of it")
        raise KeyboardInterrupt  # A second Ctrl+C, say.

    with pytest.raises(KeyboardInterrupt):
        save(100_000, write)

    assert run.newest() == before
    assert sorted(path.name for path in run.path.iterdir()) == names_before


@pytest.mark.parametrize("kind", [Kind.CHECKPOINT, Kind.MILESTONE])
def test_a_file_is_never_saved_over_one_at_the_same_step(
    run: RunDirectory, kind: Kind
) -> None:
    # A reader may have the first open, and Windows can't replace an open file.
    save = run.save_checkpoint if kind is Kind.CHECKPOINT else run.save_milestone
    first = save(50_000, writing(b"the first"))
    names_before = sorted(path.name for path in run.path.iterdir())

    with pytest.raises(FileExistsError):
        save(50_000, writing(b"the second"))

    assert first.path.read_bytes() == b"the first"
    assert sorted(path.name for path in run.path.iterdir()) == names_before


def test_an_old_checkpoint_a_reader_has_open_is_deleted_at_a_later_save(
    run: RunDirectory,
) -> None:
    # Windows can't delete an open file, and an old checkpoint is only disk
    # space: the Run carries on.
    oldest = run.save_checkpoint(50_000, writing(b"at 50000"))
    run.save_checkpoint(100_000, writing(b"at 100000"))
    with oldest.path.open("rb"):
        run.save_checkpoint(150_000, writing(b"at 150000"))

    run.save_checkpoint(200_000, writing(b"at 200000"))

    assert [saved.step for saved in run.checkpoints()] == [150_000, 200_000]


@pytest.mark.parametrize(
    "inputs",
    [
        # Stacks of four Observations, as the Agent's network takes them: 0 and
        # 255, in a pattern of their own in each place.
        np.where(
            np.arange(3 * 4 * 84 * 84).reshape(3, 4, 84, 84) % 7 == 0, 255, 0
        ).astype(np.uint8),
        # Positions, as the Position learner's takes them: no two the same.
        np.linspace(-0.123456789, 1.0, 5 * 16, dtype=np.float32).reshape(5, 16),
    ],
    ids=["observations", "positions"],
)
def test_the_q_graph_s_inputs_are_read_back_exactly_as_saved_after_reopening(
    run: RunDirectory, inputs: npt.NDArray[np.generic]
) -> None:
    run.save_q_graph_inputs(inputs)

    read_back = RunDirectory.open(run.path, current_commit=run.commit).q_graph_inputs()
    assert read_back.dtype == inputs.dtype
    assert read_back.shape == inputs.shape
    assert np.array_equal(read_back, inputs)


def regular_evaluation(*margins: int) -> Evaluation:
    """An Evaluation on the regular set, of a Game for each Score margin, rather
    than 100."""
    return Evaluation(
        games=tuple(
            GameResult(seed=1_000_000 + game, score_margin=margin, cut_off=False)
            for game, margin in enumerate(margins)
        )
    )


def test_a_new_run_has_no_evaluation(run: RunDirectory) -> None:
    assert run.evaluations() == {}


def test_each_milestone_s_evaluation_is_a_line_of_its_file_read_back_by_its_step(
    run: RunDirectory,
) -> None:
    first, second = regular_evaluation(-21, -19), regular_evaluation(3, 5)

    run.record_evaluation(first, step=0)
    run.record_evaluation(second, step=500_000)

    lines = (run.path / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        json.loads(first.json_line(seed_set=SeedSet.REGULAR, step=0)),
        json.loads(second.json_line(seed_set=SeedSet.REGULAR, step=500_000)),
    ]
    reopened = RunDirectory.open(run.path, current_commit=run.commit)
    assert reopened.evaluations() == {0: first, 500_000: second}


A_CONFIRMATION = Evaluation(
    games=(
        GameResult(seed=1_000_100, score_margin=21, cut_off=False),
        GameResult(seed=1_000_101, score_margin=15, cut_off=True),
    )
)
"""A Confirmation's result, of two Games rather than 100."""


def test_a_new_run_has_had_no_confirmation(run: RunDirectory) -> None:
    assert run.confirmation() is None


def test_a_recorded_confirmation_is_read_back_after_the_run_is_reopened(
    run: RunDirectory,
) -> None:
    run.record_confirmation(A_CONFIRMATION, step=1_500_000)

    reopened = RunDirectory.open(run.path, current_commit=run.commit)
    assert reopened.confirmation() == Confirmation(
        step=1_500_000, evaluation=A_CONFIRMATION
    )


def test_a_second_confirmation_is_refused_and_the_first_one_is_kept(
    run: RunDirectory,
) -> None:
    run.record_confirmation(A_CONFIRMATION, step=1_500_000)
    another = Evaluation(
        games=(GameResult(seed=1_000_100, score_margin=-3, cut_off=False),)
    )

    with pytest.raises(SecondConfirmationError, match="Step 1,500,000"):
        run.record_confirmation(another, step=2_000_000)

    assert run.confirmation() == Confirmation(step=1_500_000, evaluation=A_CONFIRMATION)
