"""A Run's training graphs in TensorBoard, to see while it runs whether it is
healthy (issues #10, #13).

Every graph's x-axis is the Run's Step count. The names, and how often each is
written, are the spec's (#17):

- `train/score_margin` and `train/game_length`, at the end of every Game of the
  training, with `train/score_margin_avg100` and `train/game_length_avg100`, their
  moving averages over the latest 100 Games.
- `train/max_q`, every 10,000 Steps from Step 0: the mean, over the Run's fixed
  inputs, of the highest Q of each.
- `train/epsilon`, every 1,000 Steps from Step 0; and `train/loss`, the mean since
  the previous time, and `train/steps_per_second`, every 1,000 Steps that have
  them.
- `run/resumes`, at the Run's start and at every resume.
- `eval/mean`, `eval/min` and `eval/cut_off`, the mean and lowest Score margin of
  a Milestone's Evaluation and how many of its Games were cut off, at the
  Milestone's Step.
- `confirmation/mean`, the mean Score margin of the Run's Confirmation, one point
  at the end of the Run, at the Step of the Milestone confirmed.

Each is a scalar: no hparams, whose board is broken on Windows, and no histograms,
gradients or images, until something breaks (#10). They are written through
TensorBoard's `add_scalar`, as PyTorch's SummaryWriter has it. This module never
imports it: it takes seconds to import, and only `pong`'s commands need it.
"""

import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

from pong_ai_bot.evaluation import Summary

SCALARS_EVERY = 1_000
"""How many Steps between the points of the loss, ε and Steps a second (#17)."""

MAX_Q_EVERY = 10_000
"""How many Steps between the points of the Q graph (#17)."""

GAMES_AVERAGED = 100
"""How many of the latest Games the moving averages are over (#17)."""


class ScalarWriter(Protocol):
    """What the graphs need of TensorBoard's writer."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Add the point `scalar_value` at the Step `global_step` to the graph
        `tag`."""
        ...


class TrainingGraphs:
    """A Run's training graphs, written through `writer` as the Run's Steps are
    played.

    `max_q` gives the mean, over the Run's fixed inputs, of the highest Q of each,
    by the network now, and `epsilon` gives ε at a Step count.
    """

    def __init__(
        self,
        writer: ScalarWriter,
        *,
        max_q: Callable[[], float],
        epsilon: Callable[[int], float],
    ) -> None:
        self._writer = writer
        self._max_q = max_q
        self._epsilon = epsilon
        self._score_margins: deque[float] = deque(maxlen=GAMES_AVERAGED)
        self._game_lengths: deque[int] = deque(maxlen=GAMES_AVERAGED)
        self._game_rewards = 0.0
        self._game_steps = 0
        self._loss_sum = 0.0
        self._losses = 0
        self._timed_from_step = 0
        self._timed_from = time.perf_counter()

    def begin(self, step: int, *, resumes: int) -> None:
        """The Run starts, or resumes, at its Step `step`, resumed `resumes` times
        so far, with a new Game: write how many times it was resumed, and ε and
        the Q graph if their points fall on `step`."""
        self._writer.add_scalar("run/resumes", resumes, step)
        self._write_on_their_steps(step)
        self._game_rewards = 0.0
        self._game_steps = 0
        self._loss_sum = 0.0
        self._losses = 0
        self._timed_from_step = step
        self._timed_from = time.perf_counter()

    def step_played(
        self, step: int, *, reward: float, game_ended: bool, loss: float | None
    ) -> None:
        """A Step was played, the Run's `step`th: it gave `reward`, ended its Game
        if `game_ended`, whether by an end or a cut-off, and the learner's update
        in it had `loss`, or None if it made no update. Write the points that fall
        on it."""
        self._game_rewards += reward
        self._game_steps += 1
        if loss is not None:
            self._loss_sum += loss
            self._losses += 1
        if game_ended:
            self._game_ended(step)
        if step % SCALARS_EVERY == 0:
            if self._losses:
                self._writer.add_scalar(
                    "train/loss", self._loss_sum / self._losses, step
                )
                self._loss_sum = 0.0
                self._losses = 0
            now = time.perf_counter()
            self._writer.add_scalar(
                "train/steps_per_second",
                (step - self._timed_from_step) / (now - self._timed_from),
                step,
            )
            self._timed_from_step = step
            self._timed_from = now
        self._write_on_their_steps(step)

    def milestone_evaluated(
        self, step: int, summary: Summary, *, seconds: float
    ) -> None:
        """The Milestone at the Run's Step `step` was evaluated, in `seconds`, with
        `summary`: write its points, at its Step.

        The training stopped through the Evaluation, so its time doesn't count in
        the Steps a second.
        """
        self._writer.add_scalar("eval/mean", summary.mean, step)
        self._writer.add_scalar("eval/min", summary.min, step)
        self._writer.add_scalar("eval/cut_off", summary.cut_off, step)
        self._timed_from += seconds

    def milestone_confirmed(self, step: int, summary: Summary) -> None:
        """The Milestone at the Run's Step `step` had the Run's Confirmation, with
        `summary`: write its one point, at its Step."""
        self._writer.add_scalar("confirmation/mean", summary.mean, step)

    def _game_ended(self, step: int) -> None:
        # A Game's Rewards add up to its Score margin: +1 for each Point won,
        # and -1 for each lost.
        self._score_margins.append(self._game_rewards)
        self._game_lengths.append(self._game_steps)
        for tag, latest in (
            ("train/score_margin", self._score_margins),
            ("train/game_length", self._game_lengths),
        ):
            self._writer.add_scalar(tag, latest[-1], step)
            self._writer.add_scalar(
                f"{tag}_avg{GAMES_AVERAGED}", sum(latest) / len(latest), step
            )
        self._game_rewards = 0.0
        self._game_steps = 0

    def _write_on_their_steps(self, step: int) -> None:
        """Write ε and the Q graph, if their points fall on `step`."""
        if step % SCALARS_EVERY == 0:
            self._writer.add_scalar("train/epsilon", self._epsilon(step), step)
        if step % MAX_Q_EVERY == 0:
            self._writer.add_scalar("train/max_q", self._max_q(), step)
