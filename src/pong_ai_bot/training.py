"""The training loop: Steps against any Gymnasium environment (issue #10).

Each Step, the learner chooses an Action, with ε by the Step counter alone, and
learns from the Step's transition. The loop never reads the `info`. It knows
nothing of Pong, and of files only through the Run directory: a Run's loop saves
its full checkpoints and its Milestones there, and stops at its Budget, or when
Ctrl+C is pressed. A Run that stopped resumes from its latest full checkpoint, as
the same Run. The loop writes the Run's graphs too, through the writer it is given,
and evaluates each Milestone with the Evaluation it is given, if any (#17). A Run
with Evaluations stops at its first Milestone good enough, too, and at its end
gives its best Milestone its one Confirmation (issues #13, #14).
"""

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from pong_ai_bot.ctrl_c import CtrlC, catch_ctrl_c
from pong_ai_bot.evaluation import Evaluation, EvaluationStopped, SeedSet
from pong_ai_bot.graphs import ScalarWriter, TrainingGraphs
from pong_ai_bot.run_settings import RunSettings

if TYPE_CHECKING:
    from pong_ai_bot.run_directory import RunDirectory, SavedFile

Q_GRAPH_INPUTS = 1_000
"""How many fixed inputs the Q graph averages over (#17)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EpsilonSchedule:
    """How likely the learner is to choose a random Action, by the Run's Step
    counter alone (issue #12): in a straight line from `start` to `end` over the
    first `decay_steps` Steps, and `end` from then on.

    The three come from the Run's settings `epsilon_start`, `epsilon_end` and
    `epsilon_decay_steps`, with their defaults there, in RunSettings.
    """

    start: float
    end: float
    decay_steps: int

    @classmethod
    def of_run(cls, settings: RunSettings) -> EpsilonSchedule:
        """The schedule of a Run with `settings`."""
        return cls(
            start=settings.epsilon_start,
            end=settings.epsilon_end,
            decay_steps=settings.epsilon_decay_steps,
        )

    def at(self, step: int) -> float:
        """ε at the Run's Step `step`, counting from 0."""
        done = min(step / self.decay_steps, 1.0)
        return self.start + (self.end - self.start) * done


class Transition(NamedTuple):
    """One Step, as the learner learns from it."""

    observation: Any
    """What the learner chose its Action from."""
    action: int
    reward: float
    next_observation: Any
    """What the environment gave after the Step, even if the Game ended in it."""
    terminated: bool
    """Whether the Game ended in the Step, so that it has no future."""
    truncated: bool
    """Whether the Game was cut off in the Step, though it had a future."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStop:
    """Where a Run's training stopped."""

    step: int
    """The Run's Step it stopped at: its end, unless Ctrl+C stopped it."""
    by_ctrl_c: bool
    """Whether Ctrl+C stopped it, before its end, with a full checkpoint at `step`
    to resume from. Otherwise it is over: at its Budget, or, if it has
    Evaluations, at its first Milestone of a mean of at least its stop_mean on the
    regular set, with every Milestone evaluated, and its Confirmation."""


class RunLearner(Protocol):
    """What the loop needs of a learner: the real one, or a fake in the tests."""

    def choose_action(self, observation: Any, epsilon: float) -> int:
        """The Action for `observation`, by `epsilon`."""
        ...

    def learn(self, transition: Transition) -> float | None:
        """Learn from `transition`, the Step just played, and give the loss of the
        update made, or None if none was."""
        ...

    def q_values(self, observations: Any) -> npt.NDArray[np.float32]:
        """The Q of each Action for each of a batch of inputs, one row for each."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """All of the learner's state, for a full checkpoint."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Take on `state`, as a full checkpoint holds it, to resume a Run."""
        ...

    def network_state_dict(self) -> dict[str, Any]:
        """The network's weights alone, for a Milestone."""
        ...


class MilestoneEvaluation(Protocol):
    """What the loop needs of the Evaluation: the real one, of a learning player's
    Milestones, or a fake in the tests (#17)."""

    def __call__(
        self, milestone: SavedFile, *, seed_set: SeedSet, stop: Callable[[], bool]
    ) -> Evaluation:
        """The Evaluation of `milestone`, the network's weights it holds, on
        `seed_set`. Before each of its Steps, `stop` says whether to abandon it
        there instead, with an EvaluationStopped.

        The loop asks for the Confirmation's set once in a Run, at its end, and
        nothing else ever asks for it (issue #16).
        """
        ...


def train(
    env: gym.Env[Any, Any],
    learner: RunLearner,
    *,
    steps: int,
    epsilon: EpsilonSchedule,
    seed: int,
    start: int = 0,
    after_step: Callable[[int], None] | None = None,
    stop: Callable[[], bool] | None = None,
    graphs: TrainingGraphs | None = None,
) -> int:
    """Play Steps against `env`, from the Run's Step count `start` until it is
    `steps`, a Game after another, the first reset with `seed`, with `learner`
    choosing each Action and learning from each Step.

    After each Step, `graphs`, if given, are told of it, and then `after_step`, if
    given, gets the Run's Step count. Before each Step, `stop`, if given, says
    whether to stop there instead. It returns the Run's Step count it stopped at.
    """
    observation, _ = env.reset(seed=seed)
    for step in range(start, steps):
        if stop is not None and stop():
            return step
        action = learner.choose_action(observation, epsilon.at(step))
        next_observation, reward, terminated, truncated, _ = env.step(action)
        loss = learner.learn(
            Transition(
                observation,
                action,
                float(reward),
                next_observation,
                terminated,
                truncated,
            )
        )
        if graphs is not None:
            graphs.step_played(
                step + 1,
                reward=float(reward),
                game_ended=terminated or truncated,
                loss=loss,
            )
        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation
        if after_step is not None:
            after_step(step + 1)
    return steps


def _q_graph_inputs(env: gym.Env[Any, Any], *, seed: int) -> npt.NDArray[Any]:
    """The fixed inputs of a Run's Q graph: the first Q_GRAPH_INPUTS inputs a
    player chooses from in `env`, playing Actions at random, a Game after another,
    the first reset with `seed`.

    The random Actions come from a generator of their own, drawn from `seed`, so
    the learner's are left as they were.
    """
    assert isinstance(env.action_space, spaces.Discrete)
    actions = int(env.action_space.n)
    rng = np.random.default_rng(seed)
    inputs = []
    observation, _ = env.reset(seed=seed)
    for _ in range(Q_GRAPH_INPUTS):
        inputs.append(observation)
        action = int(rng.integers(actions))
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    return np.stack(inputs)


def train_run(
    run: RunDirectory,
    env: gym.Env[Any, Any],
    learner: RunLearner,
    *,
    stay_awake: Callable[[], AbstractContextManager[object]],
    writer: ScalarWriter,
    evaluation: MilestoneEvaluation | None = None,
) -> RunStop:
    """Play the new Run `run` against `env`, from Step 0 to its Budget, with
    `learner`, by the Run's settings and from its seed, and write its graphs
    through `writer`.

    The Run saves a Milestone at Step 0, a full checkpoint every so many Steps and
    a Milestone every so many, as its settings say, and a Milestone at its end. A
    full checkpoint holds the learner's whole state (its two networks, Adam's
    state and its random generator's), the Run's Step count, and how many times
    the Run was resumed, none in a new Run. A Milestone holds the network's
    weights alone. The learner fills its replay buffer before it learns, and the
    filling's Steps count toward the Budget like any other.

    Before Step 0, the Run saves the fixed inputs of its Q graph, from a Game of
    random Actions against `env`.

    With `evaluation`, the Run evaluates each Milestone as it saves it, on the
    regular set, the training stopped meanwhile. The Evaluation's Steps aren't the
    Run's: they count in none of its Steps, and the learner neither learns from
    them nor chooses their Actions. A Run of a test environment has no Evaluation
    (#17).

    A Run with Evaluations is over at its first Milestone whose mean is at least
    its stop_mean, or at its Budget, whichever comes first (issue #14). Then its
    Milestone of the highest mean, of a tie the earliest, has the Run's one
    Evaluation on the Confirmation's set: its Confirmation, recorded in the Run's
    directory, graphed, and printed, with whether its mean is at least the Run's
    confirmation_gate (issue #13).

    One Ctrl+C stops the Run once its Step is over, with a full checkpoint at that
    Step, unless it has one there already, and no Milestone: the Run isn't over.
    In an Evaluation, it abandons the Evaluation, which isn't recorded, and stops
    the Run with a full checkpoint at the Milestone's Step, even at the Budget, so
    that the Run's resume measures the Milestone. In the Confirmation, it
    abandons it the same way, so that the resume gives it. A second Ctrl+C stops
    it at once, with a KeyboardInterrupt. It says where the Run stopped: at its
    end, or by Ctrl+C.

    All along, it holds the request `stay_awake` makes, that Windows doesn't sleep,
    and releases it however the Run ends.
    """
    with stay_awake(), catch_ctrl_c() as ctrl_c:
        q_graph_inputs = _q_graph_inputs(env, seed=run.seed)
        run.save_q_graph_inputs(q_graph_inputs)
        return _play_run(
            run,
            env,
            learner,
            start=0,
            resumes=0,
            seed=run.seed,
            q_graph_inputs=q_graph_inputs,
            writer=writer,
            ctrl_c=ctrl_c,
            evaluation=evaluation,
        )


class ConfirmedRunError(RuntimeError):
    """A Run that had its Confirmation resumed: it never goes on, so that the
    Confirmation's set never becomes a set to choose by (issues #13, #14)."""


def resume_run(
    run: RunDirectory,
    env: gym.Env[Any, Any],
    learner: RunLearner,
    *,
    stay_awake: Callable[[], AbstractContextManager[object]],
    writer: ScalarWriter,
    evaluation: MilestoneEvaluation | None = None,
) -> RunStop:
    """Resume the Run `run` against `env`, from its latest full checkpoint to its
    Budget, with `learner`, a new one, and write its graphs through `writer`. The
    Run must have a full checkpoint.

    The learner takes on the state the checkpoint holds, and the Run goes on from
    its Step, as the same Run: the Step count, and ε with it, go on from there, and
    the Run counts one more resume, in its checkpoints and its graph. The Game
    that was being played isn't kept: the Run starts a new one, the same on every
    resume from that checkpoint. The replay buffer isn't kept either, so the
    learner's new one fills by its current choices before it learns on, and the
    filling's Steps count toward the Budget, as at the Run's start (#17). The Q
    graph's inputs are the ones the Run saved.

    It saves and evaluates its Milestones, stops as `train_run` does, and holds
    the request `stay_awake` makes, and says where it stopped. A Run stopped after
    its training was over, and before its Confirmation, such as by Ctrl+C in it,
    plays no Step more: it only has its Confirmation.

    A Run that had its Confirmation never goes on: it fails with a
    ConfirmedRunError, with nothing played.
    """
    # Only here: PyTorch takes seconds to import, and the ε schedule doesn't need
    # it.
    import torch

    confirmation = run.confirmation()
    if confirmation is not None:
        raise ConfirmedRunError(
            "The Run had its Confirmation, of the Milestone at Step "
            f"{confirmation.step:,}: a Run never goes on after it."
        )
    with stay_awake(), catch_ctrl_c() as ctrl_c:
        checkpoint = torch.load(run.checkpoints()[-1].path)
        learner.load_state_dict(checkpoint["learner"])
        return _play_run(
            run,
            env,
            learner,
            start=checkpoint["step"],
            resumes=checkpoint["resumes"] + 1,
            seed=_resumed_game_seed(run.seed, step=checkpoint["step"]),
            q_graph_inputs=run.q_graph_inputs(),
            writer=writer,
            ctrl_c=ctrl_c,
            evaluation=evaluation,
        )


def _resumed_game_seed(run_seed: int, *, step: int) -> int:
    """The seed of the first Game of the Run of `run_seed` resumed at its Step
    `step`: drawn from both, as an environment draws the seed of its next Game, so
    that every resume from the same checkpoint plays the same Games, and not the
    Run's first ones."""
    return int(np.random.default_rng([run_seed, step]).integers(2**63))


def _saving(state: dict[str, Any]) -> Callable[[BinaryIO], None]:
    """A function that writes `state` into the file it is given."""
    # Only here: PyTorch takes seconds to import, and the ε schedule doesn't need
    # it.
    import torch

    return lambda file: torch.save(state, file)


def _play_run(
    run: RunDirectory,
    env: gym.Env[Any, Any],
    learner: RunLearner,
    *,
    start: int,
    resumes: int,
    seed: int,
    q_graph_inputs: npt.NDArray[Any],
    writer: ScalarWriter,
    ctrl_c: CtrlC,
    evaluation: MilestoneEvaluation | None,
) -> RunStop:
    """Play the Run `run` against `env` with `learner`, from its Step count `start`
    to its end, resumed `resumes` times so far, the first Game reset with `seed`,
    saving its files and writing its graphs, with `q_graph_inputs` the fixed
    inputs of its Q graph, and evaluating each Milestone with `evaluation`, if
    given, and then confirming the best.

    It stops early, with a full checkpoint, once `ctrl_c` is pressed, and says
    where it stopped.
    """
    settings = run.settings
    epsilon = EpsilonSchedule.of_run(settings)
    graphs = TrainingGraphs(
        writer,
        max_q=lambda: float(learner.q_values(q_graph_inputs).max(axis=1).mean()),
        epsilon=epsilon.at,
    )

    def save_checkpoint(step: int) -> None:
        checkpoint = {
            "learner": learner.state_dict(),
            "step": step,
            "resumes": resumes,
        }
        run.save_checkpoint(step, _saving(checkpoint))

    # The Step of the Run's first Milestone whose mean on the regular set is at
    # least the Run's stop_mean, once one is measured: the Run is over there
    # (issue #14).
    stop_at = _first_at_least(run.evaluations(), settings.stop_mean)

    def measure(milestone: SavedFile) -> None:
        """Evaluate `milestone`, and record its Evaluation, graph it and print its
        summary line."""
        nonlocal stop_at
        if evaluation is None:
            return
        started = time.perf_counter()
        try:
            result = evaluation(
                milestone, seed_set=SeedSet.REGULAR, stop=lambda: ctrl_c.pressed
            )
        except EvaluationStopped:
            # Not recorded, so that the Run's resume measures it (#17).
            print(
                "Ctrl+C abandoned the Evaluation of the Milestone at Step "
                f"{milestone.step:,}: resuming the Run measures it first.",
                flush=True,
            )
            return
        run.record_evaluation(result, step=milestone.step)
        graphs.milestone_evaluated(
            milestone.step, result.summary, seconds=time.perf_counter() - started
        )
        print(
            result.summary_line(seed_set=SeedSet.REGULAR, step=milestone.step),
            flush=True,
        )
        if stop_at is None and result.summary.mean >= settings.stop_mean:
            stop_at = milestone.step

    def save_milestone(step: int) -> None:
        measure(run.save_milestone(step, _saving(learner.network_state_dict())))

    def confirm(evaluation: MilestoneEvaluation) -> bool:
        """Give the Run's best Milestone on the regular set, of a tie the earliest,
        the Run's one Confirmation, with `evaluation`, and record it, graph it and
        print its summary line and whether it passed the gate. It says whether it
        did, or Ctrl+C abandoned it (issue #13)."""
        measured = run.evaluations()
        # max keeps the first of a tie.
        best = max(sorted(measured), key=lambda step: measured[step].summary.mean)
        [milestone] = [saved for saved in run.milestones() if saved.step == best]
        try:
            result = evaluation(
                milestone, seed_set=SeedSet.CONFIRMATION, stop=lambda: ctrl_c.pressed
            )
        except EvaluationStopped:
            # Not recorded, so that the Run's resume gives it.
            print(
                f"Ctrl+C abandoned the Confirmation of the Milestone at Step {best:,}: "
                "resuming the Run gives it its Confirmation.",
                flush=True,
            )
            return False
        run.record_confirmation(result, step=best)
        graphs.milestone_confirmed(best, result.summary)
        print(result.summary_line(seed_set=SeedSet.CONFIRMATION, step=best))
        mean, gate = result.summary.mean, settings.confirmation_gate
        if mean >= gate:
            verdict = f"is at least {gate:g}: the Run reached its goal."
        else:
            # The Confirmation's set would become a set to choose by (issue #14).
            verdict = (
                f"is below {gate:g}: the Run didn't reach its goal. It never goes on, "
                "and no other Milestone gets a Confirmation."
            )
        print(f"The Confirmation's mean, {mean:+.2f}, {verdict}", flush=True)
        return True

    def has_a_milestone(step: int) -> bool:
        return step % settings.milestone_every == 0 or step == settings.budget

    def after_step(step: int) -> None:
        if step % settings.checkpoint_every == 0:
            save_checkpoint(step)
        if has_a_milestone(step):
            save_milestone(step)

    # The Milestone of Step 0, and on a resume the one of the checkpoint's Step,
    # if a stop between the two files left it unsaved: the learner's network is
    # the checkpoint's, as it was at the Step.
    if has_a_milestone(start) and start not in {m.step for m in run.milestones()}:
        run.save_milestone(start, _saving(learner.network_state_dict()))
    # A Milestone saved and not measured, as when Ctrl+C abandoned its Evaluation,
    # is measured before the Run goes on (#17).
    measured = run.evaluations()
    for milestone in run.milestones():
        if milestone.step not in measured:
            measure(milestone)
    # A Milestone at the Budget is saved only at the Run's end, so with one the
    # Run trains no more, as when a crash stopped it before its Confirmation, its
    # latest checkpoint before its end. The same goes once its training stopped at
    # a Milestone of at least stop_mean.
    trained = settings.budget in {saved.step for saved in run.milestones()}
    graphs.begin(start, resumes=resumes)
    stopped_at = train(
        env,
        learner,
        steps=settings.budget,
        epsilon=epsilon,
        seed=seed,
        start=start,
        after_step=after_step,
        stop=lambda: ctrl_c.pressed or trained or stop_at is not None,
        graphs=graphs,
    )
    # Over at its first Milestone of at least stop_mean, or at its Budget with
    # its last Milestone measured too, and then its best Milestone confirmed.
    # Otherwise Ctrl+C stopped it, in a Step or in an Evaluation, the Confirmation
    # included, and it resumes from a full checkpoint at the Step, unless it has
    # one there already, as one of every so many.
    over = stop_at is not None or (
        (trained or stopped_at == settings.budget)
        and (evaluation is None or settings.budget in run.evaluations())
    )
    if over and evaluation is not None:
        over = confirm(evaluation)
    if over:
        return RunStop(
            step=settings.budget if stop_at is None else stop_at, by_ctrl_c=False
        )
    if stopped_at not in {saved.step for saved in run.checkpoints()}:
        save_checkpoint(stopped_at)
    return RunStop(step=stopped_at, by_ctrl_c=True)


def _first_at_least(evaluations: dict[int, Evaluation], mean: float) -> int | None:
    """The Step of the first Milestone of `evaluations` whose mean is at least
    `mean`, or None if none is."""
    return min(
        (step for step, result in evaluations.items() if result.summary.mean >= mean),
        default=None,
    )
