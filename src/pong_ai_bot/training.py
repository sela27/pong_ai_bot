"""The training loop: Steps against any Gymnasium environment (issue #10).

Each Step, the learner chooses an Action, with ε by the Step counter alone, and
learns from the Step's transition. The loop never reads the `info`. It knows
nothing of Pong, and of files only through the Run directory: a Run's loop saves
its full checkpoints and its Milestones there, and stops at its Budget, or when
Ctrl+C is pressed. A Run that stopped resumes from its latest full checkpoint, as
the same Run. The loop writes the Run's graphs too, through the writer it is given.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from pong_ai_bot.ctrl_c import CtrlC, catch_ctrl_c
from pong_ai_bot.graphs import ScalarWriter, TrainingGraphs
from pong_ai_bot.run_settings import RunSettings

if TYPE_CHECKING:
    from pong_ai_bot.run_directory import RunDirectory

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
) -> int:
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

    One Ctrl+C stops the Run once its Step is over, with a full checkpoint at that
    Step, unless it has one there already, and no Milestone: the Run isn't over. A
    second Ctrl+C stops it at once, with a KeyboardInterrupt. It returns the Step
    the Run stopped at: its Budget, or the Step Ctrl+C was pressed in.

    All along, it holds the request `stay_awake` makes, that Windows doesn't sleep,
    and releases it however the Run ends.
    """
    with stay_awake(), catch_ctrl_c() as ctrl_c:
        q_graph_inputs = _q_graph_inputs(env, seed=run.seed)
        run.save_q_graph_inputs(q_graph_inputs)
        run.save_milestone(0, _saving(learner.network_state_dict()))
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
        )


def resume_run(
    run: RunDirectory,
    env: gym.Env[Any, Any],
    learner: RunLearner,
    *,
    stay_awake: Callable[[], AbstractContextManager[object]],
    writer: ScalarWriter,
) -> int:
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

    It saves its files, and stops at its Budget or by Ctrl+C, holding the request
    `stay_awake` makes, as `train_run` does, and returns the Step it stopped at.
    """
    # Only here: PyTorch takes seconds to import, and the ε schedule doesn't need
    # it.
    import torch

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
) -> int:
    """Play the Run `run` against `env` with `learner`, from its Step count `start`
    to its Budget, resumed `resumes` times so far, the first Game reset with
    `seed`, saving its files and writing its graphs, with `q_graph_inputs` the
    fixed inputs of its Q graph.

    It stops early, with a full checkpoint, once `ctrl_c` is pressed, and returns
    the Step it stopped at.
    """
    settings = run.settings
    epsilon = EpsilonSchedule.of_run(settings)

    def save_checkpoint(step: int) -> None:
        checkpoint = {
            "learner": learner.state_dict(),
            "step": step,
            "resumes": resumes,
        }
        run.save_checkpoint(step, _saving(checkpoint))

    def after_step(step: int) -> None:
        if step % settings.checkpoint_every == 0:
            save_checkpoint(step)
        if step % settings.milestone_every == 0 or step == settings.budget:
            run.save_milestone(step, _saving(learner.network_state_dict()))

    graphs = TrainingGraphs(
        writer,
        max_q=lambda: float(learner.q_values(q_graph_inputs).max(axis=1).mean()),
        epsilon=epsilon.at,
    )
    graphs.begin(start, resumes=resumes)
    stopped_at = train(
        env,
        learner,
        steps=settings.budget,
        epsilon=epsilon,
        seed=seed,
        start=start,
        after_step=after_step,
        stop=lambda: ctrl_c.pressed,
        graphs=graphs,
    )
    # Unless the Step has its checkpoint already, as one of every so many.
    checkpointed = {saved.step for saved in run.checkpoints()}
    if stopped_at < settings.budget and stopped_at not in checkpointed:
        save_checkpoint(stopped_at)
    return stopped_at
