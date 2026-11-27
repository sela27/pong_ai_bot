"""The training loop: Steps against any Gymnasium environment (issue #10).

Each Step, the learner chooses an Action, with ε by the Step counter alone, and
learns from the Step's transition. The loop never reads the `info`. It knows
nothing of Pong, and of files only through the Run directory: a Run's loop saves
its full checkpoints and its Milestones there, and stops at its Budget, or when
Ctrl+C is pressed.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, NamedTuple, Protocol

import gymnasium as gym

from pong_ai_bot.ctrl_c import catch_ctrl_c
from pong_ai_bot.run_settings import RunSettings

if TYPE_CHECKING:
    from pong_ai_bot.run_directory import RunDirectory


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

    def learn(self, transition: Transition) -> None:
        """Learn from `transition`, the Step just played."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """All of the learner's state, for a full checkpoint."""
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
    after_step: Callable[[int], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> int:
    """Play `steps` Steps against `env`, a Game after another, the first reset with
    `seed`, with `learner` choosing each Action and learning from each Step.

    After each Step, `after_step`, if given, gets how many Steps have been played.
    Before each Step, `stop`, if given, says whether to stop there instead. It
    returns how many Steps were played.
    """
    observation, _ = env.reset(seed=seed)
    for step in range(steps):
        if stop is not None and stop():
            return step
        action = learner.choose_action(observation, epsilon.at(step))
        next_observation, reward, terminated, truncated, _ = env.step(action)
        learner.learn(
            Transition(
                observation,
                action,
                float(reward),
                next_observation,
                terminated,
                truncated,
            )
        )
        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation
        if after_step is not None:
            after_step(step + 1)
    return steps


def train_run(
    run: RunDirectory,
    env: gym.Env[Any, Any],
    learner: RunLearner,
    *,
    stay_awake: Callable[[], AbstractContextManager[object]],
) -> int:
    """Play the new Run `run` against `env`, from Step 0 to its Budget, with
    `learner`, by the Run's settings and from its seed.

    The Run saves a Milestone at Step 0, a full checkpoint every so many Steps and
    a Milestone every so many, as its settings say, and a Milestone at its end. A
    full checkpoint holds the learner's whole state (its two networks, Adam's
    state and its random generator's), the Run's Step count, and how many times
    the Run was resumed, none in a new Run. A Milestone holds the network's
    weights alone. The learner fills its replay buffer before it learns, and the
    filling's Steps count toward the Budget like any other.

    One Ctrl+C stops the Run once its Step is over, with a full checkpoint at that
    Step, unless it has one there already, and no Milestone: the Run isn't over. A
    second Ctrl+C stops it at once, with a KeyboardInterrupt. It returns the Step
    the Run stopped at: its Budget, or the Step Ctrl+C was pressed in.

    All along, it holds the request `stay_awake` makes, that Windows doesn't sleep,
    and releases it however the Run ends.
    """
    # Only here: PyTorch takes seconds to import, and the ε schedule doesn't need
    # it.
    import torch

    settings = run.settings

    def saving(state: dict[str, Any]) -> Callable[[BinaryIO], None]:
        return lambda file: torch.save(state, file)

    def save_checkpoint(step: int) -> None:
        checkpoint = {"learner": learner.state_dict(), "step": step, "resumes": 0}
        run.save_checkpoint(step, saving(checkpoint))

    def after_step(step: int) -> None:
        if step % settings.checkpoint_every == 0:
            save_checkpoint(step)
        if step % settings.milestone_every == 0 or step == settings.budget:
            run.save_milestone(step, saving(learner.network_state_dict()))

    with stay_awake(), catch_ctrl_c() as ctrl_c:
        run.save_milestone(0, saving(learner.network_state_dict()))
        stopped_at = train(
            env,
            learner,
            steps=settings.budget,
            epsilon=EpsilonSchedule.of_run(settings),
            seed=run.seed,
            after_step=after_step,
            stop=lambda: ctrl_c.pressed,
        )
        # Unless the Step has its checkpoint already, as one of every so many.
        checkpointed = {saved.step for saved in run.checkpoints()}
        if stopped_at < settings.budget and stopped_at not in checkpointed:
            save_checkpoint(stopped_at)
    return stopped_at
