"""How many Steps a second the environment plays, for `pong measure-env` (issue #14).

One environment, with the Opponent on the left and the Observation drawn every Step,
plays random Actions, and starts the next Game whenever one ends. The Actions are
drawn before the clock starts, so only the environment is timed.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pong_ai_bot.environment import Action, PongEnv

STEPS_PER_SECOND_GATE = 5_000
"""The fewest Steps a second the environment may play, a hard gate (issue #14)."""

MEASURED_STEPS = 100_000
"""How many Steps `pong measure-env` times: 20 seconds at the gate's speed."""


@dataclass(frozen=True, slots=True)
class EnvSpeed:
    """How many Steps the environment played, and in how many seconds."""

    steps: int
    seconds: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds

    @property
    def passes_the_gate(self) -> bool:
        return self.steps_per_second >= STEPS_PER_SECOND_GATE


def measure_env_speed(
    env: PongEnv,
    *,
    steps: int,
    seed: int,
    clock: Callable[[], float] = time.perf_counter,
) -> EnvSpeed:
    """Time `env` playing `steps` random Actions, from the Game of `seed`."""
    actions = np.random.default_rng(seed).integers(len(Action), size=steps).tolist()
    env.reset(seed=seed)
    started = clock()
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    return EnvSpeed(steps=steps, seconds=clock() - started)
