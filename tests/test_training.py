"""The training loop's ε schedule: it depends on the Run's Step counter alone
(issue #12)."""

import pytest

from pong_ai_bot.run_settings import RunSettings
from pong_ai_bot.training import EpsilonSchedule

RUN_EPSILON = EpsilonSchedule(start=1.0, end=0.01, decay_steps=1_000_000)
"""The schedule of a Run with the default settings."""


def test_epsilon_falls_in_a_straight_line_from_1_to_0_01_over_a_million_steps() -> None:
    values = [RUN_EPSILON.at(step) for step in (0, 50_000, 500_000, 1_000_000)]

    assert values == pytest.approx([1.0, 0.9505, 0.505, 0.01])


def test_epsilon_stays_at_0_01_after_the_first_million_steps() -> None:
    values = [RUN_EPSILON.at(step) for step in (1_000_001, 10_000_000)]

    assert values == pytest.approx([0.01, 0.01])


def test_a_run_s_epsilon_schedule_is_its_settings() -> None:
    settings = RunSettings(epsilon_start=0.5, epsilon_end=0.1, epsilon_decay_steps=100)

    schedule = EpsilonSchedule.of_run(settings)

    # Halfway down from 0.5 to 0.1 at Step 50, and 0.1 from Step 100 on.
    values = [schedule.at(step) for step in (0, 50, 100, 200)]
    assert values == pytest.approx([0.5, 0.3, 0.1, 0.1])
