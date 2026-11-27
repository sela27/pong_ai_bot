"""The training loop's ε schedule: it depends on the Run's Step counter alone
(issue #12)."""

import pytest

from pong_ai_bot.training import EpsilonSchedule


def test_epsilon_falls_in_a_straight_line_from_1_to_0_01_over_a_million_steps() -> None:
    epsilon = EpsilonSchedule()

    values = [epsilon.at(step) for step in (0, 50_000, 500_000, 1_000_000)]

    assert values == pytest.approx([1.0, 0.9505, 0.505, 0.01])


def test_epsilon_stays_at_0_01_after_the_first_million_steps() -> None:
    epsilon = EpsilonSchedule()

    values = [epsilon.at(step) for step in (1_000_001, 10_000_000)]

    assert values == pytest.approx([0.01, 0.01])
