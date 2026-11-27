"""The Scripted players' Evaluations on the regular set, against the Opponent at its
locked offset width (#31).

They pin the balance the width was chosen by, so a change to the physics that
breaks it shows up before a Run: the same two conditions a width passes by
(issue #62). Each plays its 100 Games through the real Evaluation, so both are
slow.
"""

import pytest

from pong_ai_bot.scripted_players import evaluate_scripted_player


@pytest.mark.slow
def test_the_edge_predictor_wins_by_at_least_20_at_the_locked_width() -> None:
    summary = evaluate_scripted_player("edge-predictor").summary

    assert summary.mean >= 20, (
        f"the edge predictor's mean Score margin is {summary.mean:+.2f}"
    )


@pytest.mark.slow
def test_the_random_player_loses_by_more_than_19_at_the_locked_width() -> None:
    summary = evaluate_scripted_player("random").summary

    assert summary.mean < -19, (
        f"the random player's mean Score margin is {summary.mean:+.2f}"
    )
