"""The Position learner: the learning player of Pong that decides from the positions
of the ball and the Paddles, as `pong train --player position-learner` trains it
(issue #9)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from pong_ai_bot.cli import main
from pong_ai_bot.environment import Action
from pong_ai_bot.evaluation import Evaluation, GameResult, SeedSet
from pong_ai_bot.game import Score
from pong_ai_bot.learning_players import LEARNING_PLAYERS, milestone_evaluation
from pong_ai_bot.replay_buffer import ReplayBuffer
from pong_ai_bot.run_directory import RunDirectory
from pong_ai_bot.transition import Transition


def test_it_plays_pong_from_the_stack_of_the_last_four_steps_positions() -> None:
    player = LEARNING_PLAYERS["position-learner"]
    env = player.make_env()

    stack, info = env.reset(seed=0)

    # A new Game: the ball waits in the center of the Court, and both Paddles are
    # there too, in each of the stack's four Steps, copies of the Game's first.
    assert (player.stack_length, stack.shape, stack.dtype) == (4, (4, 4), np.float32)
    assert stack.tolist() == [[0.5, 0.5, 0.5, 0.5]] * 4
    # Pong's three Actions, as the Agent's.
    assert env.action_space == spaces.Discrete(3)
    assert player.actions == 3
    assert info["score"] == Score(left=0, right=0)


def test_its_network_gives_a_q_for_each_action_from_a_batch_of_its_stacks() -> None:
    # Torch only here, for its seconds of import; no optimizer, which takes more.
    import torch

    player = LEARNING_PLAYERS["position-learner"]
    network = player.make_network()

    q = network(torch.zeros(5, 4, 4))

    assert q.shape == (5, 3)
    # Its 16 inputs, two hidden layers of 256, and an output of 3, each layer with
    # its biases: 16 * 256 + 256, 256 * 256 + 256, and 256 * 3 + 3.
    parameters = sum(parameter.numel() for parameter in network.parameters())
    assert parameters == 4_352 + 65_792 + 771 == 70_915


def test_the_replay_buffer_gives_its_positions_back_as_the_same_numbers() -> None:
    # Kept as they are, 16 bytes a Step, with no compression as the Agent's images
    # will have (#17). The Paddle goes up to the top, and the ball is served in
    # the 8th Step and moves from then on, so the numbers change.
    player = LEARNING_PLAYERS["position-learner"]
    env = player.make_env()
    buffer = ReplayBuffer(size=1, stack_length=player.stack_length)
    rng = np.random.default_rng(0)

    observation, _ = env.reset(seed=0)
    differences = []
    for step in range(1, 41):
        next_observation, reward, terminated, truncated, _ = env.step(Action.UP)
        latest = Transition(
            observation,
            Action.UP,
            float(reward),
            next_observation,
            terminated,
            truncated,
        )
        buffer.add(latest)
        [kept] = buffer.sample(1, rng)
        same = all(map(np.array_equal, kept, latest, strict=True))
        if not same or kept.observation.dtype != np.float32:
            differences.append(step)
        observation = next_observation

    assert differences == [], f"the Steps given back otherwise are {differences}"


@pytest.mark.slow
def test_pong_train_runs_it_with_the_real_evaluation_of_each_milestone(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Slow for the learner's optimizer, which takes about 5 seconds to make, and
    # for the Evaluations' 100 Games of Pong, at Step 0 and at the Budget. The
    # Run's Confirmation is a fake that plays no Game, so that no test ever plays
    # the Confirmation's set (issue #16).

    def confirmation_faked(network: Any, make_env: Any) -> Any:
        evaluate_milestone = milestone_evaluation(network, make_env)

        def evaluation(milestone: Any, *, seed_set: SeedSet, stop: Any) -> Evaluation:
            if seed_set is SeedSet.CONFIRMATION:
                return Evaluation(
                    games=(GameResult(seed=0, score_margin=0, cut_off=False),)
                )
            return evaluate_milestone(milestone, seed_set=seed_set, stop=stop)

        return evaluation

    exit_code = main(
        [
            "train",
            "--player",
            "position-learner",
            "--name",
            "tiny",
            "--seed",
            "1",
            "--device",
            "cpu",
            "--budget",
            "40",
            "--fill-steps",
            "32",
            "--replay-buffer-size",
            "100",
            "--checkpoint-every",
            "20",
            "--milestone-every",
            "40",
        ],
        make_milestone_evaluation=confirmation_faked,
    )

    assert exit_code == 0
    [path] = runs.iterdir()
    run = RunDirectory.open(path)
    assert run.player == "position-learner"
    # The Q graph's inputs are the stacks the player decides from.
    inputs = run.q_graph_inputs()
    assert (inputs.shape, inputs.dtype) == ((1_000, 4, 4), np.float32)
    # Each Milestone evaluated on the regular set's 100 Games of Pong.
    evaluations = run.evaluations()
    assert list(evaluations) == [0, 40]
    for evaluation in evaluations.values():
        seeds = [game.seed for game in evaluation.games]
        assert seeds == list(SeedSet.REGULAR.seeds)
    summaries = [
        line.split(":")[0]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("Evaluation on the regular set")
    ]
    assert summaries == [
        "Evaluation on the regular set at Step 0",
        "Evaluation on the regular set at Step 40",
    ]
    assert run.confirmation() is not None
