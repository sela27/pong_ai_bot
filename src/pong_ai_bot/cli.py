"""The `pong` command: one entry point, with a subcommand for each mode."""

import argparse
import secrets
from pathlib import Path

from pong_ai_bot.environment import PongEnv
from pong_ai_bot.evaluation import SeedSet, evaluate
from pong_ai_bot.scripted_players import SCRIPTED_PLAYERS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pong",
        description="A locally built Pong game and a DQN Agent that learns to play it.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser(
        "play",
        help="play as the Human against the Opponent, in a window",
        description=(
            "Play as the Human, on the right, against the Opponent. The up and down "
            "arrows move your Paddle, Space pauses and resumes, and Esc quits."
        ),
    )
    commands.add_parser(
        "measure-env",
        help="measure how many Steps a second the environment plays",
        description=(
            "Time one environment playing random Actions against the Opponent, "
            "drawing the Observation every Step, and print its Steps a second. "
            "Fails below the gate."
        ),
    )
    evaluate_command = commands.add_parser(
        "evaluate",
        help="evaluate a player in the Agent's place, on the regular set of seeds",
        description=(
            "Play an Evaluation of a player in the Agent's place: one Game against "
            "the Opponent for each of the regular set's 100 seeds. Print its "
            "summary line."
        ),
    )
    # Exactly one player to evaluate, of any kind. So far the only kind is a
    # Scripted player.
    player = evaluate_command.add_mutually_exclusive_group(required=True)
    player.add_argument(
        "--scripted",
        choices=list(SCRIPTED_PLAYERS),
        help="the Scripted player to evaluate",
    )
    evaluate_command.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help=(
            "also add the Evaluation's line, as a Run's evaluations.jsonl keeps it, "
            "to the end of FILE"
        ),
    )
    args = parser.parse_args(argv)

    if args.command == "play":
        # Only the window needs pygame, so only the window's commands import it.
        from pong_ai_bot.window import play_against_the_opponent

        play_against_the_opponent(seed=secrets.randbits(32))
    elif args.command == "measure-env":
        return _measure_env()
    elif args.command == "evaluate":
        _evaluate(args.scripted, output=args.output)
    return 0


def _measure_env() -> int:
    from pong_ai_bot.env_speed import (
        MEASURED_STEPS,
        STEPS_PER_SECOND_GATE,
        measure_env_speed,
    )

    speed = measure_env_speed(PongEnv(), steps=MEASURED_STEPS, seed=0)
    print(
        f"{speed.steps:,} Steps in {speed.seconds:.2f} seconds: "
        f"{speed.steps_per_second:,.0f} Steps a second."
    )
    if speed.passes_the_gate:
        print(f"It passes the gate of {STEPS_PER_SECOND_GATE:,}.")
        return 0
    print(
        f"It is below the gate of {STEPS_PER_SECOND_GATE:,}. Profile it and speed up "
        "only the code, not the rules, the Observation or the sizes (issue #14)."
    )
    return 1


def _evaluate(scripted: str, *, output: Path | None) -> None:
    # Never the Confirmation's set: it is played only at the end of a Run (issue #16).
    seed_set = SeedSet.REGULAR
    evaluation = evaluate(SCRIPTED_PLAYERS[scripted](), PongEnv, seed_set.seeds)
    print(evaluation.summary_line(seed_set=seed_set))
    if output is not None:
        with output.open("a", encoding="utf-8") as file:
            file.write(evaluation.json_line(seed_set=seed_set) + "\n")
