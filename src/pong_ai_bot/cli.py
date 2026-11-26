"""The `pong` command: one entry point, with a subcommand for each mode."""

import argparse
import secrets


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
    args = parser.parse_args(argv)

    if args.command == "play":
        # Only the window needs pygame, so only the window's commands import it.
        from pong_ai_bot.window import play_against_the_opponent

        play_against_the_opponent(seed=secrets.randbits(32))
    elif args.command == "measure-env":
        return _measure_env()
    return 0


def _measure_env() -> int:
    from pong_ai_bot.env_speed import (
        MEASURED_STEPS,
        STEPS_PER_SECOND_GATE,
        measure_env_speed,
    )
    from pong_ai_bot.environment import PongEnv

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
