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
    args = parser.parse_args(argv)

    if args.command == "play":
        # Only the window needs pygame, so only the window's commands import it.
        from pong_ai_bot.window import play_against_the_opponent

        play_against_the_opponent(seed=secrets.randbits(32))
    return 0
