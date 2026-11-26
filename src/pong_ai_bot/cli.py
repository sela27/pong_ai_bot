"""The `pong` command: one entry point, with a subcommand for each mode."""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pong",
        description="A locally built Pong game and a DQN Agent that learns to play it.",
    )
    parser.parse_args(argv)
    return 0
