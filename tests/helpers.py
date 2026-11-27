"""What several test modules share."""

import signal

from pong_ai_bot.run_directory import Commit

A_COMMIT = Commit(
    sha="0123456789abcdef0123456789abcdef01234567", uncommitted_changes=False
)
"""A commit that is never the code's own, for the tests' Runs to be created at,
rather than asking git for it."""


def press_ctrl_c() -> None:
    """Press Ctrl+C, as Python sees it, without a real signal: it calls the handler
    of SIGINT."""
    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler)
    handler(signal.SIGINT, None)
