"""Ctrl+C while a Run runs (issue #10): the first asks the Run to stop once its Step
is over, with a full checkpoint, and the second stops it at once.

Python calls the handler of SIGINT when Ctrl+C is pressed, in the main thread,
between two of its instructions. The handler here only notes the first press, so
the Step it lands in is played to its end; the second raises KeyboardInterrupt, as
Python's own handler does.
"""

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


class CtrlC:
    """Whether Ctrl+C has been pressed, since the `with` block of catch_ctrl_c
    began."""

    def __init__(self) -> None:
        self.pressed = False


@contextmanager
def catch_ctrl_c() -> Iterator[CtrlC]:
    """In the `with` block, the first Ctrl+C sets the CtrlC given `pressed`, and the
    second raises KeyboardInterrupt. Ctrl+C is Python's again however the block
    ends. Only the main thread may enter it, as only it may handle a signal."""
    ctrl_c = CtrlC()

    def handle(signal_number: int, frame: FrameType | None) -> None:
        if ctrl_c.pressed:
            raise KeyboardInterrupt
        ctrl_c.pressed = True

    previous = signal.signal(signal.SIGINT, handle)
    try:
        yield ctrl_c
    finally:
        signal.signal(signal.SIGINT, previous)
