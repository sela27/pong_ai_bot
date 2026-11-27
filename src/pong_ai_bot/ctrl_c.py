"""Ctrl+C while a Run runs (issue #10): the first asks the Run to stop once its Step
is over, with a full checkpoint, and the second stops it at once. Before the Run
catches Ctrl+C, in the seconds of its setup, Ctrl+C stops at once. Once a stop has
begun, or the Run is over, every later press is ignored, to the process's end.

Python calls the handler of SIGINT when Ctrl+C is pressed, in the main thread,
between two of its instructions. The Run's handler only notes the first press, so
the Step it lands in is played to its end; the second raises KeyboardInterrupt, as
Python's own handler does.

A press as the process ends, such as the developer's second, a moment after the
first stopped the Run, would stop the ending itself: Python's own handler with a
traceback, and once Python has removed its handlers, as it ends, Windows's own,
with its exit code for Ctrl+C, 0xC000013A, instead of 130. Only an ignored Ctrl+C
stays ignored then.
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
    """In the `with` block, a Run's, the first Ctrl+C sets the CtrlC given
    `pressed`, and the second raises KeyboardInterrupt. However the block ends,
    Ctrl+C is ignored from then on, to the process's end: the Run is the last thing
    its process does. Only the main thread may enter it, as only it may handle a
    signal."""
    ctrl_c = CtrlC()

    def handle(signal_number: int, frame: FrameType | None) -> None:
        if ctrl_c.pressed:
            raise KeyboardInterrupt
        ctrl_c.pressed = True

    signal.signal(signal.SIGINT, handle)
    try:
        yield ctrl_c
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)


@contextmanager
def stop_at_once_on_ctrl_c() -> Iterator[None]:
    """In the `with` block, Ctrl+C raises KeyboardInterrupt at once, as Python's own
    handler does, but only the first time: from then on Ctrl+C is ignored, after
    the block too, to the process's end, so that a second press can't stop the
    stop.

    catch_ctrl_c may be entered in the block, as a Run is, and has Ctrl+C from
    then on. Otherwise, if the block ends with no Ctrl+C, Ctrl+C is as it was
    before it. Only the main thread may enter it, as only it may handle a signal.
    """

    def handle(signal_number: int, frame: FrameType | None) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGINT, handle)
    try:
        yield
    finally:
        if signal.getsignal(signal.SIGINT) is handle:
            signal.signal(signal.SIGINT, previous)
