"""The request that Windows doesn't sleep while a Run runs, so a Run left for the
night isn't stopped (#17).

It asks with SetThreadExecutionState: the machine stays awake, though the screen
may still turn off, until the request is released or the process ends.
"""

import ctypes
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_ES_CONTINUOUS = 0x80000000
"""The request holds until it is changed, rather than only resetting the idle
timer once."""
_ES_SYSTEM_REQUIRED = 0x00000001
"""The machine must stay awake."""


def _set_thread_execution_state(flags: int) -> int:
    """Windows' own SetThreadExecutionState."""
    function = ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint32]
    function.restype = ctypes.c_uint32
    return int(function(flags))


@contextmanager
def keep_windows_awake(
    set_state: Callable[[int], int] = _set_thread_execution_state,
) -> Iterator[None]:
    """Ask Windows not to sleep while in the `with` block, and release the request
    however the block ends. `set_state` is Windows' SetThreadExecutionState, or a
    fake of it in the tests."""
    set_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        set_state(_ES_CONTINUOUS)
