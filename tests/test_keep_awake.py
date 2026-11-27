"""The request that Windows doesn't sleep while a Run runs, with a fake of Windows'
own call, SetThreadExecutionState, so no test changes how the machine sleeps."""

import pytest

from pong_ai_bot.keep_awake import keep_windows_awake

AWAKE = 0x80000001
"""The flags that ask Windows not to sleep until told otherwise: ES_CONTINUOUS and
ES_SYSTEM_REQUIRED, as Microsoft's documentation numbers them."""

RELEASED = 0x80000000
"""The flags that release the request: ES_CONTINUOUS alone."""


class ExecutionState:
    """A fake of SetThreadExecutionState, which records the flags it is given."""

    def __init__(self) -> None:
        self.given: list[int] = []

    def __call__(self, flags: int) -> int:
        self.given.append(flags)
        # The state before, as Windows returns it.
        return RELEASED


def test_windows_is_asked_not_to_sleep_until_the_request_is_released() -> None:
    state = ExecutionState()

    with keep_windows_awake(state):
        assert state.given == [AWAKE]

    assert state.given == [AWAKE, RELEASED]


def test_the_request_is_released_when_what_it_kept_awake_fails() -> None:
    state = ExecutionState()

    with pytest.raises(KeyboardInterrupt), keep_windows_awake(state):
        raise KeyboardInterrupt

    assert state.given == [AWAKE, RELEASED]
