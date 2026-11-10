from collections.abc import Generator

import pytest

import mistral_common.deprecation


@pytest.fixture(autouse=True)
def _clear_warned_keys() -> Generator[None, None, None]:
    # Deprecation warnings are emitted once per process; reset so each test sees them.
    mistral_common.deprecation._warned_keys.clear()
    try:
        yield
    finally:
        mistral_common.deprecation._warned_keys.clear()
