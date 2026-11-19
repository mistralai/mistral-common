r"""JSON helpers shared by tests."""

import json


def is_unloadable_json(payload: str) -> bool:
    r"""Whether `json.loads` rejects `payload` with `ValueError` or `RecursionError`.

    On Python 3.14+ C recursion is guarded by remaining stack space instead of the recursion limit, so deeply
    nested JSON may parse instead of raising `RecursionError`, depending on the thread stack size.
    """
    try:
        json.loads(payload)
    except (ValueError, RecursionError):
        return True
    return False
