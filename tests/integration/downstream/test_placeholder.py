r"""Sanity check that the downstream suite is wired up (marker: downstream)."""

import pytest

pytestmark = pytest.mark.downstream


def test_downstream_dir_is_wired_up() -> None:
    assert True
