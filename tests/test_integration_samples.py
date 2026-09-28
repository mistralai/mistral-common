"""Preserve the historical skip for the unresolved v2 parallel-calls sample."""

import pytest


@pytest.mark.parametrize(
    ("sample_name", "versions"),
    [("parallel_calls", (3,))],
    ids=["parallel_calls"],
)
@pytest.mark.parametrize("version", [2], ids=["v2"])
def test_samples(sample_name: str, versions: tuple[int, ...], version: int) -> None:
    if version not in versions:
        pytest.skip(f"Sample {sample_name} not available for version {version}")
