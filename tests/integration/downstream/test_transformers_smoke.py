r"""Smoke test proving the downstream Transformers lane works end to end.

Runs only via scripts/run_downstream_tests.sh --only transformers (marker:
downstream + transformers), which installs the latest Transformers from PyPI
into a uv overlay.
"""

import pytest

pytestmark = [pytest.mark.downstream, pytest.mark.transformers]

transformers = pytest.importorskip("transformers")


def test_transformers_runtime_available() -> None:
    from transformers.image_utils import load_image

    assert callable(load_image)
    print(f"transformers version under test: {transformers.__version__}")
