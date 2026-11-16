r"""Smoke test that the `ours-transformers` lane imports Transformers.

Run with `python3 scripts/run_downstream_tests.py --only ours-transformers`, which installs
the latest Transformers release next to this checkout's mistral-common.
"""

import pytest

pytestmark = [pytest.mark.downstream, pytest.mark.transformers]

transformers = pytest.importorskip("transformers")


def test_transformers_runtime_available() -> None:
    from transformers.image_utils import load_image

    assert callable(load_image)
    print(f"transformers version under test: {transformers.__version__}")
