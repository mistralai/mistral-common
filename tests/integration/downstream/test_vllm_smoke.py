r"""Smoke test that the `ours-vllm` lane imports vLLM.

Run with `python3 scripts/run_downstream_tests.py --only ours-vllm`, which installs
the latest vLLM +cpu wheel next to this checkout's mistral-common.
"""

import pytest

pytestmark = [pytest.mark.downstream, pytest.mark.vllm]

pytest.importorskip("vllm")


def test_vllm_runtime_available() -> None:
    from vllm.multimodal.image import normalize_image

    assert callable(normalize_image)
