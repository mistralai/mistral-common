r"""Smoke test proving the downstream vLLM lane works end to end.

Runs only via scripts/run_downstream_tests.sh --only vllm (marker: downstream + vllm),
which installs the latest released vLLM +cpu wheel into a uv overlay.
"""

import pytest

pytestmark = [pytest.mark.downstream, pytest.mark.vllm]

vllm = pytest.importorskip("vllm")


def test_vllm_runtime_available() -> None:
    from vllm.multimodal.image import normalize_image

    assert callable(normalize_image)
    print(f"vllm version under test: {vllm.__version__}")
