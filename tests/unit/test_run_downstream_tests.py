r"""Unit tests for scripts/run_downstream_tests.py (release lookup and wheel picking)."""

import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "run_downstream_tests.py"
spec = importlib.util.spec_from_file_location("run_downstream_tests", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
rdt = importlib.util.module_from_spec(spec)
sys.modules["run_downstream_tests"] = rdt
spec.loader.exec_module(rdt)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_resolve_latest_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: int = 0) -> _FakeResponse:
        assert url == "https://pypi.org/pypi/vllm/json"
        return _FakeResponse(json.dumps({"info": {"version": "0.33.0"}}).encode())

    monkeypatch.setattr(rdt.urllib.request, "urlopen", fake_urlopen)
    assert rdt.resolve_latest_version("vllm") == "0.33.0"


def test_resolve_latest_version_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: int = 0) -> _FakeResponse:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(rdt.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(rdt.SetupError, match="cannot resolve latest version"):
        rdt.resolve_latest_version("vllm")


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "manylinux_2_39_x86_64"),
        ("Linux", "aarch64", "manylinux_2_39_aarch64"),
        ("Linux", "arm64", "manylinux_2_39_aarch64"),
        ("Darwin", "arm64", "macosx_11_0_arm64"),
        ("Darwin", "x86_64", None),
        ("Windows", "AMD64", None),
    ],
)
def test_vllm_wheel_platform_tag(system: str, machine: str, expected: str | None) -> None:
    assert rdt.vllm_wheel_platform_tag(system, machine) == expected


INDEX_PAGE = """
<html><body>
<a href="../../../abc123/vllm-0.33.0%2Bcpu-cp312-cp312-manylinux_2_39_x86_64.whl#sha256=aaa">x86</a>
<a href="../../../abc123/vllm-0.33.0%2Bcpu-cp312-cp312-manylinux_2_39_aarch64.whl#sha256=bbb">arm</a>
</body></html>
"""


def test_resolve_vllm_cpu_wheel_picks_matching_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: int = 0) -> _FakeResponse:
        assert url == "https://wheels.vllm.ai/0.33.0/cpu/vllm/"
        return _FakeResponse(INDEX_PAGE.encode())

    monkeypatch.setattr(rdt.urllib.request, "urlopen", fake_urlopen)
    url = rdt.resolve_vllm_cpu_wheel("0.33.0", "manylinux_2_39_x86_64")
    # urljoin normalizes the ../../../ relative href against the index
    assert url == ("https://wheels.vllm.ai/abc123/vllm-0.33.0%2Bcpu-cp312-cp312-manylinux_2_39_x86_64.whl#sha256=aaa")


def test_resolve_vllm_cpu_wheel_missing_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: int = 0) -> _FakeResponse:
        return _FakeResponse(INDEX_PAGE.encode())

    monkeypatch.setattr(rdt.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(rdt.SetupError, match="no \\+cpu vLLM wheel for macosx_11_0_arm64"):
        rdt.resolve_vllm_cpu_wheel("0.33.0", "macosx_11_0_arm64")


def test_resolve_vllm_cpu_wheel_index_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: int = 0) -> _FakeResponse:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(rdt.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(rdt.SetupError, match="cannot list"):
        rdt.resolve_vllm_cpu_wheel("0.33.0", "manylinux_2_39_x86_64")


def test_main_usage_error_on_unknown_lane() -> None:
    # argparse rejects unknown --only choices with exit code 2
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--only", "bogus"],
        capture_output=True,
    )
    assert proc.returncode == rdt.EXIT_USAGE == 2
