import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_downstream_tests.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(name="run_downstream_tests", location=SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec=spec)
    spec.loader.exec_module(module)
    return module


rdt = _load_script()


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, *args: object) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes, expected_url: str | None = None) -> None:
    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        if expected_url is not None:
            assert url == expected_url
        return _FakeResponse(payload=payload)

    monkeypatch.setattr(rdt.urllib.request, "urlopen", fake_urlopen)


def _unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        raise urllib.error.URLError(reason="unreachable")

    monkeypatch.setattr(rdt.urllib.request, "urlopen", fake_urlopen)


def test_resolve_latest_version_reads_pypi(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(
        monkeypatch=monkeypatch,
        payload=json.dumps({"info": {"version": "0.33.0"}}).encode(),
        expected_url="https://pypi.org/pypi/vllm/json",
    )
    assert rdt.resolve_latest_version(package="vllm") == "0.33.0"


@pytest.mark.parametrize(
    "payload",
    [b"not json", json.dumps({"releases": {}}).encode(), json.dumps({"info": {"version": None}}).encode()],
    ids=["invalid-json", "missing-info", "non-string-version"],
)
def test_resolve_latest_version_rejects_bad_reply(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    _serve(monkeypatch=monkeypatch, payload=payload)
    with pytest.raises(rdt.SetupError):
        rdt.resolve_latest_version(package="vllm")


def test_resolve_latest_version_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _unreachable(monkeypatch=monkeypatch)
    with pytest.raises(rdt.SetupError, match="cannot resolve the latest vllm release"):
        rdt.resolve_latest_version(package="vllm")


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "manylinux_2_34_x86_64"),
        ("Linux", "aarch64", "manylinux_2_39_aarch64"),
        ("Linux", "arm64", "manylinux_2_39_aarch64"),
        ("Darwin", "arm64", "macosx_11_0_arm64"),
    ],
)
def test_vllm_wheel_platform_pattern_matches_any_tag_version(system: str, machine: str, expected: str) -> None:
    pattern = rdt.vllm_wheel_platform_pattern(system=system, machine=machine)
    assert pattern is not None
    assert rdt.re.fullmatch(pattern=pattern, string=expected)


@pytest.mark.parametrize(("system", "machine"), [("Darwin", "x86_64"), ("Windows", "AMD64")])
def test_vllm_wheel_platform_pattern_unsupported_host(system: str, machine: str) -> None:
    assert rdt.vllm_wheel_platform_pattern(system=system, machine=machine) is None


INDEX_PAGE = """
<a href="../../../abc/vllm-0.33.0rc3%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl#sha256=rc">rc</a>
<a href="../../../abc/vllm-0.33.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl#sha256=mac">mac</a>
<a href="../../../abc/vllm-0.33.0%2Bcpu-cp38-abi3-manylinux_2_34_aarch64.whl#sha256=arm">arm</a>
<a href="../../../abc/vllm-0.33.0%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl#sha256=x86">x86</a>
"""


@pytest.mark.parametrize(
    ("system", "machine", "expected_name"),
    [
        ("Linux", "x86_64", "vllm-0.33.0%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl#sha256=x86"),
        ("Linux", "aarch64", "vllm-0.33.0%2Bcpu-cp38-abi3-manylinux_2_34_aarch64.whl#sha256=arm"),
        ("Darwin", "arm64", "vllm-0.33.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl#sha256=mac"),
    ],
)
def test_resolve_vllm_cpu_wheel_picks_release_wheel_for_platform(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, expected_name: str
) -> None:
    _serve(monkeypatch=monkeypatch, payload=INDEX_PAGE.encode(), expected_url="https://wheels.vllm.ai/0.33.0/cpu/vllm/")
    pattern = rdt.vllm_wheel_platform_pattern(system=system, machine=machine)
    wheel = rdt.resolve_vllm_cpu_wheel(version="0.33.0", platform_pattern=pattern)
    assert wheel == f"https://wheels.vllm.ai/abc/{expected_name}"


def test_resolve_vllm_cpu_wheel_ignores_other_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch=monkeypatch, payload=INDEX_PAGE.encode())
    pattern = rdt.vllm_wheel_platform_pattern(system="Linux", machine="x86_64")
    with pytest.raises(rdt.SetupError, match="no vLLM 0.33 \\+cpu wheel"):
        rdt.resolve_vllm_cpu_wheel(version="0.33", platform_pattern=pattern)


def test_resolve_vllm_cpu_wheel_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _unreachable(monkeypatch=monkeypatch)
    with pytest.raises(rdt.SetupError, match="cannot list"):
        rdt.resolve_vllm_cpu_wheel(version="0.33.0", platform_pattern="manylinux_2_39_x86_64")


def test_child_env_drops_caller_virtualenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/outer/.venv")
    env = rdt._child_env(UV_TORCH_BACKEND="cpu")
    assert "VIRTUAL_ENV" not in env
    assert env["UV_TORCH_BACKEND"] == "cpu"


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ({"ours-vllm": "passed", "ours-transformers": "skipped (no wheel)"}, 0),
        ({"ours-vllm": "failed", "ours-transformers": "passed"}, 1),
        ({"ours-vllm": "failed", "ours-transformers": "setup-error"}, 3),
    ],
    ids=["pass-or-skip", "test-failure", "setup-error-wins"],
)
def test_main_exit_code(monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, str], expected: int) -> None:
    def fake_run_lane(lane: str, work: Path) -> str:
        outcome = outcomes.get(lane) or "passed"
        if outcome == "setup-error":
            raise rdt.SetupError("install failed")
        return outcome

    monkeypatch.setattr(rdt, "_run_lane", fake_run_lane)
    assert rdt.main(argv=[]) == expected


def test_main_runs_only_the_selected_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    def fake_run_lane(lane: str, work: Path) -> str:
        ran.append(lane)
        return "passed"

    monkeypatch.setattr(rdt, "_run_lane", fake_run_lane)
    assert rdt.main(argv=["--only", "upstream-transformers"]) == 0
    assert ran == ["upstream-transformers"]


def test_run_lane_reports_failed_setup_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(args: list[str], cwd: Path | None, env: dict[str, str], check: bool) -> None:
        raise subprocess.CalledProcessError(returncode=128, cmd=args)

    monkeypatch.setattr(rdt.subprocess, "run", fake_run)
    monkeypatch.setenv("TRANSFORMERS_VERSION", "5.23.0")
    with pytest.raises(rdt.SetupError, match="git clone --depth 1 ... exited with status 128"):
        rdt._run_lane(lane="upstream-transformers", work=tmp_path)


@pytest.mark.parametrize("lane", ["ours-vllm", "ours-transformers"])
def test_ours_lane_install_failure_is_a_setup_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lane: str) -> None:
    def fake_run(args: list[str], cwd: Path | None, env: dict[str, str], check: bool) -> subprocess.CompletedProcess:
        if args[:3] == ["uv", "pip", "install"]:
            raise subprocess.CalledProcessError(returncode=1, cmd=args)
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(rdt.subprocess, "run", fake_run)
    monkeypatch.setattr(rdt, "_resolve_vllm", lambda: ("0.33.0", "https://wheels.vllm.ai/vllm.whl"))
    monkeypatch.setenv("TRANSFORMERS_VERSION", "5.23.0")
    with pytest.raises(rdt.SetupError, match="uv pip install --no-config ... exited with status 1"):
        rdt._run_lane(lane=lane, work=tmp_path)


def test_unknown_lane_is_a_usage_error() -> None:
    completed = subprocess.run(
        args=[sys.executable, str(SCRIPT_PATH), "--only", "bogus"], capture_output=True, check=False
    )
    assert completed.returncode == rdt.EXIT_USAGE
