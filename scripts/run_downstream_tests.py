#!/usr/bin/env python3
r"""Run all downstream tests against the LATEST RELEASED runtimes.

Lanes:
  - ours:      tests/integration/downstream (this repo's tests) with latest vLLM +cpu / Transformers
  - upstream:  vLLM's and Transformers' own mistral test suites against this repo's mistral-common

Works identically locally and in GitHub Actions. Everything is installed with
``uv run --with`` / ``uv pip install`` into ephemeral venvs/overlays, so uv.lock
is never touched and versions are always latest-at-run-time, never lock-pinned.

Usage:
    python scripts/run_downstream_tests.py                          # all lanes
    python scripts/run_downstream_tests.py --only ours-vllm         # one lane; one of:
        ... ours-vllm | ours-transformers | upstream-vllm | upstream-transformers | all
    VLLM_VERSION=0.33.0 python scripts/run_downstream_tests.py      # pin vLLM instead of latest release

Exit codes:
    0 — every selected lane passed (or was skipped because no wheel exists)
    1 — at least one lane ran and failed
    2 — usage error (unknown lane)
    3 — setup error (network, wheel resolution, install) before any test ran
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANES = ("all", "ours-vllm", "ours-transformers", "upstream-vllm", "upstream-transformers")

EXIT_OK = 0
EXIT_TEST_FAILURE = 1
EXIT_USAGE = 2
EXIT_SETUP = 3


class SetupError(RuntimeError):
    """A lane could not be set up (network, wheel resolution, install)."""


def resolve_latest_version(package: str) -> str:
    """Latest released version of a PyPI package (its version matches the release tag)."""
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data: dict[str, object] = json.load(resp)
            info = data["info"]
            assert isinstance(info, dict)
            version = info["version"]
            assert isinstance(version, str)
            return version
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        raise SetupError(f"cannot resolve latest version of {package!r} from {url}: {exc}") from exc


def vllm_wheel_platform_tag(system: str, machine: str) -> str | None:
    """Platform tag fragment used in vLLM +cpu wheel file names, or None if unsupported."""
    if system == "Linux" and machine == "x86_64":
        return "manylinux_2_39_x86_64"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return "manylinux_2_39_aarch64"
    if system == "Darwin" and machine == "arm64":
        return "macosx_11_0_arm64"
    return None


def resolve_vllm_cpu_wheel(version: str, platform_tag: str) -> str:
    """Resolve the actual +cpu wheel URL from vLLM's per-release wheel index.

    The index hrefs are commit-pinned and percent-encode the "+", so the URL is
    scraped from the index page instead of constructed by hand.
    """
    index = f"https://wheels.vllm.ai/{version}/cpu/vllm/"
    try:
        with urllib.request.urlopen(index, timeout=30) as resp:
            page = resp.read().decode()
    except urllib.error.URLError as exc:
        raise SetupError(f"cannot list {index}: {exc}") from exc
    for href in re.findall(r'href="([^"]+)"', page):
        if platform_tag in href:
            resolved: str = urllib.parse.urljoin(index, href)
            return resolved
    raise SetupError(f"no +cpu vLLM wheel for {platform_tag} in v{version} (index {index})")


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a command, raising CalledProcessError on non-zero exit."""
    merged_env = {**os.environ, **(env or {})}
    subprocess.run(cmd, cwd=cwd, env=merged_env, check=True)


def lane_ours_vllm(vllm_version: str, wheel: str | None) -> str:
    if wheel is None:
        return "skipped (no +cpu vLLM wheel on this platform)"
    # `--no-config` bypasses pyproject (incl. exclude-newer) so the wheel URL and
    # torch resolve freely. Fully isolated env (--isolated --no-project): the
    # project's own requires-python matrix must not constrain vLLM/torch
    # resolution. mistral-common is installed from the local tree; uv.lock untouched.
    run(
        [
            "uv",
            "run",
            "--no-config",
            "--isolated",
            "--no-project",
            "--with",
            wheel,
            "--with",
            "torch",
            "--with",
            "pytest",
            "--with",
            f"mistral-common @ file://{PROJECT_ROOT}",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "--index-strategy",
            "unsafe-best-match",
            "python",
            "-m",
            "pytest",
            "-o",
            'addopts=""',
            str(PROJECT_ROOT / "tests/integration/downstream"),
            "-m",
            "downstream and vllm",
            "-v",
        ],
        env={"VLLM_TARGET_DEVICE": "empty", "UV_TORCH_BACKEND": "cpu"},
    )
    return "passed"


def lane_ours_transformers() -> str:
    # `--no-config --isolated --no-project` + unpinned `transformers` = truly
    # latest release, ignoring the project's exclude-newer clamp; lock untouched.
    run(
        [
            "uv",
            "run",
            "--no-config",
            "--isolated",
            "--no-project",
            "--with",
            "transformers",
            "--with",
            "pytest",
            "--with",
            f"mistral-common @ file://{PROJECT_ROOT}",
            "python",
            "-m",
            "pytest",
            "-o",
            'addopts=""',
            str(PROJECT_ROOT / "tests/integration/downstream"),
            "-m",
            "downstream and transformers",
            "-v",
        ]
    )
    return "passed"


def lane_upstream_vllm(vllm_version: str, wheel: str | None, work: Path) -> str:
    if wheel is None:
        return "skipped (no +cpu vLLM wheel on this platform)"
    vllm_dir = work / "vllm"
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            f"v{vllm_version}",
            "https://github.com/vllm-project/vllm.git",
            str(vllm_dir),
        ]
    )
    run(["uv", "venv", "--python", "3.12", "--no-config"], cwd=vllm_dir)
    run(
        ["uv", "pip", "install", wheel, "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
        cwd=vllm_dir,
        env={"VLLM_TARGET_DEVICE": "empty", "UV_TORCH_BACKEND": "cpu"},
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "pytest",
            "pytest-asyncio",
            "regex",
            "psutil",
            "tblib",
            "ijson",
            "partial-json-parser",
        ],
        cwd=vllm_dir,
    )
    run(["uv", "pip", "install", "--no-deps", f"mistral-common @ file://{PROJECT_ROOT}"], cwd=vllm_dir)
    run(
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-v",
            "-m",
            "not gpu",
            "tests/tokenizers_/test_mistral.py",
            "tests/tokenizers_/test_basic.py",
            "tests/tokenizers_/test_registry.py",
            "tests/parser/mistral/test_tool_calls.py",
            "tests/parser/mistral/test_reasoning.py",
            "tests/renderers/test_mistral.py",
        ],
        cwd=vllm_dir,
    )
    return "passed"


def lane_upstream_transformers(work: Path) -> str:
    transformers_version = os.environ.get("TRANSFORMERS_VERSION") or resolve_latest_version("transformers")
    tf_dir = work / "transformers"
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            f"v{transformers_version}",
            "https://github.com/huggingface/transformers.git",
            str(tf_dir),
        ]
    )
    run(["uv", "venv", "--python", "3.12", "--no-config"], cwd=tf_dir)
    run(
        ["uv", "pip", "install", "-e", ".[dev]", "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
        cwd=tf_dir,
        env={"UV_TORCH_BACKEND": "cpu"},
    )
    run(["uv", "pip", "install", "--no-deps", f"mistral-common @ file://{PROJECT_ROOT}"], cwd=tf_dir)
    run([sys.executable, "utils/fetch_hub_objects_for_ci.py"], cwd=tf_dir)
    run(
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-v",
            "tests/test_tokenization_mistral_common.py",
            "tests/integrations/mistral/test_tokenizer.py",
        ],
        cwd=tf_dir,
    )
    return "passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=LANES, default="all", help="run a single lane instead of all (default: all)")
    args = parser.parse_args()
    only = args.only

    vllm_version = os.environ.get("VLLM_VERSION") or resolve_latest_version("vllm")
    print(f"==> latest release: vllm={vllm_version}")
    plat = vllm_wheel_platform_tag(platform.system(), platform.machine())
    wheel: str | None = None
    if plat is not None:
        try:
            wheel = resolve_vllm_cpu_wheel(vllm_version, plat)
        except SetupError as exc:
            print(f"==> {exc}; vLLM lanes will be skipped", file=sys.stderr)
    else:
        print(
            f"==> no +cpu vLLM wheel platform for {platform.system()}-{platform.machine()}; vLLM lanes will be skipped",
            file=sys.stderr,
        )

    lanes: list[tuple[str, str]] = []  # (name, outcome)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        selected = [lane for lane in LANES[1:] if only == "all" or only == lane]
        for lane in selected:
            print(f"==> lane {lane}")
            try:
                if lane == "ours-vllm":
                    outcome = lane_ours_vllm(vllm_version, wheel)
                elif lane == "ours-transformers":
                    outcome = lane_ours_transformers()
                elif lane == "upstream-vllm":
                    outcome = lane_upstream_vllm(vllm_version, wheel, work)
                else:
                    outcome = lane_upstream_transformers(work)
            except SetupError as exc:
                print(f"==> lane {lane}: SETUP ERROR: {exc}", file=sys.stderr)
                lanes.append((lane, "setup-error"))
                continue
            except subprocess.CalledProcessError as exc:
                print(f"==> lane {lane}: FAILED (exit {exc.returncode})", file=sys.stderr)
                lanes.append((lane, "failed"))
                continue
            print(f"==> lane {lane}: {outcome}")
            lanes.append((lane, outcome))

    print("==> Downstream tests finished")
    for name, outcome in lanes:
        print(f"    {name}: {outcome}")
    if any(o == "setup-error" for _, o in lanes):
        return EXIT_SETUP
    if any(o == "failed" for _, o in lanes):
        return EXIT_TEST_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
