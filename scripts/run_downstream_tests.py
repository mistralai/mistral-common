#!/usr/bin/env python3
r"""Run released vLLM and Transformers Mistral suites against this checkout."""

import argparse
import html.parser
import json
import os
import platform
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLLM_CPU_INDEX = "https://wheels.vllm.ai/{version}/cpu/vllm/"
VLLM_TEST_PACKAGES = (
    "pytest",
    "pytest-asyncio",
    "regex",
    "psutil",
    "tblib",
    "ijson",
    "partial-json-parser",
)
VLLM_TESTS = (
    "tests/tokenizers_/test_mistral.py",
    "tests/tokenizers_/test_basic.py",
    "tests/tokenizers_/test_registry.py",
    "tests/parser/mistral/test_tool_calls.py",
    "tests/parser/mistral/test_reasoning.py",
    "tests/renderers/test_mistral.py",
)
TRANSFORMERS_TESTS = (
    "tests/test_tokenization_mistral_common.py",
    "tests/integrations/mistral/test_tokenizer.py",
)


class _WheelLinks(html.parser.HTMLParser):
    r"""Collect wheel hrefs from vLLM's simple package index."""

    def __init__(self) -> None:
        r"""Initialize the collected wheel links."""
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        r"""Collect the href from each anchor in the package index."""
        if tag == "a":
            href = dict(attrs).get("href")
            if href is not None:
                self.hrefs.append(href)


def _latest_pypi_version(package: str) -> str:
    r"""Read the currently published version from PyPI's package metadata."""
    request = urllib.request.Request(
        url=f"https://pypi.org/pypi/{package}/json",
        headers={"User-Agent": "mistral-common-downstream-tests"},
    )
    with urllib.request.urlopen(request) as response:
        return str(json.load(response)["info"]["version"])


def _get_vllm_cpu_wheel(version: str) -> str:
    r"""Resolve the exact CPU wheel matching this release and host platform."""
    platform_tag = {
        ("Linux", "x86_64"): "manylinux_2_39_x86_64",
        ("Linux", "aarch64"): "manylinux_2_39_aarch64",
        ("Darwin", "arm64"): "macosx_11_0_arm64",
    }.get((platform.system(), platform.machine().lower()))
    if platform_tag is None:
        raise RuntimeError(f"No vLLM CPU wheel for {platform.system()}-{platform.machine()}")

    index = VLLM_CPU_INDEX.format(version=version)
    request = urllib.request.Request(
        url=index,
        headers={"User-Agent": "mistral-common-downstream-tests"},
    )
    with urllib.request.urlopen(request) as response:
        links = _WheelLinks()
        links.feed(response.read().decode())

    expected_prefix = f"vllm-{version}+cpu-"
    wheel = next(
        (href for href in links.hrefs if expected_prefix in urllib.parse.unquote(href) and platform_tag in href),
        None,
    )
    if wheel is None:
        raise RuntimeError(f"No vLLM {version} CPU wheel for {platform_tag}")
    return urllib.parse.urljoin(base=index, url=wheel)


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    r"""Run a command from its consumer checkout with an explicit environment."""
    subprocess.run(args=command, cwd=cwd, env=env, check=True)


def _create_venv(path: Path) -> Path:
    r"""Create an isolated Python 3.12 environment and return its interpreter."""
    _run(
        command=["uv", "venv", "--python", "3.12", "--no-config", str(path)],
        cwd=path.parent,
        env=os.environ.copy(),
    )
    return path / "bin" / "python"


def _run_vllm_tests(work: Path, version: str, wheel: str) -> None:
    r"""Install the official CPU wheel and run vLLM's Mistral test set."""
    checkout = work / "vllm"
    _run(
        command=[
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            f"v{version}",
            "https://github.com/vllm-project/vllm.git",
            str(checkout),
        ],
        cwd=work,
        env=os.environ.copy(),
    )
    python = _create_venv(path=checkout / ".venv")
    install_env = {
        **os.environ,
        "VLLM_TARGET_DEVICE": "empty",
    }
    _run(
        command=[
            "uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            str(python),
            "--torch-backend",
            "cpu",
            wheel,
        ],
        cwd=checkout,
        env=install_env,
    )
    _run(
        command=["uv", "pip", "install", "--no-config", "--python", str(python), *VLLM_TEST_PACKAGES],
        cwd=checkout,
        env=install_env,
    )
    _run(
        command=[
            "uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            str(python),
            "--no-deps",
            f"mistral-common @ {ROOT.as_uri()}",
        ],
        cwd=checkout,
        env=install_env,
    )
    _run(
        command=[str(python), "-m", "pytest", "-v", "-m", "not gpu", *VLLM_TESTS],
        cwd=checkout,
        env=os.environ.copy(),
    )


def _run_transformers_tests(work: Path, version: str) -> None:
    r"""Install Transformers' CPU test environment and run its Mistral tests."""
    checkout = work / "transformers"
    _run(
        command=[
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            f"v{version}",
            "https://github.com/huggingface/transformers.git",
            str(checkout),
        ],
        cwd=work,
        env=os.environ.copy(),
    )
    python = _create_venv(path=checkout / ".venv")
    install_env = os.environ.copy()
    _run(
        command=[
            "uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            str(python),
            "--torch-backend",
            "cpu",
            "-e",
            ".[dev]",
        ],
        cwd=checkout,
        env=install_env,
    )
    # The CI fixture fetcher imports PyArrow, which requires NumPy 2.
    _run(
        command=["uv", "pip", "install", "--no-config", "--python", str(python), "numpy>=2,<2.4"],
        cwd=checkout,
        env=install_env,
    )
    _run(
        command=[
            "uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            str(python),
            "--no-deps",
            f"mistral-common @ {ROOT.as_uri()}",
        ],
        cwd=checkout,
        env=install_env,
    )
    _run(
        command=[str(python), "utils/fetch_hub_objects_for_ci.py"],
        cwd=checkout,
        env=install_env,
    )
    _run(
        command=[str(python), "-m", "pytest", "-v", *TRANSFORMERS_TESTS],
        cwd=checkout,
        env=install_env,
    )


def main() -> None:
    r"""Resolve release overrides and run the selected upstream test suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite",
        choices=("vllm", "transformers"),
        help="which upstream consumer's Mistral tests to run",
    )
    args = parser.parse_args()

    if args.suite == "vllm":
        version_override = os.environ.get("VLLM_VERSION")
        version = version_override or _latest_pypi_version(package="vllm")
        release_source = "requested" if version_override else "latest"
        print(f"Testing {release_source} vLLM release: {version}", flush=True)
        wheel = _get_vllm_cpu_wheel(version=version)
        with tempfile.TemporaryDirectory(prefix="mistral-common-vllm-") as temporary:
            _run_vllm_tests(work=Path(temporary), version=version, wheel=wheel)
    else:
        version_override = os.environ.get("TRANSFORMERS_VERSION")
        version = version_override or _latest_pypi_version(package="transformers")
        release_source = "requested" if version_override else "latest"
        print(f"Testing {release_source} Transformers release: {version}", flush=True)
        with tempfile.TemporaryDirectory(prefix="mistral-common-transformers-") as temporary:
            _run_transformers_tests(work=Path(temporary), version=version)


if __name__ == "__main__":
    main()
