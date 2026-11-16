#!/usr/bin/env python3
r"""Run downstream compatibility tests against the latest released vLLM and Transformers.

Lanes:
  ours-vllm              tests/integration/downstream with the latest vLLM +cpu wheel
  ours-transformers      tests/integration/downstream with the latest Transformers
  upstream-vllm          vLLM's own Mistral test files, cloned at the release tag
  upstream-transformers  Transformers' own Mistral test files, cloned at the release tag

Every lane installs into a fresh virtual environment in a temporary directory with
this checkout's mistral-common, so ``uv.lock`` is never touched and the runtimes are
the latest releases at run time. The script only uses the standard library and the
``uv`` and ``git`` executables, so it runs the same way locally and in GitHub Actions.

Usage:
    python3 scripts/run_downstream_tests.py                       # all lanes
    python3 scripts/run_downstream_tests.py --only upstream-vllm  # one lane
    VLLM_VERSION=0.33.0 TRANSFORMERS_VERSION=5.23.0 python3 scripts/run_downstream_tests.py

Exit codes:
    0  every selected lane passed, or was skipped because the platform has no vLLM +cpu wheel
    1  at least one lane's tests failed
    2  usage error
    3  at least one lane could not be set up (release lookup, wheel resolution, clone or install),
       including when the vLLM +cpu wheel needs a newer glibc than the host's
"""

import argparse
import functools
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
DOWNSTREAM_TESTS = PROJECT_ROOT / "tests" / "integration" / "downstream"

LANES = ("all", "ours-vllm", "ours-transformers", "upstream-vllm", "upstream-transformers")
VLLM_LANES = ("ours-vllm", "upstream-vllm")

EXIT_OK = 0
EXIT_TEST_FAILURE = 1
EXIT_USAGE = 2
EXIT_SETUP = 3

VLLM_UPSTREAM_TEST_PACKAGES = (
    "pytest",
    "pytest-asyncio",
    "regex",
    "psutil",
    "tblib",
    "ijson",
    "partial-json-parser",
)
VLLM_UPSTREAM_TESTS = (
    "tests/tokenizers_/test_mistral.py",
    "tests/tokenizers_/test_basic.py",
    "tests/tokenizers_/test_registry.py",
    "tests/parser/mistral/test_tool_calls.py",
    "tests/parser/mistral/test_reasoning.py",
    "tests/renderers/test_mistral.py",
)
TRANSFORMERS_UPSTREAM_TESTS = (
    "tests/test_tokenization_mistral_common.py",
    "tests/integrations/mistral/test_tokenizer.py",
)


class SetupError(RuntimeError):
    r"""A lane could not be set up before its tests ran."""


def resolve_latest_version(package: str) -> str:
    r"""Return the latest released version of a PyPI package.

    Args:
        package: The PyPI project name.

    Returns:
        The version PyPI reports as latest, which is also the release's git tag without the `v` prefix.

    Raises:
        SetupError: If PyPI is unreachable or its reply has no version string.
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url=url, timeout=30) as response:
            version = json.load(fp=response)["info"]["version"]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SetupError(f"cannot resolve the latest {package} release from {url}: {exc}") from exc
    if not isinstance(version, str):
        raise SetupError(f"PyPI reported a non-string {package} version: {version!r}")
    return version


def vllm_wheel_platform_pattern(system: str, machine: str) -> str | None:
    r"""Return a regex matching the platform tag of vLLM +cpu wheels for this host.

    The glibc version of the manylinux tag is left open so a new vLLM release that
    raises it is still found; `resolve_vllm_cpu_wheel` checks it against the host's glibc.

    Args:
        system: The OS name, as returned by `platform.system()`.
        machine: The CPU architecture, as returned by `platform.machine()`.

    Returns:
        The platform tag regex, or `None` when vLLM publishes no +cpu wheel for this host.
    """
    if system == "Linux" and machine == "x86_64":
        return r"manylinux_\d+_\d+_x86_64"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return r"manylinux_\d+_\d+_aarch64"
    if system == "Darwin" and machine == "arm64":
        return r"macosx_\d+_\d+_arm64"
    return None


def _host_glibc() -> tuple[int, int] | None:
    r"""Return the host's glibc version, or `None` when the host does not use glibc."""
    library, version = platform.libc_ver()
    match = re.match(pattern=r"(\d+)\.(\d+)", string=version)
    if library != "glibc" or match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def resolve_vllm_cpu_wheel(version: str, platform_pattern: str, glibc: tuple[int, int] | None = None) -> str:
    r"""Return the URL of the vLLM +cpu wheel for a release and platform.

    The index hrefs are relative to a commit directory and percent-encode the `+`,
    so the URL is read from the index page rather than built by hand.

    Args:
        version: The exact vLLM release, e.g. `0.33.0`; release candidates of it do not match.
        platform_pattern: The platform tag regex from `vllm_wheel_platform_pattern`.
        glibc: The host's glibc version; manylinux wheels that need a newer one are skipped.

    Returns:
        The absolute wheel URL, including any `#sha256=` fragment the index provides.

    Raises:
        SetupError: If the index is unreachable or lists no matching wheel the host can install.
    """
    index = f"https://wheels.vllm.ai/{version}/cpu/vllm/"
    try:
        with urllib.request.urlopen(url=index, timeout=30) as response:
            page = response.read().decode()
    except urllib.error.URLError as exc:
        raise SetupError(f"cannot list {index}: {exc}") from exc
    wheel_name = re.compile(pattern=rf"vllm-{re.escape(version)}\+cpu-.+-(?:{platform_pattern})\.whl")
    too_new: tuple[int, int] | None = None
    for href in re.findall(pattern=r'href="([^"]+)"', string=page):
        filename = urllib.parse.unquote(string=href.split(sep="#")[0].rsplit(sep="/", maxsplit=1)[-1])
        if not wheel_name.fullmatch(filename):
            continue
        tag = re.search(pattern=r"-manylinux_(\d+)_(\d+)_", string=filename)
        required = (int(tag.group(1)), int(tag.group(2))) if tag else None
        if glibc is not None and required is not None and required > glibc:
            too_new = required if too_new is None else min(too_new, required)
            continue
        wheel: str = urllib.parse.urljoin(base=index, url=href)
        return wheel
    if too_new is not None and glibc is not None:
        raise SetupError(
            f"vLLM {version} +cpu wheels need glibc >= {too_new[0]}.{too_new[1]}, this host has {glibc[0]}.{glibc[1]}"
        )
    raise SetupError(f"no vLLM {version} +cpu wheel matching {platform_pattern} in {index}")


def _child_env(**overrides: str) -> dict[str, str]:
    r"""Return the environment for a child command.

    `VIRTUAL_ENV` is dropped because a caller started under `uv run` would otherwise
    make `uv` install into the caller's environment instead of the lane's own.

    Args:
        overrides: Variables to set on top of the current environment.

    Returns:
        The child environment.
    """
    env = {name: value for name, value in os.environ.items() if name != "VIRTUAL_ENV"}
    env.update(overrides)
    return env


def _setup(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    r"""Run a setup command, turning its failure into a `SetupError`.

    Args:
        command: The command and its arguments.
        cwd: The working directory, or the current one when `None`.
        env: The environment, or `_child_env()` when `None`.

    Raises:
        SetupError: If the command exits with a non-zero status.
    """
    try:
        subprocess.run(args=command, cwd=cwd, env=env if env is not None else _child_env(), check=True)
    except subprocess.CalledProcessError as exc:
        raise SetupError(f"{' '.join(command[:4])} ... exited with status {exc.returncode}") from exc


def _test(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    r"""Run a test command and report its outcome.

    Args:
        command: The pytest command and its arguments.
        cwd: The working directory, or the current one when `None`.
        env: The environment, or `_child_env()` when `None`.

    Returns:
        `passed` when the command exits with status 0, `failed` otherwise.
    """
    completed = subprocess.run(args=command, cwd=cwd, env=env if env is not None else _child_env(), check=False)
    return "passed" if completed.returncode == 0 else "failed"


def _create_venv(path: Path) -> Path:
    r"""Create an isolated Python 3.12 virtual environment.

    Args:
        path: The directory of the new environment.

    Returns:
        The environment's Python interpreter.
    """
    _setup(command=["uv", "venv", "--no-config", "--python", "3.12", str(path)])
    return path / "bin" / "python"


def _clone(repository: str, tag: str, destination: Path, sparse_paths: tuple[str, ...] = ()) -> None:
    r"""Shallow-clone one tag of a GitHub repository.

    Args:
        repository: The `owner/name` of the repository.
        tag: The tag to check out.
        destination: The clone directory.
        sparse_paths: When set, only these directories and the files at the root are checked out.
    """
    sparse = ["--filter=blob:none", "--sparse"] if sparse_paths else []
    _setup(
        command=[
            "git",
            "clone",
            "--depth",
            "1",
            *sparse,
            "--branch",
            tag,
            f"https://github.com/{repository}.git",
            str(destination),
        ]
    )
    if sparse_paths:
        _setup(command=["git", "-C", str(destination), "sparse-checkout", "set", *sparse_paths])


def _check_test_files(checkout: Path, paths: tuple[str, ...]) -> None:
    r"""Check that a clone has the test files a lane runs.

    Args:
        checkout: The clone directory.
        paths: The test files, relative to the clone.

    Raises:
        SetupError: If any of them is missing, e.g. because the release moved it.
    """
    missing = [path for path in paths if not (checkout / path).is_file()]
    if missing:
        raise SetupError(f"test files missing from the {checkout.name} clone: {', '.join(missing)}")


def _uv_pip_install(python: Path, requirements: list[str], cwd: Path) -> None:
    r"""Install requirements into a lane's virtual environment.

    Args:
        python: The environment's interpreter.
        requirements: The requirements, optionally preceded by `uv pip install` options.
        cwd: The working directory, for editable installs.
    """
    _setup(
        command=["uv", "pip", "install", "--no-config", "--python", str(python), *requirements],
        cwd=cwd,
    )


def _mistral_common_requirement() -> str:
    r"""Return the requirement that installs this checkout's mistral-common."""
    return f"mistral-common @ {PROJECT_ROOT.as_uri()}"


def lane_ours_vllm(wheel: str, work: Path) -> str:
    r"""Run this repository's downstream tests that need vLLM.

    Args:
        wheel: The vLLM +cpu wheel URL.
        work: A temporary directory for the lane's environment.

    Returns:
        The lane outcome.
    """
    python = _create_venv(path=work / "ours-vllm")
    _uv_pip_install(
        python=python,
        requirements=["--torch-backend", "cpu", wheel, "pytest", _mistral_common_requirement()],
        cwd=PROJECT_ROOT,
    )
    return _test(
        command=[
            str(python),
            "-m",
            "pytest",
            "-o",
            "addopts=",
            str(DOWNSTREAM_TESTS),
            "-m",
            "downstream and vllm",
            "-v",
        ]
    )


def lane_ours_transformers(version: str, work: Path) -> str:
    r"""Run this repository's downstream tests that need Transformers.

    Args:
        version: The Transformers release.
        work: A temporary directory for the lane's environment.

    Returns:
        The lane outcome.
    """
    python = _create_venv(path=work / "ours-transformers")
    _uv_pip_install(
        python=python,
        requirements=[f"transformers=={version}", "pytest", _mistral_common_requirement()],
        cwd=PROJECT_ROOT,
    )
    return _test(
        command=[
            str(python),
            "-m",
            "pytest",
            "-o",
            "addopts=",
            str(DOWNSTREAM_TESTS),
            "-m",
            "downstream and transformers",
            "-v",
        ]
    )


def lane_upstream_vllm(version: str, wheel: str, work: Path) -> str:
    r"""Run vLLM's own Mistral test files at the release tag against this checkout.

    Args:
        version: The vLLM release.
        wheel: The vLLM +cpu wheel URL for that release.
        work: A temporary directory for the clone.

    Returns:
        The lane outcome.
    """
    checkout = work / "vllm"
    # Only `tests/` is checked out, so `import vllm` resolves to the installed wheel
    # rather than to the clone's source tree, which lacks the compiled extensions.
    _clone(repository="vllm-project/vllm", tag=f"v{version}", destination=checkout, sparse_paths=("tests",))
    _check_test_files(checkout=checkout, paths=VLLM_UPSTREAM_TESTS)
    python = _create_venv(path=checkout / ".venv")
    _uv_pip_install(python=python, requirements=["--torch-backend", "cpu", wheel], cwd=checkout)
    _uv_pip_install(python=python, requirements=list(VLLM_UPSTREAM_TEST_PACKAGES), cwd=checkout)
    _uv_pip_install(python=python, requirements=["--no-deps", _mistral_common_requirement()], cwd=checkout)
    return _test(
        command=[str(python), "-m", "pytest", "-v", "-m", "not gpu", *VLLM_UPSTREAM_TESTS],
        cwd=checkout,
    )


def lane_upstream_transformers(version: str, work: Path) -> str:
    r"""Run Transformers' own Mistral test files at the release tag against this checkout.

    Args:
        version: The Transformers release.
        work: A temporary directory for the clone.

    Returns:
        The lane outcome.
    """
    checkout = work / "transformers"
    _clone(repository="huggingface/transformers", tag=f"v{version}", destination=checkout)
    _check_test_files(checkout=checkout, paths=TRANSFORMERS_UPSTREAM_TESTS)
    python = _create_venv(path=checkout / ".venv")
    _uv_pip_install(python=python, requirements=["--torch-backend", "cpu", "-e", ".[dev]"], cwd=checkout)
    # The fixture fetcher imports PyArrow, which needs NumPy 2, while `.[dev]` can
    # resolve NumPy 1; the upper bound is mistral-common's own NumPy bound, which
    # `--no-deps` below would not enforce.
    _uv_pip_install(python=python, requirements=["numpy>=2,<2.4"], cwd=checkout)
    _uv_pip_install(python=python, requirements=["--no-deps", _mistral_common_requirement()], cwd=checkout)
    _setup(command=[str(python), "utils/fetch_hub_objects_for_ci.py"], cwd=checkout)
    return _test(command=[str(python), "-m", "pytest", "-v", *TRANSFORMERS_UPSTREAM_TESTS], cwd=checkout)


def _resolve_transformers() -> str:
    r"""Resolve the Transformers release under test.

    Returns:
        The release.

    Raises:
        SetupError: If the release cannot be resolved.
    """
    version = os.environ.get("TRANSFORMERS_VERSION") or resolve_latest_version(package="transformers")
    print(f"==> Transformers release under test: {version}", flush=True)
    return version


@functools.cache
def _resolve_vllm() -> tuple[str, str | None]:
    r"""Resolve the vLLM release under test and its +cpu wheel for this host.

    Returns:
        The release, and the wheel URL or `None` when vLLM publishes no +cpu wheel for this host.

    Raises:
        SetupError: If the release or the wheel cannot be resolved.
    """
    version = os.environ.get("VLLM_VERSION") or resolve_latest_version(package="vllm")
    print(f"==> vLLM release under test: {version}", flush=True)
    pattern = vllm_wheel_platform_pattern(system=platform.system(), machine=platform.machine())
    if pattern is None:
        print(f"==> vLLM publishes no +cpu wheel for {platform.system()}-{platform.machine()}", flush=True)
        return version, None
    return version, resolve_vllm_cpu_wheel(version=version, platform_pattern=pattern, glibc=_host_glibc())


def _run_lane(lane: str, work: Path) -> str:
    r"""Set up and run one lane.

    Args:
        lane: The lane name.
        work: A temporary directory for clones.

    Returns:
        The lane outcome.

    Raises:
        SetupError: If the lane cannot be set up.
    """
    if lane == "ours-transformers":
        return lane_ours_transformers(version=_resolve_transformers(), work=work)
    if lane == "upstream-transformers":
        return lane_upstream_transformers(version=_resolve_transformers(), work=work)
    vllm_version, wheel = _resolve_vllm()
    if wheel is None:
        return "skipped (no vLLM +cpu wheel for this platform)"
    if lane == "ours-vllm":
        return lane_ours_vllm(wheel=wheel, work=work)
    return lane_upstream_vllm(version=vllm_version, wheel=wheel, work=work)


def main(argv: list[str] | None = None) -> int:
    r"""Run the selected lanes and return the script's exit code.

    Args:
        argv: The command-line arguments, or `sys.argv[1:]` when `None`.

    Returns:
        The exit code described in the module docstring.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=LANES, default="all", help="run a single lane (default: all)")
    args = parser.parse_args(args=argv)

    selected = [lane for lane in LANES[1:] if args.only in ("all", lane)]
    outcomes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="mistral-common-downstream-") as temporary:
        for lane in selected:
            print(f"==> lane {lane}", flush=True)
            try:
                outcomes[lane] = _run_lane(lane=lane, work=Path(temporary))
            except SetupError as exc:
                print(f"==> lane {lane}: setup error: {exc}", file=sys.stderr, flush=True)
                outcomes[lane] = "setup-error"

    print("==> downstream lanes:", flush=True)
    for lane, outcome in outcomes.items():
        print(f"    {lane}: {outcome}", flush=True)
    if "setup-error" in outcomes.values():
        return EXIT_SETUP
    if "failed" in outcomes.values():
        return EXIT_TEST_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
