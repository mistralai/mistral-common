#!/usr/bin/env bash
# Run all downstream tests against the LATEST RELEASED runtimes:
#   - ours:      tests/integration/downstream (this repo's tests) with latest vLLM +cpu / Transformers
#   - upstream:  vLLM's and Transformers' own mistral test suites against this repo's mistral-common
#
# Works identically locally and in GitHub Actions. Everything is installed with
# `uv run --with` / `uv pip install` into ephemeral venvs/overlays, so uv.lock is
# never touched and versions are always latest-at-run-time, never lock-pinned.
#
# Usage:
#   scripts/run_downstream_tests.sh                          # all lanes
#   scripts/run_downstream_tests.sh --only ours-vllm         # one lane; one of:
#   ... ours-vllm | ours-transformers | upstream-vllm | upstream-transformers | all
#   VLLM_VERSION=0.33.0 scripts/run_downstream_tests.sh      # pin vLLM instead of latest release
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

ONLY="${1:-all}"
if [[ "$ONLY" == "--only" ]]; then
    ONLY="$2"
fi
case "$ONLY" in
    all|ours-vllm|ours-transformers|upstream-vllm|upstream-transformers) ;;
    *) echo "usage: $0 [--only all|ours-vllm|ours-transformers|upstream-vllm|upstream-transformers]" >&2; exit 2 ;;
esac

# Resolve the latest released vLLM version from PyPI (its version matches the
# release tag, and the +cpu wheel index is keyed by that same number). No GitHub
# API needed, so no rate limits. TRANSFORMERS_VERSION is only used to clone the
# upstream suite at a tag; same PyPI trick works there.
resolve_latest_version() {  # $1 = PyPI package name
    curl -sf "https://pypi.org/pypi/$1/json" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])'
}
if [[ -z "${VLLM_VERSION:-}" ]]; then
    VLLM_VERSION="$(resolve_latest_version vllm)"
fi
echo "==> latest release: vllm=${VLLM_VERSION}"
# CPU wheels are published on vLLM's own wheel index per release tag
# (https://wheels.vllm.ai/<version>/cpu/vllm/), not as GitHub release assets.
# The index hrefs are commit-pinned and percent-encode the "+", so resolve the
# actual wheel URL from the index page instead of constructing it by hand.
resolve_vllm_cpu_wheel() {  # $1 = version, $2 = platform tag fragment
    local index page url
    index="https://wheels.vllm.ai/${1}/cpu/vllm/"
    page="$(curl -sfL "$index")" || { echo "cannot list $index" >&2; return 1; }
    url="$(printf '%s' "$page" | grep -o "href=\"[^\"]*${2}[^\"]*\"" | head -1 | sed 's/href="//; s/"$//')"
    [[ -n "$url" ]] || return 1
    # hrefs are relative like ../../../<commit>/<file>%2B... ; resolve against the index
    echo "$(python3 -c 'import sys,urllib.parse; print(urllib.parse.urljoin(sys.argv[1], sys.argv[2]))' "$index" "$url")"
}
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  VLLM_WHEEL_PLAT="manylinux_2_39_x86_64" ;;
    Linux-aarch64) VLLM_WHEEL_PLAT="manylinux_2_39_aarch64" ;;
    Darwin-arm64)  VLLM_WHEEL_PLAT="macosx_11_0_arm64" ;;
    *) VLLM_WHEEL_PLAT="" ;;
esac
VLLM_CPU_WHEEL=""
if [[ -n "$VLLM_WHEEL_PLAT" ]]; then
    VLLM_CPU_WHEEL="$(resolve_vllm_cpu_wheel "$VLLM_VERSION" "$VLLM_WHEEL_PLAT")" \
        || echo "==> no +cpu vLLM wheel for ${VLLM_WHEEL_PLAT} in v${VLLM_VERSION}; vLLM lanes will be skipped" >&2
fi

if [[ "$ONLY" == "all" || "$ONLY" == "ours-vllm" ]]; then
    if [[ -z "$VLLM_CPU_WHEEL" ]]; then
        echo "==> ours-vllm: SKIPPED (no +cpu vLLM wheel on $(uname -s)-$(uname -m) for v${VLLM_VERSION})"
    else
        echo "==> ours: downstream vLLM tests (vllm ${VLLM_VERSION} +cpu wheel: ${VLLM_WHEEL_PLAT})"
        # `--no-config` bypasses pyproject (incl. exclude-newer) so the wheel URL and
        # Fully isolated env (--isolated --no-project): the project's own
        # requires-python matrix must not constrain vLLM/torch resolution.
        # mistral-common is installed from the local tree; uv.lock untouched.
        VLLM_TARGET_DEVICE=empty UV_TORCH_BACKEND=cpu uv run --no-config --isolated --no-project \
            --with "${VLLM_CPU_WHEEL}" --with torch --with pytest \
            --with "mistral-common @ file://${PROJECT_ROOT}" \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            --index-strategy unsafe-best-match \
            python -m pytest -o addopts="" "${PROJECT_ROOT}/tests/integration/downstream" -m "downstream and vllm" -v
    fi
fi

if [[ "$ONLY" == "all" || "$ONLY" == "ours-transformers" ]]; then
    echo "==> ours: downstream Transformers tests (latest PyPI, unpinned)"
    # `--no-config --isolated --no-project` + unpinned `transformers` = truly
    # latest release, ignoring the project's exclude-newer clamp; lock untouched.
    uv run --no-config --isolated --no-project \
        --with "transformers" --with pytest --with "mistral-common @ file://${PROJECT_ROOT}" \
        python -m pytest -o addopts="" "${PROJECT_ROOT}/tests/integration/downstream" -m "downstream and transformers" -v
fi

if [[ "$ONLY" == "all" || "$ONLY" == "upstream-vllm" ]]; then
    if [[ -z "$VLLM_WHEEL_PLAT" ]]; then
        echo "==> upstream-vllm: SKIPPED (no +cpu vLLM wheel on $(uname -s)-$(uname -m) for v${VLLM_VERSION})"
    else
        echo "==> upstream: vLLM's own mistral tests @ v${VLLM_VERSION} against this mistral-common"
        git clone --depth 1 --branch "v${VLLM_VERSION}" https://github.com/vllm-project/vllm.git "$WORK/vllm"
        cd "$WORK/vllm"
        uv venv --python 3.12 --no-config
        VLLM_TARGET_DEVICE=empty UV_TORCH_BACKEND=cpu uv pip install "${VLLM_CPU_WHEEL}" \
            --extra-index-url https://download.pytorch.org/whl/cpu
        uv pip install pytest pytest-asyncio regex psutil tblib ijson partial-json-parser
        uv pip install --no-deps "mistral-common @ file://${PROJECT_ROOT}"
        uv run --no-sync pytest -v -m 'not gpu' \
            tests/tokenizers_/test_mistral.py \
            tests/tokenizers_/test_basic.py \
            tests/tokenizers_/test_registry.py \
            tests/parser/mistral/test_tool_calls.py \
            tests/parser/mistral/test_reasoning.py \
            tests/renderers/test_mistral.py
        cd "$PROJECT_ROOT"
    fi
fi

if [[ "$ONLY" == "all" || "$ONLY" == "upstream-transformers" ]]; then
    TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-$(resolve_latest_version transformers)}"
    echo "==> upstream: Transformers' mistral tests @ v${TRANSFORMERS_VERSION} against this mistral-common"
    git clone --depth 1 --branch "v${TRANSFORMERS_VERSION}" https://github.com/huggingface/transformers.git "$WORK/transformers"
    cd "$WORK/transformers"
    uv venv --python 3.12 --no-config
    UV_TORCH_BACKEND=cpu uv pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip install --no-deps "mistral-common @ file://${PROJECT_ROOT}"
    python3 utils/fetch_hub_objects_for_ci.py
    uv run --no-sync pytest -v \
        tests/test_tokenization_mistral_common.py \
        tests/integrations/mistral/test_tokenizer.py
    cd "$PROJECT_ROOT"
fi

echo "==> Downstream tests finished"
