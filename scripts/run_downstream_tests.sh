#!/usr/bin/env bash
# Run the downstream integration tests (tests/integration/downstream) against
# the LATEST released vLLM (CPU wheel) and latest Transformers.
#
# Works identically locally and in GitHub Actions: it resolves the newest vLLM
# release at run time and installs it with `uv run --with ...`, so uv.lock is
# never touched and no project dependency group changes. The default test run
# (`uv run pytest tests/`) skips these tests because vllm is not installed.
#
# Usage:
#   scripts/run_downstream_tests.sh                 # vLLM + Transformers downstream tests
#   scripts/run_downstream_tests.sh --only vllm     # only the vLLM job
#   scripts/run_downstream_tests.sh --only transformers
#   VLLM_VERSION=0.33.0 scripts/run_downstream_tests.sh   # pin vLLM instead of latest
set -euo pipefail

cd "$(dirname "$0")/.."

ONLY="${1:-all}"
if [[ "$ONLY" == "--only" ]]; then
    ONLY="$2"
fi
case "$ONLY" in
    all|vllm|transformers) ;;
    *) echo "usage: $0 [--only all|vllm|transformers]" >&2; exit 2 ;;
esac

# Resolve the latest released vLLM version (not main, not nightly).
if [[ -z "${VLLM_VERSION:-}" ]]; then
    VLLM_VERSION="$(curl -sf https://api.github.com/repos/vllm-project/vllm/releases/latest \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"].lstrip("v"))')"
fi
echo "==> vLLM version under test: ${VLLM_VERSION} (latest release)"
VLLM_CPU_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl"

run_pytest() {
    # `--with` installs into an ephemeral overlay on top of the project venv:
    # uv.lock is untouched and the versions are resolved fresh on every run.
    # `--with torch --torch-backend cpu` makes uv pick CPU-only torch wheels.
    uv run --with "$1" --with torch --torch-backend cpu \
        --with pytest \
        python -m pytest tests/integration/downstream -m downstream -v "$@"
}

if [[ "$ONLY" == "all" || "$ONLY" == "vllm" ]]; then
    echo "==> Running downstream vLLM tests (${VLLM_CPU_WHEEL})"
    VLLM_TARGET_DEVICE=empty uv run \
        --with "${VLLM_CPU_WHEEL}" --with torch --torch-backend cpu --with pytest \
        python -m pytest tests/integration/downstream -m "downstream and vllm" -v
fi

if [[ "$ONLY" == "all" || "$ONLY" == "transformers" ]]; then
    echo "==> Running downstream Transformers tests (latest PyPI)"
    uv run --with "transformers" --with pytest \
        python -m pytest tests/integration/downstream -m "downstream and transformers" -v
fi

echo "==> Downstream tests finished"
