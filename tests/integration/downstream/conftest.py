import os

import pytest


def _downstream_enabled() -> bool:
    return os.environ.get("MISTRAL_COMMON_RUN_DOWNSTREAM", "").lower() in ("1", "true", "yes")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _downstream_enabled():
        return
    skip = pytest.mark.skip(
        reason="downstream tests need an external runtime; run scripts/run_downstream_tests.sh "
        "(or set MISTRAL_COMMON_RUN_DOWNSTREAM=1) to execute them"
    )
    for item in items:
        if "downstream" in item.keywords:
            item.add_marker(skip)
