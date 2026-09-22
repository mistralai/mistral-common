from pathlib import Path
from typing import Any, Callable

import pytest

from tests.helpers.accountability import (
    AccountabilityError,
    build_record,
    discover_callables,
    discover_legacy_nodes,
    verify_pr_records,
    verify_task,
)
from tests.unit.core.dispositions import LEGACY_NODE_DISPOSITIONS, SOURCE_DISPOSITIONS

OWNED_SOURCE_PATHS = list(SOURCE_DISPOSITIONS)


def test_callable_records_reject_malformed_missing_duplicate_and_stale_rows(tmp_path: Path) -> None:
    source = tmp_path / "base.py"
    source.write_text("class Model:\n    def run(self) -> None:\n        pass\n\ndef helper() -> None:\n    pass\n")
    record = build_record(source_paths=[str(source)], legacy_paths=[])
    record["callable_rows"] = [{"path": str(source), "symbol": "helper"}]
    with pytest.raises(AccountabilityError, match="missing callable evidence"):
        verify_pr_records(record, source_paths=[str(source)])

    record = build_record(source_paths=[str(source)], legacy_paths=[])
    record["callable_rows"].append(record["callable_rows"][0].copy())
    with pytest.raises(AccountabilityError, match="duplicate callable"):
        verify_pr_records(record, source_paths=[str(source)])

    record = build_record(source_paths=[str(source)], legacy_paths=[])
    record["callable_rows"][0]["symbol"] = "not_present"
    with pytest.raises(AccountabilityError, match="stale callable"):
        verify_pr_records(record, source_paths=[str(source)])

    record = build_record(source_paths=[str(source)], legacy_paths=[])
    record["callable_rows"][0]["unexpected"] = True
    with pytest.raises(AccountabilityError, match="unknown callable fields"):
        verify_pr_records(record, source_paths=[str(source)])


def test_task_verification_rejects_missing_duplicate_and_stale_selector_rows(tmp_path: Path) -> None:
    source = tmp_path / "base.py"
    source.write_text("def helper() -> None:\n    pass\n")
    legacy = tmp_path / "test_base.py"
    legacy.write_text("def test_helper() -> None:\n    pass\n")
    record = build_record(source_paths=[str(source)], legacy_paths=[str(legacy)])
    record["source_selectors"] = []
    with pytest.raises(AccountabilityError, match="unmatched callable"):
        verify_task(record, source_paths=[str(source)], legacy_paths=[str(legacy)])

    record = build_record(source_paths=[str(source)], legacy_paths=[str(legacy)])
    record["source_selectors"].append(record["source_selectors"][0].copy())
    with pytest.raises(AccountabilityError, match="overlapping source selectors"):
        verify_task(record, source_paths=[str(source)], legacy_paths=[str(legacy)])

    record = build_record(source_paths=[str(source)], legacy_paths=[str(legacy)])
    record["source_selectors"][0]["include_symbols"] = ["old_helper"]
    with pytest.raises(AccountabilityError, match="stale source symbol"):
        verify_task(record, source_paths=[str(source)], legacy_paths=[str(legacy)])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda record: record["legacy_rows"].clear(), "missing legacy rows"),
        (lambda record: record["legacy_rows"].append(record["legacy_rows"][0].copy()), "duplicate legacy row"),
        (lambda record: record["legacy_rows"][0].update(node="stale"), "stale legacy row"),
        (lambda record: record["legacy_rows"][0].update(extra=True), "unknown legacy fields"),
    ],
)
def test_task_verification_rejects_malformed_missing_duplicate_and_stale_node_rows(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    source = tmp_path / "base.py"
    source.write_text("def helper() -> None:\n    pass\n")
    legacy = tmp_path / "test_base.py"
    legacy.write_text("def test_helper() -> None:\n    pass\n")
    record = build_record(source_paths=[str(source)], legacy_paths=[str(legacy)])
    change(record)
    with pytest.raises(AccountabilityError, match=message):
        verify_task(record, source_paths=[str(source)], legacy_paths=[str(legacy)])


def test_task_and_pr_verification_success_paths(tmp_path: Path) -> None:
    source = tmp_path / "base.py"
    source.write_text("def helper() -> None:\n    pass\n")
    legacy = tmp_path / "test_base.py"
    legacy.write_text("def test_helper() -> None:\n    pass\n")
    record = build_record(source_paths=[str(source)], legacy_paths=[str(legacy)])
    assert discover_callables([str(source)])
    assert verify_task(record, source_paths=[str(source)], legacy_paths=[str(legacy)]).valid
    assert verify_pr_records(record, source_paths=[str(source)]).valid


def test_disposition_table_covers_each_owned_callable_once() -> None:
    expected = {(path, symbol) for path, symbols in SOURCE_DISPOSITIONS.items() for symbol in symbols}
    record = build_record(source_paths=OWNED_SOURCE_PATHS, legacy_paths=[])
    assert set((row["path"], row["symbol"]) for row in record["callable_rows"]) == expected
    assert verify_pr_records(record, source_paths=OWNED_SOURCE_PATHS).valid


def test_accountability_verification_rejects_wrong_task_base_sha(tmp_path: Path) -> None:
    source = tmp_path / "base.py"
    source.write_text("def helper() -> None:\n    pass\n")
    record = build_record(source_paths=[str(source)], legacy_paths=[])
    record["task_base_sha"] = "wrong"
    with pytest.raises(AccountabilityError, match="task_base_sha"):
        verify_pr_records(record, source_paths=[str(source)])


def test_legacy_disposition_table_covers_all_collected_nodes() -> None:
    assert sum(len(nodes) for nodes in LEGACY_NODE_DISPOSITIONS.values()) == 29
    assert sum("[" in node for nodes in LEGACY_NODE_DISPOSITIONS.values() for node in nodes) == 13
    assert all(not node.endswith("],") for nodes in LEGACY_NODE_DISPOSITIONS.values() for node in nodes)


def test_legacy_discovery_includes_each_parametrized_collected_node(tmp_path: Path) -> None:
    legacy = tmp_path / "test_legacy.py"
    legacy.write_text(
        "import pytest\n\n"
        'VALUES = ["first", "second"]\n\n'
        '@pytest.mark.parametrize("value", VALUES)\n'
        "def test_parameterized(value: str) -> None:\n"
        "    pass\n"
    )
    nodes = discover_legacy_nodes([str(legacy)])
    assert nodes == {
        (str(legacy), "test_parameterized[first]"),
        (str(legacy), "test_parameterized[second]"),
    }
    record = build_record(source_paths=[], legacy_paths=[str(legacy)])
    assert {selector["param_matrix"][0]["id"] for selector in record["legacy_selectors"]} == {"first", "second"}
    assert verify_task(record, source_paths=[], legacy_paths=[str(legacy)]).valid


def test_task_verification_rejects_incorrect_parametrization_evidence(tmp_path: Path) -> None:
    legacy = tmp_path / "test_legacy.py"
    legacy.write_text(
        "import pytest\n\n"
        '@pytest.mark.parametrize("value", ["first"])\n'
        "def test_parameterized(value: str) -> None:\n"
        "    pass\n"
    )
    record = build_record(source_paths=[], legacy_paths=[str(legacy)])
    record["legacy_selectors"][0]["param_matrix"][0]["values"] = {"value": "wrong"}
    with pytest.raises(AccountabilityError, match="param matrix"):
        verify_task(record, source_paths=[], legacy_paths=[str(legacy)])
