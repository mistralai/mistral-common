import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOP_LEVEL_FIELDS = {
    "spec_id",
    "contract_revision",
    "task_base_sha",
    "changed_paths",
    "source_selectors",
    "callable_rows",
    "legacy_selectors",
    "legacy_rows",
    "helper_rows",
    "blocker_records",
    "checks",
}
TASK_BASE_SHA = "2c30e17cbfe20a405b87522b1c1d51193dafe0ef"
SOURCE_SELECTOR_FIELDS = {"path", "include_symbols", "exclude_symbols", "uniform_disposition"}
LEGACY_SELECTOR_FIELDS = {"path", "node_pattern", "param_matrix", "destination", "disposition"}
CALLABLE_ROW_FIELDS = {"path", "symbol", "disposition", "evidence"}
LEGACY_ROW_FIELDS = {"path", "node", "disposition", "evidence"}
HELPER_ROW_FIELDS = {"source_symbol", "destination_symbol", "consumer_task_ids", "deletion_gate_task_ids"}
CHECK_FIELDS = {"command", "exit_code", "stdout_sha256"}
BLOCKER_FIELDS = {
    "blocker_id",
    "discovering_task_id",
    "discovering_base_sha",
    "affected_invariant_ids",
    "affected_scenario_ids",
    "source_paths",
    "source_symbols",
    "reproducer",
    "observed_result",
    "user_confirmed_intended_contract",
    "production_spec",
    "production_branch",
    "production_pr",
    "dependent_task_ids",
    "status",
    "merge_sha",
    "refreshed_base_sha",
    "passing_resumed_reproducer",
}


class AccountabilityError(ValueError):
    """Raised when an accountability record is incomplete or inconsistent."""


@dataclass(frozen=True)
class VerificationReport:
    """The result of a complete accountability verification."""

    valid: bool
    checked_callables: tuple[tuple[str, str], ...] = ()
    checked_nodes: tuple[tuple[str, str], ...] = ()


def _read_tree(path: str) -> ast.Module:
    try:
        return ast.parse(Path(path).read_text(), filename=path)
    except (OSError, SyntaxError) as error:
        raise AccountabilityError(f"cannot inspect {path}: {error}") from error


def discover_callables(paths: list[str]) -> set[tuple[str, str]]:
    """Return every source class, function, and method in the selected files."""
    found: set[tuple[str, str]] = set()

    def visit(nodes: list[ast.stmt], path: str, prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                symbol = f"{prefix}.{node.name}" if prefix else node.name
                found.add((path, symbol))
                visit(node.body, path, symbol)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = f"{prefix}.{node.name}" if prefix else node.name
                found.add((path, symbol))
                visit(node.body, path, symbol)

    for path in paths:
        visit(_read_tree(path).body, path)
    return found


def discover_legacy_nodes(paths: list[str]) -> set[tuple[str, str]]:
    """Return every collected-style test node, including parameter cases."""
    return {(path, node_id) for path in paths for node_id, _, _ in _discover_legacy_cases(path)}


def _resolve_expression(node: ast.expr, bindings: dict[str, ast.expr]) -> ast.expr:
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    return node


def _module_bindings(tree: ast.Module) -> dict[str, ast.expr]:
    bindings: dict[str, ast.expr] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.expr):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.value, ast.expr)
        ):
            bindings[statement.target.id] = statement.value
    return bindings


def _parameter_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [name.strip() for name in node.value.split(",")]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [item.value for item in node.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)]
    return []


def _value_id(node: ast.expr, bindings: dict[str, ast.expr]) -> str:
    resolved = _resolve_expression(node, bindings)
    if isinstance(resolved, ast.Name):
        return resolved.id
    if isinstance(resolved, ast.Constant):
        return str(resolved.value)
    return ast.unparse(resolved)


def _value_data(node: ast.expr, bindings: dict[str, ast.expr]) -> object:
    resolved = _resolve_expression(node, bindings)
    if isinstance(resolved, ast.Name):
        return resolved.id
    if isinstance(resolved, ast.Constant):
        return resolved.value
    try:
        return ast.literal_eval(resolved)
    except ValueError:
        return ast.unparse(resolved)


def _discover_legacy_cases(path: str) -> list[tuple[str, str, dict[str, object]]]:
    tree = _read_tree(path)
    bindings = _module_bindings(tree)
    cases: list[tuple[str, str, dict[str, object]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
            continue
        decorators = [
            decorator
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
        ]
        if not decorators:
            cases.append((node.name, "default", {}))
            continue
        decorator = decorators[0]
        if len(decorator.args) < 2:
            raise AccountabilityError(f"malformed parametrize decorator on {path}::{node.name}")
        names = _parameter_names(decorator.args[0])
        raw_values = _resolve_expression(decorator.args[1], bindings)
        if not isinstance(raw_values, (ast.List, ast.Tuple)):
            raise AccountabilityError(f"malformed parametrize values on {path}::{node.name}")
        explicit_ids: list[str] | None = None
        for keyword in decorator.keywords:
            if keyword.arg == "ids":
                ids_node = _resolve_expression(keyword.value, bindings)
                if isinstance(ids_node, (ast.List, ast.Tuple)):
                    explicit_ids = [str(_value_data(item, bindings)) for item in ids_node.elts]
        for index, raw_case in enumerate(raw_values.elts):
            resolved_case = _resolve_expression(raw_case, bindings)
            if len(names) > 1 and isinstance(resolved_case, ast.Tuple):
                values = [item for item in resolved_case.elts]
            else:
                values = [resolved_case]
            value_map = {name: _value_data(value, bindings) for name, value in zip(names, values, strict=False)}
            case_id = (
                explicit_ids[index]
                if explicit_ids is not None
                else "-".join(_value_id(value, bindings) for value in values)
            )
            cases.append((f"{node.name}[{case_id}]", case_id, value_map))
    return cases


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AccountabilityError(f"{label} must be an object")
    return value


def _check_fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise AccountabilityError(f"unknown {label} fields: {', '.join(sorted(unknown))}")


def _check_record_shape(record: dict[str, Any]) -> None:
    _check_fields(record, TOP_LEVEL_FIELDS, "record")
    missing = TOP_LEVEL_FIELDS - set(record)
    if missing:
        raise AccountabilityError(f"missing record fields: {', '.join(sorted(missing))}")
    if record["task_base_sha"] != TASK_BASE_SHA:
        raise AccountabilityError(f"task_base_sha must equal {TASK_BASE_SHA}")
    list_fields = (
        "changed_paths",
        "source_selectors",
        "callable_rows",
        "legacy_selectors",
        "legacy_rows",
        "helper_rows",
        "blocker_records",
        "checks",
    )
    for field in list_fields:
        if not isinstance(record[field], list):
            raise AccountabilityError(f"{field} must be a list")
    if record["changed_paths"] != sorted(record["changed_paths"]):
        raise AccountabilityError("changed_paths must be sorted")
    for index, selector in enumerate(record["source_selectors"]):
        item = _require_mapping(selector, f"source selector {index}")
        _check_fields(item, SOURCE_SELECTOR_FIELDS, "source selector")
        if set(item) != SOURCE_SELECTOR_FIELDS:
            raise AccountabilityError("source selector has missing fields")
        if not isinstance(item["include_symbols"], list) or not isinstance(item["exclude_symbols"], list):
            raise AccountabilityError("source selector symbol lists must be lists")
    for index, selector in enumerate(record["legacy_selectors"]):
        item = _require_mapping(selector, f"legacy selector {index}")
        _check_fields(item, LEGACY_SELECTOR_FIELDS, "legacy selector")
        if set(item) != LEGACY_SELECTOR_FIELDS:
            raise AccountabilityError("legacy selector has missing fields")
        if not isinstance(item["param_matrix"], list):
            raise AccountabilityError("legacy selector parameter matrix must be a list")
        for parameter in item["param_matrix"]:
            parameter_item = _require_mapping(parameter, "legacy parameter")
            if set(parameter_item) != {"id", "values"} or not isinstance(parameter_item["values"], dict):
                raise AccountabilityError("malformed legacy parameter matrix")
    for index, row in enumerate(record["callable_rows"]):
        item = _require_mapping(row, f"callable row {index}")
        _check_fields(item, CALLABLE_ROW_FIELDS, "callable")
        if set(item) != CALLABLE_ROW_FIELDS:
            raise AccountabilityError("missing callable evidence")
    for index, row in enumerate(record["legacy_rows"]):
        item = _require_mapping(row, f"legacy row {index}")
        _check_fields(item, LEGACY_ROW_FIELDS, "legacy")
        if set(item) != LEGACY_ROW_FIELDS:
            raise AccountabilityError("missing legacy evidence")
    for index, row in enumerate(record["helper_rows"]):
        item = _require_mapping(row, f"helper row {index}")
        _check_fields(item, HELPER_ROW_FIELDS, "helper")
        if set(item) != HELPER_ROW_FIELDS:
            raise AccountabilityError("helper row has missing fields")
    for index, check in enumerate(record["checks"]):
        item = _require_mapping(check, f"check {index}")
        _check_fields(item, CHECK_FIELDS, "check")
        if set(item) != CHECK_FIELDS:
            raise AccountabilityError("check has missing fields")
    for index, blocker in enumerate(record["blocker_records"]):
        item = _require_mapping(blocker, f"blocker record {index}")
        _check_fields(item, BLOCKER_FIELDS, "blocker")
        if set(item) != BLOCKER_FIELDS:
            raise AccountabilityError("blocker record has missing fields")


def _verify_callable_rows(record: dict[str, Any], expected: set[tuple[str, str]]) -> None:
    rows = record["callable_rows"]
    actual: list[tuple[str, str]] = []
    for row in rows:
        if set(row) - {"path", "symbol", "disposition", "evidence"} or "path" not in row or "symbol" not in row:
            raise AccountabilityError("malformed callable row")
        key = (row["path"], row["symbol"])
        if key in actual:
            raise AccountabilityError(f"duplicate callable row: {key[1]}")
        actual.append(key)
    actual_set = set(actual)
    stale = actual_set - expected
    if stale:
        raise AccountabilityError(f"stale callable row: {next(iter(stale))[1]}")
    missing = expected - actual_set
    if missing:
        raise AccountabilityError(f"missing callable rows: {next(iter(missing))[1]}")


def verify_pr_records(record: dict[str, Any], *, source_paths: list[str]) -> VerificationReport:
    """Verify the PR callable table against the Base-SHA source inventory."""
    _check_record_shape(record)
    expected = discover_callables(source_paths)
    _verify_callable_rows(record=record, expected=expected)
    return VerificationReport(valid=True, checked_callables=tuple(sorted(expected)))


def verify_task(
    record: dict[str, Any],
    *,
    source_paths: list[str],
    legacy_paths: list[str],
) -> VerificationReport:
    """Verify source selectors, legacy node selectors, and their evidence rows."""
    _check_record_shape(record)
    expected_callables = discover_callables(source_paths)
    owned: dict[tuple[str, str], int] = {key: 0 for key in expected_callables}
    source_path_set = set(source_paths)
    for selector in record["source_selectors"]:
        path = selector["path"]
        if path not in source_path_set:
            raise AccountabilityError(f"stale source selector path: {path}")
        available = {symbol for candidate_path, symbol in expected_callables if candidate_path == path}
        for symbol in selector["include_symbols"]:
            if symbol not in available:
                raise AccountabilityError(f"stale source symbol: {symbol}")
            owned[(path, symbol)] += 1
        for symbol in selector["exclude_symbols"]:
            if symbol not in available:
                raise AccountabilityError(f"stale excluded source symbol: {symbol}")
    duplicates = [key for key, count in owned.items() if count > 1]
    if duplicates:
        raise AccountabilityError(f"overlapping source selectors: {duplicates[0][1]}")
    missing = [key for key, count in owned.items() if count == 0]
    if missing:
        raise AccountabilityError(f"unmatched callable: {missing[0][1]}")

    legacy_cases = [
        (path, node_id, parameter_id, values)
        for path in legacy_paths
        for node_id, parameter_id, values in _discover_legacy_cases(path)
    ]
    expected_nodes = {(path, node_id) for path, node_id, _, _ in legacy_cases}
    node_owned: dict[tuple[str, str], int] = {key: 0 for key in expected_nodes}
    legacy_path_set = set(legacy_paths)
    for selector in record["legacy_selectors"]:
        path = selector["path"]
        if path not in legacy_path_set:
            raise AccountabilityError(f"stale legacy selector path: {path}")
        matched = {
            key for key in expected_nodes if key[0] == path and re.search(selector["node_pattern"], key[1]) is not None
        }
        if not matched:
            raise AccountabilityError(f"stale legacy selector: {selector['node_pattern']}")
        matrix = {parameter["id"]: parameter["values"] for parameter in selector["param_matrix"]}
        expected_matrix = {
            parameter_id: values
            for candidate_path, node_id, parameter_id, values in legacy_cases
            if candidate_path == path and (candidate_path, node_id) in matched
        }
        if matrix != expected_matrix:
            raise AccountabilityError(f"param matrix mismatch: {selector['node_pattern']}")
        for key in matched:
            node_owned[key] += 1
    duplicates = [key for key, count in node_owned.items() if count > 1]
    if duplicates:
        raise AccountabilityError(f"overlapping legacy selectors: {duplicates[0][1]}")
    missing = [key for key, count in node_owned.items() if count == 0]
    if missing:
        raise AccountabilityError(f"unmatched legacy node: {missing[0][1]}")
    rows: list[tuple[str, str]] = []
    for row in record["legacy_rows"]:
        if set(row) - LEGACY_ROW_FIELDS or "path" not in row or "node" not in row:
            raise AccountabilityError("malformed legacy row")
        key = (row["path"], row["node"])
        if key in rows:
            raise AccountabilityError(f"duplicate legacy row: {key[1]}")
        rows.append(key)
    actual_rows = set(rows)
    stale_rows = actual_rows - expected_nodes
    if stale_rows:
        raise AccountabilityError(f"stale legacy row: {next(iter(stale_rows))[1]}")
    missing_rows = expected_nodes - actual_rows
    if missing_rows:
        raise AccountabilityError(f"missing legacy rows: {next(iter(missing_rows))[1]}")
    return VerificationReport(
        valid=True,
        checked_callables=tuple(sorted(expected_callables)),
        checked_nodes=tuple(sorted(expected_nodes)),
    )


def build_record(*, source_paths: list[str], legacy_paths: list[str]) -> dict[str, Any]:
    """Build a complete evidence record for tests and local inventory review."""
    callables = sorted(discover_callables(source_paths))
    legacy_cases = [
        (path, node_id, parameter_id, values)
        for path in legacy_paths
        for node_id, parameter_id, values in _discover_legacy_cases(path)
    ]
    selectors = []
    for path in source_paths:
        symbols = [symbol for candidate_path, symbol in callables if candidate_path == path]
        selectors.append(
            {
                "path": path,
                "include_symbols": symbols,
                "exclude_symbols": [],
                "uniform_disposition": "migrated",
            }
        )
    legacy_selectors = [
        {
            "path": path,
            "node_pattern": f"^{re.escape(node_id)}$",
            "param_matrix": [{"id": parameter_id, "values": values}],
            "destination": "tests/unit/core",
            "disposition": "migrated",
        }
        for path, node_id, parameter_id, values in legacy_cases
    ]
    return {
        "spec_id": "TASK-1",
        "contract_revision": 21,
        "task_base_sha": TASK_BASE_SHA,
        "changed_paths": sorted([*source_paths, *legacy_paths]),
        "source_selectors": selectors,
        "callable_rows": [
            {"path": path, "symbol": symbol, "disposition": "migrated", "evidence": "unit test"}
            for path, symbol in callables
        ],
        "legacy_selectors": legacy_selectors,
        "legacy_rows": [
            {"path": path, "node": node_id, "disposition": "migrated", "evidence": "unit test"}
            for path, node_id, _, _ in legacy_cases
        ],
        "helper_rows": [],
        "blocker_records": [],
        "checks": [],
    }
