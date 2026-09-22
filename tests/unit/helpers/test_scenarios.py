import warnings
from typing import cast

import pytest

from tests.helpers.scenarios import (
    Expectation,
    Scenario,
    ScenarioAssertionError,
    ScenarioRunner,
    UnsupportedExpectationError,
)


def test_expectations_are_frozen_and_require_one_outcome() -> None:
    expectation = Expectation.result({"answer": 1})
    with pytest.raises(AttributeError):
        expectation.value = 2  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="exactly one outcome"):
        Expectation()
    with pytest.raises(ValueError, match="exactly one outcome"):
        Expectation(expected_value=1, expected_exception=ValueError)
    with pytest.raises(ValueError, match="warning matcher requires"):
        Expectation(expected_value=1, warning_match="warning")


def test_runner_accepts_result_error_and_warning_scenarios() -> None:
    runner = ScenarioRunner()
    assert runner.run(Scenario("result", lambda: {"n": 1}, lambda value: value["n"], Expectation.result(1))) == 1
    assert (
        runner.run(
            Scenario(
                "error",
                lambda: "input",
                lambda value: (_ for _ in ()).throw(ValueError(f"bad {value}")),
                Expectation.error(ValueError, match="bad input"),
            )
        )
        is None
    )
    assert (
        runner.run(
            Scenario(
                "warning",
                lambda: "input",
                lambda value: (warnings.warn("expected warning", UserWarning), value)[1],
                Expectation.result("input", warning=UserWarning, warning_match="expected"),
            )
        )
        == "input"
    )


def test_runner_rejects_wrong_result_and_unexpected_errors() -> None:
    runner = ScenarioRunner()
    with pytest.raises(ScenarioAssertionError, match="result"):
        runner.run(Scenario("wrong", lambda: None, lambda value: 2, Expectation.result(1)))
    with pytest.raises(ScenarioAssertionError, match="expected ValueError"):
        runner.run(Scenario("wrong-error", lambda: None, lambda value: 1, Expectation.error(ValueError)))
    with pytest.raises(ScenarioAssertionError, match="unexpected ZeroDivisionError"):
        runner.run(Scenario("unexpected", lambda: None, lambda value: 1 / 0, Expectation.result(1)))


def test_runner_dispatch_is_exhaustive() -> None:
    class UnknownExpectation:
        pass

    with pytest.raises(UnsupportedExpectationError, match="UnknownExpectation"):
        ScenarioRunner().run(
            Scenario("unknown", lambda: None, lambda value: None, cast(Expectation, UnknownExpectation()))
        )


def test_runner_uses_fresh_input_for_each_run() -> None:
    seen: list[dict[str, int]] = []

    def make_input() -> dict[str, int]:
        return {"count": 0}

    def mutate(value: dict[str, int]) -> int:
        seen.append(value)
        value["count"] += 1
        return value["count"]

    scenario = Scenario("isolated", make_input, mutate, Expectation.result(1))
    runner = ScenarioRunner()
    assert runner.run(scenario) == 1
    assert runner.run(scenario) == 1
    assert seen[0] is not seen[1]
