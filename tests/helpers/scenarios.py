import re
import warnings
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)
FactoryOutputT_co = TypeVar("FactoryOutputT_co", covariant=True)
_MISSING = object()


class ScenarioAssertionError(AssertionError):
    """Raised when a scenario does not produce its frozen expected evidence."""


class UnsupportedExpectationError(ScenarioAssertionError):
    """Raised when the runner receives an expectation it cannot dispatch."""


class ScenarioOperation(Protocol[InputT_contra, OutputT_co]):
    def __call__(self, value: InputT_contra) -> OutputT_co: ...


class InputFactory(Protocol[FactoryOutputT_co]):
    def __call__(self) -> FactoryOutputT_co: ...


@dataclass(frozen=True)
class Expectation:
    """One result/error outcome and an optional warning contract."""

    expected_value: object = _MISSING
    expected_exception: type[BaseException] | None = None
    error_match: str | None = None
    warning: type[Warning] | None = None
    warning_match: str | None = None

    def __post_init__(self) -> None:
        outcomes = (self.expected_value is not _MISSING, self.expected_exception is not None)
        if sum(outcomes) != 1:
            raise ValueError("expectation requires exactly one outcome")
        if self.error_match is not None and self.expected_exception is None:
            raise ValueError("error matcher requires an expected exception")
        if self.warning_match is not None and self.warning is None:
            raise ValueError("warning matcher requires an expected warning")

    @classmethod
    def result(
        cls,
        value: object,
        *,
        warning: type[Warning] | None = None,
        warning_match: str | None = None,
    ) -> "Expectation":
        return cls(expected_value=value, warning=warning, warning_match=warning_match)

    @classmethod
    def error(cls, exception: type[BaseException], *, match: str | None = None) -> "Expectation":
        return cls(expected_exception=exception, error_match=match)


@dataclass(frozen=True)
class Scenario(Generic[InputT, OutputT]):
    """A named operation with a fresh-input factory and frozen evidence."""

    scenario_id: str
    input_factory: InputFactory[InputT]
    operation: ScenarioOperation[InputT, OutputT]
    expectation: Expectation

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        if not callable(self.input_factory) or not callable(self.operation):
            raise TypeError("scenario factory and operation must be callable")


class ScenarioRunner:
    """Execute scenarios while checking their result, error, and warning evidence."""

    def run(self, scenario: Scenario[InputT, OutputT]) -> OutputT | None:
        expectation = scenario.expectation
        if not isinstance(expectation, Expectation):
            raise UnsupportedExpectationError(f"unsupported expectation type: {type(expectation).__name__}")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                result = scenario.operation(scenario.input_factory())
            except BaseException as error:
                self._check_error(scenario.scenario_id, expectation, error)
                return None

        self._check_warnings(scenario.scenario_id, expectation, captured)
        if expectation.expected_exception is not None:
            raise ScenarioAssertionError(
                f"{scenario.scenario_id}: expected {expectation.expected_exception.__name__}, got a result"
            )
        if result != expectation.expected_value:
            raise ScenarioAssertionError(
                f"{scenario.scenario_id}: result {result!r} does not match {expectation.expected_value!r}"
            )
        return result

    @staticmethod
    def _check_error(scenario_id: str, expectation: Expectation, error: BaseException) -> None:
        expected = expectation.expected_exception
        if expected is None:
            raise ScenarioAssertionError(f"{scenario_id}: unexpected {type(error).__name__}") from error
        if not isinstance(error, expected):
            raise ScenarioAssertionError(
                f"{scenario_id}: expected {expected.__name__}, got {type(error).__name__}"
            ) from error
        if expectation.error_match is not None and re.search(expectation.error_match, str(error)) is None:
            raise ScenarioAssertionError(
                f"{scenario_id}: error {error!r} does not match {expectation.error_match!r}"
            ) from error

    @staticmethod
    def _check_warnings(
        scenario_id: str,
        expectation: Expectation,
        captured: list[warnings.WarningMessage],
    ) -> None:
        if expectation.warning is None:
            if captured:
                raise ScenarioAssertionError(f"{scenario_id}: unexpected warning {captured[0].message}")
            return
        matching = [item for item in captured if issubclass(item.category, expectation.warning)]
        if not matching:
            raise ScenarioAssertionError(f"{scenario_id}: expected {expectation.warning.__name__} warning")
        if expectation.warning_match is not None and all(
            re.search(expectation.warning_match, str(item.message)) is None for item in matching
        ):
            raise ScenarioAssertionError(f"{scenario_id}: warning does not match {expectation.warning_match!r}")
