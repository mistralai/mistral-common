import copy
import pickle
from typing import Callable

import pytest

from mistral_common import exceptions
from mistral_common.exceptions import InvalidRequestException, MistralCommonException


def test_mistral_common_exception_uses_default_message() -> None:
    error = MistralCommonException()

    assert error.message == "Internal server error"


def test_mistral_common_exception_uses_custom_message() -> None:
    error = MistralCommonException(message="custom error")

    assert error.message == "custom error"


def test_subclass_uses_custom_message() -> None:
    error = InvalidRequestException(message="invalid request")

    assert error.message == "invalid request"


EXCEPTION_CLASSES = [
    obj for obj in vars(exceptions).values() if isinstance(obj, type) and issubclass(obj, MistralCommonException)
]


@pytest.mark.parametrize("cls", EXCEPTION_CLASSES)
def test_keyword_and_positional_message_are_equivalent(cls: type[MistralCommonException]) -> None:
    by_keyword = cls(message="boom")
    by_position = cls("boom")

    assert str(by_keyword) == str(by_position) == "boom"
    assert by_keyword.args == by_position.args == ("boom",)


@pytest.mark.parametrize("cls", EXCEPTION_CLASSES)
@pytest.mark.parametrize("by_keyword", [True, False])
@pytest.mark.parametrize(
    "round_trip",
    [lambda e: pickle.loads(pickle.dumps(e)), copy.copy, copy.deepcopy],
    ids=["pickle", "copy", "deepcopy"],
)
def test_exception_round_trip(
    cls: type[MistralCommonException], by_keyword: bool, round_trip: Callable[[Exception], Exception]
) -> None:
    error = cls(message="boom") if by_keyword else cls("boom")

    restored = round_trip(error)

    assert type(restored) is cls
    assert isinstance(restored, MistralCommonException)
    assert restored.message == "boom"
    assert str(restored) == "boom"


def test_default_message_is_the_exception_string() -> None:
    error = MistralCommonException()

    assert str(error) == "Internal server error"
    assert pickle.loads(pickle.dumps(error)).message == "Internal server error"
