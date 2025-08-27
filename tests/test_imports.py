import warnings
from enum import Enum
from functools import _lru_cache_wrapper
from typing import Callable
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from mistral_common.imports import (
    assert_hf_hub_installed,
    assert_opencv_installed,
    assert_package_installed,
    assert_sentencepiece_installed,
    assert_soundfile_installed,
    assert_soxr_installed,
    create_deprecate_cls_import,
    is_hf_hub_installed,
    is_opencv_installed,
    is_package_installed,
    is_sentencepiece_installed,
    is_soundfile_installed,
    is_soxr_installed,
)

_IS_INSTALLED_TO_TESTS = [
    is_hf_hub_installed,
    is_sentencepiece_installed,
    is_soundfile_installed,
    is_soxr_installed,
]

_ASSERT_TO_TESTS = {
    (
        is_hf_hub_installed,
        assert_hf_hub_installed,
        "`huggingface_hub` is not installed. Please install it with `pip install mistral-common[hf-hub]`",
    ),
    (
        is_opencv_installed,
        assert_opencv_installed,
        "`opencv` is not installed. Please install it with `pip install mistral-common[opencv]`",
    ),
    (
        is_sentencepiece_installed,
        assert_sentencepiece_installed,
        "`sentencepiece` is not installed. Please install it with `pip install mistral-common[sentencepiece]`",
    ),
    (
        is_soundfile_installed,
        assert_soundfile_installed,
        "`soundfile` is not installed. Please install it with `pip install mistral-common[soundfile]`",
    ),
    (
        is_soxr_installed,
        assert_soxr_installed,
        "`soxr` is not installed. Please install it with `pip install mistral-common[soxr]`",
    ),
}


@patch("importlib.util.find_spec")
def test_is_package_installed(mock_find_spec: MagicMock) -> None:
    mock_find_spec.return_value = True
    assert is_package_installed("package_name") is True

    mock_find_spec.return_value = None
    assert is_package_installed("package_name") is False


@patch("mistral_common.imports.is_package_installed")
def test_assert_package_installed(mock_is_package_installed: MagicMock) -> None:
    mock_is_package_installed.return_value = True
    assert_package_installed("package_name")

    mock_is_package_installed.return_value = False
    with pytest.raises(ImportError):
        assert_package_installed("package_name")


def test_is_opencv_installed() -> None:
    is_opencv_installed.cache_clear()

    with patch.dict("sys.modules", {"cv2": Mock()}):
        assert is_opencv_installed() is True
    is_opencv_installed.cache_clear()
    # TODO(Julien): Find a way to mock import for testing wrong import


@patch("mistral_common.imports.is_package_installed")
@pytest.mark.parametrize("is_installed_fn", _IS_INSTALLED_TO_TESTS)
def test_is_installed(mock_is_package_installed: MagicMock, is_installed_fn: _lru_cache_wrapper) -> None:
    is_installed_fn.cache_clear()

    mock_is_package_installed.return_value = True
    assert is_installed_fn() is True
    is_installed_fn.cache_clear()

    mock_is_package_installed.return_value = False
    assert is_installed_fn() is False
    is_installed_fn.cache_clear()


@patch("mistral_common.imports.is_package_installed")
@pytest.mark.parametrize("is_installed_fn, assert_fn, error_message", _ASSERT_TO_TESTS)
def test_assert_installed(
    mock_is_package_installed: MagicMock,
    is_installed_fn: _lru_cache_wrapper,
    assert_fn: Callable[[], None],
    error_message: str,
) -> None:
    is_installed_fn.cache_clear()
    mock_is_package_installed.return_value = True
    assert_fn()
    is_installed_fn.cache_clear()

    mock_is_package_installed.return_value = False
    with pytest.raises(ImportError) as exc_info:
        assert_fn()
    assert str(exc_info.value) == error_message
    is_installed_fn.cache_clear()


class TestCreateDeprecateClsImport(TestCase):
    def test_pydantic_model(self) -> None:
        class TestA(BaseModel):
            field1: int = 0
            field2: str

            def field1_to_str(self) -> str:
                return str(self.field1)

        class TestB(BaseModel):
            field1: int = 0
            field2: str

        DeprecatedTestA = create_deprecate_cls_import(TestA, "prev_location.package.module", __name__, "1.5.0")
        DeprecatedTestB = create_deprecate_cls_import(TestB, "prev_location.package.module", __name__, "1.6.0")  # noqa: F841

        for i in range(2):
            assert warnings.filters[i][0] == "once"
            regex_compiled = warnings.filters[i][1]
            assert regex_compiled is not None
            assert regex_compiled.pattern == (
                f"Test{'A' if i else 'B'} has moved to tests.test_imports. It will be removed in "
                f"prev_location.package.module in {'1.5.0' if i else '1.6.0'}."
            )
            assert warnings.filters[i][2] is FutureWarning

        assert issubclass(DeprecatedTestA, TestA)
        assert set(DeprecatedTestA.model_fields) == set(TestA.model_fields)

        with self.assertWarns(FutureWarning) as cm:
            instance = DeprecatedTestA(field2="A")
        assert instance.field1_to_str() == "0"
        assert isinstance(cm.warning, FutureWarning)
        assert cm.warning.args[0] == (
            "TestA has moved to tests.test_imports. It will be removed in prev_location.package.module in 1.5.0."
        )

    def test_enum(self) -> None:
        class TestEnumA(Enum):
            test_1 = 0
            test_2 = 1

        class TestEnumB(Enum):
            test_1 = 2
            test_2 = 3

        DeprecatedTestEnumA = create_deprecate_cls_import(TestEnumA, "prev_location.package.module", __name__, "1.5.0")
        DeprecatedTestEnumB = create_deprecate_cls_import(TestEnumB, "prev_location.package.module", __name__, "1.6.0")  # noqa: F841

        for i in range(2):
            assert warnings.filters[i][0] == "once"
            regex_compiled = warnings.filters[i][1]
            assert regex_compiled is not None
            assert regex_compiled.pattern == (
                f"TestEnum{'A' if i else 'B'} has moved to tests.test_imports. It will be removed in "
                f"prev_location.package.module in {'1.5.0' if i else '1.6.0'}."
            )
            assert warnings.filters[i][2] is FutureWarning

        assert set(DeprecatedTestEnumA.__members__) == set(TestEnumA.__members__)

        with self.assertWarns(FutureWarning) as cm:
            assert hasattr(DeprecatedTestEnumA, "test_1")
            DeprecatedTestEnumA.test_1.value
        assert isinstance(cm.warning, FutureWarning)
        assert cm.warning.args[0] == (
            "TestEnumA has moved to tests.test_imports. It will be removed in prev_location.package.module in 1.5.0."
        )
