from functools import _lru_cache_wrapper
from typing import Callable
from unittest.mock import MagicMock, Mock, patch

import pytest

from mistral_common.imports import (
    assert_hf_hub_installed,
    assert_opencv_installed,
    assert_package_installed,
    assert_sentencepiece_installed,
    assert_soundfile_installed,
    assert_soxr_installed,
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
