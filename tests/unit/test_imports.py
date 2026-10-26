import builtins
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from mistral_common.imports import (
    assert_hf_hub_installed,
    assert_jinja2_installed,
    assert_llguidance_installed,
    assert_opencv_installed,
    assert_package_installed,
    assert_sentencepiece_installed,
    assert_soundfile_installed,
    assert_soxr_installed,
    is_hf_hub_installed,
    is_jinja2_installed,
    is_llguidance_installed,
    is_opencv_installed,
    is_package_installed,
    is_sentencepiece_installed,
    is_soundfile_installed,
    is_soxr_installed,
)

_AVAILABILITY_CHECKS = (
    ("hf-hub", "huggingface_hub", is_hf_hub_installed),
    ("jinja2", "jinja2", is_jinja2_installed),
    ("llguidance", "llguidance", is_llguidance_installed),
    ("sentencepiece", "sentencepiece", is_sentencepiece_installed),
    ("soundfile", "soundfile", is_soundfile_installed),
    ("soxr", "soxr", is_soxr_installed),
)

_AVAILABILITY_CASES = tuple(
    pytest.param(
        check,
        package_name,
        expected,
        id=f"{case_id}-{'present' if expected else 'missing'}",
    )
    for case_id, package_name, check in _AVAILABILITY_CHECKS
    for expected in (True, False)
)

_ASSERTION_CHECKS = (
    (
        "hf-hub",
        "huggingface_hub",
        assert_hf_hub_installed,
        "`huggingface_hub` is not installed. Please install it with `pip install mistral-common[hf-hub]`",
    ),
    (
        "jinja2",
        "jinja2",
        assert_jinja2_installed,
        "`jinja2` is not installed. Please install it with `pip install mistral-common[guidance]`",
    ),
    (
        "llguidance",
        "llguidance",
        assert_llguidance_installed,
        "`llguidance` is not installed. Please install it with `pip install mistral-common[guidance]`",
    ),
    (
        "opencv",
        "cv2",
        assert_opencv_installed,
        "`opencv` is not installed. Please install it with `pip install mistral-common[opencv]`",
    ),
    (
        "sentencepiece",
        "sentencepiece",
        assert_sentencepiece_installed,
        "`sentencepiece` is not installed. Please install it with `pip install mistral-common[sentencepiece]`",
    ),
    (
        "soundfile",
        "soundfile",
        assert_soundfile_installed,
        "`soundfile` is not installed. Please install it with `pip install mistral-common[soundfile]`",
    ),
    (
        "soxr",
        "soxr",
        assert_soxr_installed,
        "`soxr` is not installed. Please install it with `pip install mistral-common[soxr]`",
    ),
)

_ASSERTION_CASES = tuple(
    pytest.param(
        package_name,
        check,
        installed,
        error_message,
        id=f"{case_id}-{'present' if installed else 'missing'}",
    )
    for case_id, package_name, check, error_message in _ASSERTION_CHECKS
    for installed in (True, False)
)

_CACHED_IMPORT_CHECKS = (
    is_hf_hub_installed,
    is_jinja2_installed,
    is_llguidance_installed,
    is_opencv_installed,
    is_sentencepiece_installed,
    is_soundfile_installed,
    is_soxr_installed,
    assert_hf_hub_installed,
    assert_jinja2_installed,
    assert_llguidance_installed,
    assert_opencv_installed,
    assert_sentencepiece_installed,
    assert_soundfile_installed,
    assert_soxr_installed,
)


@contextmanager
def _isolated_import_caches() -> Iterator[None]:
    for cached_check in _CACHED_IMPORT_CHECKS:
        cached_check.cache_clear()

    try:
        yield
    finally:
        for cached_check in _CACHED_IMPORT_CHECKS:
            cached_check.cache_clear()


@pytest.fixture(autouse=True)
def clear_import_caches() -> Iterator[None]:
    with _isolated_import_caches():
        yield


@pytest.mark.parametrize(
    ("package_spec", "expected_installed"),
    [
        pytest.param(ModuleSpec(name="package_name", loader=None), True, id="package-present"),
        pytest.param(None, False, id="package-missing"),
    ],
)
@patch("importlib.util.find_spec")
def test_is_package_installed(
    mock_find_spec: MagicMock, package_spec: ModuleSpec | None, expected_installed: bool
) -> None:
    mock_find_spec.return_value = package_spec

    assert is_package_installed("package_name") is expected_installed

    mock_find_spec.assert_called_once_with("package_name")


@pytest.mark.parametrize(
    ("package_name", "is_installed", "error_message", "expected_error_message"),
    [
        pytest.param("package_name", True, None, None, id="package-present"),
        pytest.param(
            "package_name",
            False,
            None,
            "Package 'package_name' is required but not installed.",
            id="package-missing-default-message",
        ),
        pytest.param(
            "missing_pkg",
            False,
            "Install missing_pkg for this test",
            "Install missing_pkg for this test",
            id="package-missing-custom-message",
        ),
    ],
)
@patch("mistral_common.imports.is_package_installed")
def test_assert_package_installed(
    mock_is_package_installed: MagicMock,
    package_name: str,
    is_installed: bool,
    error_message: str | None,
    expected_error_message: str | None,
) -> None:
    mock_is_package_installed.return_value = is_installed

    def call_package_check() -> None:
        if error_message is None:
            assert_package_installed(package_name=package_name)
        else:
            assert_package_installed(package_name=package_name, error_message=error_message)

    if expected_error_message is None:
        call_package_check()
    else:
        with pytest.raises(ImportError) as exc_info:
            call_package_check()
        assert str(exc_info.value) == expected_error_message

    mock_is_package_installed.assert_called_once_with(package_name)


def test_is_opencv_installed_when_cv2_import_succeeds() -> None:
    with patch.dict("sys.modules", {"cv2": Mock()}):
        assert is_opencv_installed() is True


def test_is_opencv_installed_when_cv2_import_raises_import_error(caplog: pytest.LogCaptureFixture) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cv2":
            raise ImportError("Simulated import error for cv2")
        return real_import(name, *args, **kwargs)

    with caplog.at_level(logging.WARNING, logger="mistral_common.imports"):
        with patch("builtins.__import__", side_effect=fake_import):
            assert is_opencv_installed() is False

    assert not [record for record in caplog.records if record.name == "mistral_common.imports"]


def test_is_opencv_installed_logs_broken_install_diagnostic(caplog: pytest.LogCaptureFixture) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cv2":
            raise RuntimeError("broken cv2")
        return real_import(name, *args, **kwargs)

    with caplog.at_level(logging.WARNING, logger="mistral_common.imports"):
        with patch("builtins.__import__", side_effect=fake_import):
            assert is_opencv_installed() is False

    assert "Your installation of OpenCV appears to be broken: broken cv2." in caplog.text


@pytest.mark.parametrize(
    ("is_installed_fn", "package_name", "expected_installed"),
    _AVAILABILITY_CASES,
)
@patch("mistral_common.imports.is_package_installed")
def test_dependency_availability_reports_package_state(
    mock_is_package_installed: MagicMock,
    is_installed_fn: Callable[[], bool],
    package_name: str,
    expected_installed: bool,
) -> None:
    mock_is_package_installed.return_value = expected_installed

    assert is_installed_fn() is expected_installed

    mock_is_package_installed.assert_called_once_with(package_name)


@pytest.mark.parametrize(
    ("package_name", "assert_installed_fn", "is_installed", "expected_error_message"),
    _ASSERTION_CASES,
)
@patch("mistral_common.imports.is_package_installed")
def test_dependency_assertion_reports_package_state(
    mock_is_package_installed: MagicMock,
    package_name: str,
    assert_installed_fn: Callable[[], None],
    is_installed: bool,
    expected_error_message: str,
) -> None:
    mock_is_package_installed.return_value = is_installed

    if is_installed:
        assert_installed_fn()
    else:
        with pytest.raises(ImportError) as exc_info:
            assert_installed_fn()
        assert str(exc_info.value) == expected_error_message

    mock_is_package_installed.assert_called_once_with(package_name)


def test_import_caches_are_cleared_after_a_case_fails() -> None:
    with pytest.raises(RuntimeError, match="simulated test failure"):
        with _isolated_import_caches():
            with patch("mistral_common.imports.is_package_installed", return_value=True):
                for _, _, availability_check in _AVAILABILITY_CHECKS:
                    availability_check()
                for _, _, assertion_check, _ in _ASSERTION_CHECKS:
                    assertion_check()
                assert_opencv_installed()
            with patch.dict("sys.modules", {"cv2": Mock()}):
                is_opencv_installed()

            assert all(cached_check.cache_info().currsize == 1 for cached_check in _CACHED_IMPORT_CHECKS)
            raise RuntimeError("simulated test failure")

    assert all(cached_check.cache_info().currsize == 0 for cached_check in _CACHED_IMPORT_CHECKS)
