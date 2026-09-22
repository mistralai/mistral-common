import builtins
from types import ModuleType
from typing import Any
from unittest.mock import Mock, patch

import pytest

import mistral_common.imports as imports


@pytest.mark.parametrize(
    ("find_spec_result", "expected"),
    [
        pytest.param(object(), True, id="found"),
        pytest.param(None, False, id="missing"),
    ],
)
def test_is_package_installed(find_spec_result: object | None, expected: bool) -> None:
    with patch("importlib.util.find_spec", return_value=find_spec_result):
        assert imports.is_package_installed("package_name") is expected


@pytest.mark.parametrize(
    ("installed", "expected_error"),
    [
        pytest.param(True, None, id="installed"),
        pytest.param(False, "Package 'package_name' is required but not installed", id="missing"),
    ],
)
def test_assert_package_installed(installed: bool, expected_error: str | None) -> None:
    with patch("mistral_common.imports.is_package_installed", return_value=installed):
        if expected_error is None:
            imports.assert_package_installed("package_name")
        else:
            with pytest.raises(ImportError, match=expected_error):
                imports.assert_package_installed("package_name")


def test_assert_package_installed_uses_custom_message() -> None:
    with patch("mistral_common.imports.is_package_installed", return_value=False):
        with pytest.raises(ImportError, match="custom error"):
            imports.assert_package_installed("package_name", "custom error")


@pytest.mark.parametrize(
    ("import_error", "expected", "warning_expected"),
    [
        pytest.param(None, True, False, id="clean"),
        pytest.param(ImportError("missing"), False, False, id="missing"),
        pytest.param(RuntimeError("broken"), False, True, id="broken-runtime-error"),
    ],
)
def test_is_opencv_installed(import_error: Exception | None, expected: bool, warning_expected: bool) -> None:
    imports.is_opencv_installed.cache_clear()
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "cv2" and import_error is not None:
            raise import_error
        if name == "cv2":
            return Mock()
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import), patch.object(imports.logger, "warning") as warning:
        assert imports.is_opencv_installed() is expected
    if warning_expected:
        warning.assert_called_once()
    else:
        warning.assert_not_called()
    imports.is_opencv_installed.cache_clear()


_INSTALLED_CASES = [
    pytest.param(imports.is_hf_hub_installed, "huggingface_hub", True, id="hf-hub-present"),
    pytest.param(imports.is_hf_hub_installed, "huggingface_hub", False, id="hf-hub-absent"),
    pytest.param(imports.is_jinja2_installed, "jinja2", True, id="jinja2-present"),
    pytest.param(imports.is_jinja2_installed, "jinja2", False, id="jinja2-absent"),
    pytest.param(imports.is_llguidance_installed, "llguidance", True, id="llguidance-present"),
    pytest.param(imports.is_llguidance_installed, "llguidance", False, id="llguidance-absent"),
    pytest.param(imports.is_sentencepiece_installed, "sentencepiece", True, id="sentencepiece-present"),
    pytest.param(imports.is_sentencepiece_installed, "sentencepiece", False, id="sentencepiece-absent"),
    pytest.param(imports.is_soundfile_installed, "soundfile", True, id="soundfile-present"),
    pytest.param(imports.is_soundfile_installed, "soundfile", False, id="soundfile-absent"),
    pytest.param(imports.is_soxr_installed, "soxr", True, id="soxr-present"),
    pytest.param(imports.is_soxr_installed, "soxr", False, id="soxr-absent"),
]


@pytest.mark.parametrize("predicate, package, expected", _INSTALLED_CASES)
def test_is_installed(predicate: Any, package: str, expected: bool) -> None:
    predicate.cache_clear()
    with patch("mistral_common.imports.is_package_installed", return_value=expected) as check:
        assert predicate() is expected
        check.assert_called_once_with(package)
    predicate.cache_clear()


_ASSERT_INSTALLED_CASES = [
    pytest.param(imports.assert_hf_hub_installed, "huggingface_hub", "hf-hub", True, id="hf-hub-present"),
    pytest.param(imports.assert_hf_hub_installed, "huggingface_hub", "hf-hub", False, id="hf-hub-absent"),
    pytest.param(imports.assert_jinja2_installed, "jinja2", "guidance", True, id="jinja2-present"),
    pytest.param(imports.assert_jinja2_installed, "jinja2", "guidance", False, id="jinja2-absent"),
    pytest.param(imports.assert_llguidance_installed, "llguidance", "guidance", True, id="llguidance-present"),
    pytest.param(imports.assert_llguidance_installed, "llguidance", "guidance", False, id="llguidance-absent"),
    pytest.param(imports.assert_opencv_installed, "opencv", "opencv", True, id="opencv-present"),
    pytest.param(imports.assert_opencv_installed, "opencv", "opencv", False, id="opencv-absent"),
    pytest.param(
        imports.assert_sentencepiece_installed,
        "sentencepiece",
        "sentencepiece",
        True,
        id="sentencepiece-present",
    ),
    pytest.param(
        imports.assert_sentencepiece_installed,
        "sentencepiece",
        "sentencepiece",
        False,
        id="sentencepiece-absent",
    ),
    pytest.param(imports.assert_soundfile_installed, "soundfile", "soundfile", True, id="soundfile-present"),
    pytest.param(imports.assert_soundfile_installed, "soundfile", "soundfile", False, id="soundfile-absent"),
    pytest.param(imports.assert_soxr_installed, "soxr", "soxr", True, id="soxr-present"),
    pytest.param(imports.assert_soxr_installed, "soxr", "soxr", False, id="soxr-absent"),
]


@pytest.mark.parametrize("assertion, package, group, installed", _ASSERT_INSTALLED_CASES)
def test_assert_installed(assertion: Any, package: str, group: str, installed: bool) -> None:
    assertion.cache_clear()
    with patch("mistral_common.imports.is_package_installed", return_value=installed):
        if installed:
            assertion()
        else:
            with pytest.raises(ImportError, match=rf"{package}.*mistral-common\[{group}\]"):
                assertion()
    assertion.cache_clear()


_DEPENDENCY_ASSERT_CASES = [
    pytest.param(imports.assert_hf_hub_installed, "huggingface_hub", id="hf-hub"),
    pytest.param(imports.assert_jinja2_installed, "jinja2", id="jinja2"),
    pytest.param(imports.assert_llguidance_installed, "llguidance", id="llguidance"),
    pytest.param(imports.assert_opencv_installed, "cv2", id="opencv"),
    pytest.param(imports.assert_sentencepiece_installed, "sentencepiece", id="sentencepiece"),
    pytest.param(imports.assert_soundfile_installed, "soundfile", id="soundfile"),
    pytest.param(imports.assert_soxr_installed, "soxr", id="soxr"),
]


@pytest.mark.parametrize("assertion, package", _DEPENDENCY_ASSERT_CASES)
def test_dependency_assert_helpers_are_cached(assertion: Any, package: str) -> None:
    assertion.cache_clear()
    with patch("mistral_common.imports.is_package_installed", return_value=True) as check:
        assertion()
        assertion()
    check.assert_called_once_with(package)
    assertion.cache_clear()
