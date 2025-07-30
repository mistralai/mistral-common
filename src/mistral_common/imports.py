import importlib.util
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


def is_package_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def assert_package_installed(package_name: str, error_message: Optional[str] = None) -> None:
    if not is_package_installed(package_name):
        error_message = error_message or f"Package '{package_name}' is required but not installed."
        raise ImportError(error_message)


@lru_cache()
def is_hf_hub_installed() -> bool:
    return is_package_installed("huggingface_hub")


@lru_cache()
def is_opencv_installed() -> bool:
    try:
        import cv2  # noqa: IOO1,F401

        cv2 = cv2
    except ImportError as e:
        raise e  # Should not happen, as we check for the package before importing
    except Exception as e:
        # cv2 has lots of import problems: https://github.com/opencv/opencv-python/issues/884
        # for better UX, let's simply skip all errors that might arise from import for now
        _cv2_available = False
        logger.warning(
            f"Warning: Your installation of OpenCV appears to be broken: {e}."
            "Please follow the instructions at https://github.com/opencv/opencv-python/issues/884 "
            "to correct your environment. The import of cv2 has been skipped."
        )
    else:
        _cv2_available = True
    return _cv2_available


@lru_cache()
def is_sentencepiece_installed() -> bool:
    return is_package_installed("sentencepiece")


@lru_cache()
def is_soundfile_installed() -> bool:
    return is_package_installed("soundfile")


@lru_cache()
def is_soxr_installed() -> bool:
    return is_package_installed("soxr")


def assert_hf_hub_installed() -> None:
    assert_package_installed(
        "huggingface_hub",
        "`huggingface_hub` is not installed. Please install it with `pip install mistral-common[hf-hub]`",
    )


def assert_opencv_installed() -> None:
    assert_package_installed(
        "cv2", "`opencv` is not installed. Please install it with `pip install mistral-common[opencv]`"
    )


def assert_sentencepiece_installed() -> None:
    assert_package_installed(
        "sentencepiece",
        "`sentencepiece` is not installed. Please install it with `pip install mistral-common[sentencepiece]`",
    )


def assert_soundfile_installed() -> None:
    assert_package_installed(
        "soundfile", "`soundfile` is not installed. Please install it with `pip install mistral-common[soundfile]`"
    )


def assert_soxr_installed() -> None:
    assert_package_installed(
        "soxr", "`soxr` is not installed. Please install it with `pip install mistral-common[soxr]`"
    )
