import enum
import importlib.util
import logging
import warnings
from enum import Enum
from functools import lru_cache
from typing import Any, Optional, Type, TypeVar, overload

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _get_dependency_error_message(package_name: str, dependency_group: str) -> str:
    return f"`{package_name}` is not installed. Please install it with `pip install mistral-common[{dependency_group}]`"


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
        import cv2  # noqa: F401
    except ImportError:
        _cv2_available = False
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
    assert_package_installed("huggingface_hub", _get_dependency_error_message("huggingface_hub", "hf-hub"))


def assert_opencv_installed() -> None:
    assert_package_installed("cv2", _get_dependency_error_message("opencv", "opencv"))


def assert_sentencepiece_installed() -> None:
    assert_package_installed("sentencepiece", _get_dependency_error_message("sentencepiece", "sentencepiece"))


def assert_soundfile_installed() -> None:
    assert_package_installed("soundfile", _get_dependency_error_message("soundfile", "soundfile"))


def assert_soxr_installed() -> None:
    assert_package_installed("soxr", _get_dependency_error_message("soxr", "soxr"))


T = TypeVar("T", bound=BaseModel)


@overload
def create_deprecate_cls_import(
    cls_moved: Type[T], prev_location: str, new_location: str, deprecate_version: str
) -> Type[T]: ...
@overload
def create_deprecate_cls_import(
    cls_moved: Type[Enum], prev_location: str, new_location: str, deprecate_version: str
) -> Type[Enum]: ...
def create_deprecate_cls_import(
    cls_moved: Type[T] | Type[Enum], prev_location: str, new_location: str, deprecate_version: str
) -> Type[T] | Type[Enum]:
    msg = (
        f"{cls_moved.__name__} has moved to {new_location}. "
        f"It will be removed in {prev_location} in {deprecate_version}."
    )
    warnings.filterwarnings(
        action="once",
        category=FutureWarning,
        message=msg,  # Replace `msg` with the actual warning message or a unique part of it
    )

    if isinstance(cls_moved, type) and not issubclass(cls_moved, enum.Enum):
        ClsDeprecated = type(cls_moved.__name__, (cls_moved,), {})

        def wrapped_init(self: T, *args: Any, **kwargs: Any) -> None:
            warnings.warn(msg, FutureWarning)
            cls_moved.__init__(self, *args, **kwargs)
            return None

        ClsDeprecated.__init__ = wrapped_init  # type: ignore[misc]
        ClsDeprecated.__name__ = cls_moved.__name__
        ClsDeprecated.__doc__ = cls_moved.__doc__
        ClsDeprecated.__module__ = prev_location

        return ClsDeprecated

    elif isinstance(cls_moved, type) and issubclass(cls_moved, enum.Enum):
        EnumDeprecated = Enum(str(cls_moved.__name__), {k: v.value for k, v in cls_moved.__members__.items()})  # type: ignore[misc]

        def wrapped_getattribute(self: object, name: str) -> Any:
            warnings.warn(msg, FutureWarning)
            return object.__getattribute__(self, name)

        EnumDeprecated.__getattribute__ = wrapped_getattribute  # type: ignore[method-assign]
        EnumDeprecated.__name__ = cls_moved.__name__
        EnumDeprecated.__doc__ = cls_moved.__doc__
        EnumDeprecated.__module__ = prev_location

        return EnumDeprecated

    else:
        raise TypeError(f"deprecated_import cannot be applied to object of type {type(cls_moved)}")
