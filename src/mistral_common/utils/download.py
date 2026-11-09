import math
import os

from mistral_common.types import TimeoutType

DOWNLOAD_TIMEOUT_ENV_KEY = "MISTRAL_COMMON_DOWNLOAD_TIMEOUT"
DEFAULT_DOWNLOAD_TIMEOUT_S = 10.0


def validate_download_timeout(timeout: TimeoutType, *, error_message: str) -> float:
    r"""Convert a timeout to a positive finite float or raise.

    Args:
        timeout: The timeout value in seconds.
        error_message: The error message used when the value is invalid.

    Returns:
        The timeout as a positive finite float.

    Raises:
        ValueError: If `timeout` is not a number, not finite, or not positive.
    """
    try:
        value = float(timeout)
    except (TypeError, ValueError) as e:
        raise ValueError(error_message) from e
    if not math.isfinite(value) or value <= 0:
        raise ValueError(error_message)
    return value


def resolve_env_download_timeout() -> float:
    r"""Return the default download timeout from the environment.

    Reads `MISTRAL_COMMON_DOWNLOAD_TIMEOUT` (seconds). Unset or empty uses 10.

    Raises:
        ValueError: If the variable is set but not a positive finite number.
    """
    raw = os.getenv(DOWNLOAD_TIMEOUT_ENV_KEY)
    if raw is None or raw.strip() == "":
        return DEFAULT_DOWNLOAD_TIMEOUT_S
    error_message = (
        f"Invalid environment variable {DOWNLOAD_TIMEOUT_ENV_KEY}: expected a positive finite number of seconds."
    )
    return validate_download_timeout(raw, error_message=error_message)


def resolve_download_timeout(timeout: TimeoutType | None) -> float:
    r"""Resolve a caller-provided timeout, falling back to the environment default.

    Args:
        timeout: Seconds to wait. If `None`, uses the
            `MISTRAL_COMMON_DOWNLOAD_TIMEOUT` environment variable.

    Returns:
        The timeout as a positive finite float.

    Raises:
        ValueError: If `timeout` is not a positive finite number.
    """
    if timeout is None:
        return resolve_env_download_timeout()
    return validate_download_timeout(
        timeout,
        error_message=f"timeout must be a positive finite number, got {timeout=}",
    )
