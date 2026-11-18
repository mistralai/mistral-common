import pytest

from mistral_common.utils.data_url import maybe_strip_base64_data_url_prefix


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Base64 data URLs are unwrapped.
        pytest.param("data:image/png;base64,aGVsbG8=", "aGVsbG8=", id="image-png"),
        pytest.param("data:audio/wav;base64,aGVsbG8=", "aGVsbG8=", id="audio-wav"),
        # RFC 6838 subtypes contain "-", "+" and "." (mistralai/mistral-common#340).
        pytest.param("data:image/svg+xml;base64,aGVsbG8=", "aGVsbG8=", id="image-svg-xml"),
        pytest.param("data:image/x-icon;base64,aGVsbG8=", "aGVsbG8=", id="image-x-icon"),
        pytest.param("data:image/vnd.microsoft.icon;base64,aGVsbG8=", "aGVsbG8=", id="image-vnd-microsoft-icon"),
        pytest.param("data:audio/x-wav;base64,aGVsbG8=", "aGVsbG8=", id="audio-x-wav"),
        # Media-type parameters are part of the header, not the payload.
        pytest.param("data:image/png;charset=utf-8;base64,aGVsbG8=", "aGVsbG8=", id="image-png-charset-param"),
        # Only the first comma splits header from payload.
        pytest.param("data:image/png;base64,YWJj,ZGVm", "YWJj,ZGVm", id="payload-with-comma"),
    ],
)
def test_maybe_strip_base64_data_url_prefix_unwraps_data_urls(value: str, expected: str) -> None:
    r"""Return the base64 payload of a well-formed data URL."""
    assert maybe_strip_base64_data_url_prefix(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("aGVsbG8=", id="bare-base64"),
        pytest.param("https://example.com/image.png", id="https-url"),
        pytest.param("", id="empty-string"),
    ],
)
def test_maybe_strip_base64_data_url_prefix_passes_through_non_data_urls(value: str) -> None:
    r"""Return inputs that do not start with `data:` unchanged."""
    assert maybe_strip_base64_data_url_prefix(value) == value


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("data:text/plain,hello", id="non-base64-data-url"),
        pytest.param("data:,hello", id="no-base64-header"),
        pytest.param("data:image/png;base64", id="no-comma"),
        pytest.param("data:image/png;base64,", id="empty-payload"),
    ],
)
def test_maybe_strip_base64_data_url_prefix_rejects_malformed_data_urls(value: str) -> None:
    r"""Reject `data:` inputs that are not well-formed base64 data URLs."""
    with pytest.raises(ValueError, match="Invalid base64 data URL"):
        maybe_strip_base64_data_url_prefix(value)
