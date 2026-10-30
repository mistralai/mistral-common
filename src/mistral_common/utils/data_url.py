def maybe_strip_base64_data_url_prefix(s: str) -> str:
    r"""Return the payload of a base64 data URL, or the input unchanged.

    Implements the split step of the WHATWG Fetch "data: URL processor": if the
    input starts with `data:` and the part before the first comma ends with
    `;base64`, everything after that comma is the base64 payload. Inputs that
    do not start with `data:` — plain base64, an http(s) URL, an empty string —
    are returned unchanged. An input that starts with `data:` but is not a
    well-formed base64 data URL is rejected immediately, so a malformed payload
    fails here instead of surfacing later as a decode error.

    Args:
        s: The string to strip, typically a base64 string or data URL.

    Returns:
        The base64 payload if `s` is a base64 data URL, otherwise `s` unchanged.

    Raises:
        ValueError: If `s` starts with `data:` but is not a well-formed base64
            data URL (missing comma, a header not ending in `;base64`, or an
            empty payload).

    Examples:
        >>> maybe_strip_base64_data_url_prefix("data:image/png;base64,aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("data:image/svg+xml;base64,aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("data:image/png;charset=utf-8;base64,aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("https://example.com/image.png")
        'https://example.com/image.png'
        >>> maybe_strip_base64_data_url_prefix("data:text/plain,hello")
        Traceback (most recent call last):
        ...
        ValueError: Invalid base64 data URL data:text/plain,hello: expected the form `data:<mime>;base64,<payload>`.
        >>> maybe_strip_base64_data_url_prefix("data:image/png;base64,")
        Traceback (most recent call last):
        ...
        ValueError: Invalid base64 data URL data:image/png;base64,: expected the form `data:<mime>;base64,<payload>`.
    """
    if s.startswith("data:"):
        header, sep, payload = s.partition(",")
        if sep and header.endswith(";base64") and payload:
            return payload
        raise ValueError(f"Invalid base64 data URL {s[:64]}: expected the form `data:<mime>;base64,<payload>`.")
    return s
