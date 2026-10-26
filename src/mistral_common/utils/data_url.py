def maybe_strip_base64_data_url_prefix(s: str, *, strict: bool = False) -> str:
    r"""Return the payload of a base64 data URL, or the input unchanged.

    Implements the split step of the WHATWG Fetch "data: URL processor": if the
    input starts with `data:` and the part before the first comma ends with
    `;base64`, everything after that comma is the base64 payload. Any other
    input — plain base64, a non-base64 data URL such as `data:text/plain,hello`,
    a URL, or a string without a comma — is returned unchanged.

    Args:
        s: The string to strip, typically a base64 string or data URL.
        strict: If `True`, raise `ValueError` when `s` starts with `data:` but is
            not a well-formed base64 data URL (missing comma, a header not ending
            in `;base64`, or an empty payload). Plain base64 still passes through.

    Returns:
        The base64 payload if `s` is a base64 data URL, otherwise `s` unchanged.

    Raises:
        ValueError: If `strict` is `True` and `s` is a malformed base64 data URL.

    Examples:
        >>> maybe_strip_base64_data_url_prefix("data:image/png;base64,aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("data:image/svg+xml;base64,aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("data:text/plain,hello")
        'data:text/plain,hello'
        >>> maybe_strip_base64_data_url_prefix("aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("data:text/plain,hello", strict=True)
        Traceback (most recent call last):
        ...
        ValueError: Invalid base64 data URL data:text/plain,hello: expected the form `data:<mime>;base64,<payload>`.
        >>> maybe_strip_base64_data_url_prefix("data:image/png;base64,", strict=True)
        Traceback (most recent call last):
        ...
        ValueError: Invalid base64 data URL data:image/png;base64,: expected the form `data:<mime>;base64,<payload>`.
    """
    if s.startswith("data:"):
        header, sep, payload = s.partition(",")
        if sep and header.endswith(";base64") and (payload or not strict):
            return payload
        if strict:
            raise ValueError(f"Invalid base64 data URL {s[:64]}: expected the form `data:<mime>;base64,<payload>`.")
    return s
