def maybe_strip_base64_data_url_prefix(s: str) -> str:
    r"""Return the payload of a base64 data URL, or the input unchanged.

    Implements the split step of the WHATWG Fetch "data: URL processor": if the
    input starts with `data:` and the part before the first comma ends with
    `;base64`, everything after that comma is the base64 payload. Any other
    input — plain base64, a non-base64 data URL such as `data:text/plain,hello`,
    a URL, or a string without a comma — is returned unchanged.

    Args:
        s: The string to strip, typically a base64 string or data URL.

    Returns:
        The base64 payload if `s` is a base64 data URL, otherwise `s` unchanged.

    Examples:
        >>> maybe_strip_base64_data_url_prefix("data:image/png;base64,aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("data:image/svg+xml;base64,aGVsbG8=")
        'aGVsbG8='
        >>> maybe_strip_base64_data_url_prefix("data:text/plain,hello")
        'data:text/plain,hello'
        >>> maybe_strip_base64_data_url_prefix("aGVsbG8=")
        'aGVsbG8='
    """
    if s.startswith("data:"):
        header, sep, payload = s.partition(",")
        if sep and header.endswith(";base64"):
            return payload
    return s
