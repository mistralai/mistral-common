def _split_content_and_think_chunks(
    tokens: list[int], begin_think_token_id: int, end_think_token_id: int
) -> list[tuple[list[int], bool]]:
    r"""Split the content and think chunks from a list of tokens.

    Think chunks include their begin/end think tokens in the returned token
    list; content chunks are the token runs between them.

    Args:
        tokens: The token IDs to split.
        begin_think_token_id: The token ID of the begin think token.
        end_think_token_id: The token ID of the end think token.

    Returns:
        Chunks in order, each a tuple of (`token_ids`, `is_think_chunk`) where
        `is_think_chunk` is `True` for think chunks and `False` for content chunks.
        An unclosed think chunk at the end of tokens is returned as a think
        chunk.

    Raises:
        ValueError: If think chunks are nested or an end think token appears
            without a matching begin think token.
    """
    if not tokens:
        return []

    content_chunks: list[tuple[list[int], bool]] = []
    current_content: list[int] = []

    in_think_chunk = False
    for token in tokens:
        if token == begin_think_token_id and in_think_chunk:
            raise ValueError("Nested think chunks are not allowed.")
        elif token == begin_think_token_id:
            if current_content:
                content_chunks.append((current_content, False))
                current_content = []
            in_think_chunk = True
            current_content.append(token)
        elif token == end_think_token_id:
            if not in_think_chunk:
                raise ValueError("End think token found without a begin think token.")
            current_content.append(token)
            content_chunks.append((current_content, True))
            current_content = []
            in_think_chunk = False
        else:
            current_content.append(token)

    if current_content:
        content_chunks.append((current_content, in_think_chunk))

    return content_chunks
