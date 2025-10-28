def _split_content_and_think_chunks(
    tokens: list[int], begin_think_token_id: int, end_think_token_id: int
) -> list[tuple[list[int], bool]]:
    r"""Split the content and think chunks from a list of tokens.

    Args:
        tokens: list of tokens.
        begin_think_token_id: The token id for the begin think token.
        end_think_token_id: The token id for the end think token.

    Returns:
        list of tuples, where each tuple contains a list of tokens and a boolean indicating if the chunk is a think
        chunk.
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
