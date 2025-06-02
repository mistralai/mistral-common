from typing import Iterator, List


def chunks(lst: List[str], chunk_size: int) -> Iterator[List[str]]:
    r"""Chunk a list into smaller lists of a given size.

    Args:
        lst: The list to chunk.
        chunk_size: The size of each chunk.

    Returns:
        An iterator over the chunks.

    Examples:
        >>> all_chunks = list(chunks([1, 2, 3, 4, 5], 2))
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]
