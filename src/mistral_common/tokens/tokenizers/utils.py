from typing import Iterator, List


def chunks(lst: List[str], chunk_size: int) -> Iterator[List[str]]:
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]
