import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.multimodal import MultiModalVersion

_hub_installed: bool
try:
    import huggingface_hub

    _hub_installed = True
except ImportError:
    _hub_installed = False

logger = logging.getLogger(__name__)


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


def download_tokenizer_from_hf_hub(
    repo_id: str, token: Optional[Union[bool, str]] = None, revision: Optional[str] = None
) -> str:
    r"""Download the tokenizer file of a Mistral model from the Hugging Face Hub.

    See [here](../../../../models.md#list-of-open-models) for a list of our OSS models.

    Note:
        You need to install the `huggingface_hub` package to use this method.

        Please run `pip install mistral-common[hf-hub]` to install it.

    Args:
        repo_id: The Hugging Face repo ID.
        token: The Hugging Face token to use to download the tokenizer.
        revision: The revision of the model to use. If `None`, the latest revision will be used.

    Returns:
        The downloaded tokenizer local path for the given model ID.
    """
    if not _hub_installed:
        raise ImportError(
            "Please install the `huggingface_hub` package to use this method.\n"
            "Run `pip install mistral-common[hf-hub]` to install it."
        )

    hf_api = huggingface_hub.HfApi()
    repo_files = hf_api.list_repo_files(repo_id)

    valid_tokenizer_files = []
    tokenizer_file: str

    instruct_versions = list(TokenizerVersion.__members__)
    mm_versions = list(MultiModalVersion.__members__) + [""]  # allow no mm version
    sentencepiece_suffixes = [f".model.{v}{m}" for v in instruct_versions for m in mm_versions] + [".model"]

    for repo_file in repo_files:
        pathlib_repo_file = Path(repo_file)
        file_name = pathlib_repo_file.name
        suffix = "".join(pathlib_repo_file.suffixes)
        if file_name == "tekken.json":
            valid_tokenizer_files.append(file_name)
        elif suffix in sentencepiece_suffixes:
            valid_tokenizer_files.append(file_name)

    if len(valid_tokenizer_files) == 0:
        raise ValueError(f"No tokenizer file found for model ID: {repo_id}")
    # If there are multiple tokenizer files, we use tekken.json if it exists, otherwise the versionned one.
    if len(valid_tokenizer_files) > 1:
        if "tekken.json" in valid_tokenizer_files:
            tokenizer_file = "tekken.json"
        else:
            tokenizer_file = sorted(valid_tokenizer_files)[-1]
        logger.warning(f"Multiple tokenizer files found for model ID: {repo_id}. Using {tokenizer_file}.")
    else:
        tokenizer_file = valid_tokenizer_files[0]

    tokenizer_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=tokenizer_file, token=token, revision=revision
    )
    return tokenizer_path
