import logging
import os
from pathlib import Path
from typing import Iterator

import requests

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.image import MultiModalVersion

_hub_installed: bool
try:
    import huggingface_hub
    import huggingface_hub.constants

    _hub_installed = True
except ImportError:
    _hub_installed = False

logger = logging.getLogger(__name__)


def _assert_hub_installed() -> None:
    if not _hub_installed:
        raise ImportError(
            "Please install the `huggingface_hub` package to use this method.\n"
            "Run `pip install mistral-common[hf-hub]` to install it."
        )


def chunks(lst: list[str], chunk_size: int) -> Iterator[list[str]]:
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


def list_local_hf_repo_files(repo_id: str, revision: str | None) -> list[str]:
    r"""list the files of a local Hugging Face repo.

    Args:
        repo_id: The Hugging Face repo ID.
        revision: The revision of the model to use. If `None`, the latest revision will be used.
    """
    _assert_hub_installed()

    repo_cache = Path(huggingface_hub.constants.HF_HUB_CACHE) / huggingface_hub.constants.REPO_ID_SEPARATOR.join(
        ["models", *repo_id.split("/")]
    )

    if revision is None:
        revision_file = repo_cache / "refs" / huggingface_hub.constants.DEFAULT_REVISION
        if revision_file.is_file():
            with revision_file.open("r") as file:
                revision = file.read()

    if revision:
        revision_dir = repo_cache / "snapshots" / revision
        if revision_dir.is_dir():
            return os.listdir(revision_dir)

    return []


def _filter_valid_tokenizer_files(files: list[str]) -> list[tuple[str, str]]:
    r"""Filter the valid tokenizer files from a list of files.

    Args:
        files: The list of files to filter.

    Returns:
        The list of tuples of file names and paths to the valid tokenizer files.
    """
    valid_tokenizer_files = []

    instruct_versions = list(TokenizerVersion.__members__)
    mm_versions = list(MultiModalVersion.__members__) + [""]  # allow no mm version
    sentencepiece_suffixes = [f".model.{v}{m}" for v in instruct_versions for m in mm_versions] + [".model"]

    for file in files:
        pathlib_file = Path(file)
        file_name = pathlib_file.name
        suffix = "".join(pathlib_file.suffixes)
        if file_name == "tekken.json":
            valid_tokenizer_files.append((file_name, file))
        elif suffix in sentencepiece_suffixes:
            valid_tokenizer_files.append((file_name, file))

    return valid_tokenizer_files


def get_one_valid_tokenizer_file(files: list[str]) -> str:
    r"""Get one valid tokenizer file from a list of files.

    Args:
        files: The list of files to filter.

    Returns:
        The path to the tokenizer file.
    """
    valid_tokenizer_file_names_and_files = _filter_valid_tokenizer_files(files)

    if len(valid_tokenizer_file_names_and_files) == 0:
        raise ValueError("No tokenizer file found.")
    # If there are multiple tokenizer files, we use tekken.json if it exists, otherwise the versioned one.
    if len(valid_tokenizer_file_names_and_files) > 1:
        for file_name, tokenizer_file in valid_tokenizer_file_names_and_files:
            if "tekken.json" == file_name:
                return tokenizer_file
        tokenizer_file = sorted(valid_tokenizer_file_names_and_files, key=lambda x: x[0])[-1][1]
        logger.warning(f"Multiple valid tokenizer files found. Using {tokenizer_file}.")
    else:
        tokenizer_file = valid_tokenizer_file_names_and_files[0][1]

    return tokenizer_file


def download_tokenizer_from_hf_hub(
    repo_id: str,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> str:
    r"""Download the tokenizer file of a Mistral model from the Hugging Face Hub.

    See [here](https://huggingface.co/mistralai/models) for a list of our OSS models.

    Note:
        You need to install the `huggingface_hub` package to use this method.

        Please run `pip install mistral-common[hf-hub]` to install it.

    Args:
        repo_id: The Hugging Face repo ID.
        cache_dir: The directory where the tokenizer will be cached.
        token: The Hugging Face token to use to download the tokenizer.
        revision: The revision of the model to use. If `None`, the latest revision will be used.
        force_download: Whether to force the download of the tokenizer. If `True`, the tokenizer will be downloaded
            even if it is already cached.
        local_files_only: Whether to only use local files. If `True`, the tokenizer will be downloaded only if it is
            already cached.

    Returns:
        The downloaded tokenizer local path for the given model ID.
    """
    _assert_hub_installed()

    if force_download and local_files_only:
        raise ValueError("You cannot force the download of the tokenizer if you only want to use local files.")

    if not local_files_only:
        try:
            hf_api = huggingface_hub.HfApi()
            repo_files = hf_api.list_repo_files(repo_id, revision=revision, token=token)
            local_files_only = False
        except (requests.ConnectionError, requests.HTTPError, requests.Timeout) as e:
            if force_download:
                raise e

            repo_files = list_local_hf_repo_files(repo_id=repo_id, revision=revision)
            local_files_only = True

            logger.info("Could not connect to the Hugging Face Hub. Using local files only.")

            if len(repo_files) == 0:
                raise FileNotFoundError(
                    f"Could not connect to the Hugging Face Hub and no local files were found for the repo ID {repo_id}"
                    f" and revision {revision}. Please check your internet connection and try again."
                ) from e
    else:
        repo_files = list_local_hf_repo_files(repo_id=repo_id, revision=revision)
        if len(repo_files) == 0:
            raise FileNotFoundError(
                f"No local files found for the repo ID {repo_id} and revision {revision}. Please check the repo ID and"
                " the revision or try to download the tokenizer without setting `local_files_only` to `True`."
            )

    try:
        tokenizer_file = get_one_valid_tokenizer_file(files=repo_files)
    except ValueError:
        raise ValueError(f"No valid tokenizer file found in the repo {repo_id}.")

    tokenizer_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        filename=tokenizer_file,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    return tokenizer_path
