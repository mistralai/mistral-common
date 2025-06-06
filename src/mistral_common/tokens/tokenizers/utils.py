from typing import Any, Dict, Iterator, List

from mistral_common.exceptions import TokenizerException

_hub_installed: bool
try:
    import huggingface_hub

    _hub_installed = True
except ImportError:
    _hub_installed = False


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


def download_tokenizer_from_hf_hub(model_id: str, **kwargs: Any) -> str:
    r"""Download the configuration file of an official Mistral tokenizer from the Hugging Face Hub.

    See [Models](../../../../models.md) for a list of supported models.

    Note:
        You need to install the `huggingface_hub` package to use this method.
        please run `pip install mistral-common[hf-hub]` to install it.

    Args:
        model_id: The Hugging Face model ID.
            See [Models](../../../../models.md) for a list of supported models.
        kwargs: Additional keyword arguments to pass to `huggingface_hub.hf_hub_download`.

    Returns:
        The downloaded tokenizer local path for the given model ID.
    """
    if not _hub_installed:
        raise ImportError(
            "Please install the `huggingface_hub` package to use this method.\n"
            "Run `pip install mistral-common[hf-hub]` to install it."
        )

    if model_id not in MODEL_HF_ID_TO_TOKENIZER_FILE:
        raise TokenizerException(f"Unrecognized model ID: {model_id}")

    tokenizer_file = MODEL_HF_ID_TO_TOKENIZER_FILE[model_id]
    tokenizer_path = huggingface_hub.hf_hub_download(repo_id=model_id, filename=tokenizer_file, **kwargs)
    return tokenizer_path


MODEL_HF_ID_TO_TOKENIZER_FILE: Dict[str, str] = {
    "mistralai/Mistral-7B-v0.1": "tokenizer.model",
    "mistralai/Mistral-7B-Instruct-v0.1": "tokenizer.model.v1",
    "mistralai/Mixtral-8x7B-v0.1": "tokenizer.model",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "tokenizer.model",
    "mistralai/Mistral-7B-Instruct-v0.2": "tokenizer.model",
    "mistralai/Mixtral-8x22B-v0.1": "tokenizer.model.v1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "tokenizer.model.v3",
    "mistralai/Mistral-7B-v0.3": "tokenizer.model.v3",
    "mistralai/Mistral-7B-Instruct-v0.3": "tokenizer.model.v3",
    "mistralai/Codestral-22B-v0.1": "tokenizer.model.v3",
    "mistralai/Mathstral-7B-v0.1": "tokenizer.model.v3",
    "mistralai/Mamba-Codestral-7B-v0.1": "tokenizer.model.v3",
    "mistralai/Mistral-Nemo-Base-2407": "tekken.json",
    "mistralai/Mistral-Nemo-Instruct-2407": "tekken.json",
    "mistralai/Mistral-Large-Instruct-2407": "tokenizer.model.v3",
    "mistralai/Pixtral-12B-Base-2409": "tekken.json",
    "mistralai/Pixtral-12B-2409": "tekken.json",
    "mistralai/Mistral-Large-Instruct-2411": "tokenizer.model.v7",
    "mistralai/Pixtral-Large-Instruct-2411": "tokenizer.model.v7m1",
    "mistralai/Mistral-Small-24B-Base-2501": "tekken.json",
    "mistralai/Mistral-Small-24B-Instruct-2501": "tekken.json",
    "mistralai/Mistral-Small-3.1-24B-Base-2503": "tekken.json",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "tekken.json",
    "mistralai/Devstral-Small-2505": "tekken.json",
}
