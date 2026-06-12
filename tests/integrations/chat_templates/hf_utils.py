r"""HuggingFace transformers/tokenizers utilities for chat template tests.

This module contains helpers that depend on `transformers` and
`tokenizers`.  General-purpose helpers live in `helpers.py`.
"""

import json
from pathlib import Path
from typing import Any, cast

try:
    from transformers import PreTrainedTokenizerFast
    from transformers.utils.chat_template_utils import render_jinja_template

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from mistral_common.protocol.instruct.request import ChatCompletionRequest


def _to_openai_request(chat_request: ChatCompletionRequest, keep_name_for_tools: bool = False) -> dict[str, Any]:
    r"""Convert a ChatCompletionRequest to OpenAI format.

    Args:
        chat_request: The chat completion request to convert.
        keep_name_for_tools: If True, preserve tool message `name` fields
            that `to_openai` strips by default.

    Returns:
        OpenAI-format request dictionary.
    """
    openai_request = chat_request.to_openai()
    if keep_name_for_tools:
        for openai_message, chat_message in zip(openai_request["messages"], chat_request.messages):
            if chat_message.role == "tool":
                openai_message["name"] = chat_message.name
    return openai_request


def encode_transformers(
    chat_template: str, chat_request: ChatCompletionRequest, keep_name_for_tools: bool = False
) -> str:
    r"""Encode a chat request using the transformers render_jinja_template.

    Converts a ChatCompletionRequest to OpenAI format and renders through
    the HuggingFace transformers rendering pipeline.
    """
    assert _HAS_TRANSFORMERS, "transformers is required"
    openai_request = _to_openai_request(chat_request, keep_name_for_tools)
    return _render_via_transformers(chat_template, openai_request)


def encode_transformers_from_openai(chat_template: str, openai_request: dict[str, Any]) -> str:
    r"""Encode a raw OpenAI-format dict using the transformers render_jinja_template.

    Use this for tests with hand-crafted dicts (e.g., invalid input tests)
    that cannot be represented as a ChatCompletionRequest.
    """
    assert _HAS_TRANSFORMERS, "transformers is required"
    return _render_via_transformers(chat_template, openai_request)


def _build_hf_tokenizer(tekken_path: Path, chat_template: str) -> "PreTrainedTokenizerFast":
    r"""Build a `PreTrainedTokenizerFast` from a tekken.json file.

    Converts the Tekken BPE vocabulary to a HuggingFace tokenizers format,
    preserving the Tekkenizer's ID scheme: special tokens at IDs 0 to
    `num_special - 1`, regular BPE tokens at IDs `num_special` and above.

    All special tokens in the file are unconditionally registered as
    `AddedToken` objects, matching Tekkenizer behavior where every token
    in the special range is treated as special.

    .. note::

        TODO: The ID-offset conversion done here should be fixed upstream
        in transformers' `MistralConverter` so that it produces IDs matching
        the Tekkenizer natively. Until then, we apply the offset manually to
        ensure a fair token-level comparison.

    Args:
        tekken_path: Path to the tekken.json file.
        chat_template: Jinja chat template string to set on the tokenizer.

    Returns:
        A configured `PreTrainedTokenizerFast` with matching token IDs.
    """
    import base64
    from functools import lru_cache

    from tokenizers import Regex, decoders, pre_tokenizers, processors
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from transformers import AddedToken
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    with open(tekken_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pattern: str = data["config"]["pattern"]
    special_tokens_list: list[dict[str, Any]] = data["special_tokens"]
    num_special: int = data["config"]["default_num_special_tokens"]
    vocab_size: int = data["config"]["default_vocab_size"]
    inner_vocab_size = vocab_size - num_special
    vocab_entries: list[dict[str, Any]] = data["vocab"][:inner_vocab_size]

    byte_encoder = bytes_to_unicode()

    @lru_cache
    def token_bytes_to_string(b: bytes) -> str:
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    # Decode raw bytes for each vocab entry
    raw_tokens = [base64.b64decode(entry["token_bytes"]) for entry in vocab_entries]
    rank_set = set(raw_tokens)
    token_to_rank: dict[bytes, int] = {token: rank for rank, token in enumerate(raw_tokens)}

    # Build BPE vocab with regular tokens at IDs num_special+ and special tokens at 0..num_special-1.
    # All special token slots (active and filler) must appear in the BPE vocab so the
    # HuggingFace tokenizer recognizes IDs 0 through num_special-1.
    bpe_vocab: dict[str, int] = {}
    defined_ids: set[int] = set()
    for st in special_tokens_list:
        bpe_vocab[st["token_str"]] = st["rank"]
        defined_ids.add(st["rank"])
    for i in range(num_special):
        if i not in defined_ids:
            bpe_vocab[f"<SPECIAL_{i}>"] = i
    for rank, token in enumerate(raw_tokens):
        bpe_vocab[token_bytes_to_string(token)] = rank + num_special

    # Extract BPE merges (same algorithm as MistralConverter.extract_vocab_merges_from_model)
    merges: list[tuple[bytes, bytes, int]] = []
    for rank, token in enumerate(raw_tokens):
        if len(token) == 1:
            continue
        local: list[tuple[bytes, bytes, int]] = []
        for index in range(1, len(token)):
            piece_l, piece_r = token[:index], token[index:]
            if piece_l in rank_set and piece_r in rank_set and (piece_l + piece_r) in rank_set:
                local.append((piece_l, piece_r, rank))
        local = sorted(local, key=lambda x: (token_to_rank[x[0]], token_to_rank[x[1]]))
        merges.extend(local)
    merges = sorted(merges, key=lambda val: val[2])
    bpe_merges = [(token_bytes_to_string(m[0]), token_bytes_to_string(m[1])) for m in merges]

    # Create tokenizer with BPE model
    tokenizer = HFTokenizer(BPE(bpe_vocab, bpe_merges, fuse_unk=False))
    if hasattr(tokenizer.model, "ignore_merges"):
        tokenizer.model.ignore_merges = True

    # Set pre-tokenizer, decoder, and post-processor
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Register all special tokens in added_tokens_decoder, matching Tekkenizer
    # behavior where every token in the special range is treated as special.
    added_tokens_decoder: dict[int, AddedToken] = {}
    for st in special_tokens_list:
        added_tokens_decoder[st["rank"]] = AddedToken(st["token_str"], special=True, normalized=False)
    for i in range(num_special):
        if i not in defined_ids:
            added_tokens_decoder[i] = AddedToken(f"<SPECIAL_{i}>", special=True, normalized=False)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        added_tokens_decoder=added_tokens_decoder,
        bos_token="<s>",
        eos_token="</s>",
        chat_template=chat_template,
    )


def encode_hf_tokens(
    hf_tokenizer: "PreTrainedTokenizerFast", chat_request: ChatCompletionRequest, keep_name_for_tools: bool = False
) -> list[int]:
    r"""Get token IDs from a standard HuggingFace tokenizer.

    Converts the `ChatCompletionRequest` to OpenAI format and encodes via
    `apply_chat_template(tokenize=True)`.

    Args:
        hf_tokenizer: A `PreTrainedTokenizerFast` instance.
        chat_request: The chat completion request to encode.
        keep_name_for_tools: If True, preserve tool message `name` fields.

    Returns:
        Token IDs produced by the HuggingFace tokenizer.
    """
    openai_request = _to_openai_request(chat_request, keep_name_for_tools)
    messages = openai_request["messages"]

    tools = openai_request.get("tools", None)
    if tools is not None:
        for tool in tools:
            tool["function"].pop("strict", None)

    template_kwargs: dict[str, Any] = {}
    reasoning_effort = openai_request.get("reasoning_effort")
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    result = hf_tokenizer.apply_chat_template(
        conversation=messages,
        tools=tools,
        tokenize=True,
        return_dict=False,
        padding=False,
        truncation=False,
        **template_kwargs,
    )
    return cast(list[int], result)


def _render_via_transformers(chat_template: str, openai_request: dict[str, Any]) -> str:
    r"""Shared rendering logic for transformers-based encoding."""
    for tool in openai_request.get("tools", []):
        tool["function"].pop("strict", False)

    reasoning_effort = openai_request.get("reasoning_effort")
    template_kwargs: dict[str, Any] = {}
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    encoded = render_jinja_template(
        [openai_request["messages"]],
        tools=openai_request.get("tools", None),
        chat_template=chat_template,
        bos_token="<s>",
        eos_token="</s>",
        **template_kwargs,
    )[0][0]
    assert isinstance(encoded, str), type(encoded)
    return encoded
