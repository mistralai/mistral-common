r"""Golden expected-data registry for tokenizer outputs.

A `Scenario` is the single unit of golden coverage in this test suite: a protocol, a
tokenizer key, a request name, and a callable building the request. It is the only way a
golden is produced or asserted -- there is exactly one storage layout, one loader path,
and one regeneration path (`tests/utils/regenerate_registry.py`).

The tokenizer key fully identifies a configured tokenizer, model settings included: two
scenarios that need different model settings use two different keys (see
`tests.utils.tokenizers.TOKENIZER_FACTORIES`), never a per-scenario override. Golden
outputs are stored once per protocol and tokenizer key under
`tests/data/expected/<protocol>/<key>.jsonl`, one JSON object per line, sorted by request
name so regeneration never churns the diff. Each line is:

- `name` -> the request name.
- `request` -> the request serialized with `serialize_request`, asserted against
  `scenario.build_request()` for every scenario in `tests.tokenizers.test_registry_samples`
  so the stored request can never drift from the Python builder that produces it.
- `token_ids` -> the list of ints, exact equality.
- `text` -> the decoded string, exact equality.

A scenario with `has_images` additionally stores its processed image arrays under
`tests/data/expected/<protocol>/<key>/<request>.npz`, compared with
`numpy.testing.assert_allclose` (arrays are stored, floats are never hashed).

A scenario may instead declare `raises`, asserting that encoding it fails with that
exception rather than producing a golden; this is how a protocol a version does not
support is covered (e.g. FIM on v1). Refusal scenarios have no output to store, so they
have no line in `<key>.jsonl` at all.

`SCENARIOS` is built as a capability-filtered cross product, not a hand-bound list:
`_INSTRUCT_SCENARIO_BUILDERS` declares each instruct request builder exactly once,
independent of any tokenizer key, and `tests.utils.tokenizers.KEY_CAPABILITIES` declares
each key's capabilities exactly once, independent of any scenario. `_derive_requirements`
inspects a built request's actual messages and chunk types -- never its scenario name --
to compute what it needs, and `_is_instruct_compatible` is the one predicate deciding
which (scenario, key) pairs exist; every version floor and ceiling in it was verified
empirically against the actual tokenizer/validator behavior it encodes, not guessed from a
version number. FIM has no such requirements to derive (a `FIMRequest` carries no chunks,
tools, or system prompt), so its matrix is a plain cross product over
`_FIM_CAPABLE_KEYS`. Refusal scenarios (`_REFUSAL_SCENARIOS`) stay hand-declared: a
refusal is a deliberate assertion about one specific unsupported combination, not a
capability to filter on.

Adding a new protocol (e.g. transcription) means adding an encoder to `PROTOCOL_ENCODERS`,
an entry to `SUPPORTED_PROTOCOLS` per version, and a matrix-building function alongside
`_build_instruct_scenarios`/`_build_fim_scenarios` -- never a new mechanism. Loaders are
process-cached so a whole session reads each golden file once.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import cast

import numpy as np

from mistral_common.base import MistralBase
from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURLChunk, ImageChunk, ImageURLChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    InstructRequest,
    ModelSettings,
    ReasoningEffort,
)
from mistral_common.tokens.tokenizers.base import Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils.requests.fim import prompt_suffix_request, registry_fim_requests
from tests.utils.requests.instruct import (
    abcd_multi_turn_request,
    abcd_multi_turn_tools_request,
    abcd_single_turn_continue_request,
    abcd_system_multi_turn_continue_request,
    abcd_system_multi_turn_request,
    abcd_system_single_turn_request,
    abcd_system_tools_multi_turn_request,
    assistant_prefix_tool_call_request,
    dummy_audio_chunk,
    dummy_base64_image_url_chunk,
    image_user_assistant_continue_final_message_request,
    image_user_assistant_tool_result_request,
    multimodal_system_request,
    multimodal_tool_request,
    multimodal_user_request,
    registry_image_request,
    registry_instruct_requests,
    single_turn_tool_request,
    system_image_tool_result_chat_request,
    system_two_users_tool_call_result_request,
    system_user_tool_call_request,
    tool_call_chat_request,
    tool_call_instruct_request,
    tool_call_no_id_request,
    tool_call_null_id_request,
    tool_message_chunks_request,
    tool_message_json_request,
    tool_message_multiple_shots_with_history_request,
    tool_message_multiple_shots_without_history_request,
    tool_message_plain_request,
    tool_multiple_calls_request,
    tool_response_chunks_request,
    tool_response_json_request,
    tool_response_plain_request,
    v7_assistant_tool_call_and_content_request,
    v7_truncation_full_convo_request,
    v7_truncation_keep_sys_and_last_message_request,
    v11_continue_final_message_request,
    v11_plain_text_think_request,
    v13_system_user_audio_request,
)
from tests.utils.tokenizers import (
    KEY_CAPABILITIES,
    KeyCapabilities,
    load_mistral_tokenizer,
)

# Tolerances for float array comparisons (processed images / audio). Kept tight and
# centralized so every golden comparison uses the same bounds.
RTOL = 1e-6
ATOL = 1e-6

EXPECTED_DIR = Path(__file__).parent.parent / "data" / "expected"

# Protocols known to the registry. Extending this (plus `PROTOCOL_ENCODERS` and
# `SUPPORTED_PROTOCOLS`) is the entire cost of adding a new protocol's golden coverage.
PROTOCOLS: tuple[str, ...] = ("instruct", "fim")

# Explicit, hand-authored ground truth for which protocol each tokenizer version supports.
# Never inferred by catching exceptions: this table is what `TestProtocolCoverage` checks
# scenarios against. v1 has no FIM marker tokens, so it is one unsupported combination --
# and it raises, so it gets a refusal scenario (see `_REFUSAL_SCENARIOS`). v2 is also
# unsupported for FIM (the shipped SentencePiece model has no `[SUFFIX]`/`[PREFIX]`
# pieces), but `piece_to_id` silently returns `<unk>` instead of raising, so a refusal
# scenario would assert an exception that never happens; see `SILENT_UNSUPPORTED_PROTOCOLS`
# for how that combination is excluded from this registry's coverage check. It is still
# pinned directly, by `test_encode_fim_v2_emits_unk_for_fim_markers` in
# `tests.tokenizers.test_fim`.
SUPPORTED_PROTOCOLS: dict[TokenizerVersion, frozenset[str]] = {
    TokenizerVersion.v1: frozenset({"instruct"}),
    TokenizerVersion.v2: frozenset({"instruct"}),
    TokenizerVersion.v3: frozenset({"instruct", "fim"}),
    TokenizerVersion.v7: frozenset({"instruct", "fim"}),
    TokenizerVersion.v11: frozenset({"instruct", "fim"}),
    TokenizerVersion.v13: frozenset({"instruct", "fim"}),
    TokenizerVersion.v15: frozenset({"instruct", "fim"}),
}

# Version/protocol pairs that are unsupported but do not raise, so neither a golden
# scenario nor a refusal scenario would be honest: v2's shipped SentencePiece model has no
# `[SUFFIX]`/`[PREFIX]` pieces, so `encode_fim` silently emits `<unk>` tokens for the
# markers instead of failing. `TestProtocolCoverage` excludes these pairs from its
# assertion entirely rather than invent a scenario shape that misrepresents the behavior.
SILENT_UNSUPPORTED_PROTOCOLS: frozenset[tuple[TokenizerVersion, str]] = frozenset({(TokenizerVersion.v2, "fim")})

# Explicit tokenizer-key -> version table, used by the coverage test instead of parsing
# key strings. Derived from `KEY_CAPABILITIES` so the two can never disagree.
KEY_VERSIONS: dict[str, TokenizerVersion] = {key: cap.version for key, cap in KEY_CAPABILITIES.items()}

# Keys that make up the FIM matrix: every key whose version supports the `fim` protocol
# `InstructTokenizerV2` defines `encode_fim` and `_encode_infilling`, and no later class
# overrides either, so FIM logic is frozen from v2 onwards. Across versions only the resolved
# `[SUFFIX]`/`[PREFIX]` ids and the surrounding vocabulary change, which is a property of the
# vocabulary rather than of FIM -- the instruct goldens already pin that. What FIM behaviour
# genuinely depends on is the BACKEND, because `_encode_infilling`'s `☺` sentinel interacts with
# SentencePiece's prefix space and is a near no-op on tekken. So goldens are kept for exactly one
# plain key per backend at the FIM floor version, and `TestFimVersionSmoke` guards the inheritance
# on every later version without storing redundant goldens.
FIM_GOLDEN_VERSION = TokenizerVersion.v3

_FIM_CAPABLE_KEYS: tuple[str, ...] = tuple(
    key
    for key, capabilities in KEY_CAPABILITIES.items()
    if capabilities.version == FIM_GOLDEN_VERSION
    and "fim" in SUPPORTED_PROTOCOLS[capabilities.version]
    and (capabilities.version, "fim") not in SILENT_UNSUPPORTED_PROTOCOLS
    and not (capabilities.image or capabilities.audio or capabilities.think)
)

# Every remaining key whose version supports FIM. Not golden-backed; `TestFimVersionSmoke`
# asserts FIM still functions on these, which is the only risk in not pinning them.
FIM_SMOKE_KEYS: tuple[str, ...] = tuple(
    key
    for key, capabilities in KEY_CAPABILITIES.items()
    if key not in _FIM_CAPABLE_KEYS
    and "fim" in SUPPORTED_PROTOCOLS[capabilities.version]
    and (capabilities.version, "fim") not in SILENT_UNSUPPORTED_PROTOCOLS
)


@dataclass(frozen=True)
class Scenario:
    r"""One unit of golden coverage: a protocol, tokenizer key, request name, and builder.

    Attributes:
        protocol: The protocol name (e.g. `"instruct"` or `"fim"`).
        key: A tokenizer key from `tests.utils.tokenizers.TOKENIZER_FACTORIES`, fully
            identifying the tokenizer (model settings included).
        name: The request name, unique among scenarios sharing `protocol` and `key`.
        build_request: Callable returning a fresh request to encode.
        raises: The exception type encoding this scenario must raise, or `None` for a
            scenario that stores an ordinary golden.
        raises_match: Regex the raised exception's message must contain. Required exactly
            when `raises` is set, so a refusal scenario can never assert on the exception
            type alone.
        has_images: Whether this scenario's processed image arrays are stored under
            `<key>/<name>.npz`, compared separately from token ids and decoded text.

    Raises:
        ValueError: If `raises` and `raises_match` are not both set or both unset.
    """

    protocol: str
    key: str
    name: str
    build_request: Callable[[], object]
    raises: type[BaseException] | None = None
    raises_match: str | None = None
    has_images: bool = False

    def __post_init__(self) -> None:
        if (self.raises is None) != (self.raises_match is None):
            raise ValueError("`raises` and `raises_match` must be set together, or both left as `None`.")


def _fim_request(name: str) -> FIMRequest:
    r"""Look up a named request from the curated FIM request set.

    Args:
        name: The request name.

    Returns:
        The request for that name.
    """
    return registry_fim_requests()[name]


@dataclass(frozen=True)
class _ScenarioRequirements:
    r"""Capabilities an instruct request needs from its tokenizer key.

    Every field is computed by `_derive_requirements` from a built request's actual
    messages and chunk types, never from a scenario's name, so a requirement can never
    silently drift from the request it describes. `_is_instruct_compatible` is the one
    place that turns these into a pass/fail decision against a `KeyCapabilities`.

    Attributes:
        needs_image: Any message carries an `ImageChunk`/`ImageURLChunk`.
        needs_audio: Any message carries an `AudioChunk`/`AudioURLChunk`.
        needs_think: Any message carries a `ThinkChunk`.
        uses_tools: `tools`/`available_tools` is non-empty, or any assistant message has
            tool calls, or any tool message is present.
        has_bare_tool_call: Any tool call's `id` is the default `"null"` sentinel.
        has_null_tool_message_id: Any tool message's `tool_call_id` is `None`.
        uses_model_settings: The request sets model settings explicitly (a non-default
            `settings` on a raw `InstructRequest`, or `reasoning_effort` on a
            `ChatCompletionRequest`).
        is_raw_instruct_request: The request is a raw `InstructRequest`, which bypasses
            the `ChatCompletionRequest` normalizer and validator entirely.
        uses_legacy_system_prompt: A raw `InstructRequest` sets the legacy
            `system_prompt` string field.
        uses_system_message: Any message is a `SystemMessage`.
        has_user_message: Any message is a `UserMessage`. A raw `InstructRequest` only
            reaches its tokenizer's model-settings validation from `encode_user_message`
            (`InstructTokenizerBase.encode_instruct` calls it solely for `UserMessage`
            instances), so a raw request with no `UserMessage` at all -- e.g. a lone
            assistant-message continuation -- never touches model settings regardless of
            whether it sets them.
        has_assistant_content_and_tool_calls: Any assistant message has both `content`
            and `tool_calls` set.
        has_system_think: A `SystemMessage` specifically (not an assistant message)
            carries a `ThinkChunk`.
        has_system_and_audio: Both a `SystemMessage` and an audio chunk inside a
            `UserMessage` are present anywhere in the request, mirroring
            `MistralRequestValidatorV5._validate_system_prompt_and_audio`, which only
            inspects `UserMessage` content for audio -- a `SystemMessage` carrying its own
            audio chunk directly does not trigger this restriction.
        has_tool_multimodal_content: A tool message's content list carries an image or
            audio chunk.
        has_unnormalized_tool_message_content: A raw `InstructRequest`'s tool message
            content is a chunk list rather than a plain string.
        has_orphan_tool_message: A `ChatCompletionRequest`'s tool message id has no
            matching (non-`"null"`) tool call id anywhere in the request.
    """

    needs_image: bool = False
    needs_audio: bool = False
    needs_think: bool = False
    uses_tools: bool = False
    has_bare_tool_call: bool = False
    has_null_tool_message_id: bool = False
    uses_model_settings: bool = False
    is_raw_instruct_request: bool = False
    uses_legacy_system_prompt: bool = False
    uses_system_message: bool = False
    has_user_message: bool = False
    has_assistant_content_and_tool_calls: bool = False
    has_system_think: bool = False
    has_system_and_audio: bool = False
    has_tool_multimodal_content: bool = False
    has_unnormalized_tool_message_content: bool = False
    has_orphan_tool_message: bool = False


def _derive_requirements(request: object) -> _ScenarioRequirements:
    r"""Derive an instruct request's tokenizer requirements by typed inspection.

    Walks the request's actual messages and chunk types -- never pattern-matches a
    scenario's name -- so a requirement can never drift from the request it describes.

    Args:
        request: A `ChatCompletionRequest` or `InstructRequest`.

    Returns:
        The requirements the request places on a compatible tokenizer key.
    """
    messages = list(getattr(request, "messages", None) or [])
    chunks = [chunk for message in messages if isinstance(message.content, list) for chunk in message.content]
    tool_calls = [
        tool_call
        for message in messages
        if isinstance(message, AssistantMessage) and message.tool_calls
        for tool_call in message.tool_calls
    ]
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    tools_declared = bool(getattr(request, "tools", None) or getattr(request, "available_tools", None))

    if isinstance(request, InstructRequest):
        is_raw_instruct_request = True
        uses_model_settings = request.settings != ModelSettings.none()
        uses_legacy_system_prompt = request.system_prompt is not None
    else:
        assert isinstance(request, ChatCompletionRequest)
        is_raw_instruct_request = False
        uses_model_settings = request.reasoning_effort is not None
        uses_legacy_system_prompt = False

    needs_audio = any(isinstance(chunk, (AudioChunk, AudioURLChunk)) for chunk in chunks)
    uses_system_message = any(isinstance(message, SystemMessage) for message in messages)
    has_audio_in_user_message = any(
        isinstance(message, UserMessage)
        and isinstance(message.content, list)
        and any(isinstance(chunk, (AudioChunk, AudioURLChunk)) for chunk in message.content)
        for message in messages
    )
    tool_call_ids = {tool_call.id for tool_call in tool_calls if tool_call.id != "null"}
    tool_message_ids = {message.tool_call_id for message in tool_messages if message.tool_call_id is not None}

    return _ScenarioRequirements(
        needs_image=any(isinstance(chunk, (ImageChunk, ImageURLChunk)) for chunk in chunks),
        needs_audio=needs_audio,
        needs_think=any(isinstance(chunk, ThinkChunk) for chunk in chunks),
        uses_tools=tools_declared or bool(tool_calls) or bool(tool_messages),
        has_bare_tool_call=any(tool_call.id == "null" for tool_call in tool_calls),
        has_null_tool_message_id=any(message.tool_call_id is None for message in tool_messages),
        uses_model_settings=uses_model_settings,
        is_raw_instruct_request=is_raw_instruct_request,
        uses_legacy_system_prompt=uses_legacy_system_prompt,
        uses_system_message=uses_system_message,
        has_user_message=any(isinstance(message, UserMessage) for message in messages),
        has_assistant_content_and_tool_calls=any(
            isinstance(message, AssistantMessage) and bool(message.tool_calls) and bool(message.content)
            for message in messages
        ),
        has_system_think=any(
            isinstance(message, SystemMessage)
            and isinstance(message.content, list)
            and any(isinstance(chunk, ThinkChunk) for chunk in message.content)
            for message in messages
        ),
        has_system_and_audio=uses_system_message and has_audio_in_user_message,
        has_tool_multimodal_content=any(
            isinstance(message.content, list)
            and any(
                isinstance(chunk, (ImageChunk, ImageURLChunk, AudioChunk, AudioURLChunk)) for chunk in message.content
            )
            for message in tool_messages
        ),
        has_unnormalized_tool_message_content=is_raw_instruct_request
        and any(isinstance(message.content, list) for message in tool_messages),
        has_orphan_tool_message=not is_raw_instruct_request and bool(tool_message_ids - tool_call_ids),
    )


def _is_instruct_compatible(requirements: _ScenarioRequirements, capabilities: "KeyCapabilities") -> bool:
    r"""Return whether a tokenizer key's capabilities satisfy an instruct request's requirements.

    Every check is a version floor or ceiling verified empirically against the actual
    tokenizer/validator behavior it guards (see the calling code for how each was
    verified), never inferred from a version number's parity with a feature name.

    Args:
        requirements: The request's requirements, from `_derive_requirements`.
        capabilities: The candidate tokenizer key's capabilities.

    Returns:
        Whether the key can encode a request with these requirements.
    """
    version = capabilities.version
    if requirements.needs_image and not capabilities.image:
        return False
    if requirements.needs_audio and not capabilities.audio:
        return False
    if requirements.needs_think and not capabilities.think:
        return False
    # v15's normalizer (`InstructRequestNormalizerV15.build_settings`) unconditionally
    # requires a real model settings builder to normalize *any* `ChatCompletionRequest`; a
    # v15 key without one (e.g. `"v15"`) cannot encode any `ChatCompletionRequest`. A raw
    # `InstructRequest` bypasses the normalizer entirely, and its tokenizer
    # (`InstructTokenizerV15._validate_settings`) is only reached from
    # `encode_user_message`, so a raw request needs a builder only when it has a
    # `UserMessage` at all -- one with no `UserMessage` (e.g. a lone assistant-message
    # continuation) never touches model settings.
    if version.supports_model_settings and not capabilities.model_settings:
        if not requirements.is_raw_instruct_request or requirements.has_user_message:
            return False
    if requirements.uses_model_settings and not capabilities.model_settings:
        return False
    # A model settings builder with a default (every `model_settings` key here has one)
    # only fills that default in for a `ChatCompletionRequest`, via the normalizer; a raw
    # `InstructRequest` bypasses the normalizer, so it must set the value explicitly
    # whenever it has a `UserMessage` (the only place `encode_instruct` ever reaches model
    # settings for a raw request).
    if (
        capabilities.model_settings
        and requirements.is_raw_instruct_request
        and not requirements.uses_model_settings
        and requirements.has_user_message
    ):
        return False
    # v1 silently discards a declared `tools`/`available_tools` schema (no schema is
    # rendered) and raises for an actual tool call; either way it cannot honor a scenario
    # relying on tool calling.
    if requirements.uses_tools and version < TokenizerVersion.v2:
        return False
    # v13+ asserts every tool call id is real and non-`"null"`
    # (`InstructTokenizerV13._encode_tool_calls_in_assistant_message`).
    if requirements.has_bare_tool_call and version >= TokenizerVersion.v13:
        return False
    # v3+ asserts every tool message carries a tool call id
    # (`InstructTokenizerV3._prepare_tool_result`).
    if requirements.has_null_tool_message_id and version >= TokenizerVersion.v3:
        return False
    # The legacy `InstructRequest.system_prompt` field is only encoded by v1-v3; v7+
    # asserts it is never used.
    if requirements.uses_legacy_system_prompt and version >= TokenizerVersion.v7:
        return False
    # A raw `InstructRequest` carrying `SystemMessage` objects (rather than the legacy
    # `system_prompt` field) is only supported from v7: v1-v3's raw encode path raises
    # `NotImplementedError` for a `SystemMessage`.
    if requirements.uses_system_message and requirements.is_raw_instruct_request and version < TokenizerVersion.v7:
        return False
    # An assistant message combining content and tool calls is only allowed from v7
    # (`MistralRequestValidatorV5._allow_tool_call_and_content`); pre-v7 raises either at
    # the validator (`ChatCompletionRequest`) or the tokenizer (raw `InstructRequest`).
    if requirements.has_assistant_content_and_tool_calls and version < TokenizerVersion.v7:
        return False
    # v15 rejects a `ThinkChunk` specifically inside a system message
    # (`InstructTokenizerV15.encode_system_message`); it stays valid in assistant messages.
    if requirements.has_system_think and version >= TokenizerVersion.v15:
        return False
    # A system message and an audio chunk cannot coexist before v13
    # (`MistralRequestValidatorV5._validate_system_prompt_and_audio`).
    if requirements.has_system_and_audio and version < TokenizerVersion.v13:
        return False
    # A tool message's content chunks are limited to text before v15
    # (`MistralRequestValidator._validate_tool_content_chunks`).
    if requirements.has_tool_multimodal_content and version < TokenizerVersion.v15:
        return False
    # v7 and v11 require a raw `InstructRequest`'s tool message content to already be a
    # plain string (`InstructTokenizerV7.encode_tool_message` asserts this; normalization
    # is what produces a plain string for a `ChatCompletionRequest`). v3 and v13+ both
    # aggregate a chunk list themselves.
    if requirements.has_unnormalized_tool_message_content and version in (TokenizerVersion.v7, TokenizerVersion.v11):
        return False
    # v13+'s validator (`MistralRequestValidatorV13._validate_tool_calls_followed_by_tool_messages`)
    # requires every tool message id to match a preceding tool call id in the same round;
    # only a `ChatCompletionRequest` goes through that validator.
    if requirements.has_orphan_tool_message and version >= TokenizerVersion.v13:
        return False
    return True


# Every distinct instruct request builder the golden registry covers, declared exactly
# once and independent of any tokenizer key. `_build_instruct_scenarios` crosses each of
# these against every key in `tests.utils.tokenizers.KEY_CAPABILITIES`, keeping only the
# pairs `_is_instruct_compatible` allows.
_INSTRUCT_SCENARIO_BUILDERS: dict[str, Callable[[], object]] = {
    "single_turn": lambda: registry_instruct_requests()["single_turn"],
    "single_turn_system": lambda: registry_instruct_requests()["single_turn_system"],
    "multi_turn": lambda: registry_instruct_requests()["multi_turn"],
    "multi_turn_system": lambda: registry_instruct_requests()["multi_turn_system"],
    "multi_turn_tools": lambda: registry_instruct_requests()["multi_turn_tools"],
    "tool_calls": lambda: registry_instruct_requests()["tool_calls"],
    "image": registry_image_request,
    "abcd_multi_turn": abcd_multi_turn_request,
    "abcd_system_single_turn": abcd_system_single_turn_request,
    "abcd_system_multi_turn": abcd_system_multi_turn_request,
    "abcd_system_multi_turn_continue": abcd_system_multi_turn_continue_request,
    "abcd_multi_turn_tools": abcd_multi_turn_tools_request,
    "single_turn_tool": single_turn_tool_request,
    "abcd_system_tools_multi_turn": abcd_system_tools_multi_turn_request,
    "tool_response_plain": tool_response_plain_request,
    "tool_response_json": tool_response_json_request,
    "tool_response_chunks": tool_response_chunks_request,
    "tool_message_multiple_shots_without_history": tool_message_multiple_shots_without_history_request,
    "tool_message_plain": tool_message_plain_request,
    "tool_message_json": tool_message_json_request,
    "tool_message_chunks": tool_message_chunks_request,
    "tool_call_null_id": tool_call_null_id_request,
    "tool_call_no_id": tool_call_no_id_request,
    "tool_message_multiple_shots_with_history": tool_message_multiple_shots_with_history_request,
    "tool_multiple_calls": tool_multiple_calls_request,
    "assistant_prefix_tool_call": assistant_prefix_tool_call_request,
    "system_tool_call": system_user_tool_call_request,
    "system_two_users_tool_result": system_two_users_tool_call_result_request,
    "image_tool_result_chat": system_image_tool_result_chat_request,
    "abcd_single_turn_continue": abcd_single_turn_continue_request,
    "image_tool_result": image_user_assistant_tool_result_request,
    "image_continue_final_message": image_user_assistant_continue_final_message_request,
    "truncation_keep_sys_and_last_message": v7_truncation_keep_sys_and_last_message_request,
    "truncation_full_convo": v7_truncation_full_convo_request,
    "assistant_tool_call_and_content": v7_assistant_tool_call_and_content_request,
    "continue_final_message": v11_continue_final_message_request,
    "plain_text_think": v11_plain_text_think_request,
    "tools_think": lambda: tool_call_chat_request(think=True),
    "tools_wrong_order": lambda: tool_call_chat_request(swap_tool_results=True),
    "system_user_audio": lambda: v13_system_user_audio_request(dummy_audio_chunk()),
    "tools_reasoning_high": lambda: tool_call_instruct_request(reasoning_effort=ReasoningEffort.high),
    "no_tools_reasoning_none": lambda: tool_call_instruct_request(
        reasoning_effort=ReasoningEffort.none, with_tools=False
    ),
    "tool_audio": lambda: multimodal_tool_request(dummy_audio_chunk()),
    "tool_image": lambda: multimodal_tool_request(dummy_base64_image_url_chunk()),
    "system_audio": lambda: multimodal_system_request(dummy_audio_chunk()),
    "user_audio": lambda: multimodal_user_request(dummy_audio_chunk()),
    "user_image": lambda: multimodal_user_request(dummy_base64_image_url_chunk()),
}


def _build_instruct_scenarios() -> list[Scenario]:
    r"""Build every compatible (instruct scenario, tokenizer key) pair.

    Crosses every builder in `_INSTRUCT_SCENARIO_BUILDERS` against every key in
    `tests.utils.tokenizers.KEY_CAPABILITIES`, keeping only the pairs
    `_is_instruct_compatible` allows. Each builder is called once per scenario name (not
    once per pair) since a request's requirements do not depend on the tokenizer key.

    Returns:
        One `Scenario` per compatible (name, key) pair.
    """
    scenarios: list[Scenario] = []
    for name, build_request in _INSTRUCT_SCENARIO_BUILDERS.items():
        requirements = _derive_requirements(build_request())
        for key, capabilities in KEY_CAPABILITIES.items():
            if "instruct" not in SUPPORTED_PROTOCOLS[capabilities.version]:
                continue
            if (capabilities.version, "instruct") in SILENT_UNSUPPORTED_PROTOCOLS:
                continue
            if not _is_instruct_compatible(requirements, capabilities):
                continue
            scenarios.append(
                Scenario(
                    protocol="instruct",
                    key=key,
                    name=name,
                    build_request=build_request,
                    has_images=requirements.needs_image,
                )
            )
    return scenarios


def _build_fim_scenarios() -> list[Scenario]:
    r"""Build every (FIM request, FIM-capable tokenizer key) pair.

    A `FIMRequest` carries no chunks, tools, or system prompt, so it has no requirements
    to derive: every FIM-capable key (`_FIM_CAPABLE_KEYS`) can encode every FIM request.

    Returns:
        One `Scenario` per (name, key) pair.
    """
    return [
        Scenario(protocol="fim", key=key, name=name, build_request=partial(_fim_request, name))
        for key in _FIM_CAPABLE_KEYS
        for name in registry_fim_requests()
    ]


# Scenarios covering a protocol a version does not support: encoding must raise, so there
# is no golden to store. v1 has no FIM marker tokens (see `SUPPORTED_PROTOCOLS`).
_REFUSAL_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        protocol="fim",
        key="v1_spm",
        name="fim_not_supported",
        build_request=prompt_suffix_request,
        raises=TokenizerException,
        raises_match="FIM not available for",
    ),
)

# The single source of truth for every golden and refusal scenario in the registry.
SCENARIOS: tuple[Scenario, ...] = (
    tuple(_build_instruct_scenarios()) + tuple(_build_fim_scenarios()) + _REFUSAL_SCENARIOS
)


def golden_keys(protocol: str) -> list[str]:
    r"""Return the tokenizer keys with stored goldens for a protocol, in first-seen order.

    Args:
        protocol: The protocol name.

    Returns:
        The ordered, de-duplicated list of keys with at least one golden scenario.
    """
    keys: list[str] = []
    for scenario in SCENARIOS:
        if scenario.protocol == protocol and scenario.raises is None and scenario.key not in keys:
            keys.append(scenario.key)
    return keys


def build_tokenizer(key: str) -> MistralTokenizer:
    r"""Return the (cached) `MistralTokenizer` for a registry key.

    Args:
        key: A key from `tests.utils.tokenizers.TOKENIZER_FACTORIES`.

    Returns:
        The `MistralTokenizer` for the key.

    Raises:
        KeyError: If the key is not a known tokenizer key.
    """
    return load_mistral_tokenizer(key)


def _encode_instruct(tokenizer: MistralTokenizer, request: object) -> Tokenized:
    r"""Encode an instruct scenario request.

    A raw `InstructRequest` bypasses normalization/validation (needed for requests, such
    as the abcd conversations, that assert on a specific vocabulary directly); every other
    request is a `ChatCompletionRequest` encoded through the full pipeline.

    Args:
        tokenizer: The tokenizer to encode with.
        request: The request to encode.

    Returns:
        The encoding of the request.
    """
    if isinstance(request, InstructRequest):
        encoded: Tokenized = tokenizer.instruct_tokenizer.encode_instruct(request)
        return encoded
    tokenized: Tokenized = tokenizer.encode_chat_completion(request)  # type: ignore[arg-type]
    return tokenized


def _encode_fim(tokenizer: MistralTokenizer, request: object) -> Tokenized:
    r"""Encode a FIM scenario request.

    Args:
        tokenizer: The tokenizer to encode with.
        request: The `FIMRequest` to encode.

    Returns:
        The encoding of the request.
    """
    assert isinstance(request, FIMRequest)
    encoded: Tokenized = tokenizer.encode_fim(request)
    return encoded


# Dispatch table from protocol name to its encode function. Adding a protocol means
# adding an encoder here (plus `SUPPORTED_PROTOCOLS` and scenarios), never a new mechanism.
PROTOCOL_ENCODERS: dict[str, Callable[[MistralTokenizer, object], Tokenized]] = {
    "instruct": _encode_instruct,
    "fim": _encode_fim,
}


def encode_scenario(scenario: Scenario) -> Tokenized:
    r"""Encode a scenario's request with its tokenizer.

    This is the single encode path shared by the golden regeneration script and the tests
    that assert against the goldens (or, for a `raises` scenario, the refusal).

    Args:
        scenario: The scenario to encode.

    Returns:
        The encoding of the scenario's request.
    """
    tokenizer = build_tokenizer(scenario.key)
    request = scenario.build_request()
    return PROTOCOL_ENCODERS[scenario.protocol](tokenizer, request)


REDACTED_BINARY_PAYLOAD = "<redacted binary payload>"

# Request fields whose values are base64 blobs. See `serialize_request` for why they are redacted.
_BINARY_PAYLOAD_FIELDS = frozenset({"image", "image_url", "input_audio"})

# Matches a data URL or a bare base64 run. Used to fail regeneration loudly if a new binary field
# appears that `_BINARY_PAYLOAD_FIELDS` does not cover, rather than letting an unstable payload
# reach the goldens and break on whichever CI runner encodes it differently.
_BINARY_PAYLOAD_PATTERN = re.compile(r"^(?:data:|[A-Za-z0-9+/]{40,}={0,2}$)")


def _assert_no_binary_payload_left(value: object, path: str = "request") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_binary_payload_left(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_binary_payload_left(item, f"{path}[{index}]")
    elif isinstance(value, str) and _BINARY_PAYLOAD_PATTERN.match(value):
        raise AssertionError(
            f"{path} still holds a base64 or data-URL payload. Such encodings are not byte-stable "
            f"across environments, so it must be added to `_BINARY_PAYLOAD_FIELDS` before it can be "
            f"stored in a golden."
        )


def _redact_binary_payloads(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: REDACTED_BINARY_PAYLOAD
            if key in _BINARY_PAYLOAD_FIELDS and isinstance(item, str)
            else _redact_binary_payloads(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_binary_payloads(item) for item in value]
    return value


def serialize_request(request: object) -> dict[str, object]:
    r"""Serialize a scenario request to a JSON-stable mapping for golden storage.

    Args:
        request: The request to serialize, a `ChatCompletionRequest`, `InstructRequest`,
            or `FIMRequest` -- always a `MistralBase` subclass.

    Base64 image and audio payloads are redacted. Their encodings are not byte-stable across
    environments -- PNG output depends on the zlib version, so the same pixels serialize
    differently on different Python builds -- which would make the golden fail on some CI
    runners and pass on others. The payloads themselves stay covered: processed image arrays
    are pinned by the `.npz` goldens, and audio content is pinned by the token ids, whose
    count follows the audio duration.

    Returns:
        The request dumped through Pydantic's JSON mode with unset-to-`None` fields
        dropped, so the golden stays stable as new optional fields are added, and with
        binary payloads replaced by `REDACTED_BINARY_PAYLOAD`.
    """
    assert isinstance(request, MistralBase)
    redacted = _redact_binary_payloads(request.model_dump(mode="json", exclude_none=True))
    assert isinstance(redacted, dict)
    _assert_no_binary_payload_left(redacted)
    return redacted


def _key_jsonl_path(protocol: str, key: str) -> Path:
    r"""Return the golden JSONL file for a protocol and key.

    Args:
        protocol: The protocol name (e.g. ``"instruct"`` or ``"fim"``).
        key: The tokenizer key.

    Returns:
        The path to the golden JSONL file for the protocol and key.
    """
    return EXPECTED_DIR / protocol / f"{key}.jsonl"


def _key_dir(protocol: str, key: str) -> Path:
    r"""Return the golden image directory for a protocol and key.

    Args:
        protocol: The protocol name (e.g. ``"instruct"`` or ``"fim"``).
        key: The tokenizer key.

    Returns:
        The path to the directory storing that protocol and key's `.npz` image arrays.
    """
    return EXPECTED_DIR / protocol / key


@lru_cache(maxsize=None)
def _load_key_golden(protocol: str, key: str) -> dict[str, dict[str, object]]:
    r"""Load the raw golden mapping for a protocol and key.

    Args:
        protocol: The protocol name.
        key: The tokenizer key.

    Returns:
        Mapping from request name to its `request`/`text`/`token_ids` golden entry.
    """
    data: dict[str, dict[str, object]] = {}
    with open(_key_jsonl_path(protocol, key), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record: dict[str, object] = json.loads(line)
            name = cast(str, record.pop("name"))
            data[name] = record
    return data


def load_token_ids(protocol: str, key: str) -> dict[str, list[int]]:
    r"""Load the golden token ids for a protocol and key.

    Args:
        protocol: The protocol name.
        key: The tokenizer key.

    Returns:
        Mapping from request name to golden token ids.
    """
    return {name: cast(list[int], entry["token_ids"]) for name, entry in _load_key_golden(protocol, key).items()}


def load_decoded(protocol: str, key: str) -> dict[str, str]:
    r"""Load the golden decoded text for a protocol and key.

    Args:
        protocol: The protocol name.
        key: The tokenizer key.

    Returns:
        Mapping from request name to golden decoded text.
    """
    return {name: cast(str, entry["text"]) for name, entry in _load_key_golden(protocol, key).items()}


def load_requests(protocol: str, key: str) -> dict[str, dict[str, object]]:
    r"""Load the golden serialized requests for a protocol and key.

    Args:
        protocol: The protocol name.
        key: The tokenizer key.

    Returns:
        Mapping from request name to its golden serialized request.
    """
    return {name: cast(dict[str, object], entry["request"]) for name, entry in _load_key_golden(protocol, key).items()}


def _image_array_index(key: str) -> int:
    r"""Parse the integer index out of an ``arr_{i}`` image array key.

    Args:
        key: An ``arr_{i}`` key as written by `tests.utils.regenerate_registry`.

    Returns:
        The integer index `i`.
    """
    return int(key.removeprefix("arr_"))


def load_image_arrays(protocol: str, key: str, request_name: str) -> list[np.ndarray]:
    r"""Load the golden processed image arrays for a request.

    Sorts by the integer index parsed from each ``arr_{i}`` key rather than lexicographic
    key order, so ordering stays correct at 10+ images (lexicographic order would put
    ``arr_10`` before ``arr_2``).

    Args:
        protocol: The protocol name.
        key: The tokenizer key.
        request_name: The request whose processed images to load.

    Returns:
        The processed image arrays in their stored order.
    """
    with np.load(_key_dir(protocol, key) / f"{request_name}.npz") as data:
        return [data[name] for name in sorted(data.files, key=_image_array_index)]
