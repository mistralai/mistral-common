from dataclasses import dataclass

from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion

# Module-level constants derived from SpecialTokens enum
_BOS = SpecialTokens.bos.value
_EOS = SpecialTokens.eos.value
_BEGIN_INST = SpecialTokens.begin_inst.value
_END_INST = SpecialTokens.end_inst.value
_BEGIN_TOOLS = SpecialTokens.begin_tools.value
_END_TOOLS = SpecialTokens.end_tools.value
_BEGIN_TOOL_RESULTS = SpecialTokens.begin_tool_results.value
_END_TOOL_RESULTS = SpecialTokens.end_tool_results.value
_TOOL_CALLS = SpecialTokens.tool_calls.value
_IMG = SpecialTokens.img.value
_AUDIO = SpecialTokens.audio.value
_BEGIN_SYSTEM = SpecialTokens.begin_system.value
_END_SYSTEM = SpecialTokens.end_system.value
_BEGIN_TOOL_CONTENT = SpecialTokens.begin_tool_content.value
_ARGS = SpecialTokens.args.value
_CALL_ID = SpecialTokens.call_id.value
_BEGIN_THINK = SpecialTokens.begin_think.value
_END_THINK = SpecialTokens.end_think.value
_BEGIN_MODEL_SETTINGS = SpecialTokens.begin_model_settings.value
_END_MODEL_SETTINGS = SpecialTokens.end_model_settings.value


@dataclass
class TemplateConfig:
    r"""Configuration for generating a chat template.

    This class encapsulates all the configuration options required to generate
    a Jinja2 chat template that formats conversation messages for Mistral models.
    The template handles message roles, special tokens, tool calls, and multimodal content.

    Attributes:
        version: The tokenizer version (e.g., v1, v2, v3, v7, v11, v13, v15). Determines
            special token formatting, tool call syntax, and available features.
        spm: Whether to use SentencePiece tokenizer formatting. When True, adds spaces
            after special tokens. Not supported for versions v11+ or with audio.
        image_support: Whether to enable image chunk processing in user messages.
            Adds [IMG] token support. Requires version v3+. Mutually exclusive with audio.
        audio_support: Whether to enable audio chunk processing in user messages.
            Adds [AUDIO] token support. Requires version v7+. Mutually exclusive with image.
        thinking_support: Whether to enable thinking chunks in system and assistant messages.
            Adds [THINK]/[/THINK] token support. Requires version v13+.
        plain_thinking_support: Whether to enable plain text thinking chunks using
            `<think>`/`</think>` tags instead of special tokens. Only available for v11.
            Mutually exclusive with `thinking_support`.
        use_special_token_variables: Whether to emit BOS/EOS as Jinja variable references
            (`bos_token`/`eos_token`) or as literal string values (`'<s>'`/`'</s>'`).
            When True, the template expects `bos_token` and `eos_token`
            to be passed as render kwargs.

    Raises:
        ValueError: If the configuration is invalid (e.g., conflicting options like
            spm with v11+, image and audio together, or version requirements not met).

    Examples:
        >>> config = TemplateConfig(version=TokenizerVersion.v3, image_support=True)
        >>> config.image_support
        True
        >>> config.version == TokenizerVersion.v3
        True
    """

    version: TokenizerVersion
    spm: bool = False
    image_support: bool = False
    audio_support: bool = False
    thinking_support: bool = False
    plain_thinking_support: bool = False
    use_special_token_variables: bool = False

    def __post_init__(self) -> None:
        if self.plain_thinking_support and self.thinking_support:
            raise ValueError("Plain thinking support and thinking support are mutually exclusive")
        if self.spm and (self.version >= TokenizerVersion.v11 or self.audio_support):
            raise ValueError("SPM tokenizer is not supported for tokenizer versions v11 and above or audio")
        if self.image_support and self.audio_support:
            raise ValueError("Image and audio support are mutually exclusive")
        if self.image_support and self.version < TokenizerVersion.v3:
            raise ValueError("Image support is only available for tokenizer versions v3 and above")
        if self.audio_support and self.version < TokenizerVersion.v7:
            raise ValueError("Audio support is only available for tokenizer versions v7 and above")
        if self.thinking_support and self.version < TokenizerVersion.v13:
            raise ValueError("Thinking support is only available for tokenizer versions v13 and above")
        if self.audio_support and self.thinking_support:
            raise ValueError("Audio and thinking support are mutually exclusive")
        if self.plain_thinking_support and self.version != TokenizerVersion.v11:
            raise ValueError("Plain thinking support is only available for tokenizer version v11")
        if self.audio_support and self.plain_thinking_support:
            raise ValueError("Audio and plain thinking support are mutually exclusive")

    @property
    def bos_expr(self) -> str:
        r"""Jinja expression for the BOS token."""
        if self.use_special_token_variables:
            return "bos_token"
        return f"'{_BOS}'"

    @property
    def eos_expr(self) -> str:
        r"""Jinja expression for the EOS token."""
        if self.use_special_token_variables:
            return "eos_token"
        return f"'{_EOS}'"

    @property
    def any_thinking_support(self) -> bool:
        r"""Whether any form of thinking support is enabled."""
        return self.thinking_support or self.plain_thinking_support

    @property
    def has_tools(self) -> bool:
        r"""Whether tool definitions are supported."""
        return self.version >= TokenizerVersion.v2

    @property
    def uses_system_prompt_tokens(self) -> bool:
        r"""Whether to use [SYSTEM_PROMPT] tokens vs inline system message."""
        return self.version >= TokenizerVersion.v7

    @property
    def tracks_max_idx_user(self) -> bool:
        r"""Whether to track max user index for tools definition placement."""
        return self.has_tools and self.version < TokenizerVersion.v13

    @property
    def tools_at_beginning(self) -> bool:
        r"""Whether tools definition is emitted at the beginning."""
        return self.version >= TokenizerVersion.v13

    @property
    def uses_v13_tool_format(self) -> bool:
        r"""Whether to use v13-style tool calls (name[ARGS]arguments)."""
        return self.version >= TokenizerVersion.v13

    @property
    def uses_json_tool_results(self) -> bool:
        r"""Whether tool results use JSON format with content/call_id."""
        return self.version == TokenizerVersion.v3 and self.spm

    @property
    def uses_tool_content_format(self) -> bool:
        r"""Whether tool results use [TOOL_RESULTS]id[TOOL_CONTENT]content format."""
        return self.version >= TokenizerVersion.v7 and self.version < TokenizerVersion.v13

    @property
    def uses_simple_tool_results(self) -> bool:
        r"""Whether tool results use simple [TOOL_RESULTS]content format."""
        return self.version >= TokenizerVersion.v13

    @property
    def uses_v2_spm_tool_format(self) -> bool:
        r"""Whether to use v2_spm tool format (no ID, uses name in results)."""
        return self.version == TokenizerVersion.v2 and self.spm

    @property
    def uses_v2_tool_format(self) -> bool:
        r"""Whether to use v2 tool format (no ID, uses name in results, elif branch)."""
        return self.version == TokenizerVersion.v2

    @property
    def uses_spm_space_tracking(self) -> bool:
        r"""Whether to track add_space for SPM formatting."""
        return self.spm and self.version >= TokenizerVersion.v2

    @property
    def uses_spm_prev_img_tracking(self) -> bool:
        r"""Whether to track prev_img for SPM image formatting."""
        return self.spm and self.image_support

    @property
    def spm_system_prompt_has_space(self) -> bool:
        r"""Whether SPM system prompt tokens have trailing space."""
        return self.spm and self.version >= TokenizerVersion.v7

    @property
    def uses_call_id_in_tool_calls(self) -> bool:
        r"""Whether to include [CALL_ID] in tool calls."""
        return self.version == TokenizerVersion.v11

    @property
    def tracks_has_sp_for_audio(self) -> bool:
        r"""Whether to track has_sp for audio constraint. V15+ allows audio with system prompts."""
        return self.audio_support and self.version < TokenizerVersion.v15

    @property
    def supports_model_settings(self) -> bool:
        r"""Whether model settings (reasoning_effort) are supported. V15+."""
        return self.version >= TokenizerVersion.v15

    @property
    def is_v1(self) -> bool:
        r"""Whether this is a v1 template with minimal features."""
        return self.version == TokenizerVersion.v1

    @property
    def system_supports_thinking(self) -> bool:
        r"""Whether system messages support thinking chunks. Pre-v15 only."""
        return self.any_thinking_support and self.version < TokenizerVersion.v15

    @property
    def uses_v2_v3spm_tool_branch(self) -> bool:
        r"""Whether assistant tool calls use the v2/v3-SPM inline elif branch."""
        return self.version == TokenizerVersion.v2 or (self.spm and self.version == TokenizerVersion.v3)

    @property
    def forbids_assistant_content_with_tools(self) -> bool:
        r"""Whether assistant messages cannot have both content and tool calls."""
        return self.version in [TokenizerVersion.v2, TokenizerVersion.v3]

    @property
    def validates_assistant_non_empty(self) -> bool:
        r"""Whether to validate that assistant messages have non-empty content or tool calls."""
        return self.version >= TokenizerVersion.v7 or (self.version >= TokenizerVersion.v3 and not self.spm)

    @property
    def tool_supports_multimodal(self) -> bool:
        r"""Whether tool messages can contain non-text content chunks. V15+."""
        return self.version >= TokenizerVersion.v15

    @property
    def assistant_supports_multimodal(self) -> bool:
        r"""Whether assistant messages can contain non-text content chunks. V15+."""
        return self.version >= TokenizerVersion.v15

    @property
    def system_supports_audio(self) -> bool:
        r"""Whether system messages can contain audio. V15+ with audio support."""
        return self.audio_support and self.version >= TokenizerVersion.v15


def _join_types_desc(parts: list[str]) -> str:
    r"""Join type names into a human-readable description string.

    Args:
        parts: List of type names (e.g. ["text", "thinking", "image"]).

    Returns:
        Formatted string like "text", "text and thinking", or "text, thinking and image".
    """
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _generate_header(config: TemplateConfig) -> str:
    r"""Generate template header with default system message.

    Args:
        config: Template configuration.

    Returns:
        Jinja2 header block with default system message variable.
    """
    return (
        """{#- Default system message if no system prompt is passed. #}
{%- set default_system_message = '' %}

{#- Begin of sequence token. #}
{{- """
        + config.bos_expr
        + """ }}
"""
    )


def _generate_system_prompt_handling(config: TemplateConfig) -> str:
    r"""Generate system prompt handling section.

    For pre-v7: Extract ALL system messages from anywhere in the conversation,
    merge their text content with `"\\n\\n"`, and filter them out of `loop_messages`.

    For v7+: Keep system messages in `loop_messages` so they are handled individually
    in the message processing loop. Only emit default system prompt if no system
    message is at position 0.

    Args:
        config: The template configuration.

    Returns:
        The system prompt handling section of the chat template.
    """
    lines = [
        "",
        "{#- Handle system prompt if it exists. #}",
    ]

    if config.uses_system_prompt_tokens:
        lines.extend(_generate_system_prompt_handling_v7_plus(config))
    else:
        lines.extend(_generate_system_prompt_handling_pre_v7(config))

    return "\n".join(lines)


def _generate_system_prompt_handling_pre_v7(config: TemplateConfig) -> list[str]:
    r"""Generate pre-v7 system prompt handling that extracts all system messages.

    Loops through all messages to collect system content and filters them
    out of `loop_messages`.

    Args:
        config: The template configuration.

    Returns:
        Lines of Jinja2 template code for pre-v7 system prompt handling.
    """
    lines = [
        "{#- Extract all system messages and merge into system prompt. #}",
        "{%- set ns_sys = namespace(system_parts=[], filtered=[]) %}",
        "{%- for message in messages %}",
        "    {%- if message['role'] == 'system' %}",
        "        {%- if message['content'] is string %}",
        "            {%- set ns_sys.system_parts = ns_sys.system_parts + [message['content']] %}",
        "        {%- else %}",
        "            {%- for block in message['content'] %}",
        "                {%- if block['type'] == 'text' %}",
        "                    {%- set ns_sys.system_parts = ns_sys.system_parts + [block['text']] %}",
        "                {%- else %}",
        "                    {{- raise_exception('Only text chunks are supported in system message content.') }}",
        "                {%- endif %}",
        "            {%- endfor %}",
        "        {%- endif %}",
        "    {%- else %}",
        "        {%- set ns_sys.filtered = ns_sys.filtered + [message] %}",
        "    {%- endif %}",
        "{%- endfor %}",
        "{%- if ns_sys.system_parts | length > 0 %}",
        "    {%- set system_message = ns_sys.system_parts | join('\\n\\n') %}",
        "{%- else %}",
        "    {%- set system_message = default_system_message %}",
        "{%- endif %}",
        "{%- set loop_messages = ns_sys.filtered %}",
    ]
    return lines


def _generate_system_prompt_handling_v7_plus(config: TemplateConfig) -> list[str]:
    r"""Generate v7+ system prompt handling.

    For v7+, system messages stay in `loop_messages` and are handled by the
    aggregation pre-pass (which coalesces their `TextChunks`) and the message
    processing loop (which emits `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`).

    Sets `loop_messages = messages` (no filtering needed since system messages
    are kept in place) and emits the default system prompt if the first message
    is not a system message.

    Args:
        config: The template configuration.

    Returns:
        Lines of Jinja2 template code for v7+ system prompt handling.
    """
    lines = [
        "{%- set loop_messages = messages %}",
        "{%- if messages[0]['role'] != 'system' and default_system_message != '' %}",
        "    {{- '" + _BEGIN_SYSTEM + "' + default_system_message + '" + _END_SYSTEM + "' }}",
        "{%- endif %}",
    ]

    if config.tracks_has_sp_for_audio:
        lines.extend(
            [
                "{%- if messages[0]['role'] == 'system' or default_system_message != '' %}",
                "    {%- set has_sp = true %}",
                "{%- else %}",
                "    {%- set has_sp = false %}",
                "{%- endif %}",
            ]
        )

    return lines


def _generate_available_tools_definition(config: TemplateConfig) -> str:
    r"""Generate available tools and model settings definition.

    Builds an `available_tools` string variable that contains
    `[AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]` (if tools are provided).
    For v15+, also builds a separate `model_settings` variable containing
    `[MODEL_SETTINGS]...[/MODEL_SETTINGS]` which is always emitted
    (defaults to `reasoning_effort="none"` when not specified).
    Both variables are emitted later in the message loop at the appropriate
    user message position.

    Args:
        config: The template configuration.

    Returns:
        The tools and settings definition section of the chat template.
    """
    if config.supports_model_settings:
        comment = "{#- Tools and model settings definition #}"
    else:
        comment = "{#- Tools definition #}"

    lines = [
        "",
        "",
        comment,
        "{%- set available_tools = '' %}",
        "{%- set has_tools = false %}",
    ]

    if config.has_tools:
        lines.append("{%- if tools is defined and tools is not none and tools|length > 0 %}")
        lines.append("    {%- set has_tools = true %}")
        if config.spm:
            lines.append(
                "    {%- set available_tools = '" + _BEGIN_TOOLS + " ' + (tools| tojson) + '" + _END_TOOLS + "' %}"
            )
        else:
            lines.append(
                "    {%- set available_tools = '" + _BEGIN_TOOLS + "' + (tools| tojson) + '" + _END_TOOLS + "' %}"
            )
        lines.append("{%- endif %}")

    if config.supports_model_settings:
        lines.extend(
            [
                "{%- if reasoning_effort is not defined or reasoning_effort is none %}",
                "    {%- set reasoning_effort = 'none' %}",
                "{%- endif %}",
                "{%- if reasoning_effort not in ['none', 'high'] %}",
                '    {{- raise_exception(\'reasoning_effort must be either "none" or "high"\') }}',
                "{%- endif %}",
                "{%- set model_settings = '"
                + _BEGIN_MODEL_SETTINGS
                + '{"reasoning_effort": "\' + reasoning_effort + \'"}'
                + _END_MODEL_SETTINGS
                + "' %}",  # noqa: E501
            ]
        )

    return "\n".join(lines)


def _generate_macros(config: TemplateConfig) -> str:
    r"""Generate Jinja2 macro definitions for the chat template.

    Defines a unified `render_content` macro that handles content rendering
    for all message roles. The macro accepts content as either a plain string
    or a list of blocks, iterating blocks and dispatching by type based on
    boolean flags.

    Only the branches relevant to the current template config are emitted.
    For configs with SPM image tracking (`uses_spm_prev_img_tracking`), an
    `initial_prev_img` parameter enables inter-block spacing logic.

    Args:
        config: The template configuration.

    Returns:
        The macro definitions section of the chat template. Returns empty
        string only for v1 templates.
    """
    if config.is_v1:
        return ""

    has_extra_types = config.any_thinking_support or config.image_support or config.audio_support

    # --- Build parameter list ---
    params = ["content", "context_name"]
    if has_extra_types:
        params.append("supported_types_desc")
    if config.any_thinking_support:
        params.append("support_thinking")
    if config.image_support:
        params.append("support_images")
    if config.audio_support:
        params.append("support_audio")
    if config.uses_spm_prev_img_tracking:
        params.append("initial_prev_img")

    lines = [
        "",
        "{#- Macros #}",
        "{%- macro render_content(" + ", ".join(params) + ") -%}",
        "    {%- if content is string -%}",
        "        {{- content -}}",
        "    {%- elif content -%}",
    ]

    # SPM prev_img tracking: create local namespace
    if config.uses_spm_prev_img_tracking:
        lines.append("        {%- set _ns = namespace(prev_img=initial_prev_img) -%}")

    lines.append("        {%- for block in content -%}")

    # SPM prev_img: add space before text if previous block was image
    if config.uses_spm_prev_img_tracking:
        lines.append("            {%- if _ns.prev_img and block['type'] == 'text' -%}")
        lines.append("                {{- ' ' -}}")
        lines.append("            {%- endif -%}")

    lines.extend(
        [
            "            {%- if block['type'] == 'text' -%}",
            "                {{- block['text'] -}}",
        ]
    )

    # SPM prev_img: reset after text block
    if config.uses_spm_prev_img_tracking:
        lines.append("                {%- set _ns.prev_img = false -%}")

    # Thinking branch
    if config.any_thinking_support:
        begin_tag = "<think>" if config.plain_thinking_support else _BEGIN_THINK
        end_tag = "</think>" if config.plain_thinking_support else _END_THINK
        lines.extend(
            [
                "            {%- elif support_thinking and block['type'] == 'thinking' -%}",
                "                {{- '" + begin_tag + "' + block['thinking'] -}}",
                "                {%- if block.get('closed', true) -%}{{- '" + end_tag + "' -}}{%- endif -%}",
            ]
        )

    # Image branch
    if config.image_support:
        lines.append("            {%- elif support_images and block['type'] in ['image', 'image_url'] -%}")
        lines.append("                {{- '[IMG]' -}}")
        if config.uses_spm_prev_img_tracking:
            lines.append("                {%- set _ns.prev_img = true -%}")

    # Audio branch
    if config.audio_support:
        lines.append("            {%- elif support_audio and block['type'] in ['input_audio', 'audio_url'] -%}")
        if config.tracks_has_sp_for_audio:
            lines.append("                {%- if has_sp -%}")
            lines.append(
                "                    {{- raise_exception('Audio chunks are not supported in user message content when system prompt is provided.') -}}"  # noqa: E501
            )
            lines.append("                {%- endif -%}")
        lines.append("                {{- '[AUDIO]' -}}")

    # Error branch
    if has_extra_types:
        lines.append("            {%- else -%}")
        lines.append(
            "                {{- raise_exception('Only ' + supported_types_desc + ' chunks are supported in ' + context_name + '.') -}}"  # noqa: E501
        )
    else:
        lines.append("            {%- else -%}")
        lines.append(
            "                {{- raise_exception('Only text chunks are supported in ' + context_name + '.') -}}"
        )

    lines.extend(
        [
            "            {%- endif -%}",
            "        {%- endfor -%}",
            "    {%- else -%}",
            "        {{- raise_exception(context_name + ' must have non-empty content.') -}}",
            "    {%- endif -%}",
            "{%- endmacro -%}",
        ]
    )

    return "\n".join(lines)


def _emit_int_float_parsing(indent: str) -> list[str]:
    r"""Generate the Jinja2 block for parsing `message['content']` as int or float.

    Attempts int parsing first, then float, falling back to the string
    representation. Used in tool message handling to preserve numeric types
    in JSON-serialized tool results.

    Args:
        indent: Whitespace prefix for each emitted line.

    Returns:
        Lines of Jinja2 template code for int/float content parsing.
    """
    return [
        f"{indent}{{# Try to parse 'content' as int or float if possible #}}",
        f"{indent}{{%- set tool_content = message['content']|string %}}",
        f"{indent}{{# Try to parse as int #}}",
        f"{indent}{{%- set parsed_int = message['content']|int %}}",
        f"{indent}{{% if parsed_int|string == message['content'] %}}",
        f"{indent}    {{%- set tool_content = parsed_int %}}",
        f"{indent}{{# If int fails, try to parse as float #}}",
        f"{indent}{{%- else %}}",
        f"{indent}    {{%- set parsed_float = message['content']|float %}}",
        f"{indent}    {{%- if parsed_float|string == message['content'] %}}",
        f"{indent}        {{%- set tool_content = parsed_float %}}",
        f"{indent}    {{%- endif %}}",
        f"{indent}{{%- endif %}}",
    ]


def _emit_call_id_resolution(indent: str) -> list[str]:
    r"""Generate the Jinja2 block for extracting `tool_id` from a tool message.

    Checks `message['call_id']` first, then `message['tool_call_id']`,
    requiring exactly 9 characters. Raises a template exception if neither
    field is present or valid.

    Args:
        indent: Whitespace prefix for each emitted line.

    Returns:
        Lines of Jinja2 template code for call ID resolution.
    """
    return [
        f"{indent}{{%- if message['call_id'] is not undefined and message['call_id']|length == 9 %}}",
        f"{indent}    {{%- set tool_id = message['call_id'] %}}",
        f"{indent}{{%- elif message['tool_call_id'] is not undefined and message['tool_call_id']|length == 9  %}}",
        f"{indent}    {{%- set tool_id = message['tool_call_id'] %}}",
        f"{indent}{{%- else %}}",
        f"{indent}    {{{{- raise_exception('Tool message must have a call_id or tool_call_id of 9 characters or numbers.') }}}}",  # noqa: E501
        f"{indent}{{%- endif %}}",
    ]


def _emit_argument_normalization(indent: str) -> list[str]:
    r"""Generate the Jinja2 block for normalizing tool call `arguments`.

    Converts non-string arguments to JSON via `tojson|safe`, and replaces
    empty-string arguments with `'{}'`.

    Args:
        indent: Whitespace prefix for each emitted line.

    Returns:
        Lines of Jinja2 template code for argument normalization.
    """
    return [
        f"{indent}{{%- if arguments is not string %}}",
        f"{indent}    {{%- set arguments = arguments|tojson|safe %}}",
        f"{indent}{{%- elif arguments == '' %}}",
        f"{indent}    {{%- set arguments = '{{}}' %}}",
        f"{indent}{{%- endif %}}",
    ]


def _emit_call_id_validation(indent: str) -> list[str]:
    r"""Generate the Jinja2 block for validating `tool['id']` in tool calls.

    Checks that `tool['id']` is defined and has exactly 9 characters,
    raising a template exception otherwise.

    Args:
        indent: Whitespace prefix for each emitted line.

    Returns:
        Lines of Jinja2 template code for call ID validation.
    """
    return [
        f"{indent}{{%- set id = tool['id']%}}",
        f"{indent}{{%- if id is not defined or id|length != 9 %}}",
        f"{indent}    {{{{- raise_exception('Tool call must have an id of 9 characters or numbers.') }}}}",
        f"{indent}{{%- endif %}}",
    ]


def _generate_reasoning_to_thinking_inline() -> list[str]:
    r"""Generate inline reasoning-to-thinking conversion inside the aggregation loop.

    Emitted inside the `for msg in ns_agg.current_group` loop when
    `config.any_thinking_support` is enabled. For each assistant message, converts
    a top-level `reasoning_content` or `reasoning` field into a leading
    `{"type": "thinking", "thinking": ...}` chunk prepended to the content,
    so that reasoning traces from third-party APIs (OpenAI, DeepSeek, vLLM, …)
    are aggregated uniformly with inline `ThinkChunk`\s.

    `reasoning_content` takes precedence over `reasoning` when both are present.

    Raises at template render time if the message already contains thinking chunks
    in its content alongside a `reasoning` or `reasoning_content` field.

    Returns:
        Lines of Jinja2 template code for the inline conversion.
    """
    return [
        "                {#- Convert reasoning / reasoning_content to a leading thinking chunk. #}",
        "                {%- set reasoning = msg.get('reasoning_content', msg.get('reasoning', none)) %}",
        "                {%- if reasoning is not none and reasoning != '' %}",
        "                    {%- if msg['content'] is not none and msg['content'] is not string %}",
        "                        {%- for block in msg['content'] %}",
        "                            {%- if block['type'] == 'thinking' %}",
        "                                {{- raise_exception('Message cannot have both thinking chunks in content and a top-level `reasoning` or `reasoning_content` field.') }}",  # noqa: E501
        "                            {%- endif %}",
        "                        {%- endfor %}",
        "                    {%- endif %}",
        "                    {%- set think_chunk = {'type': 'thinking', 'thinking': reasoning} %}",
        "                    {%- if msg['content'] is string and msg['content'] != '' %}",
        "                        {%- set new_content = [think_chunk, {'type': 'text', 'text': msg['content']}] %}",
        "                    {%- elif msg['content'] is not none and msg['content'] is not string and msg['content'] | length > 0 %}",  # noqa: E501
        "                        {%- set new_content = [think_chunk] + msg['content'] | list %}",
        "                    {%- else %}",
        "                        {%- set new_content = [think_chunk] %}",
        "                    {%- endif %}",
        "                    {%- if msg['tool_calls'] is defined and msg['tool_calls'] is not none %}",
        "                        {%- set msg = {'role': msg['role'], 'content': new_content, 'tool_calls': msg['tool_calls']} %}",  # noqa: E501
        "                    {%- else %}",
        "                        {%- set msg = {'role': msg['role'], 'content': new_content} %}",
        "                    {%- endif %}",
        "                {%- endif %}",
    ]


def _generate_message_aggregation(config: TemplateConfig) -> str:
    r"""Generate message aggregation pre-processing block.

    Aggregates consecutive messages with the same role before the alternation
    check to match the behavior of the mistral-common normalizer:

    - **User messages**: consecutive messages merged into one, text content joined
      with `"\\n\\n"`, non-text chunks (image, audio) preserved in order.
    - **Assistant messages**: consecutive messages merged into one, text content joined
      with `"\\n\\n"`, `tool_calls` lists concatenated.
    - **System messages**: each message coalesced individually (adjacent `TextChunks`
      joined with `"\\n\\n"`), but NOT merged across consecutive messages.
    - **Tool messages**: passed through as-is.

    Non-text chunk type validation is deferred to the message rendering loop.

    The grouping logic ensures system and tool messages always form single-message
    groups (they break any ongoing group), so the flush logic is role-agnostic:
    it always coalesces the group into one output message.

    Args:
        config: The template configuration.

    Returns:
        The message aggregation section of the chat template.
    """
    lines = [
        "",
        "{#- Aggregate consecutive messages with the same role except system and tool. #}",
        "{#- A sentinel message is appended so the last group gets flushed inside the loop. #}",
        "{%- set ns_agg = namespace(messages=[], current_group=[], current_role=none) %}",
        "{%- for message in loop_messages + [{'role': '__sentinel__'}] %}",
        "    {%- if message['role'] != ns_agg.current_role or message['role'] == 'system' or message['role'] == 'tool' %}",  # noqa: E501
    ]

    lines.extend(_generate_flush_logic(config))

    lines.extend(
        [
            "        {%- if message['role'] != '__sentinel__' %}",
            "            {%- set ns_agg.current_group = [message] %}",
            "            {%- set ns_agg.current_role = message['role'] %}",
            "        {%- endif %}",
            "    {%- else %}",
            "        {%- set ns_agg.current_group = ns_agg.current_group + [message] %}",
            "    {%- endif %}",
            "{%- endfor %}",
            "{%- set loop_messages = ns_agg.messages %}",
        ]
    )

    return "\n".join(lines)


def _generate_flush_logic(config: TemplateConfig) -> list[str]:
    r"""Generate the flush logic for aggregating a group of same-role messages.

    Called when the role changes (or at end of messages) to coalesce all messages
    in the current group into a single output message. Adjacent `TextChunk`\s are
    joined with `"\\n\\n"`, non-text chunks are preserved as barriers, and
    `tool_calls` from all messages in the group are concatenated. Chunk type
    validation is deferred to the message rendering loop.

    When `config.any_thinking_support` is enabled, `reasoning_content` /
    `reasoning` fields on assistant messages are converted to a leading thinking
    chunk prepended to the message content **inside** the aggregation loop, so that
    reasoning traces are aggregated uniformly with inline `ThinkChunk`\s.

    The logic is role-agnostic for user, assistant, and system: system messages
    always form single-message groups (enforced by the grouping logic), so they
    are effectively coalesced individually. Tool messages pass through as-is to
    preserve extra fields like `tool_call_id` and `name`.

    Args:
        config: The template configuration.

    Returns:
        Lines of Jinja2 template code for flushing the current aggregation group.
    """
    lines = [
        "        {%- if ns_agg.current_role == 'tool' %}",
        "            {%- set ns_agg.messages = ns_agg.messages + ns_agg.current_group %}",
        "        {%- elif ns_agg.current_role is not none %}",
        "            {%- set ns_c = namespace(text_parts=[], chunks=[], has_non_text=false, tool_calls=[]) %}",
        "            {%- for msg in ns_agg.current_group %}",
    ]

    if config.any_thinking_support:
        lines.extend(_generate_reasoning_to_thinking_inline())

    # Build list-content processing block: v15+ joins intra-message TextChunks with ""
    # (no separator), pre-v15 joins all TextChunks with "\n\n" (same as inter-message).
    if config.version >= TokenizerVersion.v15:
        list_content_lines = [
            "                {%- elif msg['content'] is not none %}",
            "                    {%- set ns_msg = namespace(msg_text_parts=[]) %}",
            "                    {%- for block in msg['content'] %}",
            "                        {%- if block['type'] == 'text' %}",
            "                            {%- set ns_msg.msg_text_parts = ns_msg.msg_text_parts + [block['text']] %}",
            "                        {%- else %}",
            "                            {%- if ns_msg.msg_text_parts | length > 0 %}",
            "                                {%- set ns_c.text_parts = ns_c.text_parts + [ns_msg.msg_text_parts | join('')] %}",  # noqa: E501
            "                                {%- set ns_msg.msg_text_parts = [] %}",
            "                            {%- endif %}",
            "                            {%- if ns_c.text_parts | length > 0 %}",
            "                                {%- set ns_c.chunks = ns_c.chunks + [{'type': 'text', 'text': ns_c.text_parts | join('\\n\\n')}] %}",  # noqa: E501
            "                                {%- set ns_c.text_parts = [] %}",
            "                            {%- endif %}",
            "                            {%- set ns_c.chunks = ns_c.chunks + [block] %}",
            "                            {%- set ns_c.has_non_text = true %}",
            "                        {%- endif %}",
            "                    {%- endfor %}",
            "                    {%- if ns_msg.msg_text_parts | length > 0 %}",
            "                        {%- set ns_c.text_parts = ns_c.text_parts + [ns_msg.msg_text_parts | join('')] %}",
            "                    {%- endif %}",
            "                {%- endif %}",
        ]
    else:
        list_content_lines = [
            "                {%- elif msg['content'] is not none %}",
            "                    {%- for block in msg['content'] %}",
            "                        {%- if block['type'] == 'text' %}",
            "                            {%- set ns_c.text_parts = ns_c.text_parts + [block['text']] %}",
            "                        {%- else %}",
            "                            {%- if ns_c.text_parts | length > 0 %}",
            "                                {%- set ns_c.chunks = ns_c.chunks + [{'type': 'text', 'text': ns_c.text_parts | join('\\n\\n')}] %}",  # noqa: E501
            "                                {%- set ns_c.text_parts = [] %}",
            "                            {%- endif %}",
            "                            {%- set ns_c.chunks = ns_c.chunks + [block] %}",
            "                            {%- set ns_c.has_non_text = true %}",
            "                        {%- endif %}",
            "                    {%- endfor %}",

            "                {%- endif %}",
        ]

    lines.extend(
        [
            "                {%- if msg['content'] is string %}",
            "                    {%- set ns_c.text_parts = ns_c.text_parts + [msg['content']] %}",
            *list_content_lines,
            "                {%- if msg['tool_calls'] is defined and msg['tool_calls'] is not none %}",
            "                    {%- set ns_c.tool_calls = ns_c.tool_calls + msg['tool_calls'] | list %}",
            "                {%- endif %}",
            "            {%- endfor %}",
            # Finalize: flush remaining text, build merged_content
            "            {%- if ns_c.has_non_text %}",
            "                {%- if ns_c.text_parts | length > 0 %}",
            "                    {%- set ns_c.chunks = ns_c.chunks + [{'type': 'text', 'text': ns_c.text_parts | join('\\n\\n')}] %}",  # noqa: E501
            "                {%- endif %}",
            "                {%- set merged_content = ns_c.chunks %}",
            "            {%- else %}",
            "                {%- set merged_content = ns_c.text_parts | join('\\n\\n') %}",
            "            {%- endif %}",
            # Emit the merged message with role and optional tool_calls
            "            {%- if ns_c.tool_calls | length > 0 %}",
            "                {%- set ns_agg.messages = ns_agg.messages + [{'role': ns_agg.current_role, 'content': merged_content, 'tool_calls': ns_c.tool_calls}] %}",  # noqa: E501
            "            {%- else %}",
            "                {%- set ns_agg.messages = ns_agg.messages + [{'role': ns_agg.current_role, 'content': merged_content}] %}",  # noqa: E501
            "            {%- endif %}",
            "        {%- endif %}",
        ]
    )
    return lines


def _generate_system_message_handling(config: TemplateConfig) -> str:
    r"""Generate system message handling in the message loop for v7+.

    Emits `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]` for each system message
    encountered during message processing.

    Args:
        config: The template configuration.

    Returns:
        The system message handling section for the message loop, or empty string for pre-v7.
    """
    if not config.uses_system_prompt_tokens:
        return ""

    lines = [
        "",
        "    {#- System messages. #}",
        "    {%- elif message['role'] == 'system' %}",
    ]

    if config.spm_system_prompt_has_space:
        lines.append("        {{- '" + _BEGIN_SYSTEM + " ' -}}")
    else:
        lines.append("        {{- '" + _BEGIN_SYSTEM + "' -}}")

    has_extra_types = config.any_thinking_support or config.image_support or config.audio_support
    rc_args = "message['content'], 'system message contents'"
    if has_extra_types:
        if config.system_supports_thinking:
            rc_args += ", supported_types_desc='text and thinking'"
        elif config.system_supports_audio:
            rc_args += ", supported_types_desc='text and audio'"
        else:
            rc_args += ", supported_types_desc='text'"
    if config.any_thinking_support:
        if config.system_supports_thinking:
            rc_args += ", support_thinking=true"
        else:
            rc_args += ", support_thinking=false"
    if config.image_support:
        rc_args += ", support_images=false"
    if config.audio_support:
        rc_args += f", support_audio={'true' if config.system_supports_audio else 'false'}"
    lines.append("        {{- render_content(" + rc_args + ") -}}")

    lines.append("        {{- '" + _END_SYSTEM + "' -}}")

    return "\n".join(lines)


def _generate_alternation_check(config: TemplateConfig) -> str:
    r"""Generate message ordering validation using a role transition table.

    Validates that message roles follow a valid ordering, matching the rules
    from `mistral_common.protocol.instruct.validator._validate_message_order`.

    The transition table defines which roles can follow each previous role:

    - After `system`: `user`, `assistant`, `system`
    - After `user`: `assistant`, `system`, `user`
    - After `assistant`: `assistant`, `user`, `tool`
    - After `tool`: `assistant`, `tool`, `user`

    For pre-v7 templates, system messages are extracted before this check,
    so only `user`, `assistant`, and `tool` roles are seen.

    Args:
        config: The template configuration.

    Returns:
        The message ordering validation section of the chat template.
    """
    lines = [
        "",
        "{#- Validates message ordering. #}",
    ]

    ns_vars: list[str] = []
    if config.tracks_max_idx_user:
        ns_vars.append("max_idx_user=-1")
    if config.uses_spm_space_tracking:
        ns_vars.append("add_space=false")
    if config.uses_v2_spm_tool_format:
        ns_vars.append("prev_tool=false")
    if config.tools_at_beginning:
        emitted_var = (
            "available_tools_and_settings_emitted" if config.supports_model_settings else "available_tools_emitted"
        )
        ns_vars.append(f"{emitted_var}=false")

    lines.append("{%- set ns = namespace(" + ", ".join(ns_vars) + ") %}")

    # First-message constraint
    if config.uses_system_prompt_tokens:
        # v7+: system messages remain in loop_messages, so first message can be user or system
        first_msg_cond = "{%- if loop_messages | length > 0 and loop_messages[0]['role'] not in ['user', 'system'] %}"
        first_msg_err = "    {{- raise_exception('Conversation must start with a user or system message, got ' + loop_messages[0]['role'] + '.') }}"  # noqa: E501
    else:
        # pre-v7: system messages are extracted, so first message must be user
        first_msg_cond = "{%- if loop_messages | length > 0 and loop_messages[0]['role'] != 'user' %}"
        first_msg_err = "    {{- raise_exception('Conversation must start with a user message, got ' + loop_messages[0]['role'] + '.') }}"  # noqa: E501
    lines.append(first_msg_cond)
    lines.append(first_msg_err)
    lines.append("{%- endif %}")

    # Transition table validation
    lines.append("{%- set ns_order = namespace(previous_role=none) %}")
    lines.append("{%- for message in loop_messages %}")
    lines.append("    {%- set current_role = message['role'] %}")
    lines.append("    {%- if ns_order.previous_role is not none %}")

    transition_error = "                {{- raise_exception('Unexpected role \\'' + current_role + '\\' after role \\'' + ns_order.previous_role + '\\'') }}"  # noqa: E501

    if config.uses_system_prompt_tokens:
        # v7+: full transition table including system
        lines.append("        {%- if ns_order.previous_role == 'system' %}")
        lines.append("            {%- if current_role not in ['user', 'assistant', 'system'] %}")
        lines.append(transition_error)
        lines.append("            {%- endif %}")
        lines.append("        {%- elif ns_order.previous_role == 'user' %}")
        lines.append("            {%- if current_role not in ['assistant', 'system', 'user'] %}")
    else:
        # pre-v7: no system in loop_messages
        lines.append("        {%- if ns_order.previous_role == 'user' %}")
        lines.append("            {%- if current_role not in ['assistant', 'user'] %}")

    lines.append(transition_error)
    lines.append("            {%- endif %}")
    lines.append("        {%- elif ns_order.previous_role == 'assistant' %}")
    lines.append("            {%- if current_role not in ['assistant', 'user', 'tool'] %}")
    lines.append(transition_error)
    lines.append("            {%- endif %}")
    lines.append("        {%- elif ns_order.previous_role == 'tool' %}")
    lines.append("            {%- if current_role not in ['assistant', 'tool', 'user'] %}")
    lines.append(transition_error)
    lines.append("            {%- endif %}")
    lines.append("        {%- endif %}")
    lines.append("    {%- endif %}")

    # Track max_idx_user in the same loop
    if config.tracks_max_idx_user:
        lines.append("    {%- if message.role == 'user' %}")
        lines.append("        {%- set ns.max_idx_user = ns.max_idx_user + 1 %}")
        lines.append("    {%- endif %}")

    lines.append("    {%- set ns_order.previous_role = current_role %}")
    lines.append("{%- endfor %}")

    return "\n".join(lines)


def _generate_user_message_handling(config: TemplateConfig) -> str:
    r"""Generate user message handling section.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        Jinja2 template lines for user message processing.
    """
    lines = []

    chunk_types = ["text"]
    if config.image_support:
        chunk_types.extend(["image", "image_url"])
    if config.audio_support:
        chunk_types.extend(["input_audio", "audio_url"])

    chunk_desc = ", ".join(chunk_types[:-1]) + " and " + chunk_types[-1] if len(chunk_types) > 1 else chunk_types[0]
    comment = f"{{#- User messages supports {chunk_desc} content. #}}"

    lines.append(f"    {comment}")
    lines.append("    {%- if message['role'] == 'user' %}")

    # Tools and model settings placement
    if config.tracks_max_idx_user:
        # Pre-v13: emit before the last user message
        lines.append("        {%- if (ns.index == ns.max_idx_user) and has_tools %}")
        lines.append("            {{- available_tools }}")
        lines.append("        {%- endif %}")
    elif config.tools_at_beginning:
        # v13+: emit tools and settings before the first user message
        emitted_var = (
            "available_tools_and_settings_emitted" if config.supports_model_settings else "available_tools_emitted"
        )
        lines.append(f"        {{%- if not ns.{emitted_var} %}}")
        lines.append("            {{- available_tools }}")
        if config.supports_model_settings:
            lines.append("            {{- model_settings }}")
        lines.append(f"            {{%- set ns.{emitted_var} = true %}}")
        lines.append("        {%- endif %}")

    if config.spm and not config.uses_spm_prev_img_tracking:
        inst_open = _BEGIN_INST + " "
        inst_close = _END_INST
    else:
        inst_open = _BEGIN_INST
        inst_close = _END_INST

    has_extra_types = config.any_thinking_support or config.image_support or config.audio_support

    if config.uses_system_prompt_tokens:
        # =================================================================
        # v7+ path: self-contained, no tag/render_content duplication
        # =================================================================
        needs_content_prep = config.image_support or (
            config.version >= TokenizerVersion.v13 and not config.audio_support
        )

        if needs_content_prep:
            # --- Content preparation: sort/reorder list content ---
            lines.append("        {%- if message['content'] is not string and message['content'] %}")

            if config.image_support:
                lines.append(
                    "            {#- When content has exactly one image and one text block, put image first. #}"
                )
                lines.append(
                    "            {%- if message['content'] | length == 2 and message['content'][0]['type'] == 'text' and message['content'][1]['type'] in ['image', 'image_url'] %}"  # noqa: E501
                )
                lines.append("                {%- set blocks = [message['content'][1], message['content'][0]] %}")
                lines.append("            {%- else %}")
                lines.append("                {%- set blocks = message['content'] %}")
                lines.append("            {%- endif %}")
                lines.append("            {%- set user_content = blocks %}")
            else:
                # v13+ non-audio: sort blocks by type
                lines.append("            {%- set user_content = message['content'] | sort(attribute='type') %}")

            if config.uses_spm_prev_img_tracking:
                # SPM + prev_img: list content gets [INST] without space
                lines.append("            {%- set inst_tag = '" + _BEGIN_INST + "' %}")

            lines.append("        {%- else %}")
            lines.append("            {%- set user_content = message['content'] %}")

            if config.uses_spm_prev_img_tracking:
                # SPM + prev_img: string content gets [INST] with space
                lines.append("            {%- set inst_tag = '" + _BEGIN_INST + " ' %}")

            lines.append("        {%- endif %}")

        # --- Emit [INST] tag ---
        if config.uses_spm_prev_img_tracking:
            lines.append("        {{- inst_tag -}}")
        elif config.spm:
            lines.append("        {{- '" + _BEGIN_INST + " ' -}}")
        else:
            lines.append("        {{- '" + _BEGIN_INST + "' -}}")

        # --- Build unified render_content args ---
        content_var = "user_content" if needs_content_prep else "message['content']"
        rc_args = content_var + ", 'user message content'"
        if has_extra_types:
            if config.image_support:
                rc_args += ", supported_types_desc='text, image and image_url'"
            elif config.audio_support:
                rc_args += ", supported_types_desc='text, input_audio and audio_url'"
            else:
                rc_args += ", supported_types_desc='text'"
        if config.any_thinking_support:
            rc_args += ", support_thinking=false"
        if config.image_support:
            rc_args += ", support_images=true"
        if config.audio_support:
            rc_args += ", support_audio=true"
        if config.uses_spm_prev_img_tracking:
            rc_args += ", initial_prev_img=not added_sp"

        # --- Emit render_content + [/INST] (once each) ---
        lines.append("        {{- render_content(" + rc_args + ") -}}")
        lines.append("        {{- '" + inst_close + "' }}")

    else:
        # =================================================================
        # Pre-v7 paths: uses_spm_prev_img_tracking (v3_image_spm) or plain
        # Keep existing structure with shared image/sorting code
        # =================================================================
        if config.uses_spm_prev_img_tracking:
            lines.append(f"        {{{{- '{inst_open}' }}}}")
            lines.append("        {%- set added_sp = false %}")
            lines.append("        {%- if (ns.index == ns.max_idx_user) and system_message != '' %}")
            lines.append("            {%- set added_sp = true %}")
            lines.append("            {{- ' ' + system_message + '\\n\\n' }}")
            lines.append("        {%- elif message['content'] is string %}")
            lines.append("            {{- ' '  }}")
            lines.append("        {%- endif %}")
            lines.append("")
            lines.append("")
            lines.append("        {%- if message['content'] is string %}")
            lines.append("            {{- render_content(message['content'], 'user message content') -}}")
            lines.append("        {%- elif message['content'] | length > 0 %}")
        else:
            lines.append(f"        {{{{- '{inst_open}' }}}}")
            if config.is_v1:
                lines.append("        {%- if loop.index0 == 0 and system_message != '' %}")
            else:
                lines.append("        {%- if (ns.index == ns.max_idx_user) and system_message != '' %}")
            lines.append("            {{- system_message + '\\n\\n' }}")
            lines.append("        {%- endif %}")
            lines.append("        {%- if message['content'] is string %}")
            lines.append("            {{- render_content(message['content'], 'user message content') -}}")
            lines.append("        {%- elif message['content'] | length > 0 %}")

        # Shared: image sorting + render_content for list case (pre-v7 only)
        if config.image_support:
            lines.append("            {#- When content has exactly one image and one text block, put image first. #}")
            lines.append(
                "            {%- if message['content'] | length == 2 and message['content'][0]['type'] == 'text' and message['content'][1]['type'] in ['image', 'image_url'] %}"  # noqa: E501
            )
            lines.append("                {%- set blocks = [message['content'][1], message['content'][0]] %}")
            lines.append("            {%- else %}")
            lines.append("                {%- set blocks = message['content'] %}")
            lines.append("            {%- endif %}")
            block_var = "blocks"
        else:
            block_var = "message['content']"

        rc_call_args = block_var + ", 'user message content'"
        if has_extra_types:
            if config.image_support:
                rc_call_args += ", supported_types_desc='text, image and image_url'"
            elif config.audio_support:
                rc_call_args += ", supported_types_desc='text, input_audio and audio_url'"
            else:
                rc_call_args += ", supported_types_desc='text'"
        if config.any_thinking_support:
            rc_call_args += ", support_thinking=false"
        if config.image_support:
            rc_call_args += ", support_images=true"
        if config.audio_support:
            rc_call_args += ", support_audio=true"
        if config.uses_spm_prev_img_tracking:
            rc_call_args += ", initial_prev_img=not added_sp"
        lines.append("            {{- render_content(" + rc_call_args + ") -}}")

        # Closing
        lines.append("        {%- else %}")
        lines.append(
            "            {{- raise_exception('User message must have a string or a list of chunks in content') }}"
        )
        lines.append("        {%- endif %}")
        lines.append(f"        {{{{- '{inst_close}' }}}}")

    if config.uses_spm_space_tracking:
        lines.append("        {%- if loop.index < loop.length %}")
        lines.append("            {%- set ns.add_space=true %}")
        lines.append("        {%- endif %}")
        lines.append("        {%- set ns.prev_tool=false %}")

    return "\n".join(lines)


def _generate_assistant_message_handling(config: TemplateConfig) -> str:
    r"""Generate assistant message handling section.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        Jinja2 template lines for assistant message processing.
    """
    lines = []

    if config.any_thinking_support:
        chunk_types = "text and thinking"
    else:
        chunk_types = "text"

    comment = f"{{#- Assistant messages supports {chunk_types} content. #}}"
    lines.append("")
    lines.append(f"    {comment}")
    lines.append("    {%- elif message['role'] == 'assistant' %}")

    if config.forbids_assistant_content_with_tools:
        lines.append(
            "        {%- if message['content'] is not none and message['content'] | length > 0 and message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}"  # noqa: E501
        )
        lines.append("            {{- raise_exception('Assistant message cannot have both content and tool calls.') }}")
        lines.append("        {%- endif %}")
        lines.append("")

    if config.validates_assistant_non_empty:
        lines.append(
            "        {%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}"  # noqa: E501
        )
        lines.append(
            "            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}"  # noqa: E501
        )
        lines.append("        {%- endif %}")
        lines.append("")

    has_extra_types = config.any_thinking_support or config.image_support or config.audio_support
    rc_call_args = "message['content'], 'assistant message contents'"
    if has_extra_types:
        desc_parts = ["text"]
        if config.any_thinking_support:
            desc_parts.append("thinking")
        if config.assistant_supports_multimodal and config.image_support:
            desc_parts.append("image")
        if config.assistant_supports_multimodal and config.audio_support:
            desc_parts.append("audio")
        rc_call_args += f", supported_types_desc='{_join_types_desc(desc_parts)}'"
    if config.any_thinking_support:
        rc_call_args += ", support_thinking=true"
    if config.image_support:
        rc_call_args += f", support_images={'true' if config.assistant_supports_multimodal else 'false'}"
    if config.audio_support:
        rc_call_args += f", support_audio={'true' if config.assistant_supports_multimodal else 'false'}"

    lines.append("        {%- if message['content'] %}")

    if config.uses_spm_space_tracking:
        lines.append("            {%- if ns.add_space %}")
        lines.append("                {{- ' ' }}")
        lines.append("                {%- set ns.add_space=false %}")
        lines.append("            {%- endif %}")

    lines.append("            {{- render_content(" + rc_call_args + ") -}}")

    if config.uses_v2_tool_format:
        lines.append("            {{- " + config.eos_expr + " }}")

    if config.uses_v2_v3spm_tool_branch:
        lines.append(_generate_tool_calls_elif_v2_v3(config))
        lines.append("        {%- else %}")
        lines.append(
            "            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}"  # noqa: E501
        )
        lines.append("        {%- endif %}")
    else:
        lines.append("        {%- endif %}")

    if config.has_tools and not config.uses_v2_v3spm_tool_branch:
        lines.append("")
        lines.append(_generate_tool_calls_block(config))

    if not config.uses_v2_tool_format:
        lines.append("")
        lines.append("        {{- " + config.eos_expr + " }}")

    if config.uses_spm_space_tracking:
        lines.append("        {%- set ns.prev_tool=false %}")

    return "\n".join(lines)


def _generate_tool_calls_elif_v2_v3(config: TemplateConfig) -> str:
    r"""Generate tool calls as elif branch for v2 (SPM and non-SPM) and v3_spm templates.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        Jinja2 template lines for v2/v3 tool call formatting.
    """
    lines = []

    if config.uses_v2_tool_format:
        lines.append(
            "        {%- elif message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 and ns.index > ns.max_idx_user %}"  # noqa: E501
        )
    else:
        lines.append(
            "        {%- elif message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}"  # noqa: E501
        )

    if config.spm:
        lines.append("            {%- if ns.add_space %}")
        lines.append("                {%- set ns.add_space=false %}")
        lines.append("            {%- endif %}")
        lines.append("            {{- '" + _TOOL_CALLS + " [' }}")
    else:
        lines.append("            {{- '" + _TOOL_CALLS + "[' }}")
    lines.append("            {%- for tool in message['tool_calls'] %}")
    lines.append("                {%- set name = tool['function']['name'] %}")
    lines.append("                {%- set arguments = tool['function']['arguments'] %}")

    if config.uses_v2_tool_format:
        # v2: no ID in tool calls
        lines.extend(_emit_argument_normalization("                "))
        lines.append("                {{- '{\"name\": \"' + name + '\", \"arguments\": ' + arguments + '}' }}")
    else:
        # v3_spm: has ID in tool calls
        lines.extend(_emit_call_id_validation("                "))
        lines.extend(_emit_argument_normalization("                "))
        lines.append(
            '                {{- \'{"name": "\' + name + \'", "arguments": \' + arguments + \', "id": "\' + id + \'"}\' }}'  # noqa: E501
        )

    lines.append("                {%- if loop.length > 1 and loop.index < loop.length %}")
    lines.append("                    {{- ', ' }}")
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")
    lines.append("            {{- ']' }}")
    # v2 has EOS inside the elif, v3_spm has EOS after the endif
    if config.uses_v2_tool_format:
        lines.append("            {{- " + config.eos_expr + " }}")

    if config.uses_v2_tool_format:
        # v2: additional elif for tool calls during user messages (ignored)
        lines.append(
            "        {%- elif message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 and ns.index <= ns.max_idx_user %}"  # noqa: E501
        )

    return "\n".join(lines)


def _generate_tool_calls_block(config: TemplateConfig) -> str:
    r"""Generate tool calls block (v7+ style).

    Args:
        config: Template configuration specifying version and features.

    Returns:
        Jinja2 template lines for v7+ tool call formatting.
    """
    lines = []
    lines.append(
        "        {%- if message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}"  # noqa: E501
    )

    if config.uses_spm_space_tracking:
        lines.append("            {%- if ns.add_space %}")
        lines.append("                {%- set ns.add_space=false %}")
        lines.append("            {%- endif %}")

    if config.uses_v13_tool_format:
        lines.append("            {%- for tool in message['tool_calls'] %}")
        lines.append("                {{- '" + _TOOL_CALLS + "' }}")
        lines.append("                {%- set name = tool['function']['name'] %}")
        lines.append("                {%- set arguments = tool['function']['arguments'] %}")
        lines.extend(_emit_argument_normalization("                "))
        lines.append("                {{- name + '" + _ARGS + "' + arguments }}")
        lines.append("            {%- endfor %}")
    elif config.uses_call_id_in_tool_calls:
        lines.append("            {%- for tool in message['tool_calls'] %}")
        lines.append("                {{- '" + _TOOL_CALLS + "' }}")
        lines.append("                {%- set name = tool['function']['name'] %}")
        lines.append("                {%- set arguments = tool['function']['arguments'] %}")
        lines.extend(_emit_call_id_validation("                "))
        lines.extend(_emit_argument_normalization("                "))
        lines.append("                {{- name + '" + _CALL_ID + "' + id + '" + _ARGS + "' + arguments }}")
        lines.append("            {%- endfor %}")
    else:
        if config.spm:
            lines.append("            {{- '" + _TOOL_CALLS + " [' }}")
        else:
            lines.append("            {{- '" + _TOOL_CALLS + "[' }}")
        lines.append("            {%- for tool in message['tool_calls'] %}")
        lines.append("                {%- set name = tool['function']['name'] %}")
        lines.append("                {%- set arguments = tool['function']['arguments'] %}")
        lines.extend(_emit_call_id_validation("                "))
        lines.extend(_emit_argument_normalization("                "))
        lines.append(
            '                {{- \'{"name": "\' + name + \'", "arguments": \' + arguments + \', "id": "\' + id + \'"}\' }}'  # noqa: E501
        )
        lines.append("                {%- if loop.length > 1 and loop.index < loop.length %}")
        lines.append("                    {{- ', ' }}")
        lines.append("                {%- endif %}")
        lines.append("            {%- endfor %}")
        lines.append("            {{- ']' }}")

    lines.append("        {%- endif %}")

    return "\n".join(lines)


def _generate_tool_message_handling(config: TemplateConfig) -> str:
    r"""Generate tool message handling section.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        Jinja2 template lines for tool result message processing.
    """
    if not config.has_tools:
        return ""

    lines = []
    lines.append("")

    if config.uses_v2_tool_format:
        lines.append("    {#- Tool messages supports int, float or text content. #}")
        lines.append("    {%- elif message['role'] == 'tool' and ns.index > ns.max_idx_user %}")
    else:
        lines.append("    {#- Tool messages only supports text content. #}")
        lines.append("    {%- elif message['role'] == 'tool' %}")

    if config.uses_spm_space_tracking:
        lines.append("        {%- if ns.add_space %}")
        lines.append("            {%- if not ns.prev_tool %}")
        lines.append("                {{- ' '}}")
        lines.append("            {%- endif %}")
        lines.append("            {%- set ns.add_space=false %}")
        lines.append("        {%- endif %}")

    if config.uses_v2_tool_format:
        lines.extend(_emit_int_float_parsing("        "))
        lines.append("        ")
        lines.append("        {%- if message['name'] is undefined or message['name'] is none %}")
        lines.append("            {{- raise_exception('Tool message must have a name.') }}")
        lines.append("        {%- endif %}")
        lines.append("")
        lines.append("        {%- set tool_message = {'name': message['name'], 'content': tool_content} %}")
        lines.append("        ")
        if config.spm:
            lines.append(
                "        {{- '" + _BEGIN_TOOL_RESULTS + " [' + (tool_message|tojson) + ']" + _END_TOOL_RESULTS + "' }}"
            )  # noqa: E501
        else:
            lines.append(
                "        {{- '" + _BEGIN_TOOL_RESULTS + "[' + (tool_message|tojson) + ']" + _END_TOOL_RESULTS + "' }}"
            )  # noqa: E501
    elif config.uses_json_tool_results:
        lines.extend(_emit_int_float_parsing("        "))
        lines.append("        ")
        lines.extend(_emit_call_id_resolution("        "))
        lines.append("")
        lines.append("        {%- set tool_message = {'content': tool_content, 'call_id': tool_id} %}")
        lines.append("        ")
        lines.append(
            "        {{- '" + _BEGIN_TOOL_RESULTS + " ' + (tool_message|tojson) + '" + _END_TOOL_RESULTS + "' }}"
        )  # noqa: E501
    elif config.uses_tool_content_format:
        lines.extend(_emit_call_id_resolution("        "))
        if config.spm:
            lines.append(
                "        {{- '"
                + _BEGIN_TOOL_RESULTS
                + " ' + tool_id + '"
                + _BEGIN_TOOL_CONTENT
                + " ' + message['content']|string + '"
                + _END_TOOL_RESULTS
                + "' }}"  # noqa: E501
            )
        else:
            lines.append(
                "        {{- '"
                + _BEGIN_TOOL_RESULTS
                + "' + tool_id + '"
                + _BEGIN_TOOL_CONTENT
                + "' + message['content']|string + '"
                + _END_TOOL_RESULTS
                + "' }}"  # noqa: E501
            )
    elif config.uses_simple_tool_results:
        if config.tool_supports_multimodal:
            tool_rc_args = "message['content'], 'tool message contents'"
            if config.image_support or config.audio_support:
                desc_parts = ["text"]
                if config.image_support:
                    desc_parts.append("image")
                if config.audio_support:
                    desc_parts.append("audio")
                tool_rc_args += f", supported_types_desc='{_join_types_desc(desc_parts)}'"
            if config.image_support:
                tool_rc_args += ", support_images=true"
            if config.audio_support:
                tool_rc_args += ", support_audio=true"
            lines.append("        {{- '" + _BEGIN_TOOL_RESULTS + "' -}}")
            lines.append("        {{- render_content(" + tool_rc_args + ") -}}")
            lines.append("        {{- '" + _END_TOOL_RESULTS + "' }}")
        else:
            lines.append(
                "        {{- '" + _BEGIN_TOOL_RESULTS + "' + message['content']|string + '" + _END_TOOL_RESULTS + "' }}"
            )  # noqa: E501
    else:
        # v3 non-spm style
        lines.extend(_emit_int_float_parsing("        "))
        lines.append("        ")
        lines.extend(_emit_call_id_resolution("        "))
        lines.append("")
        lines.append("        {%- set tool_message = {'content': tool_content, 'call_id': tool_id} %}")
        lines.append("        ")
        lines.append(
            "        {{- '" + _BEGIN_TOOL_RESULTS + "' + (tool_message|tojson) + '" + _END_TOOL_RESULTS + "' }}"
        )  # noqa: E501

    # SPM tracking
    if config.uses_spm_space_tracking:
        lines.append("        {%- set ns.prev_tool=true %}")
        lines.append("        {%- set ns.add_space=true %}")

    return "\n".join(lines)


def _generate_else_role_block(config: TemplateConfig) -> str:
    r"""Generate else block for unsupported roles.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        Jinja2 template lines for the fallback error on unknown roles.
    """
    lines = []
    lines.append("")
    lines.append("    {#- Raise exception for unsupported roles. #}")

    if config.uses_system_prompt_tokens:
        # v7+: system messages are handled in the loop, so 'system' is a valid role.
        # v7+ always has tools (has_tools is True for v2+), but we keep the branch
        # for clarity in case the property semantics change.
        lines.append("    {%- else %}")
        if config.has_tools:
            lines.append(
                "        {{- raise_exception('Only user, assistant, system and tool roles are supported, got ' + message['role'] + '.') }}"  # noqa: E501
            )
        else:
            lines.append(
                "        {{- raise_exception('Only user, assistant and system roles are supported, got ' + message['role'] + '.') }}"  # noqa: E501
            )
    else:
        # pre-v7: system messages are already filtered out
        if config.has_tools:
            if config.uses_v2_v3spm_tool_branch:
                lines.append("    {%- elif message['role'] != 'tool' or ns.index > ns.max_idx_user %}")
            else:
                lines.append("    {%- else %}")
            lines.append(
                "        {{- raise_exception('Only user, assistant and tool roles are supported, got ' + message['role'] + '.') }}"  # noqa: E501
            )
        else:
            lines.append("    {%- else %}")
            lines.append(
                "        {{- raise_exception('Only user and assistant roles are supported, got ' + message['role'] + '.') }}"  # noqa: E501
            )

    lines.append("    {%- endif %}")

    return "\n".join(lines)


def _generate_message_loop(config: TemplateConfig) -> str:
    r"""Generate the main message processing loop.

    Args:
        config: The template configuration.

    Returns:
        A string representing the main message processing loop of the chat template.
    """
    lines = []

    if config.tracks_max_idx_user:
        lines.append("")
        lines.append("{%- set ns.index = 0 %}")

    lines.append("{#- Handle conversation messages. #}")
    lines.append("{%- for message in loop_messages %}")

    lines.append(_generate_user_message_handling(config))
    lines.append(_generate_assistant_message_handling(config))
    lines.append(_generate_tool_message_handling(config))
    lines.append(_generate_system_message_handling(config))
    lines.append(_generate_else_role_block(config))

    # Index tracking for v3/v7 style
    if config.tracks_max_idx_user:
        lines.append("    {%- if message['role'] == 'user' %}")
        lines.append("        {%- set ns.index = ns.index + 1 %}")
        lines.append("    {%- endif %}")

    lines.append("{%- endfor %}")

    return "\n".join(lines)


def _generate_v1_template(config: TemplateConfig) -> str:
    r"""Generate v1 template which has unique structure.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        The complete Jinja2 template string for v1 tokenizer.
    """
    lines = []
    lines.append(_generate_header(config))
    lines.append(_generate_system_prompt_handling(config))
    lines.append(_generate_message_aggregation(config))
    lines.append("")
    lines.append("{#- Validates message ordering. #}")
    lines.append("{%- if loop_messages | length > 0 and loop_messages[0]['role'] != 'user' %}")
    lines.append(
        "    {{- raise_exception('Conversation must start with a user message, got ' + loop_messages[0]['role'] + '.') }}"  # noqa: E501
    )
    lines.append("{%- endif %}")
    lines.append("{%- set ns_order = namespace(previous_role=none) %}")
    lines.append("{%- for message in loop_messages %}")
    lines.append("    {%- set current_role = message['role'] %}")
    lines.append("    {%- if ns_order.previous_role is not none %}")
    lines.append("        {%- if ns_order.previous_role == 'user' %}")
    lines.append("            {%- if current_role not in ['assistant', 'user'] %}")
    lines.append(
        "                {{- raise_exception('Unexpected role \\'' + current_role + '\\' after role \\'' + ns_order.previous_role + '\\'') }}"  # noqa: E501
    )
    lines.append("            {%- endif %}")
    lines.append("        {%- elif ns_order.previous_role == 'assistant' %}")
    lines.append("            {%- if current_role not in ['assistant', 'user'] %}")
    lines.append(
        "                {{- raise_exception('Unexpected role \\'' + current_role + '\\' after role \\'' + ns_order.previous_role + '\\'') }}"  # noqa: E501
    )
    lines.append("            {%- endif %}")
    lines.append("        {%- endif %}")
    lines.append("    {%- endif %}")
    lines.append("    {%- set ns_order.previous_role = current_role %}")
    lines.append("{%- endfor %}")
    lines.append("")
    lines.append("{#- Handle conversation messages. #}")
    lines.append("{%- for message in loop_messages %}")
    lines.append("    {#- User messages supports text content. #}")
    lines.append("    {%- if message['role'] == 'user' %}")
    if config.spm:
        lines.append("        {{- ' " + _BEGIN_INST + " ' }}")
    else:
        lines.append("        {{- '" + _BEGIN_INST + " ' }}")
    lines.append("        {%- if loop.index0 == 0 and system_message != '' %}")
    lines.append("            {{- system_message + '\\n\\n' }}")
    lines.append("        {%- endif %}")
    lines.append("        {%- if message['content'] is string %}")
    lines.append("            {{- message['content']}}")
    lines.append("        {%- elif message['content'] | length > 0 %}")
    lines.append("            {%- for block in  message['content'] %}")
    lines.append("                {%- if block['type'] == 'text' %}")
    lines.append("                    {{- block['text'] }}")
    lines.append("                {%- else %}")
    lines.append(
        "                    {{- raise_exception('Only text chunks are supported in user message content.') }}"
    )
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")
    lines.append("        {%- else %}")
    lines.append("            {{- raise_exception('User message must have a string or a list of chunks in content') }}")
    lines.append("        {%- endif %}")
    lines.append("        {{- ' " + _END_INST + "' }}")
    if config.spm:
        lines.append("        {%- if loop.index < loop.length %}")
        lines.append('            {{- " " }}')
        lines.append("        {%- endif %}")
    lines.append("")
    lines.append("    {#- Assistant messages supports text content or text chunks. #}")
    lines.append("    {%- elif message['role'] == 'assistant' %}")
    lines.append("        {%- if message['content'] is string and message['content'] != '' %}")
    lines.append("            {{- message['content'] }}")
    lines.append("        {%- elif message['content'] | length > 0 %}")
    lines.append("            {%- for block in message['content'] %}")
    lines.append("                {%- if block['type'] == 'text' %}")
    lines.append("                    {{- block['text'] }}")
    lines.append("                {%- else %}")
    lines.append(
        "                    {{- raise_exception('Only text chunks are supported in assistant message contents.') }}"
    )
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")
    lines.append("            {#- End of sequence token for each assistant messages. #}")
    lines.append("        {%- else %}")
    lines.append("            {{- raise_exception('Assistant message content must be non-empty.') }}")
    lines.append("        {%- endif %}")
    lines.append("        {{- " + config.eos_expr + "}}")
    lines.append("")
    lines.append("")
    lines.append("    {#- Raise exception for unsupported roles. #}")
    lines.append("    {%- else %}")
    lines.append(
        "        {{- raise_exception('Only user and assistant roles are supported, got ' + message['role'] + '.') }}"
    )
    lines.append("    {%- endif %}")
    lines.append("{%- endfor %}")

    return "\n".join(lines) + "\n"


def build_chat_template(config: TemplateConfig) -> str:
    r"""Generate a complete chat template based on configuration.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        The complete Jinja2 template as a string.

    Examples:
        >>> config = TemplateConfig(version=TokenizerVersion.v3, image_support=True, use_special_token_variables=True)
        >>> template = build_chat_template(config)
        >>> "bos_token" in template
        True
    """
    if config.is_v1:
        return _generate_v1_template(config)

    parts = []
    parts.append(_generate_header(config))
    parts.append(_generate_system_prompt_handling(config))
    parts.append(_generate_available_tools_definition(config))

    macros = _generate_macros(config)
    if macros:
        parts.append(macros)

    parts.append(_generate_message_aggregation(config))
    parts.append(_generate_alternation_check(config))
    parts.append("")
    parts.append(_generate_message_loop(config))
    parts.append("")

    return "\n".join(parts)
