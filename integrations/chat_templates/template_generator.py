from dataclasses import dataclass

from mistral_common.tokens.tokenizers.base import TokenizerVersion


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

    Raises:
        ValueError: If the configuration is invalid (e.g., conflicting options like
            spm with v11+, image and audio together, or version requirements not met).

    Examples:
        >>> config = TemplateConfig(
        ...     version=TokenizerVersion.v3,
        ...     spm=False,
        ...     image_support=True
        ... )
    """

    version: TokenizerVersion
    spm: bool = False
    image_support: bool = False
    audio_support: bool = False
    thinking_support: bool = False

    def __post_init__(self) -> None:
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

    @property
    def has_tools(self) -> bool:
        return self.version >= TokenizerVersion.v2

    @property
    def uses_system_prompt_tokens(self) -> bool:
        """Whether to use [SYSTEM_PROMPT] tokens vs inline system message."""
        return self.version >= TokenizerVersion.v7

    @property
    def tracks_max_idx_user(self) -> bool:
        """Whether to track max user index for tools definition placement."""
        return self.has_tools and self.version < TokenizerVersion.v13

    @property
    def tools_at_beginning(self) -> bool:
        """Whether tools definition is emitted at the beginning."""
        return self.version >= TokenizerVersion.v13

    @property
    def uses_tool_id(self) -> bool:
        """Whether tool calls require an ID."""
        return self.version >= TokenizerVersion.v3 and self.version < TokenizerVersion.v13

    @property
    def uses_v13_tool_format(self) -> bool:
        """Whether to use v13-style tool calls (name[ARGS]arguments)."""
        return self.version >= TokenizerVersion.v13

    @property
    def uses_json_tool_results(self) -> bool:
        """Whether tool results use JSON format with content/call_id."""
        return self.version == TokenizerVersion.v3 and self.spm

    @property
    def uses_tool_content_format(self) -> bool:
        """Whether tool results use [TOOL_RESULTS]id[TOOL_CONTENT]content format."""
        return self.version >= TokenizerVersion.v7 and self.version < TokenizerVersion.v13

    @property
    def uses_simple_tool_results(self) -> bool:
        """Whether tool results use simple [TOOL_RESULTS]content format."""
        return self.version >= TokenizerVersion.v13

    @property
    def uses_v2_spm_tool_format(self) -> bool:
        """Whether to use v2_spm tool format (no ID, uses name in results)."""
        return self.version == TokenizerVersion.v2 and self.spm

    @property
    def uses_v2_tool_format(self) -> bool:
        """Whether to use v2 tool format (no ID, uses name in results, elif branch)."""
        return self.version == TokenizerVersion.v2

    @property
    def uses_spm_space_tracking(self) -> bool:
        """Whether to track add_space for SPM formatting."""
        return self.spm and self.version >= TokenizerVersion.v2

    @property
    def uses_spm_prev_tool_tracking(self) -> bool:
        """Whether to track prev_tool for SPM formatting."""
        return self.spm and self.version >= TokenizerVersion.v2

    @property
    def uses_spm_prev_img_tracking(self) -> bool:
        """Whether to track prev_img for SPM image formatting."""
        return self.spm and self.image_support

    @property
    def spm_system_prompt_has_space(self) -> bool:
        """Whether SPM system prompt tokens have trailing space."""
        return self.spm and self.version >= TokenizerVersion.v7

    @property
    def assistant_can_have_content_and_tools(self) -> bool:
        """Whether assistant messages can have both content and tool calls."""
        return self.version >= TokenizerVersion.v11

    @property
    def uses_call_id_in_tool_calls(self) -> bool:
        """Whether to include [CALL_ID] in tool calls."""
        return self.version == TokenizerVersion.v11

    @property
    def uses_json_array_tool_calls(self) -> bool:
        """Whether tool calls are in JSON array format."""
        return self.version >= TokenizerVersion.v3 and self.version < TokenizerVersion.v13

    @property
    def tracks_has_sp_for_audio(self) -> bool:
        """Whether to track has_sp for audio constraint. V15+ allows audio with system prompts."""
        return self.audio_support and self.version < TokenizerVersion.v15


def _generate_header() -> str:
    r"""Generate template header with default system message."""
    return """{#- Default system message if no system prompt is passed. #}
{%- set default_system_message = '' %}

{#- Begin of sequence token. #}
{{- '<s>' }}
"""


def _generate_system_prompt_handling(config: TemplateConfig) -> str:
    r"""Generate system prompt handling section.

    For pre-v7: Extract ALL system messages from anywhere in the conversation,
    merge their text content with ``"\\n\\n"``, and filter them out of ``loop_messages``.

    For v7+: Keep system messages in ``loop_messages`` so they are handled individually
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
    out of ``loop_messages``.

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

    For v7+, system messages stay in ``loop_messages`` and are handled by the
    aggregation pre-pass (which coalesces their ``TextChunks``) and the message
    processing loop (which emits ``[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]``).

    Only the default system prompt needs special handling: if the first message
    is not a system message, emit the default before the loop.

    Args:
        config: The template configuration.

    Returns:
        Lines of Jinja2 template code for v7+ system prompt handling.
    """
    lines = [
        "{#- System messages are handled in the message loop. #}",
        "{#- Only emit default system prompt if no system message is at position 0. #}",
        "{%- set loop_messages = messages %}",
        "{%- if messages[0]['role'] != 'system' and default_system_message != '' %}",
        "    {{- '[SYSTEM_PROMPT]' + default_system_message + '[/SYSTEM_PROMPT]' }}",
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


def _generate_tools_and_settings_definition(config: TemplateConfig) -> str:
    r"""Generate tools and model settings definition.

    Builds a ``tools_and_settings`` string variable that contains
    ``[AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]`` (if tools are provided).
    For v15+, also builds a separate ``model_settings`` variable containing
    ``[MODEL_SETTINGS]...[/MODEL_SETTINGS]`` which is always emitted
    (defaults to ``reasoning_effort="none"`` when not specified).
    Both variables are emitted later in the message loop at the appropriate
    user message position.

    Args:
        config: The template configuration.

    Returns:
        The tools and settings definition section of the chat template.
    """
    lines = [
        "",
        "",
        "{#- Tools and model settings definition #}",
        "{%- set tools_and_settings = '' %}",
        "{%- set has_tools = false %}",
    ]

    if config.has_tools:
        lines.append("{%- if tools is defined and tools is not none and tools|length > 0 %}")
        lines.append("    {%- set has_tools = true %}")
        if config.spm:
            lines.append(
                "    {%- set tools_and_settings = '[AVAILABLE_TOOLS] ' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}"
            )
        else:
            lines.append(
                "    {%- set tools_and_settings = '[AVAILABLE_TOOLS]' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}"
            )
        lines.append("{%- endif %}")

    if config.version >= TokenizerVersion.v15:
        lines.extend(
            [
                # model_settings is kept separate from tools_and_settings to avoid
                # Jinja2 Markup escaping issues when tojson returns a Markup object.
                "{%- if reasoning_effort is not defined or reasoning_effort is none %}",
                "    {%- set reasoning_effort = 'none' %}",
                "{%- endif %}",
                "{%- if reasoning_effort not in ['none', 'high'] %}",
                '    {{- raise_exception(\'reasoning_effort must be either "none" or "high"\') }}',
                "{%- endif %}",
                "{%- set model_settings = '[MODEL_SETTINGS]{\"reasoning_effort\": \"' + reasoning_effort + '\"}[/MODEL_SETTINGS]' %}",  # noqa: E501
            ]
        )

    return "\n".join(lines)


def _generate_message_aggregation(config: TemplateConfig) -> str:
    r"""Generate message aggregation pre-processing block.

    Aggregates consecutive messages with the same role before the alternation
    check to match the behavior of the mistral-common normalizer:

    - **User messages**: consecutive messages merged into one, text content joined
      with ``"\\n\\n"``, non-text chunks (image, audio) preserved in order.
    - **Assistant messages**: consecutive messages merged into one, text content joined
      with ``"\\n\\n"``, ``tool_calls`` lists concatenated.
    - **System messages**: each message coalesced individually (adjacent ``TextChunks``
      joined with ``"\\n\\n"``), but NOT merged across consecutive messages.
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
        "{#- Aggregate consecutive messages with the same role. #}",
        "{#- System and tool messages are never merged across consecutive messages. #}",
        "{#- A sentinel message is appended so the last group gets flushed inside the loop #}",
        "{#- without duplicating the flush logic after the loop. #}",
        "{%- set ns_agg = namespace(messages=[], current_group=[], current_role=none) %}",
        "{%- for message in loop_messages + [{'role': '__sentinel__'}] %}",
        "    {%- if message['role'] != ns_agg.current_role or message['role'] == 'system' or message['role'] == 'tool' %}",  # noqa: E501
    ]

    lines.extend(_generate_flush_logic())

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


def _generate_flush_logic() -> list[str]:
    r"""Generate the flush logic for aggregating a group of same-role messages.

    Called when the role changes (or at end of messages) to coalesce all messages
    in the current group into a single output message. Adjacent ``TextChunks`` are
    joined with ``"\\n\\n"``, non-text chunks are preserved as barriers, and
    ``tool_calls`` from all messages in the group are concatenated. Chunk type
    validation is deferred to the message rendering loop.

    The logic is role-agnostic for user, assistant, and system: system messages
    always form single-message groups (enforced by the grouping logic), so they
    are effectively coalesced individually. Tool messages pass through as-is to
    preserve extra fields like ``tool_call_id`` and ``name``.

    Returns:
        Lines of Jinja2 template code for flushing the current aggregation group.
    """
    lines = [
        "        {%- if ns_agg.current_role == 'tool' %}",
        "            {%- set ns_agg.messages = ns_agg.messages + ns_agg.current_group %}",
        "        {%- elif ns_agg.current_role is not none %}",
        "            {%- set ns_c = namespace(text_parts=[], chunks=[], has_non_text=false, tool_calls=[]) %}",
        "            {%- for msg in ns_agg.current_group %}",
        "                {%- if msg['content'] is string %}",
        "                    {%- set ns_c.text_parts = ns_c.text_parts + [msg['content']] %}",
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
    return lines


def _generate_system_message_handling(config: TemplateConfig) -> str:
    r"""Generate system message handling in the message loop for v7+.

    Emits ``[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`` for each system message
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
        lines.append("        {{- '[SYSTEM_PROMPT] ' -}}")
    else:
        lines.append("        {{- '[SYSTEM_PROMPT]' -}}")

    lines.append("        {%- if message['content'] is string %}")
    lines.append("            {{- message['content'] -}}")
    lines.append("        {%- else %}")
    lines.append("            {%- for block in message['content'] %}")

    if config.thinking_support and config.version < TokenizerVersion.v15:
        lines.extend(
            [
                "                {%- if block['type'] == 'text' %}",
                "                    {{- block['text'] }}",
                "                {%- elif block['type'] == 'thinking' %}",
                "                    {{- '[THINK]' + block['thinking'] + '[/THINK]' }}",
                "                {%- else %}",
                "                    {{- raise_exception('Only text and thinking chunks are supported in system message contents.') }}",  # noqa: E501
                "                {%- endif %}",
            ]
        )
    else:
        lines.extend(
            [
                "                {%- if block['type'] == 'text' %}",
                "                    {{- block['text'] }}",
                "                {%- else %}",
                "                    {{- raise_exception('Only text chunks are supported in system message contents.') }}",  # noqa: E501
                "                {%- endif %}",
            ]
        )

    lines.extend(
        [
            "            {%- endfor %}",
            "        {%- endif %}",
            "        {{- '[/SYSTEM_PROMPT]' -}}",
        ]
    )

    return "\n".join(lines)


def _generate_alternation_check(config: TemplateConfig) -> str:
    r"""Generate message alternation validation.

    For v7+, system messages are skipped in the alternation check since they
    can appear between user and assistant messages without breaking alternation.

    Args:
        config: The template configuration.

    Returns:
        The alternation check section of the chat template.
    """
    lines = [
        "",
        "{#- Checks for alternating user/assistant messages. #}",
    ]

    ns_vars = ["index=0"]
    if config.tracks_max_idx_user:
        ns_vars.append("max_idx_user=-1")
    if config.uses_spm_space_tracking:
        ns_vars.append("add_space=false")
    if config.uses_v2_spm_tool_format:
        ns_vars.append("prev_tool=false")
    if config.uses_spm_prev_img_tracking:
        ns_vars.append("prev_img=false")
    if config.tools_at_beginning:
        ns_vars.append("tools_and_settings_emitted=false")

    lines.append("{%- set ns = namespace(" + ", ".join(ns_vars) + ") %}")

    lines.append("{%- for message in loop_messages %}")

    # For v7+, system messages are transparent in the alternation check — skip them entirely.
    if config.uses_system_prompt_tokens:
        if config.has_tools:
            condition = "    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}"  # noqa: E501
        else:
            condition = "    {%- if message.role == 'user' or message.role == 'assistant' %}"
    elif config.has_tools:
        condition = "    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}"  # noqa: E501
    else:
        condition = "    {%- if message.role == 'user' or message.role == 'assistant' %}"

    lines.append(condition)

    if config.has_tools:
        error_msg = "After the optional system message, conversation roles must alternate user and assistant roles except for tool calls, results and system messages."  # noqa: E501
    else:
        error_msg = "After the optional system message, conversation roles must alternate user and assistant except for system messages."  # noqa: E501

    lines.append("        {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}")
    lines.append(f"            {{{{- raise_exception('{error_msg}') }}}}")
    lines.append("        {%- endif %}")
    lines.append("        {%- set ns.index = ns.index + 1 %}")

    if config.tracks_max_idx_user:
        lines.append("        {%- if message.role == 'user' %}")
        lines.append("            {%- set ns.max_idx_user = ns.max_idx_user + 1 %}")
        lines.append("        {%- endif %}")

    lines.append("    {%- endif %}")
    lines.append("{%- endfor %}")

    return "\n".join(lines)


def _generate_user_message_handling(config: TemplateConfig) -> str:
    r"""Generate user message handling section."""
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
        lines.append("            {{- tools_and_settings }}")
        lines.append("        {%- endif %}")
    elif config.tools_at_beginning:
        # v13+: emit tools and settings before the first user message
        lines.append("        {%- if not ns.tools_and_settings_emitted %}")
        lines.append("            {{- tools_and_settings }}")
        if config.version >= TokenizerVersion.v15:
            lines.append("            {{- model_settings }}")
        lines.append("            {%- set ns.tools_and_settings_emitted = true %}")
        lines.append("        {%- endif %}")

    if config.spm and not config.uses_spm_prev_img_tracking:
        inst_open = "[INST] "
        inst_close = "[/INST]"
    else:
        inst_open = "[INST]"
        inst_close = "[/INST]"

    if config.uses_system_prompt_tokens:
        lines.append("        {%- if message['content'] is string %}")
        if config.spm:
            lines.append(f"            {{{{- '[INST] ' + message['content'] + '{inst_close}' }}}}")
        else:
            lines.append(f"            {{{{- '{inst_open}' + message['content'] + '{inst_close}' }}}}")
        lines.append("        {%- elif message['content'] | length > 0 %}")
        if config.uses_spm_prev_img_tracking:
            lines.append("            {{- '[INST]' }}")
        elif config.spm:
            lines.append("            {{- '[INST] ' }}")
        else:
            lines.append(f"            {{{{- '{inst_open}' }}}}")
    elif config.uses_spm_prev_img_tracking:
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
        lines.append("            {{- message['content']}}")
        lines.append("        {%- elif message['content'] | length > 0 %}")
    else:
        lines.append(f"        {{{{- '{inst_open}' }}}}")
        if config.version == TokenizerVersion.v1:
            lines.append("        {%- if loop.index0 == 0 and system_message != '' %}")
        else:
            lines.append("        {%- if (ns.index == ns.max_idx_user) and system_message != '' %}")
        lines.append("            {{- system_message + '\\n\\n' }}")
        lines.append("        {%- endif %}")
        lines.append("        {%- if message['content'] is string %}")
        lines.append("            {{- message['content']}}")
        lines.append("        {%- elif message['content'] | length > 0 %}")

    if config.image_support:
        lines.append("            {%- if message['content'] | length == 2 %}")
        lines.append("                {%- set blocks = message['content'] | sort(attribute='type') %}")
        lines.append("            {%- else %}")
        lines.append("                {%- set blocks = message['content'] %}")
        lines.append("            {%- endif %}")
        block_var = "blocks"
    elif config.version >= TokenizerVersion.v13 and not config.audio_support:
        lines.append("            {%- set sorted_blocks = message['content'] | sort(attribute='type') %}")
        block_var = "sorted_blocks"
    else:
        block_var = "message['content']"

    if config.uses_spm_prev_img_tracking:
        lines.append("            {% set ns.prev_img = not added_sp %}")

    lines.append(f"            {{%- for block in {block_var} %}}")

    if config.uses_spm_prev_img_tracking:
        lines.append("                {%- if ns.prev_img and block['type'] == 'text' %}")
        lines.append("                    {{- ' ' }}")
        lines.append("                {%- endif %}")

    lines.append("                {%- if block['type'] == 'text' %}")
    lines.append("                    {{- block['text'] }}")
    if config.uses_spm_prev_img_tracking:
        lines.append("                    {%- set ns.prev_img = false %}")

    if config.image_support:
        lines.append("                {%- elif block['type'] in ['image', 'image_url'] %}")
        lines.append("                    {{- '[IMG]' }}")
        if config.uses_spm_prev_img_tracking:
            lines.append("                    {%- set ns.prev_img = true %}")

    if config.audio_support:
        lines.append("                {%- elif block['type'] in ['input_audio', 'audio_url'] %}")
        if config.tracks_has_sp_for_audio:
            lines.append("                    {%- if has_sp %}")
            lines.append(
                "                        {{- raise_exception('Audio chunks are not supported in user message content when system prompt is provided.') }}"  # noqa: E501
            )
            lines.append("                    {%- endif %}")
        lines.append("                    {{- '[AUDIO]' }}")

    lines.append("                {%- else %}")
    lines.append(
        f"                    {{{{- raise_exception('Only {chunk_desc} chunks are supported in user message content.') }}}}"  # noqa: E501
    )
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")

    if config.uses_system_prompt_tokens:
        lines.append(f"            {{{{- '{inst_close}' }}}}")
        lines.append("        {%- else %}")
        lines.append(
            "            {{- raise_exception('User message must have a string or a list of chunks in content') }}"
        )
        lines.append("        {%- endif %}")
    else:
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
    r"""Generate assistant message handling section."""
    lines = []

    if config.thinking_support:
        chunk_types = "text and thinking"
    else:
        chunk_types = "text"

    comment = f"{{#- Assistant messages supports {chunk_types} content. #}}"
    lines.append("")
    lines.append(f"    {comment}")
    lines.append("    {%- elif message['role'] == 'assistant' %}")

    if config.version in [TokenizerVersion.v2, TokenizerVersion.v3]:
        lines.append(
            "        {%- if message['content'] is not none and message['content'] | length > 0 and message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}"  # noqa: E501
        )
        lines.append("            {{- raise_exception('Assistant message cannot have both content and tool calls.') }}")
        lines.append("        {%- endif %}")
        lines.append("")

    if (
        config.version >= TokenizerVersion.v7
        or (config.version >= TokenizerVersion.v3 and not config.spm)
        or config.version == TokenizerVersion.v13
    ):
        lines.append(
            "        {%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}"  # noqa: E501
        )
        lines.append(
            "            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}"  # noqa: E501
        )
        lines.append("        {%- endif %}")
        lines.append("")

    lines.append("        {%- if message['content'] is string and message['content'] != '' %}")

    if config.uses_spm_space_tracking:
        lines.append("            {%- if ns.add_space %}")
        lines.append("                {{- ' ' }}")
        lines.append("                {%- set ns.add_space=false %}")
        lines.append("            {%- endif %}")

    lines.append("            {{- message['content'] }}")

    if config.uses_v2_tool_format:
        lines.append("            {{- '</s>' }}")

    lines.append("        {%- elif message['content'] | length > 0 %}")

    if config.uses_spm_space_tracking:
        lines.append("            {%- if ns.add_space %}")
        lines.append("                {{- ' ' }}")
        lines.append("                {%- set ns.add_space=false %}")
        lines.append("            {%- endif %}")

    lines.append("            {%- for block in message['content'] %}")
    lines.append("                {%- if block['type'] == 'text' %}")
    lines.append("                    {{- block['text'] }}")

    if config.thinking_support:
        lines.append("                {%- elif block['type'] == 'thinking' %}")
        lines.append("                    {{- '[THINK]' + block['thinking'] + '[/THINK]' }}")

    lines.append("                {%- else %}")
    lines.append(
        f"                    {{{{- raise_exception('Only {chunk_types} chunks are supported in assistant message contents.') }}}}"  # noqa: E501
    )
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")

    if config.uses_v2_tool_format:
        lines.append("            {{- '</s>' }}")

    if config.version == TokenizerVersion.v2 or (config.spm and config.version == TokenizerVersion.v3):
        lines.append(_generate_tool_calls_elif_v2_v3(config))
        lines.append("        {%- else %}")
        lines.append(
            "            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}"  # noqa: E501
        )
        lines.append("        {%- endif %}")
    else:
        lines.append("        {%- endif %}")

    if config.has_tools and not (
        config.version == TokenizerVersion.v2 or (config.spm and config.version == TokenizerVersion.v3)
    ):
        lines.append("")
        lines.append(_generate_tool_calls_block(config))

    if not config.uses_v2_tool_format:
        lines.append("")
        lines.append("        {{- '</s>' }}")

    if config.uses_spm_space_tracking:
        lines.append("        {%- set ns.prev_tool=false %}")

    return "\n".join(lines)


def _generate_tool_calls_elif_v2_v3(config: TemplateConfig) -> str:
    r"""Generate tool calls as elif branch for v2 (SPM and non-SPM) and v3_spm templates."""
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

    if config.spm:
        lines.append("            {{- '[TOOL_CALLS] [' }}")
    else:
        lines.append("            {{- '[TOOL_CALLS][' }}")
    lines.append("            {%- for tool in message['tool_calls'] %}")
    lines.append("                {%- set name = tool['function']['name'] %}")
    lines.append("                {%- set arguments = tool['function']['arguments'] %}")

    if config.uses_v2_tool_format:
        # v2: no ID in tool calls
        lines.append("                {%- if arguments is not string %}")
        lines.append("                    {%- set arguments = arguments|tojson|safe %}")
        lines.append("                {%- elif arguments == '' %}")
        lines.append("                    {%- set arguments = '{}' %}")
        lines.append("                {%- endif %}")
        lines.append("                {{- '{\"name\": \"' + name + '\", \"arguments\": ' + arguments + '}' }}")
    else:
        # v3_spm: has ID in tool calls
        lines.append("                {%- set id = tool['id']%}")
        lines.append("                {%- if id is not defined or id|length != 9 %}")
        lines.append(
            "                    {{- raise_exception('Tool call must have an id of 9 characters or numbers.') }}"
        )
        lines.append("                {%- endif %}")
        lines.append("                {%- if arguments is not string %}")
        lines.append("                    {%- set arguments = arguments|tojson|safe %}")
        lines.append("                {%- elif arguments == '' %}")
        lines.append("                    {%- set arguments = '{}' %}")
        lines.append("                {%- endif %}")
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
        lines.append("            {{- '</s>' }}")

    if config.uses_v2_tool_format:
        # v2: additional elif for tool calls during user messages (ignored)
        lines.append(
            "        {%- elif message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 and ns.index <= ns.max_idx_user %}"  # noqa: E501
        )

    return "\n".join(lines)


def _generate_tool_calls_block(config: TemplateConfig) -> str:
    r"""Generate tool calls block (v7+ style)."""
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
        lines.append("                {{- '[TOOL_CALLS]' }}")
        lines.append("                {%- set name = tool['function']['name'] %}")
        lines.append("                {%- set arguments = tool['function']['arguments'] %}")
        lines.append("                {%- if arguments is not string %}")
        lines.append("                    {%- set arguments = arguments|tojson|safe %}")
        lines.append("                {%- elif arguments == '' %}")
        lines.append("                    {%- set arguments = '{}' %}")
        lines.append("                {%- endif %}")
        lines.append("                {{- name + '[ARGS]' + arguments }}")
        lines.append("            {%- endfor %}")
    elif config.uses_call_id_in_tool_calls:
        lines.append("            {%- for tool in message['tool_calls'] %}")
        lines.append("                {{- '[TOOL_CALLS]' }}")
        lines.append("                {%- set name = tool['function']['name'] %}")
        lines.append("                {%- set arguments = tool['function']['arguments'] %}")
        lines.append("                {%- set id = tool['id']%}")
        lines.append("                {%- if id is not defined or id|length != 9 %}")
        lines.append(
            "                    {{- raise_exception('Tool call must have an id of 9 characters or numbers.') }}"
        )
        lines.append("                {%- endif %}")
        lines.append("                {%- if arguments is not string %}")
        lines.append("                    {%- set arguments = arguments|tojson|safe %}")
        lines.append("                {%- elif arguments == '' %}")
        lines.append("                    {%- set arguments = '{}' %}")
        lines.append("                {%- endif %}")
        lines.append("                {{- name + '[CALL_ID]' + id + '[ARGS]' + arguments }}")
        lines.append("            {%- endfor %}")
    else:
        if config.spm:
            lines.append("            {{- '[TOOL_CALLS] [' }}")
        else:
            lines.append("            {{- '[TOOL_CALLS][' }}")
        lines.append("            {%- for tool in message['tool_calls'] %}")
        lines.append("                {%- set name = tool['function']['name'] %}")
        lines.append("                {%- set arguments = tool['function']['arguments'] %}")
        lines.append("                {%- set id = tool['id']%}")
        lines.append("                {%- if id is not defined or id|length != 9 %}")
        lines.append(
            "                    {{- raise_exception('Tool call must have an id of 9 characters or numbers.') }}"
        )
        lines.append("                {%- endif %}")
        lines.append("                {%- if arguments is not string %}")
        lines.append("                    {%- set arguments = arguments|tojson|safe %}")
        lines.append("                {%- elif arguments == '' %}")
        lines.append("                    {%- set arguments = '{}' %}")
        lines.append("                {%- endif %}")
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
    r"""Generate tool message handling section."""
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
        lines.append("        {# Try to parse 'content' as int or float if possible #}")
        lines.append("        {%- set tool_content = message['content']|string %}")
        lines.append("        {# Try to parse as int #}")
        lines.append("        {%- set parsed_int = message['content']|int %}")
        lines.append("        {% if parsed_int|string == message['content'] %}")
        lines.append("            {%- set tool_content = parsed_int %}")
        lines.append("        {# If int fails, try to parse as float #}")
        lines.append("        {%- else %}")
        lines.append("            {%- set parsed_float = message['content']|float %}")
        lines.append("            {%- if parsed_float|string == message['content'] %}")
        lines.append("                {%- set tool_content = parsed_float %}")
        lines.append("            {%- endif %}")
        lines.append("        {%- endif %}")
        lines.append("        ")
        lines.append("        {%- if message['name'] is undefined or message['name'] is none %}")
        lines.append("            {{- raise_exception('Tool message must have a name.') }}")
        lines.append("        {%- endif %}")
        lines.append("")
        lines.append("        {%- set tool_message = {'name': message['name'], 'content': tool_content} %}")
        lines.append("        ")
        if config.spm:
            lines.append("        {{- '[TOOL_RESULTS] [' + (tool_message|tojson) + '][/TOOL_RESULTS]' }}")
        else:
            lines.append("        {{- '[TOOL_RESULTS][' + (tool_message|tojson) + '][/TOOL_RESULTS]' }}")
    elif config.uses_json_tool_results:
        lines.append("        {# Try to parse 'content' as int or float if possible #}")
        lines.append("        {%- set tool_content = message['content']|string %}")
        lines.append("        {# Try to parse as int #}")
        lines.append("        {%- set parsed_int = message['content']|int %}")
        lines.append("        {% if parsed_int|string == message['content'] %}")
        lines.append("            {%- set tool_content = parsed_int %}")
        lines.append("        {# If int fails, try to parse as float #}")
        lines.append("        {%- else %}")
        lines.append("            {%- set parsed_float = message['content']|float %}")
        lines.append("            {%- if parsed_float|string == message['content'] %}")
        lines.append("                {%- set tool_content = parsed_float %}")
        lines.append("            {%- endif %}")
        lines.append("        {%- endif %}")
        lines.append("        ")
        lines.append("        {%- if message['call_id'] is not undefined and message['call_id']|length == 9 %}")
        lines.append("            {%- set tool_id = message['call_id'] %}")
        lines.append(
            "        {%- elif message['tool_call_id'] is not undefined and message['tool_call_id']|length == 9  %}"
        )
        lines.append("            {%- set tool_id = message['tool_call_id'] %}")
        lines.append("        {%- else %}")
        lines.append(
            "            {{- raise_exception('Tool message must have a call_id or tool_call_id of 9 characters or numbers.') }}"  # noqa: E501
        )
        lines.append("        {%- endif %}")
        lines.append("")
        lines.append("        {%- set tool_message = {'content': tool_content, 'call_id': tool_id} %}")
        lines.append("        ")
        lines.append("        {{- '[TOOL_RESULTS] ' + (tool_message|tojson) + '[/TOOL_RESULTS]' }}")
    elif config.uses_tool_content_format:
        lines.append("        {%- if message['call_id'] is not undefined and message['call_id']|length == 9 %}")
        lines.append("            {%- set tool_id = message['call_id'] %}")
        lines.append(
            "        {%- elif message['tool_call_id'] is not undefined and message['tool_call_id']|length == 9  %}"
        )
        lines.append("            {%- set tool_id = message['tool_call_id'] %}")
        lines.append("        {%- else %}")
        lines.append(
            "            {{- raise_exception('Tool message must have a call_id or tool_call_id of 9 characters or numbers.') }}"  # noqa: E501
        )
        lines.append("        {%- endif %}")
        if config.spm:
            lines.append(
                "        {{- '[TOOL_RESULTS] ' + tool_id + '[TOOL_CONTENT] ' + message['content']|string + '[/TOOL_RESULTS]' }}"  # noqa: E501
            )
        else:
            lines.append(
                "        {{- '[TOOL_RESULTS]' + tool_id + '[TOOL_CONTENT]' + message['content']|string + '[/TOOL_RESULTS]' }}"  # noqa: E501
            )
    elif config.uses_simple_tool_results:
        lines.append("        {{- '[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]' }}")
    else:
        # v3 non-spm style
        lines.append("        {# Try to parse 'content' as int or float if possible #}")
        lines.append("        {%- set tool_content = message['content']|string %}")
        lines.append("        {# Try to parse as int #}")
        lines.append("        {%- set parsed_int = message['content']|int %}")
        lines.append("        {% if parsed_int|string == message['content'] %}")
        lines.append("            {%- set tool_content = parsed_int %}")
        lines.append("        {# If int fails, try to parse as float #}")
        lines.append("        {%- else %}")
        lines.append("            {%- set parsed_float = message['content']|float %}")
        lines.append("            {%- if parsed_float|string == message['content'] %}")
        lines.append("                {%- set tool_content = parsed_float %}")
        lines.append("            {%- endif %}")
        lines.append("        {%- endif %}")
        lines.append("        ")
        lines.append("        {%- if message['call_id'] is not undefined and message['call_id']|length == 9 %}")
        lines.append("            {%- set tool_id = message['call_id'] %}")
        lines.append(
            "        {%- elif message['tool_call_id'] is not undefined and message['tool_call_id']|length == 9  %}"
        )
        lines.append("            {%- set tool_id = message['tool_call_id'] %}")
        lines.append("        {%- else %}")
        lines.append(
            "            {{- raise_exception('Tool message must have a call_id or tool_call_id of 9 characters or numbers.') }}"  # noqa: E501
        )
        lines.append("        {%- endif %}")
        lines.append("")
        lines.append("        {%- set tool_message = {'content': tool_content, 'call_id': tool_id} %}")
        lines.append("        ")
        lines.append("        {{- '[TOOL_RESULTS]' + (tool_message|tojson) + '[/TOOL_RESULTS]' }}")

    # SPM tracking
    if config.uses_spm_space_tracking:
        lines.append("        {%- set ns.prev_tool=true %}")
        lines.append("        {%- set ns.add_space=true %}")

    return "\n".join(lines)


def _generate_else_role_block(config: TemplateConfig) -> str:
    r"""Generate else block for unsupported roles."""
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
            if config.version == TokenizerVersion.v2 or (config.spm and config.version == TokenizerVersion.v3):
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
    r"""Generate v1 template which has unique structure."""
    lines = []
    lines.append(_generate_header())
    lines.append(_generate_system_prompt_handling(config))
    lines.append(_generate_message_aggregation(config))
    lines.append("")
    lines.append("{#- Checks for alternating user/assistant messages. #}")
    lines.append("{%- set ns = namespace(index=0) %}")
    lines.append("{%- for message in loop_messages %}")
    lines.append("    {%- if message.role == 'user' or message.role == 'assistant' %}")
    lines.append("        {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}")
    lines.append(
        "            {{- raise_exception('After the optional system message, conversation roles must alternate user and assistant.') }}"  # noqa: E501
    )
    lines.append("        {%- endif %}")
    lines.append("        {%- set ns.index = ns.index + 1 %}")
    lines.append("    {%- endif %}")
    lines.append("{%- endfor %}")
    lines.append("")
    lines.append("{#- Handle conversation messages. #}")
    lines.append("{%- for message in loop_messages %}")
    lines.append("    {#- User messages supports text content. #}")
    lines.append("    {%- if message['role'] == 'user' %}")
    if config.spm:
        lines.append("        {{- ' [INST] ' }}")
    else:
        lines.append("        {{- '[INST] ' }}")
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
    lines.append("        {{- ' [/INST]' }}")
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
    lines.append("        {{- eos_token}}")
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


def generate_chat_template(config: TemplateConfig) -> str:
    r"""Generate a complete chat template based on configuration.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        The complete Jinja2 template as a string.

    Examples:
        >>> config = TemplateConfig(version=TokenizerVersion.v3, image_support=True)
        >>> template = generate_chat_template(config)
    """
    if config.version == TokenizerVersion.v1:
        return _generate_v1_template(config)

    parts = []
    parts.append(_generate_header())
    parts.append(_generate_system_prompt_handling(config))
    parts.append(_generate_tools_and_settings_definition(config))
    parts.append(_generate_message_aggregation(config))
    parts.append(_generate_alternation_check(config))
    parts.append("")
    parts.append(_generate_message_loop(config))
    parts.append("")

    return "\n".join(parts)
