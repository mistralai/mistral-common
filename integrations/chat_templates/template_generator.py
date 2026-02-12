from dataclasses import dataclass

from mistral_common.tokens.tokenizers.base import TokenizerVersion


@dataclass
class TemplateConfig:
    r"""Configuration for generating a chat template.

    This class encapsulates all the configuration options required to generate
    a Jinja2 chat template that formats conversation messages for Mistral models.
    The template handles message roles, special tokens, tool calls, and multimodal content.

    Attributes:
        version: The tokenizer version (e.g., v1, v2, v3, v7, v11, v13). Determines
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
        """Whether to track has_sp for audio constraint."""
        return self.audio_support


def _generate_header() -> str:
    r"""Generate template header with default system message."""
    return """{#- Default system message if no system prompt is passed. #}
{%- set default_system_message = '' %}

{#- Begin of sequence token. #}
{{- '<s>' }}
"""


def _generate_system_prompt_handling(config: TemplateConfig) -> str:
    r"""Generate system prompt handling section."""
    lines = [
        "",
        "{#- Handle system prompt if it exists. #}",
    ]

    if config.thinking_support:
        chunk_comment = "{#- System prompt supports text content or text and thinking chunks. #}"
        chunk_handler = """            {%- if block['type'] == 'text' %}
                {{- block['text'] }}
            {%- elif block['type'] == 'thinking' %}
                {{- '[THINK]' + block['thinking'] + '[/THINK]' }}
            {%- else %}
                {{- raise_exception('Only text and thinking chunks are supported in system message contents.') }}
            {%- endif %}"""
    else:
        chunk_comment = "{#- System prompt supports text content or text chunks. #}"
        chunk_handler = """            {%- if block['type'] == 'text' %}
                {{- block['text'] }}
            {%- else %}
                {{- raise_exception('Only text chunks are supported in system message contents.') }}
            {%- endif %}"""

    lines.append(chunk_comment)

    if config.uses_system_prompt_tokens:
        if config.tracks_has_sp_for_audio:
            lines.append("{%- if messages[0]['role'] == 'system' %}")
            lines.append("    {%- set has_sp = true %}")
        else:
            lines.append("{%- if messages[0]['role'] == 'system' %}")
        if config.spm_system_prompt_has_space:
            lines.append("    {{- '[SYSTEM_PROMPT] ' -}}")
        else:
            lines.append("    {{- '[SYSTEM_PROMPT]' -}}")
        lines.append("    {%- if messages[0]['content'] is string %}")
        lines.append("        {{- messages[0]['content'] -}}")
        lines.append("    {%- else %}        ")
        lines.append("        {%- for block in messages[0]['content'] %}")
        lines.append(chunk_handler)
        lines.append("        {%- endfor %}")
        lines.append("    {%- endif %}")
        lines.append("    {{- '[/SYSTEM_PROMPT]' -}}")
        lines.append("    {%- set loop_messages = messages[1:] %}")
        lines.append("{%- else %}")
        lines.append("    {%- set loop_messages = messages %}")
        lines.append("    {%- if default_system_message != '' %}")
        lines.append("        {{- '[SYSTEM_PROMPT]' + default_system_message + '[/SYSTEM_PROMPT]' }}")
        if config.tracks_has_sp_for_audio:
            lines.append("        {%- set has_sp = true %}")
            lines.append("    {%- else %}")
            lines.append("        {%- set has_sp = false %}")
        lines.append("    {%- endif %}")
        lines.append("{%- endif %}")
    else:
        lines.append("{%- if messages[0]['role'] == 'system' %}")
        lines.append("    {%- if messages[0]['content'] is string %}")
        lines.append("        {%- set system_message = messages[0]['content'] %}")
        lines.append("    {%- else %}        ")
        lines.append("        {%- for block in messages[0]['content'] %}")
        lines.append("            {%- set system_message = '' %}")
        lines.append("            {%- if block['type'] == 'text' %}")
        lines.append("                {% set system_message = system_message + block['text'] %}")
        lines.append("            {%- else %}")
        lines.append(
            "                {{- raise_exception('Only text chunks are supported in system message contents.') }}"
        )
        lines.append("            {%- endif %}")
        lines.append("        {%- endfor %}")
        lines.append("    {%- endif %}")
        lines.append("    {%- set loop_messages = messages[1:] %}")
        lines.append("{%- else %}")
        lines.append("    {%- set system_message = default_system_message %}")
        lines.append("    {%- set loop_messages = messages %}")
        lines.append("{%- endif%}")

    return "\n".join(lines)


def _generate_tools_definition(config: TemplateConfig) -> str:
    r"""Generate tools definition section."""
    if not config.has_tools:
        return ""

    lines = [
        "",
        "",
        "{#- Tools definition #}",
        "{%- set tools_definition = '' %}",
        "{%- set has_tools = false %}",
        "{%- if tools is defined and tools is not none and tools|length > 0 %}",
        "    {%- set has_tools = true %}",
    ]

    if config.spm:
        lines.append("    {%- set tools_definition = '[AVAILABLE_TOOLS] ' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}")
    else:
        lines.append("    {%- set tools_definition = '[AVAILABLE_TOOLS]' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}")

    if config.tools_at_beginning:
        lines.append("    {{- tools_definition }}")

    lines.append("{%- endif %}")

    return "\n".join(lines)


def _generate_alternation_check(config: TemplateConfig) -> str:
    r"""Generate message alternation validation."""
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

    lines.append("{%- set ns = namespace(" + ", ".join(ns_vars) + ") %}")

    lines.append("{%- for message in loop_messages %}")

    if config.has_tools:
        lines.append(
            "    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}"  # noqa: E501
        )
    else:
        lines.append("    {%- if message.role == 'user' or message.role == 'assistant' %}")

    if config.has_tools:
        error_msg = "After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results."  # noqa: E501
    else:
        error_msg = "After the optional system message, conversation roles must alternate user and assistant."

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

    # Tools definition for non-v11+ templates (placed before last user message)
    if config.tracks_max_idx_user:
        lines.append("        {%- if (ns.index == ns.max_idx_user) and has_tools %}")
        lines.append("            {{- tools_definition }}")
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
    parts.append(_generate_tools_definition(config))
    parts.append(_generate_alternation_check(config))
    parts.append("")
    parts.append(_generate_message_loop(config))
    parts.append("")

    return "\n".join(parts)
