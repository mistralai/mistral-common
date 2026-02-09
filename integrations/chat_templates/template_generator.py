"""Dynamic chat template generator for Mistral tokenizers.

This module generates Jinja2 chat templates dynamically based on tokenizer version
and feature flags, avoiding code repetition across the 24 static template files.
"""

from dataclasses import dataclass

from mistral_common.tokens.tokenizers.base import TokenizerVersion


@dataclass
class TemplateConfig:
    """Configuration for generating a chat template."""

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
    """Generate template header with default system message."""
    return """{#- Default system message if no system prompt is passed. #}
{%- set default_system_message = '' %}

{#- Begin of sequence token. #}
{{- '<s>' }}
"""


def _generate_system_prompt_handling(config: TemplateConfig) -> str:
    """Generate system prompt handling section."""
    lines = [
        "",
        "{#- Handle system prompt if it exists. #}",
    ]

    # Determine supported chunk types in system message
    if config.thinking_support:
        chunk_comment = "{#- System prompt supports text content or text and thinking chunks. #}"
        supported_chunks = "text and thinking"
        chunk_handler = """            {%- if block['type'] == 'text' %}
                {{- block['text'] }}
            {%- elif block['type'] == 'thinking' %}
                {{- '[THINK]' + block['thinking'] + '[/THINK]' }}
            {%- else %}
                {{- raise_exception('Only text and thinking chunks are supported in system message contents.') }}
            {%- endif %}"""
    else:
        chunk_comment = "{#- System prompt supports text content or text chunks. #}"
        supported_chunks = "text"
        chunk_handler = """            {%- if block['type'] == 'text' %}
                {{- block['text'] }}
            {%- else %}
                {{- raise_exception('Only text chunks are supported in system message contents.') }}
            {%- endif %}"""

    lines.append(chunk_comment)

    if config.uses_system_prompt_tokens:
        # Modern style: [SYSTEM_PROMPT]...[/SYSTEM_PROMPT]
        if config.tracks_has_sp_for_audio:
            lines.append("{%- if messages[0]['role'] == 'system' %}")
            lines.append("    {%- set has_sp = true %}")
        else:
            lines.append("{%- if messages[0]['role'] == 'system' %}")
        # SPM v7+ has space after [SYSTEM_PROMPT]
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
        # Legacy style: system message injected into first user message
        lines.append("{%- if messages[0]['role'] == 'system' %}")
        lines.append("    {%- if messages[0]['content'] is string %}")
        lines.append("        {%- set system_message = messages[0]['content'] %}")
        lines.append("    {%- else %}        ")
        lines.append("        {%- for block in messages[0]['content'] %}")
        lines.append("            {%- set system_message = '' %}")
        # For v1/v2 SPM style, only text is supported
        lines.append("            {%- if block['type'] == 'text' %}")
        lines.append("                {% set system_message = system_message + block['text'] %}")
        lines.append("            {%- else %}")
        lines.append("                {{- raise_exception('Only text chunks are supported in system message contents.') }}")
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
    """Generate tools definition section."""
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

    # SPM style has spaces in tools definition
    if config.spm:
        lines.append("    {%- set tools_definition = '[AVAILABLE_TOOLS] ' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}")
    else:
        lines.append("    {%- set tools_definition = '[AVAILABLE_TOOLS]' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}")

    # v11+ emit tools at the beginning
    if config.tools_at_beginning:
        lines.append("    {{- tools_definition }}")

    lines.append("{%- endif %}")

    return "\n".join(lines)


def _generate_alternation_check(config: TemplateConfig) -> str:
    """Generate message alternation validation."""
    lines = [
        "",
        "{#- Checks for alternating user/assistant messages. #}",
    ]

    # Determine namespace variables
    ns_vars = ["index=0"]
    if config.tracks_max_idx_user:
        ns_vars.append("max_idx_user=-1")
    if config.uses_spm_space_tracking:
        ns_vars.append("add_space=false")
    if config.uses_v2_spm_tool_format:
        # v2_spm also tracks prev_tool in namespace initialization
        ns_vars.append("prev_tool=false")
    if config.uses_spm_prev_img_tracking:
        ns_vars.append("prev_img=false")

    lines.append("{%- set ns = namespace(" + ", ".join(ns_vars) + ") %}")

    lines.append("{%- for message in loop_messages %}")

    # Condition for counting messages
    if config.has_tools:
        lines.append("    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}")
    else:
        lines.append("    {%- if message.role == 'user' or message.role == 'assistant' %}")

    # Error message
    if config.has_tools:
        error_msg = "After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results."
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
    """Generate user message handling section."""
    lines = []

    # Determine supported chunk types
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

    # SPM-style INST tokens have trailing space on open
    # For all SPM: [INST] content[/INST] (space after INST, no space before /INST)
    # Exception: SPM image templates don't have space after INST for list content
    if config.spm and not config.uses_spm_prev_img_tracking:
        inst_open = "[INST] "
        inst_close = "[/INST]"
    else:
        inst_open = "[INST]"
        inst_close = "[/INST]"

    # For v7+ with system prompt tokens (no system injection needed in user message)
    # We can use the compact string format
    if config.uses_system_prompt_tokens:
        lines.append("        {%- if message['content'] is string %}")
        if config.spm:
            lines.append(f"            {{{{- '[INST] ' + message['content'] + '{inst_close}' }}}}")
        else:
            lines.append(f"            {{{{- '{inst_open}' + message['content'] + '{inst_close}' }}}}")
        lines.append("        {%- elif message['content'] | length > 0 %}")
        if config.uses_spm_prev_img_tracking:
            # SPM image: no space after INST for list content
            lines.append("            {{- '[INST]' }}")
        elif config.spm:
            lines.append("            {{- '[INST] ' }}")
        else:
            lines.append(f"            {{{{- '{inst_open}' }}}}")
    elif config.uses_spm_prev_img_tracking:
        # SPM image templates have special INST/space handling
        lines.append(f"        {{{{- '{inst_open}' }}}}")
        # Add space tracking for system message
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
        # For v1, v2, v3: need to inject system message, so always use separate INST output
        lines.append(f"        {{{{- '{inst_open}' }}}}")

        # System message injection for legacy style
        if config.version == TokenizerVersion.v1:
            # v1 injects system at first user message
            lines.append("        {%- if loop.index0 == 0 and system_message != '' %}")
        else:
            # v2, v3, v3_spm inject system at last user message
            lines.append("        {%- if (ns.index == ns.max_idx_user) and system_message != '' %}")
        lines.append("            {{- system_message + '\\n\\n' }}")
        lines.append("        {%- endif %}")

        lines.append("        {%- if message['content'] is string %}")
        lines.append("            {{- message['content']}}")
        lines.append("        {%- elif message['content'] | length > 0 %}")

    # Block sorting for image templates (when exactly 2 blocks)
    if config.image_support:
        lines.append("            {%- if message['content'] | length == 2 %}")
        lines.append("                {%- set blocks = message['content'] | sort(attribute='type') %}")
        lines.append("            {%- else %}")
        lines.append("                {%- set blocks = message['content'] %}")
        lines.append("            {%- endif %}")
        block_var = "blocks"
    elif config.version >= TokenizerVersion.v13 and not config.audio_support:
        # v13 non-audio sorts blocks
        lines.append("            {%- set sorted_blocks = message['content'] | sort(attribute='type') %}")
        block_var = "sorted_blocks"
    else:
        block_var = "message['content']"

    # SPM image templates need prev_img initialization
    if config.uses_spm_prev_img_tracking:
        lines.append("            {% set ns.prev_img = not added_sp %}")

    # Use double space for block content iteration (in elif branch)
    lines.append(f"            {{%- for block in {block_var} %}}")

    # SPM image templates add space before text after image
    if config.uses_spm_prev_img_tracking:
        lines.append("                {%- if ns.prev_img and block['type'] == 'text' %}")
        lines.append("                    {{- ' ' }}")
        lines.append("                {%- endif %}")

    lines.append("                {%- if block['type'] == 'text' %}")
    lines.append("                    {{- block['text'] }}")
    if config.uses_spm_prev_img_tracking:
        lines.append("                    {%- set ns.prev_img = false %}")

    # Image chunk handling
    if config.image_support:
        lines.append("                {%- elif block['type'] in ['image', 'image_url'] %}")
        lines.append("                    {{- '[IMG]' }}")
        if config.uses_spm_prev_img_tracking:
            lines.append("                    {%- set ns.prev_img = true %}")

    # Audio chunk handling
    if config.audio_support:
        lines.append("                {%- elif block['type'] in ['input_audio', 'audio_url'] %}")
        if config.tracks_has_sp_for_audio:
            lines.append("                    {%- if has_sp %}")
            lines.append("                        {{- raise_exception('Audio chunks are not supported in user message content when system prompt is provided.') }}")
            lines.append("                    {%- endif %}")
        lines.append("                    {{- '[AUDIO]' }}")

    # Error for unsupported types
    lines.append("                {%- else %}")
    lines.append(f"                    {{{{- raise_exception('Only {chunk_desc} chunks are supported in user message content.') }}}}")
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")

    # Close the content handling
    if config.uses_system_prompt_tokens:
        # For v7+: the string case has inline [/INST], list case needs [/INST] here
        lines.append(f"            {{{{- '{inst_close}' }}}}")
        lines.append("        {%- else %}")
        lines.append("            {{- raise_exception('User message must have a string or a list of chunks in content') }}")
        lines.append("        {%- endif %}")
    else:
        # For v1/v2/v3: shared [/INST] at the end for all cases
        lines.append("        {%- else %}")
        lines.append("            {{- raise_exception('User message must have a string or a list of chunks in content') }}")
        lines.append("        {%- endif %}")
        lines.append(f"        {{{{- '{inst_close}' }}}}")

    # SPM-specific space tracking (v2_spm and above)
    if config.uses_spm_space_tracking:
        lines.append("        {%- if loop.index < loop.length %}")
        lines.append("            {%- set ns.add_space=true %}")
        lines.append("        {%- endif %}")
        lines.append("        {%- set ns.prev_tool=false %}")

    return "\n".join(lines)


def _generate_assistant_message_handling(config: TemplateConfig) -> str:
    """Generate assistant message handling section."""
    lines = []

    # Determine supported chunk types
    if config.thinking_support:
        chunk_types = "text and thinking"
    else:
        chunk_types = "text"

    comment = f"{{#- Assistant messages supports {chunk_types} content. #}}"
    lines.append("")
    lines.append(f"    {comment}")
    lines.append("    {%- elif message['role'] == 'assistant' %}")

    # v2 and v3: cannot have both content and tool calls
    if config.version in [TokenizerVersion.v2, TokenizerVersion.v3]:
        lines.append("        {%- if message['content'] is not none and message['content'] | length > 0 and message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}")
        lines.append("            {{- raise_exception('Assistant message cannot have both content and tool calls.') }}")
        lines.append("        {%- endif %}")
        lines.append("")

    # v7+ style validation (must have content or tool calls)
    if config.version >= TokenizerVersion.v7 or (config.version >= TokenizerVersion.v3 and not config.spm) or config.version == TokenizerVersion.v13:
        lines.append("        {%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}")
        lines.append("            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}")
        lines.append("        {%- endif %}")
        lines.append("")

    # String content
    lines.append("        {%- if message['content'] is string and message['content'] != '' %}")

    # SPM space handling
    if config.uses_spm_space_tracking:
        lines.append("            {%- if ns.add_space %}")
        lines.append("                {{- ' ' }}")
        lines.append("                {%- set ns.add_space=false %}")
        lines.append("            {%- endif %}")

    lines.append("            {{- message['content'] }}")

    # v2 outputs EOS after string content (both SPM and non-SPM)
    if config.uses_v2_tool_format:
        lines.append("            {{- '</s>' }}")

    lines.append("        {%- elif message['content'] | length > 0 %}")

    # SPM space handling for list content
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
    lines.append(f"                    {{{{- raise_exception('Only {chunk_types} chunks are supported in assistant message contents.') }}}}")
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")

    # v2 outputs EOS after list content (both SPM and non-SPM)
    if config.uses_v2_tool_format:
        lines.append("            {{- '</s>' }}")

    # v2 and v3_spm style: tool calls in elif branch
    if config.version == TokenizerVersion.v2 or (config.spm and config.version == TokenizerVersion.v3):
        lines.append(_generate_tool_calls_elif_v2_v3(config))
        lines.append("        {%- else %}")
        lines.append("            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}")
        lines.append("        {%- endif %}")
    else:
        lines.append("        {%- endif %}")

    # v7+ style: tool calls after content
    if config.has_tools and not (config.version == TokenizerVersion.v2 or (config.spm and config.version == TokenizerVersion.v3)):
        lines.append("")
        lines.append(_generate_tool_calls_block(config))

    # EOS token (not for v2 which has EOS in each branch)
    if not config.uses_v2_tool_format:
        lines.append("")
        lines.append("        {{- '</s>' }}")

    # SPM tracking
    if config.uses_spm_space_tracking:
        lines.append("        {%- set ns.prev_tool=false %}")

    return "\n".join(lines)


def _generate_tool_calls_elif_v2_v3(config: TemplateConfig) -> str:
    """Generate tool calls as elif branch for v2 (SPM and non-SPM) and v3_spm templates."""
    lines = []

    if config.uses_v2_tool_format:
        # v2: tool calls only valid after all user messages
        lines.append("        {%- elif message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 and ns.index > ns.max_idx_user %}")
    else:
        # v3_spm: tool calls always valid
        lines.append("        {%- elif message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}")

    # SPM space handling (only for SPM templates)
    if config.spm:
        lines.append("            {%- if ns.add_space %}")
        lines.append("                {%- set ns.add_space=false %}")
        lines.append("            {%- endif %}")

    # Token format differs between SPM (with space) and non-SPM (no space)
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
        lines.append("                    {{- raise_exception('Tool call must have an id of 9 characters or numbers.') }}")
        lines.append("                {%- endif %}")
        lines.append("                {%- if arguments is not string %}")
        lines.append("                    {%- set arguments = arguments|tojson|safe %}")
        lines.append("                {%- elif arguments == '' %}")
        lines.append("                    {%- set arguments = '{}' %}")
        lines.append("                {%- endif %}")
        lines.append("                {{- '{\"name\": \"' + name + '\", \"arguments\": ' + arguments + ', \"id\": \"' + id + '\"}' }}")

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
        lines.append("        {%- elif message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 and ns.index <= ns.max_idx_user %}")

    return "\n".join(lines)


def _generate_tool_calls_elif(config: TemplateConfig) -> str:
    """Generate tool calls as elif branch (v3_spm style). DEPRECATED: use _generate_tool_calls_elif_spm."""
    lines = []
    lines.append("        {%- elif message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}")

    # SPM space handling
    lines.append("            {%- if ns.add_space %}")
    lines.append("                {%- set ns.add_space=false %}")
    lines.append("            {%- endif %}")

    lines.append("            {{- '[TOOL_CALLS] [' }}")
    lines.append("            {%- for tool in message['tool_calls'] %}")
    lines.append("                {%- set name = tool['function']['name'] %}")
    lines.append("                {%- set arguments = tool['function']['arguments'] %}")
    lines.append("                {%- set id = tool['id']%}")
    lines.append("                {%- if id is not defined or id|length != 9 %}")
    lines.append("                    {{- raise_exception('Tool call must have an id of 9 characters or numbers.') }}")
    lines.append("                {%- endif %}")
    lines.append("                {%- if arguments is not string %}")
    lines.append("                    {%- set arguments = arguments|tojson|safe %}")
    lines.append("                {%- elif arguments == '' %}")
    lines.append("                    {%- set arguments = '{}' %}")
    lines.append("                {%- endif %}")
    lines.append("                {{- '{\"name\": \"' + name + '\", \"arguments\": ' + arguments + ', \"id\": \"' + id + '\"}' }}")
    lines.append("                {%- if loop.length > 1 and loop.index < loop.length %}")
    lines.append("                    {{- ', ' }}")
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")
    lines.append("            {{- ']' }}")

    return "\n".join(lines)


def _generate_tool_calls_block(config: TemplateConfig) -> str:
    """Generate tool calls block (v7+ style)."""
    lines = []
    lines.append("        {%- if message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}")

    # SPM templates need to reset add_space without outputting space
    if config.uses_spm_space_tracking:
        lines.append("            {%- if ns.add_space %}")
        lines.append("                {%- set ns.add_space=false %}")
        lines.append("            {%- endif %}")

    if config.uses_v13_tool_format:
        # v13 style: [TOOL_CALLS]name[ARGS]arguments
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
        # v11 style: [TOOL_CALLS]name[CALL_ID]id[ARGS]arguments
        lines.append("            {%- for tool in message['tool_calls'] %}")
        lines.append("                {{- '[TOOL_CALLS]' }}")
        lines.append("                {%- set name = tool['function']['name'] %}")
        lines.append("                {%- set arguments = tool['function']['arguments'] %}")
        lines.append("                {%- set id = tool['id']%}")
        lines.append("                {%- if id is not defined or id|length != 9 %}")
        lines.append("                    {{- raise_exception('Tool call must have an id of 9 characters or numbers.') }}")
        lines.append("                {%- endif %}")
        lines.append("                {%- if arguments is not string %}")
        lines.append("                    {%- set arguments = arguments|tojson|safe %}")
        lines.append("                {%- elif arguments == '' %}")
        lines.append("                    {%- set arguments = '{}' %}")
        lines.append("                {%- endif %}")
        lines.append("                {{- name + '[CALL_ID]' + id + '[ARGS]' + arguments }}")
        lines.append("            {%- endfor %}")
    else:
        # v3, v7 style: [TOOL_CALLS][{json}, {json}]
        # SPM has space after [TOOL_CALLS]
        if config.spm:
            lines.append("            {{- '[TOOL_CALLS] [' }}")
        else:
            lines.append("            {{- '[TOOL_CALLS][' }}")
        lines.append("            {%- for tool in message['tool_calls'] %}")
        lines.append("                {%- set name = tool['function']['name'] %}")
        lines.append("                {%- set arguments = tool['function']['arguments'] %}")
        lines.append("                {%- set id = tool['id']%}")
        lines.append("                {%- if id is not defined or id|length != 9 %}")
        lines.append("                    {{- raise_exception('Tool call must have an id of 9 characters or numbers.') }}")
        lines.append("                {%- endif %}")
        lines.append("                {%- if arguments is not string %}")
        lines.append("                    {%- set arguments = arguments|tojson|safe %}")
        lines.append("                {%- elif arguments == '' %}")
        lines.append("                    {%- set arguments = '{}' %}")
        lines.append("                {%- endif %}")
        lines.append("                {{- '{\"name\": \"' + name + '\", \"arguments\": ' + arguments + ', \"id\": \"' + id + '\"}' }}")
        lines.append("                {%- if loop.length > 1 and loop.index < loop.length %}")
        lines.append("                    {{- ', ' }}")
        lines.append("                {%- endif %}")
        lines.append("            {%- endfor %}")
        lines.append("            {{- ']' }}")

    lines.append("        {%- endif %}")

    return "\n".join(lines)


def _generate_tool_message_handling(config: TemplateConfig) -> str:
    """Generate tool message handling section."""
    if not config.has_tools:
        return ""

    lines = []
    lines.append("")

    if config.uses_v2_tool_format:
        # v2: tool messages with index check
        lines.append("    {#- Tool messages supports int, float or text content. #}")
        lines.append("    {%- elif message['role'] == 'tool' and ns.index > ns.max_idx_user %}")
    else:
        lines.append("    {#- Tool messages only supports text content. #}")
        lines.append("    {%- elif message['role'] == 'tool' %}")

    # SPM space handling
    if config.uses_spm_space_tracking:
        lines.append("        {%- if ns.add_space %}")
        lines.append("            {%- if not ns.prev_tool %}")
        lines.append("                {{- ' '}}")
        lines.append("            {%- endif %}")
        lines.append("            {%- set ns.add_space=false %}")
        lines.append("        {%- endif %}")

    if config.uses_v2_tool_format:
        # v2 style: JSON with name instead of call_id
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
        # SPM has space after [TOOL_RESULTS], non-SPM doesn't
        if config.spm:
            lines.append("        {{- '[TOOL_RESULTS] [' + (tool_message|tojson) + '][/TOOL_RESULTS]' }}")
        else:
            lines.append("        {{- '[TOOL_RESULTS][' + (tool_message|tojson) + '][/TOOL_RESULTS]' }}")
    elif config.uses_json_tool_results:
        # v3_spm style: JSON with type parsing
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
        lines.append("        {%- elif message['tool_call_id'] is not undefined and message['tool_call_id']|length == 9  %}")
        lines.append("            {%- set tool_id = message['tool_call_id'] %}")
        lines.append("        {%- else %}")
        lines.append("            {{- raise_exception('Tool message must have a call_id or tool_call_id of 9 characters or numbers.') }}")
        lines.append("        {%- endif %}")
        lines.append("")
        lines.append("        {%- set tool_message = {'content': tool_content, 'call_id': tool_id} %}")
        lines.append("        ")
        lines.append("        {{- '[TOOL_RESULTS] ' + (tool_message|tojson) + '[/TOOL_RESULTS]' }}")
    elif config.uses_tool_content_format:
        # v7, v11 style: [TOOL_RESULTS]id[TOOL_CONTENT]content
        lines.append("        {%- if message['call_id'] is not undefined and message['call_id']|length == 9 %}")
        lines.append("            {%- set tool_id = message['call_id'] %}")
        lines.append("        {%- elif message['tool_call_id'] is not undefined and message['tool_call_id']|length == 9  %}")
        lines.append("            {%- set tool_id = message['tool_call_id'] %}")
        lines.append("        {%- else %}")
        lines.append("            {{- raise_exception('Tool message must have a call_id or tool_call_id of 9 characters or numbers.') }}")
        lines.append("        {%- endif %}")
        if config.spm:
            # v7_spm has spaces in tokens
            lines.append("        {{- '[TOOL_RESULTS] ' + tool_id + '[TOOL_CONTENT] ' + message['content']|string + '[/TOOL_RESULTS]' }}")
        else:
            lines.append("        {{- '[TOOL_RESULTS]' + tool_id + '[TOOL_CONTENT]' + message['content']|string + '[/TOOL_RESULTS]' }}")
    elif config.uses_simple_tool_results:
        # v13 style: simple [TOOL_RESULTS]content
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
        lines.append("        {%- elif message['tool_call_id'] is not undefined and message['tool_call_id']|length == 9  %}")
        lines.append("            {%- set tool_id = message['tool_call_id'] %}")
        lines.append("        {%- else %}")
        lines.append("            {{- raise_exception('Tool message must have a call_id or tool_call_id of 9 characters or numbers.') }}")
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


def _generate_else_block(config: TemplateConfig) -> str:
    """Generate else block for unsupported roles."""
    lines = []
    lines.append("")
    lines.append("    {#- Raise exception for unsupported roles. #}")

    if config.has_tools:
        if config.version == TokenizerVersion.v2 or (config.spm and config.version == TokenizerVersion.v3):
            lines.append("    {%- elif message['role'] != 'tool' or ns.index > ns.max_idx_user %}")
        else:
            lines.append("    {%- else %}")
        lines.append("        {{- raise_exception('Only user, assistant and tool roles are supported, got ' + message['role'] + '.') }}")
    else:
        lines.append("    {%- else %}")
        lines.append("        {{- raise_exception('Only user and assistant roles are supported, got ' + message['role'] + '.') }}")

    lines.append("    {%- endif %}")

    return "\n".join(lines)


def _generate_message_loop(config: TemplateConfig) -> str:
    """Generate the main message processing loop."""
    lines = []

    # Reset index for v3/v7 SPM style
    if config.tracks_max_idx_user:
        lines.append("")
        lines.append("{%- set ns.index = 0 %}")

    lines.append("{#- Handle conversation messages. #}")
    lines.append("{%- for message in loop_messages %}")

    # User message
    lines.append(_generate_user_message_handling(config))

    # Assistant message
    lines.append(_generate_assistant_message_handling(config))

    # Tool message
    lines.append(_generate_tool_message_handling(config))

    # Else block
    lines.append(_generate_else_block(config))

    # Index tracking for v3/v7 style
    if config.tracks_max_idx_user:
        lines.append("    {%- if message['role'] == 'user' %}")
        lines.append("        {%- set ns.index = ns.index + 1 %}")
        lines.append("    {%- endif %}")

    lines.append("{%- endfor %}")

    return "\n".join(lines)


def _generate_v1_template(config: TemplateConfig) -> str:
    """Generate v1 template which has unique structure."""
    lines = []
    lines.append(_generate_header())
    lines.append(_generate_system_prompt_handling(config))
    lines.append("")
    lines.append("{#- Checks for alternating user/assistant messages. #}")
    lines.append("{%- set ns = namespace(index=0) %}")
    lines.append("{%- for message in loop_messages %}")
    lines.append("    {%- if message.role == 'user' or message.role == 'assistant' %}")
    lines.append("        {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}")
    lines.append("            {{- raise_exception('After the optional system message, conversation roles must alternate user and assistant.') }}")
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
    lines.append("                    {{- raise_exception('Only text chunks are supported in user message content.') }}")
    lines.append("                {%- endif %}")
    lines.append("            {%- endfor %}")
    lines.append("        {%- else %}")
    lines.append("            {{- raise_exception('User message must have a string or a list of chunks in content') }}")
    lines.append("        {%- endif %}")
    lines.append("        {{- ' [/INST]' }}")
    if config.spm:
        lines.append("        {%- if loop.index < loop.length %}")
        lines.append("            {{- \" \" }}")
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
    lines.append("                    {{- raise_exception('Only text chunks are supported in assistant message contents.') }}")
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
    lines.append("        {{- raise_exception('Only user and assistant roles are supported, got ' + message['role'] + '.') }}")
    lines.append("    {%- endif %}")
    lines.append("{%- endfor %}")

    return "\n".join(lines) + "\n"


def generate_chat_template(config: TemplateConfig) -> str:
    """Generate a complete chat template based on configuration.

    Args:
        config: Template configuration specifying version and features.

    Returns:
        The complete Jinja2 template as a string.
    """
    # v1 has a unique structure, handle separately
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
