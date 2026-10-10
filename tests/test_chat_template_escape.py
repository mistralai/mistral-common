from jinja2.sandbox import ImmutableSandboxedEnvironment
from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion

def test_dict_tool_call_arguments_no_html_escape():
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v3,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        default_system_prompt=None,
        plain_thinking_support=False,
        use_special_token_variables=False,
    )
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    out = env.from_string(template).render(
        messages=[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "123456789", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ]},
        ],
        bos_token="<s>",
        eos_token="</s>",
    )
    
    assert "&#34;" not in out, "Detected HTML entity escaping in tool call JSON output"
    assert '[{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "123456789"}]' in out
