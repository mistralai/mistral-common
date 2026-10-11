from jinja2.sandbox import ImmutableSandboxedEnvironment
from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion

def test_dict_tool_call_arguments_not_html_escaped():
    for version in [TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7]:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
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
            bos_token="<s>", eos_token="</s>",
        )
        assert "&#34;" not in out, f"Trovate entità HTML &#34; nel rendering della versione {version}"
        assert '"name": "get_weather"' in out, f"Stringa JSON corrotta nella versione {version}"
