# Chat templates

`mistral-common` can generate a [Jinja](https://jinja.palletsprojects.com/) chat template for a Mistral tokenizer, such as the `chat_template.jinja` file of a Hugging Face model repository. The template renders OpenAI-format messages to the same prompt text as [MistralTokenizer.encode_chat_completion][mistral_common.tokens.tokenizers.mistral.MistralTokenizer.encode_chat_completion], so libraries that render chat templates, such as Transformers, build the same prompts as `mistral-common`.

## From a tokenizer file

[convert_tokenizer_to_chat_template][mistral_common.integrations.chat_templates.chat_templates.convert_tokenizer_to_chat_template] loads a tokenizer file (`tekken.json` or a SentencePiece `.model.vX` file), detects its version and features (images, audio, thinking, model settings) and returns the matching template:

```python
from huggingface_hub import hf_hub_download

from mistral_common.integrations.chat_templates.chat_templates import convert_tokenizer_to_chat_template

repo_id = "mistralai/Ministral-3-3B-Instruct-2512"
tokenizer_file = hf_hub_download(repo_id=repo_id, filename="tekken.json")

chat_template = convert_tokenizer_to_chat_template(
    tokenizer_file=tokenizer_file, system_prompt="You are a helpful assistant."
)

with open("chat_template.jinja", "w", encoding="utf-8") as f:
    f.write(chat_template)
```

- `system_prompt` is the default system prompt, rendered when the conversation does not start with a system message.
- By default the template reads the BOS and EOS tokens from the `bos_token` and `eos_token` variables. Pass `use_special_token_variables=False` to write them as literals.
- A v11 tokenizer file does not say whether the model thinks with plain `<think>` tags, so plain thinking is enabled for every v11 tokenizer without audio. Pass `plain_thinking_support=False` for a v11 model that does not think.

## Render the template

Render the template with the conversation as `messages` in the OpenAI format, and with `tools`, `bos_token` and `eos_token`. Templates for tokenizers with model settings also read `reasoning_effort` or `response_format`, depending on the settings that the tokenizer configures. For example, with Transformers:

```python
from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=repo_id, fix_mistral_regex=True)
hf_tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": "Hello, how are you?"}], chat_template=chat_template, tokenize=False
)
# '<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST]Hello, how are you?[/INST]'
```

For a conversation that `mistral-common` rejects, such as a system message after an assistant message, the template calls `raise_exception` with an error message, here `Unexpected role 'system' after role 'assistant'`. Transformers provides `raise_exception`; with another Jinja engine, define it as a function that raises.

The repository's tests check that Transformers' `apply_chat_template` produces the same tokens as `MistralTokenizer.encode_chat_completion`, and render every template variant with [minijinja](https://github.com/mitsuhiko/minijinja).

## Generation blocks

The template wraps each assistant turn, up to and including its EOS token, in a `{% generation %}` block. Transformers uses these blocks to mark the assistant tokens when you pass `return_assistant_tokens_mask=True` to `apply_chat_template`, for example to train on the assistant tokens only. The blocks do not change the rendered text.

Jinja engines that do not know the `generation` tag fail to parse the template: plain Jinja2 raises `Encountered unknown tag 'generation'` and minijinja raises `unknown statement generation`. For these engines, generate the template without the blocks:

```python
chat_template = convert_tokenizer_to_chat_template(tokenizer_file=tokenizer_file, use_generation_blocks=False)
```

## Without a tokenizer file

[generate_chat_template][mistral_common.integrations.chat_templates.chat_templates.generate_chat_template] builds a template from the tokenizer version and features instead. Every argument except `use_generation_blocks` is required:

```python
from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion

chat_template = generate_chat_template(
    spm=False,
    tokenizer_version=TokenizerVersion.v7,
    image_support=True,
    audio_support=False,
    thinking_support=False,
    default_system_prompt=None,
    plain_thinking_support=False,
    use_special_token_variables=True,
    model_settings_fields=None,
    reasoning_effort_values=None,
    reasoning_effort_default=None,
)
```

Combinations that no tokenizer supports raise a `ValueError`, for example `spm=True` with v11 or later. See [TemplateConfig][mistral_common.integrations.chat_templates.template_generator.TemplateConfig] for the rules.

## Command line

The repository's [`scripts/generate_chat_template.py`](https://github.com/mistralai/mistral-common/blob/main/scripts/generate_chat_template.py) writes a template to a file, from a tokenizer file or from flags:

```bash
python scripts/generate_chat_template.py --tokenizer_file tekken.json --default_system_prompt "You are a helpful assistant." --saving_path chat_template.jinja
python scripts/generate_chat_template.py --version v7 --image --saving_path chat_template.jinja
```

`--no_generation_blocks` leaves out the generation blocks, `--no_special_token_variables` writes literal BOS and EOS tokens, and `--plain_thinking` or `--no_plain_thinking` set plain thinking support, overriding the v11 detection with `--tokenizer_file`. Run the script with `--help` for all the options.
