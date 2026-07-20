# AGENTS.md

## Project overview

Mistral-Common is a preprocessing library for Mistral's Large Language Models (LLMs).
It encodes requests for Instruct, Transcription or Fill-In-The-Middle (FIM) tasks to tokens and optionally processed images or audios.
- Language: Python 3.10 to 3.14
- Package Manager: uv
- Testing: pytest
- Formatting and Linting: Ruff
- Type checker: mypy
- CI: GitHub Actions

## Project Structure

```
mistral-common/
├── src/
│   └── mistral_common/
│       ├── guidance/
│       ├── integrations/
│       ├── protocol/
│       ├── tokens/
│       └── ...
├── scripts/
├── tests/
├── docs/
├── .github/
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

### Root-level files in src/mistral_common/
- `audio.py`: Audio processing utilities including Audio class and mel-scale conversions
- `base.py`: Base Pydantic model configuration
- `deprecation.py`: Deprecation utilities (`deprecated_import`, `warn_once`) for emitting one-shot warnings on moved or removed symbols
- `exceptions.py`: Custom exception classes for the library
- `image.py`: Image processing utilities including download and serialization
- `imports.py`: Import utilities and dependency checks
- `multimodal.py`: Multimodal processing utilities deprecated in favor to `image.py`

### Protocol
- `src/mistral_common/protocol/`: Protocol handling
  - `instruct/`: Instruct protocol
    - `chunk.py`: Chunks content used by messages
    - `converters.py`: Converter helpers between `ChatCompletionRequest` and `Openai` requests
    - `messages.py`: Instruct messages definition
    - `normalize.py`: Normalizers for `ChatCompletionRequest`
    - `request.py`: Definition of `ChatCompletionRequest`. This is the entry point of user queries for Instruct requests
    - `tool_calls.py`: Tool calling logic
    - `validator.py`: Validators for `ChatCompletionRequest`
  - `fim/`: Fill-in-the-middle protocol
    - `request.py`: Definition of `FIMRequest`. This is the entry point of user queries for FIM requests
  - `transcription/`: Transcription protocol
    - `request.py`: Definition of `TranscriptionRequest`. This is the entry point of user queries for Transcription requests
  - `speech/`: Speech protocol
    - `request.py`: Definition of `SpeechRequest`. This is the entry point of user queries for Speech requests
  - `base.py`: Definition of `BaseCompletionRequest` subclassed by FIM and Instruct requests
  - `utils.py`: Utility functions

### Tokenization
- `src/mistral_common/tokens/`: Tokenization
  - `tokenizers/`: Tokenizer implementations
    - `audio.py`: Audio processing
    - `base.py`: Base Tokenizer implementation
    - `image.py`: Image processing
    - `instruct.py`: Instruct Tokenizer that encodes requests via Tekken tokenizer
    - `mistral.py`: Mistral Tokenizer that normalizes and validates requests to pass them to an instruct tokenizer
    - `model_settings_builder.py`: Builders (`FieldBuilder`, `EnumBuilder`, `ModelSettingsBuilder`) for validating and constructing model settings
    - `multimodal.py`: deprecated in favor of `image.py`
    - `sentencepiece.py`: Sentence Piece tokenizer (deprecated)
    - `tekken.py`: Tekken tokenizer used by all recent models
    - `utils.py`: Utility functions for the tokenizers
  - `instruct/`: deprecated in favor of `src/mistral_common/protocol/instruct/request.py`

### Guidance (Grammar)
- `src/mistral_common/guidance/`: Creates Lark grammars for tool calls, JSON schema and reasoning using llguidance
  - `grammar_factory.py`: `GrammarFactory` that builds and renders Lark grammars from Jinja templates
  - `tokenizer.py`: Adapts Tekken tokenizer for llguidance
  - `data/`: Jinja-templated Lark grammar files for base, thinking (special tokens) and thinking (plain text) modes

### Integrations
- `src/mistral_common/integrations/`: Third-party framework integrations
  - `chat_templates/`: Chat template generation for HuggingFace Transformers
    - `chat_templates.py`: Public API for generating chat templates (`generate_chat_template`)
    - `template_generator.py`: Core template generation engine with `TemplateConfig` and `build_chat_template`

### Scripts
- `scripts/generate_chat_template.py`: CLI for generating and saving chat templates

## Experimental
- `src/mistral_common/experimental/`: Experimental features
  - `utils.py`: Utility functions
  - `tools.py`: Tool calls parser
  - `think.py`: Thinking parser
  - `app/`: FastAPI application
    - `routers.py`: API routers
    - `main.py`: Application entry point
    - `models.py`: Pydantic models

## Data
- `src/mistral_common/data/`: Data files for tokenizers

### Other files and directories
- `tests/`: Test suite
- `docs/`: Documentation
- `.github/workflows/`: CI/CD workflows
- `.pre-commit-config.yaml`: Pre-commit hooks
- `pyproject.toml`: Project configuration

## Code Style Guidelines

### Style
- Respect ruff and mypy rules
- Naming: snake_case for functions/variables, PascalCase for classes
- Use Python functionalities supported by Python 3.10
- Call function arguments explicitly by keyword, not implicitly by position (e.g. `fn(x=1, y=2)`, not `fn(1, 2)`)

### Imports
- Use absolute imports for modules within the project
- Do NOT use wildcard imports
- Do NOT add import inside `__init__`
- Use `TYPE_CHECKING` blocks for type-only imports
- Do NOT use `from __future__ import annotations`

### Type Hints (Required)
- Use Python's type hints extensively
- Use modern (Python 3.10+) typing module types

### Error Handling
- Use custom exceptions from `mistral_common.exceptions`
- Provide meaningful error messages

### Docstrings (Required)
- Use Google-style docstrings
- One-liner for simple functions
- Multi-line with Args/Returns for complex ones
- Use `r"""` for raw docstrings
- Document all parameters and return values.
  - Do NOT put types in the docstring for parameters and return values.
  - For the returns sections, only describe the returned value and do not write its name
- Include examples where appropriate that can be tested via `doctest`

## Development Workflow

1. Set up using uv

```bash
uv sync --frozen --all-extras --group dev --python 3.12
source .venv/bin/activate
uv run pre-commit install
```

2.  Make Changes
- Follow code style guidelines
- Write tests for new functionality
- Update documentation
- When adding dependencies, modify root `pyproject.toml`, then run `uv lock` followed by `uv sync --frozen`
- Backward Compatibility: Don't break existing functionality

3. Run linter, formatter (Ruff), type checker (mypy) and tests (pytest), including doctests

### Commit
- After adding your changes before committing ensure pre-commit is installed or run it manually.
- Use imperative grammar, start with a verb and be concise.

## Additional Resources

- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
