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

### Comments
- Do NOT write comments that paraphrase or restate what the code already says.
- Comments should explain the "why" (intent, rationale, non-obvious constraints), not the "what".

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

### Coverage
- Aim for high, meaningful coverage. Prioritise tests that verify behaviour over hitting a line-count target.
- New and changed code should be covered by tests.
- Avoid coverage-only comments. Prefer restructuring so branches are genuinely reachable and tested (e.g. validate inputs and test the error path) over excluding lines.
- Retain existing local `pytest-cov`, coverage.py, and diff-cover workflows and the unit-only CI coverage output unchanged. The unit-only percentage is not an authoritative whole-suite signal.
- Defer combined coverage measurement, thresholds, baselines, historical comparisons, and the 90% aspiration until the test architecture stabilises. Do not distort production design or behavior to improve a coverage number.

### Test Suite Modernization

#### Layout and classification
- New tests belong under `tests/unit/`, `tests/integration/`, `tests/utils/`, or `tests/utils_tests/`. Unit tests mirror the source package structure; integration tests follow public workflows rather than private implementation boundaries.
- During migration, `tests/integrations/` and `tests/integration/` are both integration roots. Legacy tests outside those roots remain in the unit lane until they move to a target path. A test move must preserve its existing coverage atomically; no test ID may disappear or be collected by both lanes.
- Unit tests are hermetic and exercise one module or a narrowly bounded unit with controlled inputs. Integration tests exercise public multi-module workflows with real artifacts and resources appropriate to that workflow.
- Split test packages by behavior and ownership, not by arbitrary file size. Group related tests by class or function, and use source-mirroring paths for unit tests. There is no numeric file-size limit; split a file when its behavior or ownership becomes unclear.

#### Test cases, fixtures, and compatibility
- Parametrization IDs describe the semantic feature, input, or version under test. Use complete compatible feature and version matrices; do not omit supported combinations merely to shorten a run.
- Error tests assert the exact exception type and stable message or error details that form part of the contract.
- Fixtures own data at the nearest scope that uses it. Return fresh mutable values for each test. Use session-scoped fixtures only for expensive immutable resources, and promote a fixture to a broader scope after real reuse justifies it.

#### Parallel execution
- Both unit and integration lanes are required for every supported Python version and must run with `pytest-xdist` using its default `load` distribution. Keep xdist activation explicit for those suites rather than making every pytest invocation parallel by default.
- Add a serial or grouped exception only after reproducing and documenting a process-safety failure. Do not add ordering constraints, sleeps, serial markers, or shared global state to make parallel tests pass.

#### Migration and production behavior
- Replace legacy test behavior atomically while retaining coverage and observable behavior. Do not combine test-path migration with an unrelated production defect fix.
- Production defects found during migration require a separate PR with a characterized test that records the existing behavior before the production change.

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
