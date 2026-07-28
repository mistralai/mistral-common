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

## Layout

- `src/mistral_common/protocol/<instruct|fim|transcription|speech>/request.py` — the request
  types are the entry points for user queries.
- `src/mistral_common/tokens/tokenizers/` — `mistral.py` normalizes and validates a request, then
  delegates to `instruct.py`. Backends: `tekken.py` (all recent models) and `sentencepiece.py`.
- `src/mistral_common/guidance/` — Lark grammars for tool calls, JSON schema and reasoning,
  built with llguidance from Jinja templates in `guidance/data/`.
- `src/mistral_common/integrations/chat_templates/` — HuggingFace chat-template generation;
  public API is `generate_chat_template`.
- `src/mistral_common/experimental/` — unstable, includes a FastAPI app under `app/`.

Deprecated shims — re-export only, do not add to them:

| Shim | Use instead |
|---|---|
| `mistral_common/multimodal.py` | `mistral_common/image.py` |
| `tokens/tokenizers/multimodal.py` | `tokens/tokenizers/image.py` |

`sentencepiece.py` is the legacy backend, not deprecated: it is still the real implementation for
v1–v7 models. New work targets `tekken.py`.

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

```bash
uv sync --frozen --all-extras --group dev --python 3.12
source .venv/bin/activate
uv run pre-commit install
```

- Adding a dependency: edit the root `pyproject.toml`, then `uv lock` and `uv sync --frozen`.
- Don't break existing functionality; keep backward compatibility.
- Before committing, run Ruff (lint + format), mypy, and pytest including doctests, and make sure
  pre-commit has run.
- Commit messages: imperative, start with a verb, concise.
