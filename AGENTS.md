# AGENTS.md

## Project overview

Mistral-Common is a preprocessing library for Mistral's Large Language Models (LLMs).
It encodes requests for Instruct, Transcription or Fill-In-The-Middle (FIM) tasks to tokens and optionally processed images or audios.
- Language: Python 3.10 to 3.13
- Package Manager: UV (not pip/conda)
- Testing: pytest with distributed testing support
- Linting: Ruff (formatting + linting) + mypy
- CI: GitHub Actions

## Project Structure

```
mistral-common/
├── src/
│   └── mistral_common/
│       ├── protocol/
│       ├── tokens/
│       └── ...
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
    - `multimodal.py`: deprecated in favor of `image.py`
    - `sentencepiece.py`: Sentence Piece tokenizer (deprecated)
    - `tekken.py`: Tekken tokenizer used by all recent models
    - `utils.py`: Utility functions for the tokenizers
  - `instruct/`: deprecated in favor of `src/mistral_common/protocol/instruct/request.py`

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
- **Formatter:** Ruff with double quotes
- **Line length:** 120 characters
- **Naming:** snake_case for functions/variables, PascalCase for classes

### Imports
- Use absolute imports for modules within the project
- Do NOT use wildcard imports
- Sorted by ruff
- Use `TYPE_CHECKING` blocks for type-only imports

### Type Hints (Required)
- Use Python's type hints extensively
- Use modern typing module types

```python
# Use modern union syntax (Python 3.10+)
def process(data: str | None) -> tuple[int, float]:
    ...

# NOT this:
def process(data: Union[str, None]) -> Tuple[int, float]:
    ...
```

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

#### Examples

##### Function One-liner 
```python
def add(a: int, b: int) -> int:
    r"""Returns the sum of two integers."""
    return a + b
```
##### Function Multi-Line
```python
def calculate_area(length: float, width: float) -> float:
    r"""Calculates the area of a rectangle.

    Args:
        length: The length of the rectangle.
        width: The width of the rectangle.

    Returns:
        The area of the rectangle.

    Raises:
        ValueError: If either length or width is negative.

    Examples:
        >>> calculate_area(5.0, 3.0)
    """
    if length < 0 or width < 0:
        raise ValueError("Dimensions cannot be negative.")
    return length * width
```

##### Pydantic Model
```python
from pydantic import BaseModel

class UserRequest(BaseModel):
    r"""Represents a user request with validation and serialization.

    This model is used to parse and validate incoming user requests,
    ensuring all required fields are present and properly formatted.

    Attributes:
        user_id: Unique identifier for the user making the request.
        query: The user's input query or prompt.
        max_tokens: Maximum number of tokens to generate in the response.
        temperature: Controls the randomness of the output.

    Examples:
        >>> request = UserRequest(
        ...     user_id="user123",
        ...     query="Hello, how are you?",
        ...     max_tokens=100,
        ...     temperature=0.7
        ... )
    """
    user_id: str
    query: str
    max_tokens: int = 100
    temperature: float = 0.7
```

## Development Workflow

### Setup
```bash
uv sync --frozen --all-extras --group dev --python 3.12
source .venv/bin/activate
uv run pre-commit install
```

### Make Changes
- Follow code style guidelines
  - Add type hints to all functions
  - Add docstrings to all public functions and classes
- Write tests for new functionality
- Update documentation
- When adding dependencies, modify root `pyproject.toml`, then run `uv sync`

### Run Checks
```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy .
uv run pytest
uv run pytest --doctest-modules ./src
```

### Commit
After adding your changes run:
```bash
uv run pre-commit run --all-files
git commit -m "Your descriptive commit message"
```

## Best Practices

- Type Safety: Use type hints, ensure mypy passes
- Test Coverage: High test coverage for new features
- Documentation: Keep documentation up-to-date and add docstrings to public functions and classes
- Backward Compatibility: Don't break existing functionality

## Additional Resources

- [Mistral AI Documentation](https://mistralai.github.io/mistral-common/)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
