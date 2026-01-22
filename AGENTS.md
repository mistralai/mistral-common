# Agent Guidelines for Mistral-Common

This document provides guidelines for agentic coding in the Mistral-Common repository.

## Install/Lint/Test Commands

This repository uses `uv` for package management. Use `uv` to install dependencies, run linters, formatters, and tests. Don't forget to install pre-commit hooks.

### Install
```bash
# Install all dependencies
uv sync --frozen --all-extras --group dev
uv pip install pre-commit
```

### Lint
```bash
# Run Ruff linter
uv run ruff check .

# Run Ruff formatter
uv run ruff format . --check
```

### Test
```bash
# Run all tests
uv run pytest --cov=mistral_common . --cov-report "xml:coverage.xml"

# Run doctests
uv run pytest --doctest-modules ./src

# Run a single test file
uv run pytest tests/test_tokenize_v3.py

# Run a specific test
uv run pytest tests/test_tokenize_v3.py::test_tools_singleturn
```

## Code Style Guidelines

### Imports
- Use absolute imports for modules within the project
- Do not perform wildcard imports (`from module import *`)

### Formatting
- Line length: 120 characters (configured in Ruff)
- Use 4 spaces for indentation
- Follow PEP 8 guidelines for naming and spacing
- Use consistent spacing around operators and after commas
- Use double quotes for strings unless the string contains double quotes

### Types
- Use Python's type hints extensively
- All functions must have type annotations
- Use modern `typing` module types (e.g., `list`, `dict`, `optional`) where appropriate
- For complex types, consider using `TypeVar` or `Protocol`

### Naming Conventions
- **Variables and Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Classes**: `PascalCase`
- **Private members**: `_leading_underscore`
- **Protected members**: `_leading_underscore` (same as private)
- **Test functions**: `test_` prefix followed by descriptive name in snake_case

### Error Handling
- Use custom exceptions defined in `mistral_common.exceptions`
- Provide meaningful error messages
- Handle exceptions at the appropriate level
- Use context managers for resource cleanup

### Documentation
- Use Google-style docstrings for all public modules, classes, and functions
- Use `r"""` for raw docstrings to avoid escaping issues
- Don't include type information in docstrings (types are handled by type hints)
- Document all parameters and return values
- Include examples where appropriate
- For simple functions, use one-line docstrings
- Always include examples in docstrings to demonstrate usage and ensure they are tested by doctest

#### Docstring Examples

**Class docstring:**
```python
class Audio:
    r"""Audio processing and manipulation utilities.
    
    This class provides methods for loading, processing, and converting audio data
    in various formats. It supports operations like resampling, format conversion,
    and base64 encoding/decoding.
    
    Attributes:
        audio_array: The audio data as a numpy array
        sampling_rate: The sampling rate of the audio in Hz
        format: The format of the audio file
    
    Examples:
        >>> audio = Audio.from_file("audio.wav")
        >>> audio.resample(16000)
    """
```

**Enum docstring:**
```python
class ResponseFormats(str, Enum):
    r"""Enum of the different formats of an instruct response.
    
    Attributes:
        text: The response is a plain text.
        json: The response is a JSON object.
    
    Examples:
        >>> response_format = ResponseFormats.text
    """
```

**Method docstring:**
```python
def resample(self, new_sampling_rate: int) -> None:
    r"""Resample audio data to a new sampling rate.
    
    Args:
        new_sampling_rate: The new sampling rate to resample the audio to.
    
    Examples:
        >>> audio = Audio.from_file("audio.wav")
        >>> audio.resample(16000)
    """
```

**Function docstring (multi-line):**
```python
def download_image(url: str) -> Image.Image:
    r"""Download an image from a URL and return it as a PIL Image.
    
    Args:
        url: The URL of the image to download.
    
    Returns:
       The downloaded image as a PIL Image object.
    
    Raises:
        RuntimeError: If the download fails or image conversion fails.
    """
```

**Function docstring (one-line):**
```python
def is_package_installed(package_name: str) -> bool:
    r"""Check if a package is installed in the current environment."""
```

### Testing
- Write comprehensive tests for all new features
- Follow the existing test patterns in the repository
- Use pytest for testing
- Test both happy paths and edge cases
- Include integration tests where appropriate

### Code Organization
- Keep related functionality together
- Follow the existing module structure
- Use appropriate abstraction levels
- Avoid circular dependencies

### Version Control
- Follow GitHub Flow for branching
- Use descriptive commit messages
- Reference issues in commit messages when applicable
- Keep commits focused on a single change

### Continuous Integration
- All changes must pass the CI pipeline
- CI includes linting, formatting, type checking, and tests
- Ensure your changes work across all supported Python versions (3.10-3.13)

### Python Version Compatibility
- This repository supports Python versions 3.10 to 3.13
- Write code that works natively across all supported versions
- Avoid using version-specific features that aren't available in all supported versions
- Don't use backported libraries (e.g., `futures`) for features available in standard library

## Project Structure

```
mistral-common/
├── src/
│   └── mistral_common/
│       ├── protocol/
│       ├── tokens/
│       └── ...
├── tests/
│   ├── test_*.py
│   └── ...
├── docs/
├── .github/
│   └── workflows/
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

## Key Files and Directories

### Protocol-related functionality
- `src/mistral_common/protocol/`: Contains protocol-related functionality for handling different types of requests
  - `base.py`: Base classes for completion requests
  - `instruct/`: Instruct protocol implementation
    - `request.py`: Chat completion and instruct request classes with OpenAI compatibility
    - `messages.py`: Message types (UserMessage, AssistantMessage, SystemMessage, ToolMessage)
    - `tool_calls.py`: Tool and function definitions for tool usage
    - `validator.py`: Request validation logic for different tokenizer versions
    - `normalize.py`: Request normalization for tokenization
    - `converters.py`: Conversion utilities between Mistral and OpenAI formats
  - `fim/`: Fill-in-the-middle protocol implementation
    - `request.py`: FIM request classes
  - `transcription/`: Transcription protocol implementation
    - `request.py`: Transcription request classes
  - `utils.py`: Utility functions for protocol handling

### Tokenization logic
- `src/mistral_common/tokens/`: Contains tokenization logic and tokenizer implementations
  - `tokenizers/`: Tokenizer implementations
    - `base.py`: Base tokenizer classes and interfaces
    - `mistral.py`: Main Mistral tokenizer implementation that wraps instruct tokenizers
    - `sentencepiece.py`: SentencePiece tokenizer implementation (legacy, outdated)
    - `tekken.py`: Tekken tokenizer implementation (current standard for all models)
    - `instruct.py`: Instruct-specific tokenizer implementations for different versions
    - `image.py`: Image encoding functionality
    - `audio.py`: Audio encoding functionality
    - `multimodal.py`: Multimodal token handling
    - `utils.py`: Tokenizer utility functions
  - `instruct/`: Instruct-specific tokenization
    - `request.py`: Instruct request tokenization

### Other important files and directories
- `tests/`: Test suite containing comprehensive tests for all functionality
- `docs/`: Documentation for the project
- `.github/workflows/`: CI/CD workflows for automated testing and deployment
- `.pre-commit-config.yaml`: Pre-commit hooks configuration for code quality
- `pyproject.toml`: Project configuration and dependencies

### Root-level modules
- `audio.py`: Audio processing utilities including format conversion, resampling, and mel-scale transformations
- `base.py`: Base Pydantic model configuration for all Mistral models
- `exceptions.py`: Custom exception classes for various error scenarios
- `image.py`: Image handling utilities including download, serialization, and base64 encoding
- `imports.py`: Package import utilities and dependency checking functions
- `multimodal.py`: Multimodal functionality (deprecated, use image.py instead)

### Experimental features
- `src/mistral_common/experimental/`: Contains experimental features not yet stable for main library use
  - `tools.py`: Tool call parsing and handling utilities
  - `think.py`: Reasoning/think chunk parsing functionality
  - `app/`: Web application components
    - `main.py`: FastAPI application setup and CLI
    - `routers.py`: API route definitions
    - `models.py`: Application models and settings

## Development Workflow

1. **Setup**:
   ```bash
   git clone https://github.com/mistralai/mistral-common.git
   cd mistral-common
   uv venv
   source .venv/bin/activate
   uv sync --frozen --all-extras --group dev
   uv run pre-commit install
   ```

2. **Make Changes**:
   - Follow the code style guidelines
   - Write tests for new functionality
   - Update documentation as needed

3. **Run Checks**:
   ```bash
   uv run ruff check .
   uv run ruff format . --check
   uv run mypy .
   uv run pytest
   ```

4. **Commit**:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

5. **Push and Create PR**:
   ```bash
   git push origin your-branch-name
   # Create PR on GitHub
   ```

## Best Practices

1. **Type Safety**: Always use type hints and ensure mypy passes
2. **Test Coverage**: Aim for high test coverage, especially for new features
3. **Documentation**: Keep documentation up-to-date
4. **Backward Compatibility**: Ensure changes don't break existing functionality
5. **Performance**: Consider performance implications of changes
6. **Security**: Follow security best practices
7. **Code Reviews**: Participate in code reviews and address feedback

## Additional Resources

- [Mistral AI Documentation](https://mistralai.github.io/mistral-common/)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
