# Install

## Pip

You can install the library using pip:
```sh
pip install mistral-common
```

## From source

To build it for source, you can clone the repository and install it using [uv](https://github.com/astral-sh/uv) or pip. We recommend using uv for faster and more reliable dependency resolution:
```sh
git clone https://github.com/mistralai/mistral-common.git
cd mistral-common
uv sync --frozen
uv pip install . # or `uv pip install -e .` for development
```
