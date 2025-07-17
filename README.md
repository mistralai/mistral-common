<div align="center">

<img src="./docs/assets/logo.svg" alt="Mistral AI" height="100"/>

<br/>
<br/>

# Mistral-common

[![PyPI version](https://img.shields.io/pypi/v/mistral-common?label=release&logo=pypi&logoColor=white)](https://pypi.org/project/mistral-common/)
[![Tests](https://img.shields.io/github/actions/workflow/status/mistralai/mistral-common/lint_build_test.yaml?label=tests&branch=main)](https://github.com/mistralai/mistral-common/actions/workflows/lint_build_test.yaml)
[![Documentation](https://img.shields.io/website?url=https%3A%2F%2Fmistralai.github.io%2Fmistral-common%2F&up_message=online&down_message=offline&label=docs)](https://mistralai.github.io/mistral-common/)
[![Python version](https://img.shields.io/badge/dynamic/json?query=info.requires_python&label=python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fmistral-common%2Fjson)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENCE)

</div>

## What is it? 

**mistral-common** is a set of tools to help you work with [Mistral AI](https://mistral.ai/) models.

We open-source the tokenizers, validation and normalization code that can be used with our models.

This ensures that you can take full advantage of our models for the following features:

- **tokenization** of text, images and tools calls.
- **validation and normalization** of requests, messages, tool calls, and responses. This is built on top of the [Pydantic](https://docs.pydantic.dev/latest/) library.

We also version our tokenizers to guarantee backward compatibility for the models that we release.

## For who ?

This library is for you if you want to:

- use our models in your own application.
- build your own models and want to use the same tokenization and validation code as we do.

## How to use it ?

You can install the library using pip:
```sh
pip install mistral-common[opencv]
```

For more information, please refer to the [documentation](https://mistralai.github.io/mistral-common/).

## How to contribute ?

We welcome contributions to this library. Please open an issue on our [GitHub repository](https://github.com/mistralai/mistral-common/issues) if you have any questions or suggestions.

All of our features are tested to ensure best usage. But if you encounter a bug, find difficulties in using `mistral-common`. Please also open an issue.

## License

This library is licensed under the Apache 2.0 License. See the [LICENCE](./LICENCE) file for more information.
