# Introduction

## What is it? 

**mistral-common** is a set of tools to help you work with Mistral models.

We open-source the tokenizers, validation and normalization code that can be used with our models.

This ensures that you can take full advantage of our models for the following features:

- **tokenization** of text, images and tools calls.
- **validation and normalization** of requests, messages, tool calls, and responses. This is built on top of the [Pydantic](https://docs.pydantic.dev/latest/) library.

We also version our tokenizers to guarantee backward compatibility for the models that we release.

## For who ?

This library is for you if you want to:

- use our models in your own application.
- build your own models and want to use the same tokenization and validation code as we do.

## Table of contents

Explore the following sections:

- [Quickstart](./examples/inference.md) for simple code snippets to use the library for different tasks.
- [Usage](./usage/index.md) section to install and use the library.
- [Code Reference][mistral_common.base] section to see the code documentation.

## Launch the documentation locally

To launch the documentation locally, simply run the following command at the root of the repository:
```sh
mkdocs serve
```
