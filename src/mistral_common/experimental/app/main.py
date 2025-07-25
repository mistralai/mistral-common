from functools import lru_cache
from pathlib import Path
from typing import Union

import click
import uvicorn
from fastapi import FastAPI

from mistral_common.experimental.app.models import (
    GenerationBackend,
    Settings,
    get_settings,
)
from mistral_common.experimental.app.routers import (
    decode_router,
    main_router,
    tokenize_router,
)
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def create_app(
    tokenizer: Union[str, Path, MistralTokenizer],
    validation_mode: ValidationMode = ValidationMode.test,
    generation_host: str = "127.0.0.1",
    generation_port: int = 8080,
    generation_backend: GenerationBackend = GenerationBackend.llama_cpp,
    api_key: str = "",
    timeout: int = 60,
) -> FastAPI:
    r"""Create a Mistral-common FastAPI app with the given tokenizer and validation mode.

    Args:
        tokenizer: The tokenizer path or a MistralTokenizer instance.
        validation_mode: The validation mode to use.
        generation_host: The host of the generation API.
        generation_port: The port of the generation API.
        generation_backend: The backend of the generation API.
        api_key: The API key of the generation API.
        timeout: The timeout of the generation API.

    Returns:
        The Mistral-common FastAPI app.
    """
    if not isinstance(tokenizer, (MistralTokenizer, str, Path)):
        raise ValueError("Tokenizer must be a path or a MistralTokenizer instance.")

    app = FastAPI()
    app.include_router(tokenize_router)
    app.include_router(decode_router)
    app.include_router(main_router)

    @lru_cache
    def get_settings_override() -> Settings:
        settings = Settings(
            generation_host=generation_host,
            generation_port=generation_port,
            generation_backend=generation_backend,
            api_key=api_key,
            timeout=timeout,
        )
        if isinstance(tokenizer, MistralTokenizer):
            settings.tokenizer = tokenizer
        else:
            settings._load_tokenizer(tokenizer, validation_mode)
        return settings

    get_settings_override()
    app.dependency_overrides[get_settings] = get_settings_override


    return app


@click.group()
def cli() -> None:
    r"""Mistral-common CLI."""
    pass


@cli.command(name="serve", context_settings={"auto_envvar_prefix": "UVICORN"})
@click.argument("tokenizer_path", type=str)
@click.argument(
    "validation_mode",
    type=click.Choice([mode.value for mode in ValidationMode], case_sensitive=False),
    default=ValidationMode.test.value,
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Mistral-common API host",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=0,
    help="Mistral-common API port",
    show_default=True,
)
@click.option(
    "--generation-host",
    type=str,
    default="127.0.0.1",
    help="Generation API host",
    show_default=True,
)
@click.option(
    "--generation-port",
    type=int,
    default=8080,
    help="Generation API port",
    show_default=True,
)
@click.option(
    "--generation-backend",
    type=click.Choice([mode.value for mode in GenerationBackend], case_sensitive=False),
    default=GenerationBackend.llama_cpp.value,
    help="Generation API backend",
    show_default=True,
)
@click.option(
    "--api-key",
    type=str,
    default="",
    help="Generation API key",
    show_default=True,
)
@click.option(
    "--timeout",
    type=int,
    default=60,
    help="Generation API key",
    show_default=True,
)
def serve(
    tokenizer_path: Union[str, Path],
    validation_mode: Union[ValidationMode, str] = ValidationMode.test,
    host: str = "127.0.0.1",
    port: int = 0,
    generation_host: str = "127.0.0.1",
    generation_port: int = 8080,
    generation_backend: str = GenerationBackend.llama_cpp.value,
    api_key: str = "",
    timeout: int = 60,
) -> None:
    r"""Serve the Mistral-common API with the given tokenizer path and validation mode."""
    app = create_app(
        tokenizer=tokenizer_path,
        validation_mode=ValidationMode(validation_mode),
        generation_host=generation_host,
        generation_port=generation_port,
        generation_backend=GenerationBackend(generation_backend),
        api_key=api_key,
        timeout=timeout,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
