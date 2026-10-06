import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormats,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import (
    JSONSchemaBuilder,
    ModelSettingsBuilder,
    ReasoningEffortEnumBuilder,
)
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.utils import decode_keep

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"value": {"type": "string"}}}
IMAGE_CONFIG = {"image_patch_size": 14, "max_image_size": 1540, "spatial_merge_size": 2}


def _response_format_builder() -> ModelSettingsBuilder:
    return ModelSettingsBuilder(
        reasoning_effort=ReasoningEffortEnumBuilder(values=[ReasoningEffort.high], accepts_none=False, default=None),
        json_schema=JSONSchemaBuilder(accepts_none=False, default=None),
    )


def get_v15_tekken_mm_tokenizer(tmp_path: Path, builder: ModelSettingsBuilder) -> MistralTokenizer:
    r"""Load a synthetic V15 fixture for loader and builder-shape checks."""
    tokenizer_path = tmp_path / "synthetic_tekken_mm.json"
    fixture: dict[str, Any] = {
        "vocab": quick_vocab(),
        "config": {
            "pattern": r".+",
            "default_num_special_tokens": 100,
            "default_vocab_size": 356,
            "version": "v15",
        },
        "special_tokens": get_special_tokens(TokenizerVersion.v15, add_think=True),
        "version": 1,
        "type": "Tekken",
        "image": IMAGE_CONFIG,
        "model_settings_builder": builder.model_dump(mode="json"),
    }
    with tokenizer_path.open("w", encoding="utf-8") as fixture_file:
        json.dump(fixture, fixture_file, ensure_ascii=False)

    return MistralTokenizer.from_file(tokenizer_filename=tokenizer_path)


def test_v15_tekken_mm_from_file_loads_response_format_builder(tmp_path: Path) -> None:
    r"""Check loader shape using synthetic data, not released vocabulary."""
    tokenizer = get_v15_tekken_mm_tokenizer(tmp_path=tmp_path, builder=_response_format_builder())

    with (tmp_path / "synthetic_tekken_mm.json").open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)

    assert fixture["model_settings_builder"]["json_schema"] == {
        "type": "json_schema",
        "accepts_none": False,
        "default": None,
    }
    assert tokenizer.version == TokenizerVersion.v15
    model_settings_builder = tokenizer.instruct_tokenizer.tokenizer.model_settings_builder
    assert model_settings_builder is not None
    assert model_settings_builder.json_schema is not None
    image_encoder = tokenizer.instruct_tokenizer.image_encoder
    assert image_encoder is not None
    assert asdict(image_encoder.image_config) == IMAGE_CONFIG


@pytest.mark.parametrize(
    ("response_format", "settings_schema"),
    [
        pytest.param(
            ResponseFormat(type=ResponseFormats.json),
            {"anyOf": [{"type": "object"}, {"type": "array"}]},
            id="json",
        ),
        pytest.param(
            ResponseFormat(
                type=ResponseFormats.json_schema,
                json_schema=JsonSchema(name="synthetic", schema=SCHEMA, strict=False),
            ),
            SCHEMA,
            id="json-schema-non-strict",
        ),
        pytest.param(
            ResponseFormat(
                type=ResponseFormats.json_schema,
                json_schema=JsonSchema(name="synthetic", schema=SCHEMA, strict=True),
            ),
            SCHEMA,
            id="json-schema-strict",
        ),
    ],
)
def test_v15_tekken_mm_encodes_json_schema_settings(
    tmp_path: Path,
    response_format: ResponseFormat,
    settings_schema: dict[str, Any],
) -> None:
    r"""Encode schema settings with a synthetic fixture, not released vocabulary."""
    tokenizer = get_v15_tekken_mm_tokenizer(tmp_path=tmp_path, builder=_response_format_builder())
    request = ChatCompletionRequest(
        messages=[UserMessage(content="hi")],
        reasoning_effort=ReasoningEffort.high,
        response_format=response_format,
    )

    assert response_format.get_schema() == settings_schema

    encoded = tokenizer.encode_chat_completion(request=request)
    text = decode_keep(tokenizer=tokenizer, tokenized=encoded)
    expected = (
        f'<s>[MODEL_SETTINGS]{{"json_schema": {json.dumps(settings_schema, ensure_ascii=False)}, '
        f'"reasoning_effort": "high"}}[/MODEL_SETTINGS][INST]hi[/INST]'
    )
    assert text == expected
