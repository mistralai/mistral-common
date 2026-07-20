from pathlib import Path

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.integrations.chat_templates.chat_templates import (
    convert_tokenizer_to_chat_template,
    generate_chat_template,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, _build_spm_path, _build_tekken_json


class TestConvertTokenizerToChatTemplate:
    def test_tekken_v13_thinking(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v13, think=True)
        path = _build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(tokenizer_file=path)
        expected = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        assert result == expected
        assert "[THINK]" in result

    def test_tekken_v3_image(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v3, image=True)
        path = _build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(tokenizer_file=path)
        expected = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=True,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        assert result == expected

    def test_tekken_v7_audio(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v7, audio=True)
        path = _build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(tokenizer_file=path)
        expected = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=True,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        assert result == expected

    def test_tekken_v11_plain_thinking(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v11)
        path = _build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(tokenizer_file=path)
        expected = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )
        assert result == expected
        assert "<think>" in result
        assert "[THINK]" not in result

    def test_spm_v7(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v7, spm=True)
        path = _build_spm_path(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(tokenizer_file=path)
        expected = generate_chat_template(
            spm=True,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        assert result == expected

    def test_system_prompt_embedded(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v7)
        path = _build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(
            tokenizer_file=path,
            system_prompt="You are helpful.",
        )
        assert "You are helpful." in result

    def test_use_special_token_variables_true(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v7)
        path = _build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(
            tokenizer_file=path,
            use_special_token_variables=True,
        )
        assert "bos_token" in result
        assert "'<s>'" not in result

    def test_use_special_token_variables_false(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v7)
        path = _build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(
            tokenizer_file=path,
            use_special_token_variables=False,
        )
        assert "'<s>'" in result
        assert "bos_token" not in result

    def test_unrecognized_file_raises_tokenizer_exception(self, tmp_path: Path) -> None:
        invalid_path = tmp_path / "tokenizer.txt"
        invalid_path.write_text("not a tokenizer")
        with pytest.raises(TokenizerException):
            convert_tokenizer_to_chat_template(tokenizer_file=invalid_path)
