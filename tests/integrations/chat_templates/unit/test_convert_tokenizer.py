import json
from pathlib import Path

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.integrations.chat_templates.chat_templates import (
    convert_tokenizer_to_chat_template,
    generate_chat_template,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, build_spm_path, build_tekken_json


class TestConvertTokenizerToChatTemplate:
    def test_tekken_v13_thinking(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v13, think=True)
        path = build_tekken_json(config=config, output_dir=tmp_path)
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

    def test_tekken_v15_derives_reasoning_effort_only(self, tmp_path: Path) -> None:
        path = build_tekken_json(config=TestConfig(version=TokenizerVersion.v15), output_dir=tmp_path)

        result = convert_tokenizer_to_chat_template(tokenizer_file=path)
        expected = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=frozenset({"reasoning_effort"}),
        )

        assert result == expected

    def test_tekken_v15_json_schema_builder_derives_both_fields(self, tmp_path: Path) -> None:
        path = build_tekken_json(
            config=TestConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
            ),
            output_dir=tmp_path,
        )

        result = convert_tokenizer_to_chat_template(tokenizer_file=path)
        expected = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
        )

        assert result == expected

    def test_tekken_v15_without_model_settings_builder_raises(self, tmp_path: Path) -> None:
        path = build_tekken_json(config=TestConfig(version=TokenizerVersion.v15), output_dir=tmp_path)
        tekken_data = json.loads(path.read_text(encoding="utf-8"))
        del tekken_data["model_settings_builder"]
        path.write_text(json.dumps(obj=tekken_data, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ValueError, match="Model-settings-capable tokenizer has no model settings builder"):
            convert_tokenizer_to_chat_template(tokenizer_file=path)

    def test_tekken_v15_builder_without_configured_fields_raises(self, tmp_path: Path) -> None:
        path = build_tekken_json(
            config=TestConfig(version=TokenizerVersion.v15, model_settings_fields=frozenset()),
            output_dir=tmp_path,
        )

        with pytest.raises(
            ValueError,
            match="Model settings builder must configure at least one field",
        ):
            convert_tokenizer_to_chat_template(tokenizer_file=path)

    def test_tekken_v3_image(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v3, image=True)
        path = build_tekken_json(config=config, output_dir=tmp_path)
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
        path = build_tekken_json(config=config, output_dir=tmp_path)
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
        path = build_tekken_json(config=config, output_dir=tmp_path)
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

    def test_tekken_v11_plain_thinking_override_false(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v11)
        path = build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(tokenizer_file=path, plain_thinking_support=False)
        expected = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        assert result == expected
        assert "<think>" not in result
        assert "[THINK]" not in result

    def test_tekken_v11_plain_thinking_override_true_matches_autodetect(self, tmp_path: Path) -> None:
        # Forcing `True` cannot widen what is producible: `TemplateConfig` only accepts plain
        # thinking where the heuristic already returns `True`, so it restates the default.
        config = TestConfig(version=TokenizerVersion.v11)
        path = build_tekken_json(config=config, output_dir=tmp_path)
        forced = convert_tokenizer_to_chat_template(tokenizer_file=path, plain_thinking_support=True)
        autodetected = convert_tokenizer_to_chat_template(tokenizer_file=path)
        assert forced == autodetected

    def test_tekken_v11_plain_thinking_override_true_with_audio_raises(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v11, audio=True)
        path = build_tekken_json(config=config, output_dir=tmp_path)
        with pytest.raises(ValueError, match="Audio and plain thinking support are mutually exclusive"):
            convert_tokenizer_to_chat_template(tokenizer_file=path, plain_thinking_support=True)

    def test_spm_v7(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v7, spm=True)
        path = build_spm_path(config=config, output_dir=tmp_path)
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
        path = build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(
            tokenizer_file=path,
            system_prompt="You are helpful.",
        )
        assert "You are helpful." in result

    def test_use_special_token_variables_true(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v7)
        path = build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(
            tokenizer_file=path,
            use_special_token_variables=True,
        )
        assert "bos_token" in result
        assert "'<s>'" not in result

    def test_use_special_token_variables_false(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v7)
        path = build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(
            tokenizer_file=path,
            use_special_token_variables=False,
        )
        assert "'<s>'" in result
        assert "bos_token" not in result

    def test_tekken_v11_audio_no_plain_thinking(self, tmp_path: Path) -> None:
        config = TestConfig(version=TokenizerVersion.v11, audio=True)
        path = build_tekken_json(config=config, output_dir=tmp_path)
        result = convert_tokenizer_to_chat_template(tokenizer_file=path)
        expected = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=True,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        assert result == expected
        assert "[AUDIO]" in result
        assert "<think>" not in result

    def test_unrecognized_file_raises_tokenizer_exception(self, tmp_path: Path) -> None:
        invalid_path = tmp_path / "tokenizer.txt"
        invalid_path.write_text("not a tokenizer")
        with pytest.raises(TokenizerException, match="Unrecognized tokenizer file:"):
            convert_tokenizer_to_chat_template(tokenizer_file=invalid_path)
