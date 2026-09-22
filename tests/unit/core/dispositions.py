SOURCE_DISPOSITIONS = {
    "src/mistral_common/base.py": [
        "MistralBase",
        "MistralBase._filter_cls_fields",
        "MistralBase.model_validate_ignore_extra",
    ],
    "src/mistral_common/deprecation.py": ["deprecated_import", "warn_once"],
    "src/mistral_common/imports.py": [
        "_get_dependency_error_message",
        "is_package_installed",
        "assert_package_installed",
        "is_hf_hub_installed",
        "is_jinja2_installed",
        "is_llguidance_installed",
        "is_opencv_installed",
        "is_sentencepiece_installed",
        "is_soundfile_installed",
        "is_soxr_installed",
        "assert_hf_hub_installed",
        "assert_jinja2_installed",
        "assert_llguidance_installed",
        "assert_opencv_installed",
        "assert_sentencepiece_installed",
        "assert_soundfile_installed",
        "assert_soxr_installed",
    ],
    "src/mistral_common/exceptions.py": [
        "MistralCommonException",
        "MistralCommonException.__init__",
        "TokenizerException",
        "TokenizerException.__init__",
        "UnsupportedTokenizerFeatureException",
        "UnsupportedTokenizerFeatureException.__init__",
        "InvalidRequestException",
        "InvalidRequestException.__init__",
        "InvalidSystemPromptException",
        "InvalidSystemPromptException.__init__",
        "InvalidMessageStructureException",
        "InvalidMessageStructureException.__init__",
        "InvalidAssistantMessageException",
        "InvalidAssistantMessageException.__init__",
        "InvalidToolMessageException",
        "InvalidToolMessageException.__init__",
        "InvalidToolSchemaException",
        "InvalidToolSchemaException.__init__",
        "InvalidUserMessageException",
        "InvalidUserMessageException.__init__",
        "InvalidFunctionCallException",
        "InvalidFunctionCallException.__init__",
        "InvalidToolException",
        "InvalidToolException.__init__",
    ],
    "src/mistral_common/protocol/utils.py": ["random_uuid"],
}

LEGACY_DISPOSITIONS = {
    "tests/test_base.py": "migrated to tests/unit/core/test_base.py",
    "tests/test_deprecation.py": "migrated to tests/unit/core/test_deprecation.py",
    "tests/test_imports.py": "migrated to tests/unit/core/test_imports.py",
}

LEGACY_NODE_DISPOSITIONS = {
    "tests/test_base.py": [
        "test_filter_cls_fields",
        "test_model_validate_ignore_extra_filters_and_validates",
        "test_model_validate_ignore_extra_no_extra_keys",
        "test_model_validate_ignore_extra_raises_on_missing_required",
    ],
    "tests/test_deprecation.py": [
        "test_deprecated_import_returns_correct_object",
        "test_deprecated_import_emits_deprecation_warning",
        "test_deprecated_import_warns_only_once",
        "test_deprecated_import_different_pairs_each_warn",
        "test_deprecated_import_raises_attribute_error",
        "test_deprecated_import_raises_module_not_found_error",
        "test_warn_once_emits_warning",
        "test_warn_once_does_not_repeat",
        "test_warn_once_different_keys_each_warn",
    ],
    "tests/test_imports.py": [
        "test_is_package_installed",
        "test_assert_package_installed",
        "test_is_opencv_installed",
        "test_is_installed[is_hf_hub_installed]",
        "test_is_installed[is_jinja2_installed]",
        "test_is_installed[is_llguidance_installed]",
        "test_is_installed[is_sentencepiece_installed]",
        "test_is_installed[is_soundfile_installed]",
        "test_is_installed[is_soxr_installed]",
        "test_assert_installed[is_hf_hub_installed-assert_hf_hub_installed-`huggingface_hub` is not installed. "
        "Please install it with `pip install mistral-common[hf-hub]`]",
        "test_assert_installed[is_jinja2_installed-assert_jinja2_installed-`jinja2` is not installed. "
        "Please install it with `pip install mistral-common[guidance]`]",
        "test_assert_installed[is_llguidance_installed-assert_llguidance_installed-`llguidance` is not installed. "
        "Please install it with `pip install mistral-common[guidance]`]",
        "test_assert_installed[is_opencv_installed-assert_opencv_installed-`opencv` is not installed. "
        "Please install it with `pip install mistral-common[opencv]`]",
        "test_assert_installed[is_sentencepiece_installed-assert_sentencepiece_installed-`sentencepiece` "
        "is not installed. "
        "Please install it with `pip install mistral-common[sentencepiece]`]",
        "test_assert_installed[is_soundfile_installed-assert_soundfile_installed-`soundfile` is not installed. "
        "Please install it with `pip install mistral-common[soundfile]`]",
        "test_assert_installed[is_soxr_installed-assert_soxr_installed-`soxr` is not installed. "
        "Please install it with `pip install mistral-common[soxr]`]",
    ],
}
