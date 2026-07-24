import numpy as np
import pytest

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils import registry
from tests.utils.requests.fim import fim_request
from tests.utils.versions import ALL_VERSIONS

_GOLDEN_SCENARIOS = [s for s in registry.SCENARIOS if s.raises is None]
_REFUSAL_SCENARIOS = [s for s in registry.SCENARIOS if s.raises is not None]
_INSTRUCT_SCENARIOS = [s for s in _GOLDEN_SCENARIOS if s.protocol == "instruct"]
_INSTRUCT_IMAGE_SCENARIOS = [s for s in _INSTRUCT_SCENARIOS if s.has_images]
_FIM_SCENARIOS = [s for s in _GOLDEN_SCENARIOS if s.protocol == "fim"]

# Every (version, protocol) pair except those in `SILENT_UNSUPPORTED_PROTOCOLS`, which are
# unsupported without raising and therefore cannot be honestly covered by either a golden
# or a refusal scenario.
_COVERAGE_PAIRS: tuple[tuple[TokenizerVersion, str], ...] = tuple(
    (version, protocol)
    for version in ALL_VERSIONS
    for protocol in registry.PROTOCOLS
    if (version, protocol) not in registry.SILENT_UNSUPPORTED_PROTOCOLS
)


def _scenario_id(scenario: registry.Scenario) -> str:
    return f"{scenario.key}-{scenario.name}"


def _scenario_case(scenarios: list[registry.Scenario]) -> pytest.MarkDecorator:
    return pytest.mark.parametrize(
        ("scenario", "keyed_mistral_tokenizer"),
        [(scenario, scenario.key) for scenario in scenarios],
        ids=[_scenario_id(scenario) for scenario in scenarios],
        indirect=["keyed_mistral_tokenizer"],
    )


def _request_case(scenarios: list[registry.Scenario]) -> pytest.MarkDecorator:
    return pytest.mark.parametrize(
        "scenario",
        scenarios,
        ids=[_scenario_id(scenario) for scenario in scenarios],
    )


_instruct_case = _scenario_case(_INSTRUCT_SCENARIOS)
_instruct_image_case = _scenario_case(_INSTRUCT_IMAGE_SCENARIOS)
_fim_case = _scenario_case(_FIM_SCENARIOS)
_instruct_request_case = _request_case(_INSTRUCT_SCENARIOS)
_fim_request_case = _request_case(_FIM_SCENARIOS)


class TestInstructGoldens:
    @_instruct_request_case
    def test_request_matches_golden(
        self,
        scenario: registry.Scenario,
        instruct_request_goldens: dict[str, dict[str, dict[str, object]]],
    ) -> None:
        assert (
            registry.serialize_request(scenario.build_request())
            == instruct_request_goldens[scenario.key][scenario.name]
        )

    @_instruct_case
    def test_token_ids_match_golden(
        self,
        scenario: registry.Scenario,
        keyed_mistral_tokenizer: MistralTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        encoded = registry.PROTOCOL_ENCODERS[scenario.protocol](keyed_mistral_tokenizer, scenario.build_request())
        assert encoded.tokens == instruct_token_id_goldens[scenario.key][scenario.name]

    @_instruct_case
    def test_decoded_text_match_golden(
        self,
        scenario: registry.Scenario,
        keyed_mistral_tokenizer: MistralTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        encoded = registry.PROTOCOL_ENCODERS[scenario.protocol](keyed_mistral_tokenizer, scenario.build_request())
        assert encoded.text == instruct_decoded_goldens[scenario.key][scenario.name]

    @_instruct_image_case
    def test_image_arrays_match_golden(
        self, scenario: registry.Scenario, keyed_mistral_tokenizer: MistralTokenizer
    ) -> None:
        encoded = registry.PROTOCOL_ENCODERS[scenario.protocol](keyed_mistral_tokenizer, scenario.build_request())
        golden = registry.load_image_arrays(scenario.protocol, scenario.key, scenario.name)
        assert len(encoded.images) == len(golden)
        for produced, expected in zip(encoded.images, golden):
            np.testing.assert_allclose(produced, expected, rtol=registry.RTOL, atol=registry.ATOL)


class TestFimGoldens:
    @_fim_request_case
    def test_request_matches_golden(
        self,
        scenario: registry.Scenario,
        fim_request_goldens: dict[str, dict[str, dict[str, object]]],
    ) -> None:
        assert registry.serialize_request(scenario.build_request()) == fim_request_goldens[scenario.key][scenario.name]

    @_fim_case
    def test_token_ids_match_golden(
        self,
        scenario: registry.Scenario,
        keyed_mistral_tokenizer: MistralTokenizer,
        fim_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        encoded = registry.PROTOCOL_ENCODERS[scenario.protocol](keyed_mistral_tokenizer, scenario.build_request())
        assert encoded.tokens == fim_token_id_goldens[scenario.key][scenario.name]

    @_fim_case
    def test_decoded_text_match_golden(
        self,
        scenario: registry.Scenario,
        keyed_mistral_tokenizer: MistralTokenizer,
        fim_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        encoded = registry.PROTOCOL_ENCODERS[scenario.protocol](keyed_mistral_tokenizer, scenario.build_request())
        assert encoded.text == fim_decoded_goldens[scenario.key][scenario.name]


class TestRefusalScenarios:
    @pytest.mark.parametrize(
        "scenario",
        _REFUSAL_SCENARIOS,
        ids=[f"{s.protocol}-{s.key}-{s.name}" for s in _REFUSAL_SCENARIOS],
    )
    def test_encode_scenario_raises(self, scenario: registry.Scenario) -> None:
        assert scenario.raises is not None
        with pytest.raises(scenario.raises, match=scenario.raises_match):
            registry.encode_scenario(scenario)


class TestProtocolCoverage:
    @pytest.mark.parametrize(
        ("version", "protocol"),
        _COVERAGE_PAIRS,
        ids=[f"{version.value}-{protocol}" for version, protocol in _COVERAGE_PAIRS],
    )
    def test_version_protocol_has_scenario(self, version: TokenizerVersion, protocol: str) -> None:
        matching = [
            scenario
            for scenario in registry.SCENARIOS
            if scenario.protocol == protocol and registry.KEY_VERSIONS[scenario.key] == version
        ]
        if protocol in registry.SUPPORTED_PROTOCOLS[version]:
            if protocol == "fim" and version != registry.FIM_GOLDEN_VERSION:
                # FIM logic is frozen in `InstructTokenizerV2` and inherited unchanged, so later
                # versions are covered by `TestFimVersionSmoke` rather than by redundant goldens.
                assert any(registry.KEY_VERSIONS[key] == version for key in registry.FIM_SMOKE_KEYS), (
                    f"No FIM smoke key covers {version}"
                )
                return
            assert any(scenario.raises is None for scenario in matching), (
                f"No successful {protocol} scenario covers {version}"
            )
        else:
            assert any(scenario.raises is not None for scenario in matching), (
                f"No refusal {protocol} scenario covers {version}"
            )

    @pytest.mark.parametrize("key", sorted(registry.KEY_VERSIONS))
    def test_key_has_at_least_one_scenario(self, key: str) -> None:
        assert any(scenario.key == key for scenario in registry.SCENARIOS), (
            f"Key {key!r} has no golden or refusal scenario covering it"
        )


class TestFimVersionSmoke:
    @pytest.mark.parametrize("key", registry.FIM_SMOKE_KEYS)
    def test_encode_fim_still_supported(self, key: str) -> None:
        encoded = registry.load_mistral_tokenizer(key).encode_fim(fim_request())
        assert encoded.text.startswith("<s>[SUFFIX]")
        assert encoded.tokens
