from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.guidance.grammar_factory import GrammarFactory
from mistral_common.protocol.instruct.request import ToolChoiceEnum
from tests.guidance.test_guidance import _build_tekken_mistral_tokenizer

def test_grammar_factory_empty_schema_serialization():
    tokenizer = _build_tekken_mistral_tokenizer(TokenizerVersion.v11)
    factory = GrammarFactory(tokenizer)

    # Verifichiamo che uno schema JSON vuoto {} venga trattato correttamente da GrammarFactory
    grammar = factory.get_lark_from_jinja(
        template=factory.select_jinja_template(),
        mode=ToolChoiceEnum.auto,
        tools=None,
        json_schema={},
        parallel_tool_calls=False,
    )
    assert grammar is not None
