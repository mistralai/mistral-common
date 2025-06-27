import unittest
import multiprocessing

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def _worker_decode_function(tokenizer_instance: MistralTokenizer) -> str:
    """
    A simple worker that receives a tokenizer, performs a decode operation,
    and returns the result.

    Example tokens were taken from `tests/test_tokenize_v1.py.test_system_multiturn`.
    """
    tokens_to_decode = [1, 733, 16289, 28793, 17121, 22526, 13, 13, 28708, 733, 28748, 16289, 28793]
    return tokenizer_instance.decode(tokens_to_decode)


class TestTokenizerPickling(unittest.TestCase):
    """
    Test suite to ensure MistralTokenizer can be pickled and used across
    different processes, which is essential for multiprocessing.
    """

    def test_tokenizer_is_pickleable_with_multiprocessing(self) -> None:
        tokenizer_path = str(MistralTokenizer._data_path() / "tokenizer.model.v1")
        tokenizer = MistralTokenizer.from_file(tokenizer_path)

        with multiprocessing.Pool(processes=2) as pool:
            results = pool.map(_worker_decode_function, [tokenizer])

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], str)
        self.assertEqual(results[0], "[INST] SYSTEM\n\na [/INST]")
