import re

from mistral_common.protocol.utils import random_uuid


def test_random_uuid_is_a_new_32_character_hex_value() -> None:
    first = random_uuid()
    second = random_uuid()
    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert re.fullmatch(r"[0-9a-f]{32}", second)
    assert first != second
