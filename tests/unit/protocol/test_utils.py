import uuid
from unittest.mock import patch

from mistral_common.protocol.utils import random_uuid


def test_random_uuid_returns_uuid4_hex() -> None:
    known_uuid = uuid.UUID("12345678-1234-4abc-8def-1234567890ab")
    with patch("mistral_common.protocol.utils.uuid.uuid4", return_value=known_uuid):
        assert random_uuid() == "1234567812344abc8def1234567890ab"
