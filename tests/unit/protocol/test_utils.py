import uuid
from unittest.mock import patch

from mistral_common.protocol.utils import random_uuid


def test_random_uuid_returns_distinct_uuid_hex_values_in_order() -> None:
    generated_uuids = [
        uuid.UUID("00112233-4455-6677-8899-aabbccddeeff"),
        uuid.UUID("ffeeddcc-bbaa-9988-7766-554433221100"),
    ]

    with patch("mistral_common.protocol.utils.uuid.uuid4", side_effect=generated_uuids) as mock_uuid4:
        results = [random_uuid(), random_uuid()]

    assert results == ["00112233445566778899aabbccddeeff", "ffeeddccbbaa99887766554433221100"]
    assert mock_uuid4.call_count == 2
