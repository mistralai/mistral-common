import uuid


def random_uuid() -> str:
    r"""Generate a random UUID as a 32-character hex string."""
    return str(uuid.uuid4().hex)
