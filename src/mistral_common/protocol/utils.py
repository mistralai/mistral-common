import uuid


def random_uuid() -> str:
    """Generate a random UUID."""
    return str(uuid.uuid4().hex)
