import uuid


def random_uuid() -> str:
    return str(uuid.uuid4().hex)
