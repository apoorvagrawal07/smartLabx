import hashlib
import uuid


def generate_document_id(content: str) -> uuid.UUID:
    digest = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return uuid.UUID(digest[:32])