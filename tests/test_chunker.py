
from app.rag.chunker import chunk_text


def test_chunker_preserves_headings():
    document = """
# Kafka

Kafka is a distributed event streaming platform.

## Consumer Failure

When a consumer crashes, Kafka reassigns its
partitions to another consumer in the group.

## Offset Management

Consumers commit offsets so they can resume
processing after a restart.
"""

    chunks = chunk_text(
        document,
        chunk_size=30,
        overlap=5,
    )

    assert chunks
    assert any(
        "Consumer Failure" in chunk
        for chunk in chunks
    )
    assert any(
        "Offset Management" in chunk
        for chunk in chunks
    )


def test_empty_document():
    assert chunk_text("") == []
