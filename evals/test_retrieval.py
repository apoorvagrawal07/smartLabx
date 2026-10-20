import pytest

from app.rag.reranker import Reranker
from app.rag.retrieval_pipeline import RetrievalPipeline
from app.rag.retriever import HybridRetriever


@pytest.fixture
def rag_pipeline():
    retriever = HybridRetriever()
    store = retriever.vector_store
    documents = [
        ("reliability-guide.md", "Retry transient failures with exponential backoff "
         "and jitter. Limit retry attempts and use idempotency keys to avoid "
         "duplicating side effects."),
        ("kafka.md", "Kafka consumers commit offsets to track processed records."),
        ("database.md", "PostgreSQL indexes speed up queries on indexed columns."),
        ("kubernetes.md", "Readiness probes determine whether a pod receives traffic."),
        ("caching.md", "Cache entries expire after their configured time to live."),
        ("authentication.md", "Authentication verifies the identity of a user."),
        ("logging.md", "Request IDs correlate log entries across services."),
    ]

    try:
        store.connection.execute(
            "CREATE TEMP TABLE documents (LIKE public.documents INCLUDING ALL)"
        )
        for source, content in documents:
            store.add_document(
                document_id=source,
                chunk_id=0,
                source=source,
                document_type="markdown",
                title=source,
                section=None,
                content=content,
                embedding=retriever.embedding_service.embed(content),
                embedding_model="all-MiniLM-L6-v2",
                metadata={"evaluation_fixture": True},
            )

        yield RetrievalPipeline(retriever, Reranker())
    finally:
        store.connection.close()


def reciprocal_rank(retrieved_sources: list[str], expected_source: str) -> float:
    """Return reciprocal rank of the first matching source."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1.0 / rank

    return 0.0


def test_expected_document_is_retrieved(rag_pipeline):
    query = "What does our internal document say about retries?"
    expected_source = "reliability-guide.md"

    chunks = rag_pipeline.retrieve(
        query,
        candidate_count=20,
        final_count=5,
    )

    retrieved_sources = [
        chunk.get("source", "")
        for chunk in chunks
    ]

    assert expected_source in retrieved_sources
    assert len(chunks) == 5
    assert all(chunk["id"] and "rrf_score" in chunk and "rerank_score" in chunk for chunk in chunks)


def test_reciprocal_rank():
    assert reciprocal_rank(
        ["other.md", "reliability-guide.md"],
        "reliability-guide.md",
    ) == 0.5

    assert reciprocal_rank(
        ["other.md"],
        "reliability-guide.md",
    ) == 0.0
