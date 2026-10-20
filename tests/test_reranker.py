
from app.rag.reranker import Reranker


def test_reranker_returns_top_results():
    reranker = Reranker()

    documents = [
        {
            "id": "1",
            "source": "kafka.md",
            "content": (
                "Kafka consumers use offsets to track "
                "which records have been processed."
            ),
        },
        {
            "id": "2",
            "source": "kubernetes.md",
            "content": (
                "Kubernetes readiness probes determine "
                "whether a pod can receive traffic."
            ),
        },
        {
            "id": "3",
            "source": "consumer.md",
            "content": (
                "When a Kafka consumer crashes, its "
                "partitions may be reassigned to other "
                "consumers in the group."
            ),
        },
    ]

    results = reranker.rerank(
        "What happens when a Kafka consumer crashes?",
        documents,
        limit=2,
    )

    assert len(results) == 2
    assert all("rerank_score" in item for item in results)
    assert results[0]["id"] == "3"
