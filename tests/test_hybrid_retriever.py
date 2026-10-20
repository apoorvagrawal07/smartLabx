
from unittest.mock import patch

from app.rag.retriever import HybridRetriever


def test_rrf_merges_results():
    merged = {}

    def add_results(results):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            if chunk_id not in merged:
                merged[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                }

            merged[chunk_id]["rrf_score"] += (
                1 / (60 + rank)
            )

    add_results([
        {"id": "A", "content": "consumer recovery"},
        {"id": "B", "content": "offset management"},
    ])

    add_results([
        {"id": "B", "content": "offset management"},
        {"id": "C", "content": "partition reassignment"},
    ])

    ranked = sorted(
        merged.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    assert ranked[0]["id"] == "B"
    assert len(ranked) == 3
    assert ranked[0]["rrf_score"] > ranked[1]["rrf_score"]


def test_hybrid_retriever_merges_real_search_paths_by_chunk_id():
    with (
        patch("app.rag.retriever.EmbeddingService") as embedding_type,
        patch("app.rag.retriever.VectorStore") as store_type,
    ):
        retriever = HybridRetriever()
    embedding_type.return_value.embed.return_value = [0.1, 0.2]
    shared = {"id": "shared", "source": "kafka.md", "content": "recovery"}
    store = store_type.return_value
    store.vector_search.return_value = [
        {"id": "vector-only", "source": "vector.md", "content": "semantic match"}, shared,
    ]
    store.keyword_search.return_value = [shared, {"id": "keyword-only", "source": "exact.md", "content": "exact match"}]

    results = retriever.retrieve("consumer recovery", limit=3)

    assert [result["id"] for result in results] == ["shared", "vector-only", "keyword-only"]
    assert results[0]["rrf_score"] == 1 / 62 + 1 / 61
    assert results[0]["source"] == "kafka.md"
    store.vector_search.assert_called_once_with([0.1, 0.2], limit=3)
    store.keyword_search.assert_called_once_with("consumer recovery", limit=3)


def test_hybrid_retriever_handles_empty_results_and_zero_limit():
    with patch("app.rag.retriever.EmbeddingService"), patch("app.rag.retriever.VectorStore"):
        retriever = HybridRetriever()
    assert retriever.retrieve("query", limit=0) == []
    retriever.embedding_service.embed.assert_not_called()
    retriever.vector_store.vector_search.return_value = []
    retriever.vector_store.keyword_search.return_value = []
    assert retriever.retrieve("query") == []


def test_hybrid_retriever_keeps_keyword_only_matches():
    with patch("app.rag.retriever.EmbeddingService"), patch("app.rag.retriever.VectorStore"):
        retriever = HybridRetriever()
    retriever.vector_store.vector_search.return_value = []
    retriever.vector_store.keyword_search.return_value = [
        {"id": "exact", "source": "codes.md", "content": "ERR_471"},
    ]
    assert retriever.retrieve("ERR_471", limit=1)[0]["id"] == "exact"
