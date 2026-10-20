from app.rag.embedding_service import (
    EmbeddingService
)

from app.rag.vector_store import (
    VectorStore
)


class Retriever:

    def __init__(self):

        self.embedding_service = (
            EmbeddingService()
        )

        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        limit: int = 3
    ):

        query_embedding = (
            self.embedding_service
            .embed(query)
        )

        return self.vector_store.search(
            query_embedding,
            limit
        )


class HybridRetriever(Retriever):
    def retrieve(self, query: str, limit: int = 20) -> list[dict]:
        if limit <= 0:
            return []
        embedding = self.embedding_service.embed(query)
        vector_results = self.vector_store.vector_search(embedding, limit=limit)
        keyword_results = self.vector_store.keyword_search(query, limit=limit)
        merged = {}
        for results in (vector_results, keyword_results):
            for rank, document in enumerate(results, start=1):
                chunk_id = document["id"]
                if chunk_id not in merged:
                    merged[chunk_id] = {**document, "rrf_score": 0.0}
                merged[chunk_id]["rrf_score"] += 1 / (60 + rank)
        return sorted(
            merged.values(), key=lambda document: (-document["rrf_score"], document["id"])
        )[:limit]