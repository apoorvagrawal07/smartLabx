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