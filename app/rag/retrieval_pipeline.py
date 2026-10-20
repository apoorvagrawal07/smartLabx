
class RetrievalPipeline:
    def __init__(self, hybrid_retriever, reranker):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        candidate_count: int = 20,
        final_count: int = 5,
    ) -> list[dict]:
        candidates = self.hybrid_retriever.retrieve(
            query,
            limit=candidate_count,
        )

        return self.reranker.rerank(
            query,
            candidates,
            limit=final_count,
        )
