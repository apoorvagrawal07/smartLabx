from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        limit: int = 5,
    ) -> list[dict]:
        if not documents:
            return []

        pairs = [
            (query, document["content"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = []

        for document, score in zip(documents, scores):
            ranked.append({
                **document,
                "rerank_score": float(score),
            })

        ranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked[:limit]
