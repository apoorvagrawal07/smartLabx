from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed(self, text: str) -> list[float]:

        vector = self.model.encode(text)

        return vector.tolist()