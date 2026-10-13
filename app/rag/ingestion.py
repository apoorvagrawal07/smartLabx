from pathlib import Path

from app.rag.chunker import chunk_text
from app.rag.document_id import generate_document_id
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStore


class IngestionPipeline:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def ingest_file(self, path: str):

        text = Path(path).read_text(
            encoding="utf-8"
        )

        document_id = generate_document_id(text)

        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):

            embedding = self.embedding_service.embed(
                chunk
            )

            self.vector_store.add_document(
                document_id=document_id,
                chunk_id=chunk_id,
                source=path,
                document_type="markdown",
                title=Path(path).stem,
                section=None,
                content=chunk,
                embedding=embedding,
                embedding_model="all-MiniLM-L6-v2",
                metadata={}
            )