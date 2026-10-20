import uuid
import os
import psycopg
from psycopg.types.json import Jsonb
from sympy import limit


class VectorStore:

    def __init__(self, database_url: str | None = None):
        self.connection = psycopg.connect(database_url or os.environ["DATABASE_URL"])

    @staticmethod
    def setup(connection):
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    document_type TEXT,
                    title TEXT,
                    section TEXT,
                    content TEXT NOT NULL,
                    embedding VECTOR(384),
                    embedding_model TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    search_vector TSVECTOR GENERATED ALWAYS AS
                        (to_tsvector('english', content)) STORED,
                    UNIQUE (document_id, chunk_id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS documents_search_vector_idx
                ON documents USING GIN (search_vector)
            """)

    def add_document(
        self,
        document_id,
        chunk_id,
        source,
        document_type,
        title,
        section,
        content,
        embedding,
        embedding_model,
        metadata,
    ):
        with self.connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO documents (
                    id,
                    document_id,
                    chunk_id,
                    source,
                    document_type,
                    title,
                    section,
                    content,
                    embedding,
                    embedding_model,
                    metadata
                )
                VALUES (
                    gen_random_uuid(),
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                ON CONFLICT (document_id, chunk_id)
                DO NOTHING
                """,
                (
                    document_id,
                    chunk_id,
                    source,
                    document_type,
                    title,
                    section,
                    content,
                    embedding,
                    embedding_model,
                    Jsonb(metadata),
                )
            )

        self.connection.commit()


    def search(
        self,
        query_embedding: list[float],
        limit: int = 3
    ):
        return [
            (document["source"], document["content"], document["distance"])
            for document in self.vector_search(query_embedding, limit)
        ]

    def vector_search(self, query_embedding: list[float], limit: int = 20):
        with self.connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    source,
                    content,
                    embedding <=> %s::vector AS distance
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector, id
                LIMIT %s
                """,
                (
                    str(query_embedding),
                    str(query_embedding),
                    limit
                )
            )

            rows = cursor.fetchall()

        return [
            {"id": str(row[0]), "source": row[1], "content": row[2], "distance": float(row[3])}
            for row in rows
        ]

    
    def keyword_search(self, query: str, limit: int = 10):
        sql = """
            SELECT
                id,
                source,
                content,
                ts_rank_cd(
                    search_vector,
                    websearch_to_tsquery('english', %s)
                ) AS score
            FROM documents
            WHERE search_vector @@
                websearch_to_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """

        with self.connection.cursor() as cursor:
            cursor.execute(sql, (query, query, limit))
            rows = cursor.fetchall()

        return [
            {
                "id": str(row[0]),
                "source": row[1],
                "content": row[2],
                "score": float(row[3]),
            }
            for row in rows
        ]
