import uuid
import psycopg
from psycopg.types.json import Jsonb
from sympy import limit


class VectorStore:

    def __init__(self):

        self.connection = psycopg.connect(
            "host=localhost "
            "port=5432 "
            "dbname=aidb "
            "user=aiuser "
            "password=aipassword"
        )

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

        with self.connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    source,
                    content,
                    embedding <=> %s::vector AS distance
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    str(query_embedding),
                    str(query_embedding),
                    limit
                )
            )

            return cursor.fetchall()