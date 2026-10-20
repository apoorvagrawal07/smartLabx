from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.services.conversation_access import PostgresConversationAccess
from app.services.memory_service import MemoryService
from app.rag.vector_store import VectorStore


@contextmanager
def open_persistence(database_url: str):
    with ConnectionPool(
        database_url,
        min_size=1,
        max_size=5,
        timeout=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "connect_timeout": 5,
        },
    ) as pool:
        pool.wait(timeout=5)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        with pool.connection() as connection:
            VectorStore.setup(connection)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversation_owners (
                    conversation_id TEXT PRIMARY KEY,
                    session_hash TEXT
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS conversation_owners_session_hash_idx
                ON conversation_owners (session_hash)
            """)
            connection.execute("""
                INSERT INTO conversation_owners (conversation_id, session_hash)
                SELECT DISTINCT thread_id, NULL FROM checkpoints
                ON CONFLICT (conversation_id) DO NOTHING
            """)
        MemoryService(database_url).setup()
        yield checkpointer, PostgresConversationAccess(pool)