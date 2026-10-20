
import os
import psycopg


class MemoryService:
    def __init__(self, database_url: str | None = None):
        self.database_url = (
            database_url or os.environ["DATABASE_URL"]
        )

    def setup(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (user_id, memory_key)
                )
            """)

    def save(
        self,
        user_id: str,
        memory_key: str,
        memory_value: str,
    ) -> None:
        query = """
            INSERT INTO user_memories (
                user_id, memory_key, memory_value
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, memory_key)
            DO UPDATE SET
                memory_value = EXCLUDED.memory_value,
                updated_at = NOW()
        """

        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                query,
                (user_id, memory_key, memory_value),
            )

    def get_all(self, user_id: str, limit: int | None = None) -> list[dict]:
        query = """
            SELECT memory_key, memory_value
            FROM user_memories
            WHERE user_id = %s
            ORDER BY updated_at DESC, memory_key
            LIMIT %s
        """

        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(query, (user_id, limit)).fetchall()

        return [
            {"key": key, "value": value}
            for key, value in rows
        ]

    def delete(self, user_id: str, memory_key: str) -> None:
        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                """
                DELETE FROM user_memories
                WHERE user_id = %s AND memory_key = %s
                """,
                (user_id, memory_key),
            )
