import unittest
from unittest.mock import patch
from uuid import UUID

from psycopg.types.json import Jsonb

from app.rag.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_connection_uses_database_url(self):
        with (
            patch.dict("os.environ", {"DATABASE_URL": "postgresql://postgres/test"}),
            patch("app.rag.vector_store.psycopg.connect") as connect,
        ):
            VectorStore()
        connect.assert_called_once_with("postgresql://postgres/test")

    def test_add_document_adapts_metadata_to_jsonb(self):
        for metadata in ({}, {"tags": ["kafka"], "page": 1}):
            with self.subTest(metadata=metadata):
                with patch("app.rag.vector_store.psycopg.connect") as connect:
                    store = VectorStore(database_url="postgresql://test.invalid/test")
                    connection = connect.return_value
                    cursor = connection.cursor.return_value.__enter__.return_value

                    store.add_document(
                        document_id=UUID("00000000-0000-0000-0000-000000000001"),
                        chunk_id=0,
                        source="knowledge/kafka.md",
                        document_type="markdown",
                        title="Kafka",
                        section=None,
                        content="Kafka reassigns partitions after consumer failure.",
                        embedding=[0.0] * 384,
                        embedding_model="all-MiniLM-L6-v2",
                        metadata=metadata,
                    )

                    cursor.execute.assert_called_once()
                    sql, parameters = cursor.execute.call_args.args
                    self.assertIn("ON CONFLICT (document_id, chunk_id)", sql)
                    self.assertIsInstance(parameters[-1], Jsonb)
                    self.assertEqual(parameters[-1].obj, metadata)
                    connection.commit.assert_called_once_with()


    def test_vector_results_preserve_chunk_ids_and_legacy_tuple_contract(self):
        with patch("app.rag.vector_store.psycopg.connect") as connect:
            store = VectorStore(database_url="postgresql://test.invalid/test")
        cursor = connect.return_value.cursor.return_value.__enter__.return_value
        chunk_id = UUID("00000000-0000-0000-0000-000000000001")
        cursor.fetchall.return_value = [(chunk_id, "kafka.md", "recovery", 0.2)]
        assert store.vector_search([0.1], limit=5) == [{
            "id": str(chunk_id), "source": "kafka.md", "content": "recovery", "distance": 0.2,
        }]
        assert cursor.execute.call_args.args[1] == ("[0.1]", "[0.1]", 5)
        assert store.search([0.1]) == [("kafka.md", "recovery", 0.2)]


if __name__ == "__main__":
    unittest.main()
