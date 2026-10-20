from uuid import uuid4
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.services.memory_service import MemoryService
from app.services.conversation_access import ConversationAccess, PostgresConversationAccess, SESSION_COOKIE


@pytest.fixture
def memory_users():
    memory = MemoryService()
    user_id = f"test-memory-{uuid4().hex}"
    other_user_id = f"test-memory-{uuid4().hex}"

    try:
        yield memory, user_id, other_user_id
    finally:
        for test_user_id in (user_id, other_user_id):
            for item in memory.get_all(test_user_id):
                memory.delete(test_user_id, item["key"])


def test_memory_round_trip(memory_users):
    memory, user_id, _ = memory_users

    memory.save(
        user_id,
        "preferred_language",
        "Python",
    )

    results = memory.get_all(user_id)

    assert {
        "key": "preferred_language",
        "value": "Python",
    } in results

    memory.save(
        user_id,
        "preferred_language",
        "Go",
    )

    results = memory.get_all(user_id)
    matches = [
        item for item in results
        if item["key"] == "preferred_language"
    ]

    assert len(matches) == 1
    assert matches[0]["value"] == "Go"

    memory.delete(user_id, "preferred_language")

    assert memory.get_all(user_id) == []


def test_get_all_isolates_users(memory_users):
    memory, user_id, other_user_id = memory_users
    memory.save(user_id, "preferred_language", "Python")

    assert memory.get_all(other_user_id) == []

    memory.save(other_user_id, "preferred_language", "Go")

    assert memory.get_all(user_id) == [
        {"key": "preferred_language", "value": "Python"}
    ]
    assert memory.get_all(other_user_id) == [
        {"key": "preferred_language", "value": "Go"}
    ]


def test_delete_removes_only_selected_memory(memory_users):
    memory, user_id, other_user_id = memory_users
    memory.save(user_id, "preferred_language", "Python")
    memory.save(user_id, "timezone", "UTC")
    memory.save(other_user_id, "preferred_language", "Go")

    memory.delete(user_id, "preferred_language")

    assert memory.get_all(user_id) == [
        {"key": "timezone", "value": "UTC"}
    ]
    assert memory.get_all(other_user_id) == [
        {"key": "preferred_language", "value": "Go"}
    ]


@pytest.fixture
def memory_api(monkeypatch):
    from app import main

    monkeypatch.delenv("DATABASE_URL", raising=False)
    records = {}
    service = Mock()
    service.save.side_effect = lambda owner, key, value: records.__setitem__((owner, key), value)
    service.get_all.side_effect = lambda owner, limit=None: [
        {"key": key, "value": value}
        for (saved_owner, key), value in records.items() if saved_owner == owner
    ][:limit]
    service.delete.side_effect = lambda owner, key: records.pop((owner, key), None)
    workflow = Mock()
    workflow.get_state.return_value.values = {"messages": []}
    workflow.invoke.return_value = {"answer": "ok", "sources": []}
    with (
        patch.object(main, "workflow", workflow),
        patch.object(main, "conversation_access", ConversationAccess()),
        patch.object(main.app.state, "memory_service", service, create=True),
    ):
        yield main.app, service, workflow


def test_memory_api_isolates_sessions_and_supports_upsert_and_delete(memory_api):
    app, service, _ = memory_api
    with TestClient(app) as owner, TestClient(app) as stranger:
        for client in (owner, stranger):
            assert client.post("/chat", json={"message": "hello", "conversation_id": str(uuid4())}).status_code == 200
        assert owner.post("/memories", json={"key": "language", "value": "Python"}).status_code == 200
        assert owner.post("/memories", json={"key": "language", "value": "Rust"}).status_code == 200
        assert stranger.get("/memories").json() == []
        assert stranger.delete("/memories/language").status_code == 204
        assert owner.get("/memories").json() == [{"key": "language", "value": "Rust"}]
        assert owner.delete("/memories/language").status_code == 204
        assert owner.get("/memories").json() == []
        assert service.save.call_args.args[0] != owner.cookies[SESSION_COOKIE]


@pytest.mark.parametrize("cookie", [None, "FORGED_COOKIE"])
def test_memory_api_requires_a_valid_session_before_storage_access(memory_api, cookie):
    app, service, _ = memory_api
    with TestClient(app) as client:
        headers = {"Cookie": f"{SESSION_COOKIE}={cookie}"} if cookie else {}
        assert client.get("/memories", headers=headers).status_code == 403
        assert client.post("/memories", headers=headers, json={"key": "language", "value": "Rust"}).status_code == 403
        assert client.delete("/memories/language", headers=headers).status_code == 403
    service.get_all.assert_not_called()
    service.save.assert_not_called()
    service.delete.assert_not_called()


@pytest.mark.parametrize("payload", [
    {"key": "", "value": "Rust"}, {"key": "x" * 65, "value": "Rust"},
    {"key": "..", "value": "Rust"},
    {"key": "language", "value": " \n\t"}, {"key": "language", "value": "x" * 1001},
    {"key": "language", "value": "Rust", "user_id": "another-owner"},
])
def test_memory_api_rejects_invalid_or_forged_fields(memory_api, payload):
    app, service, _ = memory_api
    with TestClient(app) as client:
        assert client.post("/memories", json=payload).status_code == 422
    service.save.assert_not_called()


def test_memory_api_recalls_across_conversations_and_clears_deleted_facts(memory_api):
    app, _, workflow = memory_api
    with TestClient(app) as owner, TestClient(app) as stranger:
        original = str(uuid4())
        fresh = str(uuid4())
        assert owner.post("/chat", json={"message": "hello", "conversation_id": original}).status_code == 200
        assert owner.post("/memories", json={"key": "language", "value": "Rust"}).status_code == 200
        assert owner.post("/chat", json={"message": "What do I prefer?", "conversation_id": fresh}).status_code == 200
        assert workflow.invoke.call_args.args[0]["user_memories"] == [{"key": "language", "value": "Rust"}]
        assert stranger.post("/chat", json={"message": "hello", "conversation_id": str(uuid4())}).status_code == 200
        assert workflow.invoke.call_args.args[0]["user_memories"] == []
        assert owner.delete("/memories/language").status_code == 204
        assert owner.post("/chat", json={"message": "hello again", "conversation_id": fresh}).status_code == 200
        assert workflow.invoke.call_args.args[0]["user_memories"] == []


def test_memory_bounded_reads_use_owner_and_parameterized_limit():
    with patch("app.services.memory_service.psycopg.connect") as connect:
        connection = connect.return_value.__enter__.return_value
        connection.execute.return_value.fetchall.return_value = [("language", "Rust")]
        result = MemoryService("postgresql://test.invalid/test").get_all("owner-hash", limit=20)
    assert result == [{"key": "language", "value": "Rust"}]
    query, parameters = connection.execute.call_args.args
    assert "WHERE user_id = %s" in query and "LIMIT %s" in query
    assert parameters == ("owner-hash", 20)


def test_postgres_owner_identity_requires_a_registered_session():
    from hashlib import sha256

    pool = MagicMock()
    connection = pool.connection.return_value.__enter__.return_value
    ownership = PostgresConversationAccess(pool)
    with pytest.raises(PermissionError):
        ownership.owner_id(None)
    connection.execute.assert_not_called()
    connection.execute.return_value.fetchone.return_value = None
    with pytest.raises(PermissionError):
        ownership.owner_id("forged")
    connection.execute.return_value.fetchone.return_value = {"exists": 1}
    assert ownership.owner_id("valid-session") == sha256(b"valid-session").hexdigest()
    assert connection.execute.call_args.args[1] == (sha256(b"valid-session").hexdigest(),)
