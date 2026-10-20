import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.graph.workflow import build_graph
from app.services.memory_service import MemoryService
from app.services.persistence import open_persistence


@pytest.fixture
def persistent_conversation():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for real PostgreSQL persistence tests")
    conversation_id = f"persistence-test-{uuid4()}"
    try:
        yield database_url, conversation_id
    finally:
        thread_ids = [conversation_id, conversation_id + "-fresh"]
        with open_persistence(database_url) as (checkpointer, _):
            for thread_id in thread_ids:
                checkpointer.delete_thread(thread_id)
        with psycopg.connect(database_url) as connection:
            connection.execute(
                "DELETE FROM user_memories WHERE user_id IN "
                "(SELECT session_hash FROM conversation_owners WHERE conversation_id = ANY(%s))",
                (thread_ids,),
            )
            connection.execute("DELETE FROM conversation_owners WHERE conversation_id = ANY(%s)", (thread_ids,))


PROCESS_SCRIPT = """
import json
import sys
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from langgraph.checkpoint.postgres import PostgresSaver
from app.services.memory_service import MemoryService
from app.services.conversation_access import SESSION_COOKIE

payload = json.load(sys.stdin)
gateway = Mock()
gateway.generate.return_value = "Recorded."
with patch("app.services.model_gateway.ModelGateway", return_value=gateway):
    from app import main

with TestClient(main.app) as client:
    graph = main.app.state.workflow
    assert isinstance(graph.checkpointer, PostgresSaver)
    if payload["mode"] == "write":
        message = "My favorite language is Rust."
    else:
        with patch.object(graph, "get_state", wraps=graph.get_state) as get_state:
            denied = client.post("/chat", json={"message": "hello", "conversation_id": payload["id"]})
            assert denied.status_code == 403
            get_state.assert_not_called()
        client.cookies.set(SESSION_COOKIE, payload["cookie"])
        message = "What language did I mention?"
    response = client.post("/chat", json={"message": message, "conversation_id": payload["id"]})
    assert response.status_code == 200, response.status_code
    if payload["mode"] == "write":
        saved = client.post("/memories", json={"key": "language", "value": "Rust"})
        assert saved.status_code == 200
    else:
        assert "User: My favorite language is Rust." in gateway.generate.call_args.kwargs["prompt"]
        assert "Assistant: Recorded." in gateway.generate.call_args.kwargs["prompt"]
        assert '"value": "Rust"' in gateway.generate.call_args.kwargs["prompt"]
        fresh = client.post("/chat", json={"message": "What language do I prefer?", "conversation_id": payload["id"] + "-fresh"})
        assert fresh.status_code == 200
        prompt = gateway.generate.call_args.kwargs["prompt"]
        assert "Saved user facts" in prompt and '"value": "Rust"' in prompt
        assert "My favorite language is Rust." not in prompt
    state = graph.get_state({"configurable": {"thread_id": payload["id"]}}).values
    cookie = response.cookies[SESSION_COOKIE]
    owner_id = main.app.state.conversation_access.owner_id(cookie)
    print(json.dumps({"cookie": cookie, "owner_id": owner_id, "messages": state["messages"], "memory": client.get("/memories").json()}))
"""


def test_app_checkpoint_ownership_and_memory_survive_fresh_process(persistent_conversation):
    _, conversation_id = persistent_conversation

    def run_process(payload):
        result = subprocess.run(
            [sys.executable, "-c", PROCESS_SCRIPT],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            cwd=Path(__file__).resolve().parents[1],
        )
        return json.loads(result.stdout)

    first = run_process({"mode": "write", "id": conversation_id})
    second = run_process({"mode": "read", "id": conversation_id, "cookie": first["cookie"]})
    assert second["cookie"] == first["cookie"]
    assert second["messages"][:2] == first["messages"]
    assert len(second["messages"]) == 4
    assert second["memory"] == [{"key": "language", "value": "Rust"}]
    database_url, _ = persistent_conversation
    with open_persistence(database_url) as (checkpointer, ownership):
        checkpointer.delete_thread(conversation_id)
        with ownership.pool.connection() as connection:
            owner = connection.execute(
                "SELECT session_hash FROM conversation_owners WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchone()
        assert owner["session_hash"] != first["cookie"]
    assert MemoryService(database_url).get_all(first["owner_id"]) == second["memory"]


def test_postgres_ownership_claim_is_atomic(persistent_conversation):
    database_url, conversation_id = persistent_conversation
    with open_persistence(database_url) as (_, ownership):
        barrier = Barrier(2)

        def claim():
            barrier.wait(timeout=5)
            try:
                return ownership.authorize(conversation_id, None)
            except PermissionError:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: claim(), range(2)))
        owners = [result for result in results if result is not None]
        assert len(owners) == 1
        assert ownership.authorize(conversation_id, owners[0]) == owners[0]
        with pytest.raises(PermissionError):
            ownership.authorize(conversation_id, "FORGED_COOKIE")
    assert ownership.pool.closed


def test_existing_checkpoint_without_owner_cannot_be_claimed(persistent_conversation):
    database_url, conversation_id = persistent_conversation
    with open_persistence(database_url) as (checkpointer, _):
        gateway = Mock()
        gateway.generate.return_value = "hello"
        graph = build_graph(gateway, Mock(), Mock(), checkpointer=checkpointer)
        graph.invoke(
            {"user_message": "hello"},
            config={"configurable": {"thread_id": conversation_id}},
        )
    with open_persistence(database_url) as (_, ownership):
        with pytest.raises(PermissionError):
            ownership.authorize(conversation_id, None)


def test_missing_database_disables_chat_instead_of_using_volatile_state(monkeypatch):
    from app import main

    monkeypatch.delenv("DATABASE_URL", raising=False)
    with (
        patch.object(main, "open_persistence") as open_database,
        patch.object(main, "build_graph") as build,
        TestClient(main.app) as client,
    ):
        response = client.post("/chat", json={"message": "hello", "conversation_id": str(uuid4())})
    assert response.status_code == 503
    open_database.assert_not_called()
    build.assert_not_called()