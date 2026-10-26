import re
import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import main
from app.graph.nodes import DIRECT_SYSTEM_PROMPT


@pytest.fixture
def client():
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL is required for durable chat integration tests")
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def conversation_id(client):
    thread_id = f"e2e-{uuid4().hex}"
    yield thread_id
    main.app.state.workflow.checkpointer.delete_thread(thread_id)
    with main.app.state.conversation_access.pool.connection() as connection:
        connection.execute(
            "DELETE FROM conversation_owners WHERE conversation_id = %s", (thread_id,)
        )


def checkpoint(conversation_id):
    return main.app.state.workflow.get_state(
        {"configurable": {"thread_id": conversation_id}}
    ).values


def test_chat_endpoint_returns_answer(client, conversation_id):
    response = client.post(
        "/chat",
        json={
            "message": "Explain what an API gateway is",
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body.get("answer"), str)
    assert body["answer"].strip()
    assert body["sources"] == []
    assert body["conversation_id"] == conversation_id
    assert checkpoint(conversation_id)["route_decision"]["intent"] == "direct"


def test_rag_response_contains_sources(client, conversation_id):
    response = client.post(
        "/chat",
        json={
            "message": "What happens when a Kafka consumer crashes?",
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body.get("answer"), str)
    assert isinstance(body.get("sources"), list)
    assert body["sources"]
    source_ids = {source["id"] for source in body["sources"]}
    assert len(source_ids) == len(body["sources"])
    assert all(re.fullmatch(r"S[1-9]\d*", source_id) for source_id in source_ids)
    assert all(source["source"] and source["content"] for source in body["sources"])
    cited_ids = set(re.findall(r"\[(S\d+)\]", body["answer"]))
    assert cited_ids <= source_ids
    assert checkpoint(conversation_id)["route_decision"]["intent"] == "rag"


def test_tool_route_executes_calculator(client, conversation_id):
    response = client.post(
        "/chat",
        json={"message": "Calculate 125 * 42", "conversation_id": conversation_id},
    )

    assert response.status_code == 200
    assert "5250" in response.json()["answer"].replace(",", "")
    state = checkpoint(conversation_id)
    assert state["route_decision"]["intent"] == "tool"
    assert {"tool": "calculator", "result": "5250"} in state["tool_results"]


def test_multiple_turns_share_checkpoint_thread(client, conversation_id):
    first = client.post(
        "/chat",
        json={"message": "My name is Shanu.", "conversation_id": conversation_id},
    )
    assert first.status_code == 200
    previous = checkpoint(conversation_id)["messages"]

    with patch.object(
        main.model_gateway, "generate", wraps=main.model_gateway.generate
    ) as generate:
        second = client.post(
            "/chat",
            json={"message": "What is my name?", "conversation_id": conversation_id},
        )

    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id
    assert "shanu" in second.json()["answer"].lower()
    expected_messages = previous + [{"role": "user", "content": "What is my name?"}]
    generate.assert_called_once_with(model="llama", messages=[
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT}, *expected_messages,
    ])
    messages = checkpoint(conversation_id)["messages"]
    assert messages[:len(previous)] == previous
    assert messages[len(previous)] == {"role": "user", "content": "What is my name?"}
    assert messages[-1]["role"] == "assistant"


def test_invalid_request_is_rejected(client):
    response = client.post(
        "/chat",
        json={"message": ""},
    )

    assert response.status_code in (400, 422)
