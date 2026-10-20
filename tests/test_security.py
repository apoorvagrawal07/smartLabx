
import pytest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.agents.permissions import AgentRole, authorize_tool
from app.services.conversation_access import ConversationAccess, SESSION_COOKIE


def test_rag_agent_cannot_execute_calculator():
    with pytest.raises(PermissionError):
        authorize_tool(AgentRole.RAG, "calculator")


def test_unknown_tool_is_rejected():
    with pytest.raises(PermissionError):
        authorize_tool(AgentRole.TOOL, "arbitrary_shell")


def test_reasoning_agent_cannot_execute_tools():
    with pytest.raises(PermissionError):
        authorize_tool(AgentRole.REASONING, "knowledge_search")


@pytest.fixture
def secured_app():
    from app import main

    workflow = Mock()
    workflow.get_state.return_value.values = {"messages": []}
    workflow.invoke.return_value = {"answer": "ok", "sources": []}
    with (
        patch.object(main, "workflow", workflow),
        patch.object(main, "conversation_access", ConversationAccess()),
    ):
        yield main.app, workflow


@pytest.mark.parametrize("field, value", [
    ("message", ""), ("message", " \n\t"), ("message", "x" * 10_001),
    ("conversation_id", ""), ("conversation_id", " \n\t"),
    ("conversation_id", "x" * 129),
    ("message", None), ("message", 42),
    ("conversation_id", None), ("conversation_id", 42),
])
def test_invalid_chat_input_never_accesses_checkpoints(secured_app, field, value):
    app, workflow = secured_app
    payload = {"message": "hello", "conversation_id": "validation"}
    payload[field] = value
    with TestClient(app) as client:
        response = client.post("/chat", json=payload)
    assert response.status_code == 422
    workflow.get_state.assert_not_called()
    workflow.invoke.assert_not_called()


def test_validation_response_does_not_echo_input(secured_app):
    app, workflow = secured_app
    with TestClient(app) as client:
        response = client.post("/chat", json={
            "message": {"secret": "PRIVATE_SECRET"},
            "conversation_id": "validation",
        })
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}
    assert "PRIVATE_SECRET" not in response.text
    workflow.get_state.assert_not_called()


def test_other_client_cannot_access_owned_checkpoint(secured_app):
    app, workflow = secured_app
    from uuid import uuid4

    payload = {"message": "hello", "conversation_id": str(uuid4())}
    with TestClient(app) as owner, TestClient(app) as stranger:
        assert owner.post("/chat", json=payload).status_code == 200
        workflow.reset_mock()
        denied = stranger.post("/chat", json=payload)
        assert denied.status_code == 403
        assert denied.json() == {"detail": "Forbidden"}
        workflow.get_state.assert_not_called()
        workflow.invoke.assert_not_called()
        assert owner.post("/chat", json=payload).status_code == 200


def test_maximum_chat_input_sizes_are_accepted(secured_app):
    app, workflow = secured_app
    with TestClient(app) as client:
        response = client.post("/chat", json={
            "message": "x" * 10_000, "conversation_id": "x" * 128,
        })
    assert response.status_code == 200
    workflow.get_state.assert_called_once()
    workflow.invoke.assert_called_once()


@pytest.mark.parametrize("forged", [False, True])
def test_another_session_or_forged_cookie_cannot_read_checkpoint(secured_app, forged):
    app, workflow = secured_app
    payload = {"message": "hello", "conversation_id": str(uuid4())}
    with TestClient(app) as owner, TestClient(app) as attacker:
        assert owner.post("/chat", json=payload).status_code == 200
        assert attacker.post("/chat", json={
            **payload, "conversation_id": str(uuid4()),
        }).status_code == 200
        workflow.reset_mock()
        headers = {"Cookie": f"{SESSION_COOKIE}=FORGED_TOKEN"} if forged else {}
        response = attacker.post("/chat", json=payload, headers=headers)
    assert response.status_code == 403
    workflow.get_state.assert_not_called()
    workflow.invoke.assert_not_called()


def test_client_chosen_cookie_is_replaced_with_server_token(secured_app):
    app, _ = secured_app
    payload = {"message": "hello", "conversation_id": str(uuid4())}
    headers = {"Cookie": f"{SESSION_COOKIE}=CLIENT_CHOSEN_TOKEN"}
    with TestClient(app) as owner, TestClient(app) as attacker:
        response = owner.post("/chat", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.cookies[SESSION_COOKIE] != "CLIENT_CHOSEN_TOKEN"
        assert attacker.post("/chat", json=payload, headers=headers).status_code == 403
        assert owner.post("/chat", json=payload).status_code == 200


def test_https_session_cookie_has_security_attributes(secured_app):
    app, _ = secured_app
    with TestClient(app, base_url="https://testserver") as client:
        response = client.post("/chat", json={
            "message": "hello", "conversation_id": str(uuid4()),
        })
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "secure" in cookie
    assert "samesite=strict" in cookie
    assert "path=/" in cookie
    assert response.cookies[SESSION_COOKIE] not in response.text


def test_first_request_failure_keeps_session_for_retry(secured_app):
    app, workflow = secured_app
    payload = {"message": "hello", "conversation_id": str(uuid4())}
    workflow.get_state.side_effect = RuntimeError("PRIVATE_SECRET")
    with TestClient(app) as client:
        response = client.post("/chat", json=payload)
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}
        assert SESSION_COOKIE in response.cookies
        workflow.invoke.assert_not_called()
        workflow.get_state.side_effect = None
        assert client.post("/chat", json=payload).status_code == 200


@pytest.mark.parametrize("failure, expected_status, detail", [
    (ValueError("PRIVATE_SECRET"), 400, "Invalid request"),
    (RuntimeError("PRIVATE_SECRET"), 503, "Model inference is temporarily unavailable"),
    (OSError("PRIVATE_SECRET"), 500, "Internal server error"),
])
def test_generate_errors_have_safe_consistent_responses(secured_app, failure, expected_status, detail):
    from app import main

    app, _ = secured_app
    gateway = Mock()
    gateway.generate.side_effect = failure
    with patch.object(main, "get_model_gateway", return_value=gateway), TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "PRIVATE_PROMPT"})
    assert response.status_code == expected_status
    assert response.json() == {"detail": detail}
    assert "PRIVATE_" not in response.text


def test_tool_denial_has_safe_http_response(secured_app):
    app, workflow = secured_app
    workflow.invoke.side_effect = PermissionError("PRIVATE_TOOL_DETAILS")
    with TestClient(app) as client:
        response = client.post("/chat", json={
            "message": "Calculate 2 + 3", "conversation_id": str(uuid4()),
        })
    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}


def test_simultaneous_claims_allow_only_one_owner():
    access = ConversationAccess()
    barrier = Barrier(2)

    def claim():
        barrier.wait(timeout=5)
        try:
            return access.authorize("shared", None)
        except PermissionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim(), range(2)))
    owners = [result for result in results if result is not None]
    assert len(owners) == 1
    assert access.authorize("shared", owners[0]) == owners[0]


@pytest.mark.parametrize("role, method, tool_name", [
    (AgentRole.TOOL, "_run_tool", "calculator"),
    (AgentRole.RAG, "_run_rag", "knowledge_search"),
])
def test_legacy_agent_authorizes_before_dispatch(monkeypatch, role, method, tool_name):
    from app.agents.agent import Agent
    from app.agents.permissions import AGENT_TOOL_ALLOWLIST
    from app.agents.state import AgentState

    monkeypatch.setitem(AGENT_TOOL_ALLOWLIST, role, set())
    tool_executor = Mock()
    rag_service = Mock()
    agent = Agent(rag_service, tool_executor, Mock())
    with pytest.raises(PermissionError):
        getattr(agent, method)(AgentState("Calculate 2 + 3"))
    tool_executor.execute.assert_not_called()
    rag_service.answer.assert_not_called()


@pytest.mark.parametrize("classification", [True, None, "false", "missing"])
def test_side_effecting_or_unclassified_tools_fail_before_execution(monkeypatch, classification):
    from app.agents.authorized_executor import AuthorizedExecutor
    from app.tools.registry import TOOLS
    from app.tools.tool_executor import ToolExecutor

    function = Mock()
    tool = {**TOOLS["calculator"], "function": function, "side_effecting": classification}
    if classification == "missing":
        del tool["side_effecting"]
    monkeypatch.setitem(TOOLS, "calculator", tool)
    with pytest.raises(PermissionError, match="idempotency"):
        AuthorizedExecutor(ToolExecutor()).execute(
            AgentRole.TOOL, "calculator", {"expression": "2 + 3"},
        )
    function.assert_not_called()
