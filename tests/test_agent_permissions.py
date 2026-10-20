from unittest.mock import Mock

import pytest

from unittest.mock import Mock

import pytest

from app.agents.authorized_executor import AuthorizedToolExecutor
from app.agents.permissions import (
    AgentRole,
    ToolPermissionError,
)

from app.agents.authorized_executor import AuthorizedToolExecutor
from app.agents.permissions import (
    AgentRole,
    ToolPermissionError,
    authorize_tool,
)
from app.graph.workflow import build_graph
from app.models.llm import LLMResponse


def test_tool_agent_can_calculate():
    authorize_tool(AgentRole.TOOL, "calculator")


def test_rag_agent_can_search_knowledge():
    authorize_tool(AgentRole.RAG, "knowledge_search")


def test_rag_agent_cannot_calculate():
    with pytest.raises(ToolPermissionError):
        authorize_tool(AgentRole.RAG, "calculator")


def test_reasoning_agent_has_no_tools():
    with pytest.raises(ToolPermissionError):
        authorize_tool(AgentRole.REASONING, "current_time")


def test_unknown_tool_is_rejected():
    with pytest.raises(ToolPermissionError):
        authorize_tool(AgentRole.TOOL, "execute_shell")


def test_denied_tool_never_reaches_executor():
    underlying = Mock()
    executor = AuthorizedToolExecutor(underlying)

    with pytest.raises(ToolPermissionError):
        executor.execute(
            agent_role=AgentRole.RAG,
            tool_name="calculator",
            arguments={"expression": "125 * 42"},
        )

    underlying.execute.assert_not_called()

def test_authorized_executor_reuses_existing_executor():
    executor = Mock()
    executor.execute.return_value = "5"

    result = AuthorizedToolExecutor(executor).execute(
        AgentRole.TOOL, "calculator", {"expression": "2 + 3"}
    )

    assert result == "5"
    executor.execute.assert_called_once_with(
        tool_name="calculator", arguments={"expression": "2 + 3"}
    )


@pytest.mark.parametrize("role, tool_name", [
    (AgentRole.RAG, "calculator"),
    (AgentRole.REASONING, "current_time"),
    (AgentRole.TOOL, "knowledge_search"),
    (AgentRole.TOOL, "execute_shell"),
])
def test_permission_denial_prevents_dispatch(role, tool_name):
    executor = Mock()

    with pytest.raises(ToolPermissionError):
        AuthorizedToolExecutor(executor).execute(role, tool_name, {})

    executor.execute.assert_not_called()


@pytest.mark.parametrize("tool_name", ["knowledge_search", "execute_shell"])
def test_graph_rejects_disallowed_tool_before_dispatch(tool_name):
    gateway = Mock()
    gateway.generate.return_value = LLMResponse(
        content="",
        tool_calls=[{"tool_name": tool_name, "arguments": {}}],
    )
    executor = Mock()
    graph = build_graph(gateway, Mock(), executor)

    with pytest.raises(ToolPermissionError):
        graph.invoke(
            {"user_message": "Calculate 2 + 3"},
            config={"configurable": {"thread_id": "denied-tool"}},
        )

    executor.execute.assert_not_called()
