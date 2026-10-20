
import pytest

from app.agents.permissions import AgentRole, authorize_tool


def test_rag_agent_cannot_execute_calculator():
    with pytest.raises(PermissionError):
        authorize_tool(AgentRole.RAG, "calculator")


def test_reasoning_agent_cannot_execute_tools():
    with pytest.raises(PermissionError):
        authorize_tool(AgentRole.REASONING, "knowledge_search")


def test_tool_agent_can_execute_calculator():
    authorize_tool(AgentRole.TOOL, "calculator")
