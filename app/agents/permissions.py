
from enum import StrEnum


class AgentRole(StrEnum):
    RAG = "rag"
    TOOL = "tool"
    REASONING = "reasoning"


AGENT_TOOL_ALLOWLIST: dict[AgentRole, set[str]] = {
    AgentRole.RAG: {"knowledge_search"},
    AgentRole.TOOL: {"calculator", "current_time"},
    AgentRole.REASONING: set(),
}


class ToolPermissionError(PermissionError):
    pass


def authorize_tool(
    agent_role: AgentRole,
    tool_name: str,
) -> None:
    allowed_tools = AGENT_TOOL_ALLOWLIST[agent_role]

    if tool_name not in allowed_tools:
        raise ToolPermissionError(
            f"Agent '{agent_role.value}' cannot use '{tool_name}'."
        )
