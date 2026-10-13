from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    user_message: str
    route: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None