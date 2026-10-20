from typing import Any
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):

    user_message: str

    route: str

    route_decision: dict[str, Any]

    messages: list[dict[str, Any]]

    user_memories: list[dict[str, str]]

    retrieved_documents: list[dict[str, Any]]

    tool_calls: list[dict[str, Any]]

    tool_results: list[dict[str, Any]]

    answer: str
    
    sources: list[dict[str, Any]]

    iteration_count: int