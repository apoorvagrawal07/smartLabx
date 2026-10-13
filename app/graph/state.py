from typing import Any
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):

    user_message: str

    route: str

    messages: list[dict[str, Any]]

    retrieved_documents: list[dict[str, Any]]

    tool_calls: list[dict[str, Any]]

    tool_results: list[dict[str, Any]]

    answer: str

    iteration_count: int