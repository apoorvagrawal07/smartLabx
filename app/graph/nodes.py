from app.agents.router import AgentRouter
from app.models.llm import LLMMessage
from app.tools.calculator import _extract_expression
from app.tools.registry import get_tool_schemas

router = AgentRouter()

AGENT_SYSTEM_PROMPT = (
    "You are a tool-using assistant. Use calculator for arithmetic and "
    "current_time for time questions. When tool results are present, "
    "use them as the source of truth and never claim you lack access "
    "to information returned by a tool."
)


def _as_text(value):
    return value.content if hasattr(value, "content") else value


def _coerce_tool_calls(tool_calls):
    normalized = []

    for call in tool_calls or []:
        if isinstance(call, dict):
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments", {})
            else:
                name = call.get("tool_name") or call.get("name")
                arguments = call.get("arguments", {})

            if name is not None:
                normalized.append({
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                })
        elif hasattr(call, "function"):
            normalized.append({
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
            })

    return normalized


def _normalize_message(value):
    if isinstance(value, dict):
        return {
            "role": value.get("role", "assistant"),
            "content": value.get("content"),
            "tool_calls": _coerce_tool_calls(value.get("tool_calls", [])),
        }

    if hasattr(value, "content"):
        return {
            "role": "assistant",
            "content": value.content,
            "tool_calls": _coerce_tool_calls(getattr(value, "tool_calls", []) or []),
        }

    return {
        "role": "user",
        "content": str(value),
        "tool_calls": []
    }


def router_node(state):

    user_message = state.get("user_message")

    if user_message is None:
        user_message = next(
            message["content"]
            for message in reversed(state.get("messages", []))
            if message.get("role") == "user"
        )

    route = router.route(
        user_message
    )

    return {
        "route": route,
        "user_message": user_message,
    }


def direct_node(state, model_gateway):

    result = model_gateway.generate(
        model="llama",
        messages=[
            LLMMessage(
                role="user",
                content=state["user_message"]
            )
        ]
    )

    return {
        "answer": _as_text(result)
    }


def rag_node(state, rag_service):

    result = rag_service.answer(
        state["user_message"]
    )

    return {
        "answer": _as_text(result)
    }


def tool_node(state, tool_executor):

    expression = _extract_expression(state["user_message"])

    result = tool_executor.execute(
        tool_name="calculator",
        arguments={
            "expression": expression
        }
    )

    return {
        "tool_results": [
            {
                "tool": "calculator",
                "result": result
            }
        ],
        "answer": result
    }


def route_decision(state):
    return state["route"]


def agent_llm_node(
    state,
    model_gateway
):

    messages = state.get("messages") or [
        {
            "role": "user",
            "content": state["user_message"],
        }
    ]

    if not any(
        message.get("role") == "system"
        and message.get("content") == AGENT_SYSTEM_PROMPT
        for message in messages
    ):
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *messages,
        ]

    response = model_gateway.generate(
        model="llama",
        messages=messages,
        tools=get_tool_schemas()
    )

    normalized = _normalize_message(response)
    answer = normalized.get("content") or _as_text(response)

    return {
        "messages": messages + [normalized],
        "answer": answer,
        "iteration_count": state.get("iteration_count", 0),
    }


def execute_tool_node(
    state,
    tool_executor
):

    last_message = state["messages"][-1]
    tool_calls = _coerce_tool_calls(last_message.get("tool_calls", []))

    tool_messages = []
    tool_results = []

    for call in tool_calls:
        function = call.get("function", {})
        tool_name = function.get("name")
        arguments = function.get("arguments", {})

        if tool_name is None:
            continue

        result = tool_executor.execute(
            tool_name=tool_name,
            arguments=arguments,
        )

        tool_messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "content": str(result),
        })

        tool_results.append({
            "tool": tool_name,
            "result": result,
        })

    return {
        "messages": state["messages"] + tool_messages,
        "tool_results": tool_results,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tool_decision(state):

    if state.get("iteration_count", 0) >= 5:
        return "end"

    last_message = state["messages"][-1]
    tool_calls = last_message.get("tool_calls", [])

    if tool_calls:
        return "tool"

    return "end"