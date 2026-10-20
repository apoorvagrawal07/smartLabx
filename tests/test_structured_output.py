
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from app.agents.router import AgentRouter
from app.graph.nodes import route_decision
from app.graph.workflow import build_graph
from app.models.llm import LLMResponse
from app.models.structured_outputs import RouteDecision, ToolCallRequest
from app.tools.registry import TOOLS
from app.tools.tool_executor import ToolExecutor
from app.services.structured_output_service import (
    StructuredOutputError,
    parse_structured_output,
)


def test_valid_route():
    result = parse_structured_output(
        """
        {
          "intent": "rag",
          "confidence": 0.9,
          "requires_retrieval": true,
          "reason": "The question requires internal documentation."
        }
        """,
        RouteDecision,
    )

    assert isinstance(result, RouteDecision)
    assert result.intent == "rag"
    assert result.requires_retrieval is True


def test_invalid_intent_is_rejected():
    with pytest.raises(StructuredOutputError):
        parse_structured_output(
            """
            {
              "intent": "hack_everything",
              "confidence": 0.9,
              "requires_retrieval": false,
              "reason": "Invalid intent."
            }
            """,
            RouteDecision,
        )


def test_invalid_json_is_rejected():
    with pytest.raises(StructuredOutputError):
        parse_structured_output(
            "This is not JSON",
            RouteDecision,
        )


@pytest.mark.parametrize("raw_output", [
    "{}",
    "[]",
    "null",
    '{"intent": "rag", "confidence": 2, "requires_retrieval": true, "reason": "test"}',
    '{"intent": "rag", "confidence": 0.5, "requires_retrieval": "invalid", "reason": "test"}',
])
def test_invalid_schema_is_rejected(raw_output):
    with pytest.raises(StructuredOutputError) as error:
        parse_structured_output(raw_output, RouteDecision)

    assert isinstance(error.value.__cause__, ValidationError)


@pytest.mark.parametrize("question, intent", [
    ("Explain Kafka partitions.", "rag"),
    ("Hello!", "direct"),
])
def test_router_produces_typed_decisions(question, intent):
    decision = AgentRouter().decide(question)

    assert isinstance(decision, RouteDecision)
    assert decision.intent == intent
    assert decision.requires_retrieval is (intent == "rag")


@pytest.mark.parametrize("intent", ["direct", "rag", "tool", "clarification"])
def test_graph_routes_using_validated_intent(intent):
    decision = parse_structured_output(
        '{"intent": "' + intent + '", "confidence": 0.9, '
        '"requires_retrieval": false, "reason": "Test decision"}',
        RouteDecision,
    )
    gateway = Mock()
    gateway.generate.return_value = LLMResponse(content="Chat answer")
    rag_service = Mock()
    rag_service.answer.return_value = "Retrieved answer"
    executor = Mock()
    graph = build_graph(gateway, rag_service, executor)

    with patch("app.graph.nodes.router.decide", return_value=decision):
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Hello!"}]},
            config={"configurable": {"thread_id": "validated-route"}},
        )

    assert result["route_decision"] == decision.model_dump()
    if intent == "rag":
        assert result["answer"] == "Retrieved answer"
        gateway.generate.assert_not_called()
        rag_service.answer.assert_called_once_with("Hello!")
    else:
        assert result["answer"] == "Chat answer"
        rag_service.answer.assert_not_called()
        gateway.generate.assert_called_once()
    executor.execute.assert_not_called()


def test_graph_rejects_invalid_intent_before_branching():
    with pytest.raises(ValidationError):
        route_decision({
            "route": "rag",
            "route_decision": {
                "intent": "unknown",
                "confidence": 0.9,
                "requires_retrieval": False,
                "reason": "Invalid decision",
            },
        })


def test_unknown_structured_tool_is_rejected():
    with pytest.raises(StructuredOutputError):
        parse_structured_output(
            '{"tool_name": "shell", "arguments": {}}',
            ToolCallRequest,
        )


@pytest.mark.parametrize("tool_name", ["shell", "knowledge_search"])
def test_unregistered_tools_cannot_execute(tool_name):
    function = Mock()
    with patch.dict(TOOLS["calculator"], {"function": function}):
        with pytest.raises(ValueError, match="Unknown tool"):
            ToolExecutor().execute(tool_name, {})

    function.assert_not_called()


@pytest.mark.parametrize("tool_name, arguments", [
    ("calculator", {}),
    ("calculator", {"expression": 12}),
    ("calculator", {"expression": True}),
    ("calculator", {"expression": None}),
    ("calculator", {"expression": ""}),
    ("calculator", {"expression": "2 + 3", "unexpected": "value"}),
    ("calculator", ["2 + 3"]),
    ("current_time", {"unexpected": "value"}),
])
def test_invalid_arguments_rejected_before_execution(tool_name, arguments):
    function = Mock()
    with patch.dict(TOOLS[tool_name], {"function": function}):
        with pytest.raises(ValidationError):
            ToolExecutor().execute(tool_name, arguments)

    function.assert_not_called()


def test_valid_structured_tool_request_executes():
    request = parse_structured_output(
        '{"tool_name": "calculator", "arguments": {"expression": "2 + 3"}}',
        ToolCallRequest,
    )

    assert isinstance(request, ToolCallRequest)
    assert ToolExecutor().execute(request.tool_name, request.arguments) == "5"
