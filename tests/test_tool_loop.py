import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from ollama import Message

from app.graph.nodes import (
    AGENT_SYSTEM_PROMPT,
    MAX_TOOL_ITERATIONS,
    agent_llm_node,
    execute_tool_node,
)
from app.graph.workflow import build_graph
from app.models.llm import LLMResponse
from app.rag.rag_service import RAGService
from app.tools.tool_executor import ToolExecutor


class FakeModelGateway:
    def __init__(self):
        self.calls = []

    def generate(self, model, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})

        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                tool_calls=[{
                    "tool_name": "calculator",
                    "arguments": {"expression": "125 * 42"},
                }],
            )

        return LLMResponse(
            content="The answer is 5250.",
            tool_calls=[],
        )


class ToolLoopTests(unittest.TestCase):
    def test_repeated_tool_calls_stop_at_hard_limit(self):
        gateway = Mock()
        gateway.generate.return_value = LLMResponse(
            content="",
            tool_calls=[{
                "tool_name": "calculator",
                "arguments": {"expression": "2 + 3"},
            }],
        )
        executor = Mock()
        executor.execute.return_value = "5"
        graph = build_graph(gateway, Mock(), executor)
        config = {"configurable": {"thread_id": "iteration-limit"}}

        result = graph.invoke({"user_message": "Calculate 2 + 3"}, config=config)

        self.assertEqual(gateway.generate.call_count, MAX_TOOL_ITERATIONS)
        self.assertEqual(executor.execute.call_count, MAX_TOOL_ITERATIONS)
        self.assertEqual(result["iteration_count"], MAX_TOOL_ITERATIONS)
        self.assertIn("iteration limit", result["answer"])
        self.assertEqual(result["messages"][-1]["tool_calls"], [])
        self.assertEqual(graph.get_state(config).values["answer"], result["answer"])

    def test_exhausted_budget_prevents_direct_dispatch(self):
        executor = Mock()

        with self.assertRaisesRegex(RuntimeError, "iteration limit"):
            execute_tool_node({"iteration_count": MAX_TOOL_ITERATIONS}, executor)

        executor.execute.assert_not_called()

    def test_tools_are_limited_to_the_latest_request(self):
        cases = [
            ("Why is name", []),
            ("Hello!", []),
            ("Calculate 125 * 42", ["calculator"]),
            ("What time is it?", ["current_time"]),
            ("What is today's date?", ["current_time"]),
            ("What is 125 * 42 and what time is it?",
             ["calculator", "current_time"]),
        ]
        gateway = Mock()
        gateway.generate.return_value = LLMResponse(content="Answer")

        for question, expected_tools in cases:
            with self.subTest(question=question):
                agent_llm_node(
                    {"messages": [
                        {"role": "user", "content": "Calculate 2 * 3"},
                        {"role": "assistant", "content": "6"},
                        {"role": "user", "content": question},
                    ]},
                    gateway,
                )

                tools = gateway.generate.call_args.kwargs["tools"]
                self.assertEqual(
                    [tool["function"]["name"] for tool in tools or []],
                    expected_tools,
                )

    def test_chat_messages_include_system_prompt(self):
        gateway = Mock()
        gateway.generate.return_value = LLMResponse(content="Hello!")
        messages = [{"role": "user", "content": "Hello!"}]

        result = agent_llm_node(
            {"messages": messages, "user_message": "Hello!"},
            gateway,
        )

        self.assertEqual(
            gateway.generate.call_args.kwargs["messages"],
            [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *messages,
            ],
        )
        self.assertEqual(messages, [{"role": "user", "content": "Hello!"}])

        agent_llm_node(result, gateway)
        self.assertEqual(
            sum(
                message["role"] == "system"
                for message in gateway.generate.call_args.kwargs["messages"]
            ),
            1,
        )

    def test_chat_executes_calculator_and_time_with_named_results(self):
        gateway = Mock()
        timestamp = datetime(2026, 10, 14, 11, 5, 32)
        answer = f"125 * 42 is 5250. The current time is {timestamp.isoformat()}."
        gateway.generate.side_effect = [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "tool_name": "calculator",
                        "arguments": {"expression": "125 * 42"},
                    },
                    {"tool_name": "current_time", "arguments": {}},
                ],
            ),
            LLMResponse(content=answer),
        ]
        graph = build_graph(
            model_gateway=gateway,
            rag_service=RAGService(retriever=None, model_gateway=gateway),
            tool_executor=ToolExecutor(),
        )

        with patch("app.tools.time.datetime") as clock:
            clock.now.return_value = timestamp
            result = graph.invoke(
                {
                    "messages": [{
                        "role": "user",
                        "content": "What is 125 * 42 and what time is it?",
                    }],
                },
                config={"configurable": {"thread_id": "math-time-test"}},
            )

        self.assertEqual(result["answer"], answer)
        self.assertEqual(
            result["tool_results"],
            [
                {"tool": "calculator", "result": "5250"},
                {"tool": "current_time", "result": timestamp.isoformat()},
            ],
        )
        sent_messages = gateway.generate.call_args.kwargs["messages"]
        tool_messages = [
            Message.model_validate(message)
            for message in sent_messages
            if message["role"] == "tool"
        ]
        self.assertEqual(
            [(message.tool_name, message.content) for message in tool_messages],
            [
                ("calculator", "5250"),
                ("current_time", timestamp.isoformat()),
            ],
        )

    def test_llm_calls_calculator_then_finalizes(self):
        gateway = FakeModelGateway()
        graph = build_graph(
            model_gateway=gateway,
            rag_service=RAGService(retriever=None, model_gateway=gateway),
            tool_executor=ToolExecutor(),
        )

        result = graph.invoke(
            {"user_message": "Calculate 125 * 42"},
            config={
                "configurable": {
                    "thread_id": "tool-loop-test"
                }
            },
        )

        self.assertEqual(result["answer"], "The answer is 5250.")
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(
            gateway.calls[0]["tools"][0]["function"]["name"],
            "calculator",
        )
        self.assertEqual(result["tool_results"][0]["result"], "5250")


if __name__ == "__main__":
    unittest.main()
