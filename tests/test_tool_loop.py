import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from ollama import Message

from app.graph.nodes import AGENT_SYSTEM_PROMPT, agent_llm_node
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
