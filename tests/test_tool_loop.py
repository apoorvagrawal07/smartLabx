import unittest

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
