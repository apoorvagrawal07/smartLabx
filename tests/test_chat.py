import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import main
from app.graph.workflow import build_graph
from app.models.llm import LLMResponse
from app.tools.tool_executor import ToolExecutor


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.gateway = Mock()
        self.gateway.generate.return_value = LLMResponse(content="Hello!")
        self.rag_service = Mock()
        self.rag_service.answer.return_value = "Kafka uses partitions."
        self.workflow = build_graph(
            model_gateway=self.gateway,
            rag_service=self.rag_service,
            tool_executor=ToolExecutor(),
        )
        workflow_patch = patch.object(main, "workflow", self.workflow)
        workflow_patch.start()
        self.addCleanup(workflow_patch.stop)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)
        self.config = {"configurable": {"thread_id": "chat-test"}}

    def send(self, message, conversation_id="chat-test"):
        response = self.client.post(
            "/chat",
            json={"message": message, "conversation_id": conversation_id},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_same_conversation_preserves_history(self):
        self.send("My name is Apoorva.")
        previous = self.workflow.get_state(self.config).values["messages"]
        self.gateway.generate.return_value = LLMResponse(
            content="Your name is Apoorva."
        )

        response = self.send("What is my name?")

        self.assertEqual(
            response,
            {
                "answer": "Your name is Apoorva.",
                "conversation_id": "chat-test",
            },
        )
        self.assertEqual(
            self.gateway.generate.call_args.kwargs["messages"],
            previous + [{"role": "user", "content": "What is my name?"}],
        )
        state = self.workflow.get_state(self.config).values
        self.assertEqual(state["user_message"], "What is my name?")
        self.assertEqual(
            state["messages"][-1]["content"], "Your name is Apoorva."
        )

    def test_new_conversation_has_separate_history(self):
        self.send("My name is Apoorva.")
        previous = self.workflow.get_state(self.config).values["messages"]

        response = self.send("Hello!", conversation_id="new-chat")

        self.assertEqual(response["conversation_id"], "new-chat")
        sent_messages = self.gateway.generate.call_args.kwargs["messages"]
        self.assertEqual(
            [message for message in sent_messages if message["role"] != "system"],
            [{"role": "user", "content": "Hello!"}],
        )
        self.assertEqual(
            self.workflow.get_state(self.config).values["messages"], previous
        )

    def test_router_uses_latest_message(self):
        self.send("Explain Kafka partitions.")
        self.gateway.generate.assert_not_called()

        self.send("Hello!")

        self.rag_service.answer.assert_called_once_with(
            "Explain Kafka partitions."
        )
        self.gateway.generate.assert_called_once()
        state = self.workflow.get_state(self.config).values
        self.assertEqual(state["route"], "llm")
        self.assertEqual(state["user_message"], "Hello!")

    def test_new_turn_resets_tool_loop_budget(self):
        self.send("Hello!")
        self.workflow.update_state(self.config, {"iteration_count": 5})
        self.gateway.generate.side_effect = [
            LLMResponse(
                content="",
                tool_calls=[{
                    "tool_name": "calculator",
                    "arguments": {"expression": "125 * 42"},
                }],
            ),
            LLMResponse(content="5250"),
        ]

        response = self.send("Calculate 125 * 42.")

        self.assertEqual(response["answer"], "5250")
        state = self.workflow.get_state(self.config).values
        self.assertEqual(state["iteration_count"], 1)
        self.assertEqual(
            state["tool_results"], [{"tool": "calculator", "result": "5250"}]
        )


if __name__ == "__main__":
    unittest.main()
