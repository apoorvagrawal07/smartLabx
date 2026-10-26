import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from fastapi.testclient import TestClient

from app import main
from app.graph.nodes import DIRECT_SYSTEM_PROMPT
from app.graph.workflow import build_graph
from app.models.llm import LLMResponse
from app.services.conversation_access import ConversationAccess
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
        ownership_patch = patch.object(main, "conversation_access", ConversationAccess())
        ownership_patch.start()
        self.addCleanup(ownership_patch.stop)
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

    def test_request_context_headers_remain_intact(self):
        request_ids = []
        for message in ("Hello!", "Hello again!"):
            response = self.client.post(
                "/chat",
                json={"message": message, "conversation_id": "chat-test"},
            )

            self.assertEqual(response.status_code, 200, response.text)
            request_id = response.headers["X-Request-ID"]
            self.assertEqual(str(UUID(request_id)), request_id)
            self.assertGreaterEqual(float(response.headers["X-Process-Time"]), 0)
            request_ids.append(request_id)

        self.assertNotEqual(request_ids[0], request_ids[1])

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
                "sources": [],
            },
        )
        self.assertEqual(
            self.gateway.generate.call_args.kwargs["messages"],
            [{"role": "system", "content": DIRECT_SYSTEM_PROMPT},
             *previous, {"role": "user", "content": "What is my name?"}],
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
        self.assertEqual(
            self.gateway.generate.call_args.kwargs["messages"],
            [{"role": "system", "content": DIRECT_SYSTEM_PROMPT},
             {"role": "user", "content": "Hello!"}],
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

    def test_rag_accepts_plain_model_and_cited_answers(self):
        answer = "Kafka reassigns the affected partitions."
        sources = [{"source": "knowledge/kafka.md"}]
        cases = [
            (answer, []),
            (LLMResponse(content=answer), []),
            ({"answer": answer, "sources": sources}, sources),
        ]

        for result, expected_sources in cases:
            with self.subTest(result=result):
                self.rag_service.answer.return_value = result

                response = self.send("What happens when a Kafka consumer crashes?")

                self.assertEqual(response["answer"], answer)
                self.assertEqual(response["sources"], expected_sources)
                state = self.workflow.get_state(self.config).values
                self.assertEqual(state["sources"], expected_sources)
                self.assertEqual(
                    state["messages"][-1],
                    {"role": "assistant", "content": answer},
                )

    def test_chat_after_rag_receives_complete_exchanges(self):
        question = "Why do consumers commit offsets?"
        answer = "Consumers commit offsets to resume processing."
        self.rag_service.answer.return_value = LLMResponse(content=answer)
        self.send(question)
        self.send(question)

        self.send("Why is name")

        self.assertEqual(
            self.gateway.generate.call_args.kwargs["messages"],
            [{"role": "system", "content": DIRECT_SYSTEM_PROMPT},
             {"role": "user", "content": question},
             {"role": "assistant", "content": answer},
             {"role": "user", "content": question},
             {"role": "assistant", "content": answer},
             {"role": "user", "content": "Why is name"}],
        )

    def test_direct_route_bypasses_tool_reasoning_node(self):
        with patch("app.graph.workflow.agent_llm_node") as tool_reasoning:
            response = self.send("Explain what an API gateway is.")

        tool_reasoning.assert_not_called()
        self.gateway.generate.assert_called_once_with(
            model="llama",
            messages=[
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": "Explain what an API gateway is."},
            ],
        )
        self.assertEqual(set(response), {"answer", "conversation_id", "sources"})

    def test_time_question_still_uses_tool_loop(self):
        self.gateway.generate.side_effect = [
            LLMResponse(content="", tool_calls=[{
                "tool_name": "current_time", "arguments": {},
            }]),
            LLMResponse(content="The current time was retrieved."),
        ]

        with patch("app.tools.time.datetime") as clock:
            clock.now.return_value.isoformat.return_value = "2026-10-17T12:00:00"
            response = self.send("What time is it?")

        self.assertEqual(response["answer"], "The current time was retrieved.")
        self.assertEqual(
            self.workflow.get_state(self.config).values["route_decision"]["intent"],
            "tool",
        )
        self.assertEqual(
            self.workflow.get_state(self.config).values["tool_results"],
            [{"tool": "current_time", "result": "2026-10-17T12:00:00"}],
        )

    def test_direct_answer_clears_previous_rag_sources(self):
        self.rag_service.answer.return_value = {
            "answer": "Kafka uses partitions [S1].",
            "sources": [{"id": "S1", "source": "knowledge/kafka.md"}],
        }
        self.send("Explain Kafka partitions.")

        response = self.send("Hello!")

        self.assertEqual(response["sources"], [])
        self.assertEqual(self.workflow.get_state(self.config).values["sources"], [])

    def test_uncited_rag_answer_clears_previous_sources(self):
        self.rag_service.answer.return_value = {
            "answer": "Kafka uses partitions.",
            "sources": [{"source": "knowledge/kafka.md"}],
        }
        self.send("Explain Kafka partitions.")
        self.rag_service.answer.return_value = LLMResponse(
            content="Kafka reassigns the affected partitions."
        )

        self.send("What happens when a Kafka consumer crashes?")

        state = self.workflow.get_state(self.config).values
        self.assertEqual(state["sources"], [])

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
