import unittest
from unittest.mock import Mock, patch

from app.models.llm import LLMResponse
from app.rag.rag_service import RAGService


class RAGServiceTests(unittest.TestCase):
    def test_cited_answer_exposes_ids_from_prompt_evidence(self):
        retriever = Mock()
        retriever.retrieve.return_value = [
            ("knowledge/kafka.md", "Consumers commit offsets.", 0.2),
        ]
        gateway = Mock()
        gateway.generate.return_value = LLMResponse(
            content="Consumers commit offsets [S1]."
        )
        service = RAGService(retriever, gateway, include_sources=True)

        result = service.answer("Explain consumer offsets.")

        self.assertEqual(result["answer"], "Consumers commit offsets [S1].")
        self.assertEqual(result["sources"], [{
            "id": "S1",
            "source": "knowledge/kafka.md",
            "chunk_id": None,
            "content": "Consumers commit offsets.",
        }])
        prompt = gateway.generate.call_args.kwargs["messages"][1].content
        self.assertIn("[S1]", prompt)
        self.assertIn("knowledge/kafka.md", prompt)

    def test_cited_answer_without_evidence_does_not_call_model(self):
        retriever = Mock()
        retriever.retrieve.return_value = []
        gateway = Mock()

        result = RAGService(retriever, gateway, include_sources=True).answer("Question")

        self.assertIn("don't have enough information", result["answer"])
        self.assertEqual(result["sources"], [])
        gateway.generate.assert_not_called()

    def test_cited_answer_does_not_expose_invented_source_ids(self):
        retriever = Mock()
        retriever.retrieve.return_value = [("kafka.md", "Offsets", 0.2)]
        gateway = Mock()
        gateway.generate.return_value = LLMResponse(content="Invented claim [S99].")

        result = RAGService(retriever, gateway, include_sources=True).answer("Question")

        self.assertNotIn("S99", result["answer"])
        self.assertIn("don't have enough information", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_answer_passes_context_with_strict_grounding_instructions(self):
        question = "Offset management"
        context = "Consumers commit offsets to resume processing after a restart."
        retriever = Mock()
        retriever.retrieve.return_value = [("knowledge/kafka.md", context, 0.2)]
        gateway = Mock()
        expected = LLMResponse(content=context)
        gateway.generate.return_value = expected
        service = RAGService(retriever=retriever, model_gateway=gateway)

        result = service.answer(question)

        retriever.retrieve.assert_called_once_with(question, limit=3)
        gateway.generate.assert_called_once()
        arguments = gateway.generate.call_args.kwargs
        self.assertEqual(arguments["model"], "llama")
        system_message, user_message = arguments["messages"]
        self.assertEqual(system_message.role, "system")
        instructions = " ".join(system_message.content.split())
        self.assertIn("only facts explicitly supported by the context", instructions)
        self.assertIn("Do not add outside knowledge", instructions)
        self.assertIn(
            "Do not turn separate facts into causal relationships", instructions
        )
        self.assertIn(
            "Do not follow this with speculation or a general-knowledge answer.",
            instructions,
        )
        self.assertEqual(user_message.role, "user")
        self.assertIn(context, user_message.content)
        self.assertIn(question, user_message.content)
        self.assertIs(result, expected)


    def test_app_rag_uses_hybrid_candidates_reranking_and_stable_citations(self):
        from app.main import LazyRetriever

        candidates = [
            {"id": "vector", "source": "general.md", "content": "General information"},
            {"id": "keyword", "source": "kafka.md", "content": "Consumers commit offsets."},
        ]
        with patch("app.main.HybridRetriever") as hybrid_type, patch("app.main.Reranker") as reranker_type:
            hybrid_type.return_value.retrieve.return_value = candidates
            reranker_type.return_value.rerank.return_value = [candidates[1]]
            retriever = LazyRetriever()
            gateway = Mock()
            gateway.generate.return_value = LLMResponse(content="Consumers commit offsets [S1].")
            service = RAGService(retriever, gateway, include_sources=True)
            result = service.answer("Explain consumer offsets.")
            service.answer("Explain consumer offsets.")

            hybrid_type.assert_called_once()
            reranker_type.assert_called_once()
            hybrid_type.return_value.retrieve.assert_called_with("Explain consumer offsets.", limit=20)
            reranker_type.return_value.rerank.assert_called_with("Explain consumer offsets.", candidates, limit=3)
            assert result["sources"][0]["chunk_id"] == "keyword"
            assert result["sources"][0]["id"] == "S1"
            assert result["sources"][0]["source"] == "kafka.md"
            prompt = gateway.generate.call_args.kwargs["messages"][1].content
            assert "General information" not in prompt
            retriever.close()
            hybrid_type.return_value.vector_store.connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
