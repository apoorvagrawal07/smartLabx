
from app.rag.answer_generator import generate_answer


def test_answer_includes_source_reference():
    documents = [
        {
            "id": "kafka_consumer_failure",
            "content": (
                "When a consumer crashes, another consumer "
                "in the group may take over its partitions."
            ),
            "source": "kafka.md",
        }
    ]

    def fake_llm(prompt: str) -> str:
        assert "[S1]" in prompt
        return "Another consumer may take over its partitions [S1]."

    result = generate_answer(
        "What happens when a consumer crashes?",
        documents,
        fake_llm,
    )

    assert "[S1]" in result["answer"]
    assert result["sources"][0]["id"] == "S1"
    assert result["sources"][0]["source"] == "kafka.md"


def test_empty_context_returns_safe_answer():
    result = generate_answer(
        "What happens when a consumer crashes?",
        [],
        lambda prompt: "This should not be called",
    )

    assert "enough retrieved information" in result["answer"]
    assert result["sources"] == []
