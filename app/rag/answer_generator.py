
# app/rag/answer_generator.py

from typing import Callable


def build_context(documents: list[dict]) -> str:
    """Assign stable source IDs to retrieved documents."""
    sections = []

    for index, document in enumerate(documents, start=1):
        source_id = f"S{index}"
        content = document["content"]
        metadata = document.get("metadata", {})
        source = metadata.get("source", document.get("source", "unknown"))

        sections.append(
            f"[{source_id}]\n"
            f"Source: {source}\n"
            f"Content: {content}"
        )

    return "\n\n".join(sections)


def build_prompt(query: str, documents: list[dict]) -> str:
    context = build_context(documents)

    return f"""
You are a precise technical assistant.

Answer the question using ONLY the supplied context.

Rules:
1. Cite supporting claims using source IDs, e.g. [S1] or [S2].
2. Never invent sources, facts, or citations.
3. If the context does not contain enough information, say so.
4. Distinguish facts explicitly stated in the context from inferences.
5. Be concise and technically accurate.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()


def generate_answer(
    query: str,
    documents: list[dict],
    llm: Callable[[str], str],
) -> dict:
    if not documents:
        return {
            "answer": "I don't have enough retrieved information to answer that.",
            "sources": [],
        }

    prompt = build_prompt(query, documents)
    answer = llm(prompt)

    sources = []
    for index, document in enumerate(documents, start=1):
        sources.append({
            "id": f"S{index}",
            "source": document.get(
                "metadata", {}
            ).get("source", document.get("source", "unknown")),
            "content": document["content"],
        })

    return {
        "answer": answer,
        "sources": sources,
    }
