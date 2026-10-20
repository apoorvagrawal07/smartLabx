
def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    evidence = []
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"

        evidence.append({
            "id": source_id,
            "source": chunk.get("source", "unknown"),
            "chunk_id": chunk.get("id"),
            "content": chunk["content"],
        })

        context_parts.append(
            f"[{source_id}]\n"
            f"Source: {chunk.get('source', 'unknown')}\n"
            f"Content: {chunk['content']}"
        )

    return "\n\n".join(context_parts), evidence
