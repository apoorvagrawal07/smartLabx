
def evaluate_retrieval(
    retriever,
    dataset: list[dict],
    k: int = 5,
) -> dict:
    if not dataset:
        raise ValueError("Evaluation dataset cannot be empty")

    hits = []
    recalls = []

    for item in dataset:
        relevant_ids = set(item["relevant_doc_ids"])
        results = retriever.retrieve(item["question"], limit=k)

        retrieved_ids = {
            str(doc["id"]) for doc in results[:k]
        }

        relevant_found = relevant_ids & retrieved_ids

        hits.append(1.0 if relevant_found else 0.0)
        recalls.append(len(relevant_found) / len(relevant_ids))

    return {
        f"hit_at_{k}": sum(hits) / len(hits),
        f"recall_at_{k}": sum(recalls) / len(recalls),
        "questions_evaluated": len(dataset),
    }
