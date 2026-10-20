
from app.rag.context_builder import build_context


class CitedRAGService:

    def __init__(self, rag_pipeline, answer_service):
        self.rag_pipeline = rag_pipeline
        self.answer_service = answer_service

    def answer(self, question: str) -> dict:
        chunks = self.rag_pipeline.retrieve(
            question,
            candidate_limit=20,
            final_limit=5,
        )

        if not chunks:
            return {
                "answer": (
                    "I don't have enough information "
                    "in the knowledge base to answer that reliably."
                ),
                "sources": [],
            }

        context, evidence = build_context(chunks)

        answer = self.answer_service.answer(
            question=question,
            context=context,
        )

        return {
            "answer": answer,
            "sources": evidence,
        }
