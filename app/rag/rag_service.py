from app.models.llm import LLMMessage


class RAGService:

    def __init__(
        self,
        retriever,
        model_gateway
    ):

        self.retriever = retriever
        self.model_gateway = model_gateway

    def answer(
        self,
        question: str
    ):

        results = self.retriever.retrieve(
            question,
            limit=3
        )

        context = "\n\n".join(
            content
            for _, content, _ in results
        )

        messages = [

            LLMMessage(
                role="system",
                content="""
You are a helpful AI assistant.

Answer the user's question using
the provided context.

If the context does not contain
enough information, say that you
don't have enough information.
"""
            ),

            LLMMessage(
                role="user",
                content=f"""
Context:

{context}

Question:

{question}
"""
            )
        ]

        return self.model_gateway.generate(
            model="llama",
            messages=messages
        )