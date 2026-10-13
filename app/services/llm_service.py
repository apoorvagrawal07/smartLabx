from app.models.llm import LLMMessage


SYSTEM_PROMPT = "You are a helpful assistant."


class LLMService:

    def __init__(
        self,
        model_gateway,
        model_router
    ):
        self.model_gateway = model_gateway
        self.model_router = model_router

    def generate(
        self,
        message: str,
        request_id: str,
        mode: str
    ) -> str:

        model = self.model_router.select_model(
            mode
        )

        messages = [
            LLMMessage(
                role="system",
                content=SYSTEM_PROMPT
            ),
            LLMMessage(
                role="user",
                content=message
            )
        ]

        return self.model_gateway.generate(
            model=model,
            messages=messages
        )