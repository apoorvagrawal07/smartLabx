from app.models.llm import LLMMessage
from app.providers.ollama_provider import OllamaProvider


class ModelGateway:

    def __init__(self):

        self.providers = {
            "llama": OllamaProvider(
                "llama3.2:3b"
            ),
            "phi": OllamaProvider(
                "phi4-mini"
            )
        }

    def generate(
        self,
        model: str,
        messages,
        tools=None
    ):

        try:

            provider = self.providers[model]

            return provider.generate(
                messages,
                tools=tools,
            )

        except Exception as error:

            print(
                f"Model {model} failed: {error}"
            )

            fallback = self.providers["phi"]

            return fallback.generate(
                messages,
                tools=tools,
            )