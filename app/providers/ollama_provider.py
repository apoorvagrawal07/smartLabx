from ollama import Client

from app.models.llm import (
    LLMProvider,
    LLMResponse
)


class OllamaProvider(LLMProvider):

    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.client = Client(host=host or "http://localhost:11434")

    @staticmethod
    def _normalize_messages(messages):
        normalized = []

        for message in messages:
            if isinstance(message, dict):
                normalized.append(message)
            elif hasattr(message, "role") and hasattr(message, "content"):
                normalized.append({
                    "role": message.role,
                    "content": message.content,
                })
            else:
                normalized.append({
                    "role": "user",
                    "content": str(message),
                })

        return normalized

    def generate(
        self,
        messages,
        tools=None
    ):

        response = self.client.chat(
            model=self.model,
            messages=self._normalize_messages(messages),
            tools=tools,
        )

        tool_calls = []

        for call in getattr(response.message, "tool_calls", []) or []:
            if hasattr(call, "function"):
                tool_calls.append({
                    "tool_name": call.function.name,
                    "arguments": call.function.arguments,
                })
            elif isinstance(call, dict):
                tool_calls.append({
                    "tool_name": call.get("function", {}).get("name"),
                    "arguments": call.get("function", {}).get("arguments", {}),
                })

        return LLMResponse(
            content=getattr(response.message, "content", None),
            tool_calls=tool_calls,
        )