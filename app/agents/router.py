import re

from app.models.structured_outputs import RouteDecision


class AgentRouter:

    def decide(self, message: str) -> RouteDecision:
        route = self.route(message)
        if route == "llm" and (
            self._is_calculation(message.lower())
            or self._is_time_question(message.lower())
        ):
            route = "tool"

        return RouteDecision(
            intent="direct" if route == "llm" else route,
            confidence=1.0,
            requires_retrieval=route == "rag",
            reason="Selected by the keyword-based router.",
        )

    def route(self, message: str) -> str:

        message_lower = message.lower()

        if any(
            keyword in message_lower
            for keyword in [
                "kafka",
                "consumer",
                "partition",
                "offset",
                "documentation",
                "internal document",
                "knowledge base",
                "our docs",
            ]
        ):
            return "rag"

        return "llm"

    @staticmethod
    def _is_time_question(message: str) -> bool:
        return bool(re.search(r"\b(time|date|today|clock)\b", message))

    @staticmethod
    def _is_calculation(message: str) -> bool:
        if re.search(r"\b(calculate|compute)\b", message):
            return True

        return bool(
            re.search(
                r"\d+\s*(?:[+*/-]|plus|minus|times|multiplied\s+by|divided\s+by)\s*\d+",
                message,
            )
        )