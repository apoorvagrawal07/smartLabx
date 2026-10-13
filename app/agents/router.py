import re


class AgentRouter:

    def route(self, message: str) -> str:

        message_lower = message.lower()

        if any(
            keyword in message_lower
            for keyword in [
                "kafka",
                "consumer",
                "partition",
                "offset"
            ]
        ):
            return "rag"

        return "llm"

    @staticmethod
    def _is_calculation(message: str) -> bool:
        if re.search(r"\b(calculate|compute)\b", message):
            return True

        return bool(
            re.search(r"\d+\s*[+*/-]\s*\d+", message)
        )