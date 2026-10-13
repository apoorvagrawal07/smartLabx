from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass, field
from typing import Any

class LLMMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        messages,
        tools=None
    ):
        pass

@dataclass
class LLMResponse:

    content: str | None = None

    tool_calls: list[dict[str, Any]] = field(
        default_factory=list
    )