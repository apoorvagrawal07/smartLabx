
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RouteDecision(BaseModel):
    intent: Literal[
        "direct",
        "rag",
        "tool",
        "clarification",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_retrieval: bool
    reason: str = Field(max_length=300)


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal[
        "calculator",
        "current_time",
        "knowledge_search",
    ]
    arguments: dict[str, str | int | float | bool | None]


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expression: str = Field(min_length=1)


class CurrentTimeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class StructuredAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    value: str = Field(min_length=1, max_length=1000)

    @field_validator("value")
    @classmethod
    def reject_blank_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Memory cannot be blank")
        return value
