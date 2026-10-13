from pydantic import BaseModel
from typing import Any


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]