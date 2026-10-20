from app.models.structured_outputs import ToolCallRequest
from app.tools.registry import TOOLS


class ToolExecutor:

    def execute(
        self,
        tool_name: str,
        arguments: dict
    ):

        if tool_name not in TOOLS:
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        tool = TOOLS[tool_name]

        request = ToolCallRequest.model_validate({
            "tool_name": tool_name,
            "arguments": arguments,
        })
        validated_arguments = tool["arguments_model"].model_validate(
            request.arguments
        )

        if tool.get("side_effecting") is not False:
            raise PermissionError(
                "Side-effecting tools require a durable idempotency strategy"
            )

        function = tool["function"]

        return function(**validated_arguments.model_dump())