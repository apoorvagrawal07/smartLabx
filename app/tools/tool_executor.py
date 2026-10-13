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

        function = tool["function"]

        return function(**arguments)