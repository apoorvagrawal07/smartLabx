from app.tools.calculator import calculate
from app.tools.time import get_current_time


TOOLS = {
    "calculator": {
        "function": calculate,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Calculate a mathematical expression."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": (
                                "Mathematical expression "
                                "to calculate."
                            )
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    },

    "current_time": {
        "function": get_current_time,
        "schema": {
            "type": "function",
            "function": {
                "name": "current_time",
                "description": (
                    "Get the current date and time."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    }
}

def execute_tool(
    tool_name: str,
    arguments: dict
):
    tool = TOOLS.get(tool_name)

    if tool is None:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    return tool(**arguments)

def get_tool_schemas():

    return [
        tool["schema"]
        for tool in TOOLS.values()
    ]