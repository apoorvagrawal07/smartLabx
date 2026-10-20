from app.models.structured_outputs import CalculatorArguments, CurrentTimeArguments
from app.tools.calculator import calculate
from app.tools.time import get_current_time


TOOLS = {
    "calculator": {
        "side_effecting": False,
        "function": calculate,
        "arguments_model": CalculatorArguments,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Calculate a mathematical expression."
                ),
                "parameters": CalculatorArguments.model_json_schema(),
            }
        }
    },

    "current_time": {
        "side_effecting": False,
        "function": get_current_time,
        "arguments_model": CurrentTimeArguments,
        "schema": {
            "type": "function",
            "function": {
                "name": "current_time",
                "description": (
                    "Get the current date and time."
                ),
                "parameters": CurrentTimeArguments.model_json_schema(),
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