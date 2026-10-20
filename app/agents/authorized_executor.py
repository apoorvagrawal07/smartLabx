
from app.agents.permissions import (
    AgentRole,
    authorize_tool,
)

class AuthorizedExecutor:
    def __init__(self, executor):
        self.executor = executor

    def execute(self, role, tool_name, arguments):
        # Fail closed: authorize before executing anything.
        authorize_tool(role, tool_name)

        # Delegate to your existing executor only after authorization.
        return self.executor.execute(
            tool_name=tool_name,
            arguments=arguments,
        )


class AuthorizedToolExecutor(AuthorizedExecutor):
    def execute(self, agent_role, tool_name, arguments):
        return super().execute(
            role=agent_role,
            tool_name=tool_name,
            arguments=arguments,
        )
