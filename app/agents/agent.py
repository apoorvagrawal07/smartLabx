import re

from app.models.llm import LLMMessage
from app.agents.state import AgentState
from app.agents.router import AgentRouter


class Agent:

    def __init__(
        self,
        rag_service,
        tool_executor,
        model_gateway
    ):
        self.rag_service = rag_service
        self.tool_executor = tool_executor
        self.model_gateway = model_gateway

        self.router = AgentRouter()

    def run(self, message: str):

        state = AgentState(message)

        # Step 1: Decide route
        state.route = self.router.route(message)

        # Step 2: Execute route
        if state.route == "rag":
            return self._run_rag(state)

        if state.route == "tool":
            return self._run_tool(state)

        return self._run_direct(state)

    def _run_direct(self, state):

        result = self.model_gateway.generate(
            model="llama",
            messages=[
                LLMMessage(
                    role="user",
                    content=state.user_message
                )
            ]
        )

        state.answer = result.content if hasattr(result, "content") else result

        return state

    def _run_rag(self, state):

        result = self.rag_service.answer(
            state.user_message
        )

        state.answer = result.content if hasattr(result, "content") else result

        return state

    def _run_tool(self, state):

        result = self.tool_executor.execute(
            tool_name="calculator",
            arguments={
                "expression": self._extract_expression(
                    state.user_message
                )
            }
        )

        state.tool_results.append({
            "tool": "calculator",
            "result": result
        })

        state.answer = result

        return state

    @staticmethod
    def _extract_expression(message: str) -> str:
        expression = re.sub(
            r"^\s*(?:calculate|compute)\s+",
            "",
            message,
            flags=re.IGNORECASE
        )
        return expression.rstrip("?.!")