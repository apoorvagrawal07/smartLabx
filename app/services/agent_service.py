 

class AgentService:

    def run(self, user_message):

        messages = [
            {
                "role": "user",
                "content": user_message
            }
        ]

        while True:

            response = llm.generate(
                messages
            )

            if response.is_final:
                return response.answer

            tool_result = execute_tool(
                response.tool_name,
                response.arguments
            )

            messages.append(
                response.tool_call
            )

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result
                }
            )