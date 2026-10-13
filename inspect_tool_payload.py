from app.services.model_gateway import ModelGateway
from app.tools.registry import get_tool_schemas

m = ModelGateway()
resp = m.generate(
    'llama',
    [{'role': 'user', 'content': 'what is 135*7?'}],
    tools=get_tool_schemas(),
)
print(type(resp))
print(resp)
print('tool_calls:', getattr(resp, 'tool_calls', None))
print('content:', getattr(resp, 'content', None))
