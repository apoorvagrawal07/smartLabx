
from app.services.model_gateway import ModelGateway

gateway = ModelGateway()

response = gateway.generate(
    "A Kafka consumer can recover from a crash by",
    max_new_tokens=60,
    temperature=0.8,
)

print("Generated response:")
print(response)

assert isinstance(response, str)
assert response.strip()