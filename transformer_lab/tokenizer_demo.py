
from transformers import AutoTokenizer

MODEL_NAME = "distilbert/distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

examples = [
    "the server handles requests",
    "unbelievable",
    "Kafka consumer crashed unexpectedly",
]

for text in examples:
    token_ids = tokenizer.encode(text)

    print(f"\nText:      {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens:    {tokenizer.convert_ids_to_tokens(token_ids)}")
    print(f"Decoded:   {tokenizer.decode(token_ids)}")
    print(f"Token count: {len(token_ids)}")
