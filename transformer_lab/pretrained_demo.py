
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilbert/distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

prompt = "A Kafka consumer can recover from a crash by"

inputs = tokenizer(prompt, return_tensors="pt")

print("Prompt:", prompt)
print("Parameters:", model.num_parameters())
print("Vocabulary size:", len(tokenizer))
print("Layers / heads / width:", model.config.n_layer, model.config.n_head, model.config.n_embd)
print("Context window:", model.config.n_positions)

with torch.no_grad():
    for temperature in (0.3, 0.8, 1.2):
        torch.manual_seed(42)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        continuation_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        print(f"\nTemperature {temperature:.1f}:")
        print(tokenizer.decode(continuation_ids, skip_special_tokens=True))
