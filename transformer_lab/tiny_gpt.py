import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import TransformerBlock

torch.manual_seed(42)

# A tiny training corpus. Repetition helps the model learn patterns.
text = (
    "the server handles requests. "
    "the database stores records. "
    "the api returns responses. "
    "the worker processes tasks. "
    "the server handles requests. "
) * 100

# Character-level tokenizer
vocab = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join(itos[int(i)] for i in ids)


# Model configuration
SEQ_LEN = 64
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "transformer_lab/tiny_gpt_best.pt"



def get_batch(batch_size=16, split="train"):
    source = train_data if split == "train" else val_data

    starts = torch.randint(
        0, len(source) - SEQ_LEN - 1, (batch_size,)
    )

    x = torch.stack([source[s:s + SEQ_LEN] for s in starts])
    y = torch.stack([
        source[s + 1:s + SEQ_LEN + 1] for s in starts
    ])

    return x.to(DEVICE), y.to(DEVICE)



class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(SEQ_LEN, D_MODEL)

        self.blocks = nn.ModuleList([
            TransformerBlock(D_MODEL, NUM_HEADS)
            for _ in range(NUM_LAYERS)
        ])

        self.final_norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature: float = 1.0):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")

        self.eval()

        for _ in range(max_new_tokens):
            # Keep only the context the model can process.
            context = idx[:, -SEQ_LEN:]
            logits, _ = self(context)

            # Predict from the final position.
            next_token_logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(next_token_logits, dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

        
    @torch.no_grad()
    def estimate_loss(model, batches=10):
        model.eval()
        results = {}

        for split in ("train", "val"):
            losses = []

            for _ in range(batches):
                x, y = get_batch(split=split)
                _, loss = model(x, y)
                losses.append(loss.item())

            results[split] = sum(losses) / len(losses)

        model.train()
        return results



model = TinyGPT(len(vocab)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Device:", DEVICE)
print("Vocabulary size:", len(vocab))
print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))
print("Parameters:", sum(p.numel() for p in model.parameters()))

# Train the model to predict the next character.
model.train()

best_val_loss = float("inf")

for step in range(301):
    model.train()

    x, y = get_batch(split="train")
    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Prevent unusually large gradients from destabilizing training.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    if step % 50 == 0:
        metrics = model.estimate_loss()
        print(
            f"Step {step:3d} | "
            f"train: {metrics['train']:.4f} | "
            f"val: {metrics['val']:.4f}"
        )

        if metrics["val"] < best_val_loss:
            best_val_loss = metrics["val"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "validation_loss": best_val_loss,
                    "vocab": vocab,
                    "config": {
                        "seq_len": SEQ_LEN,
                        "d_model": D_MODEL,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                    },
                },
                CHECKPOINT_PATH,
            )


checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
print(
    f"\nLoaded best checkpoint: step {checkpoint['step']} | "
    f"val: {checkpoint['validation_loss']:.4f}"
)

# Generate text from a starting prompt.
prompt = "the "
context = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)

for temperature in (0.5, 0.8, 1.0, 1.3):
    torch.manual_seed(42)
    generated = model.generate(context, max_new_tokens=200, temperature=temperature)
    assert generated.shape == (1, context.size(1) + 200)
    print(f"\nTemperature {temperature:.1f}:")
    print(decode(generated[0].tolist()))
