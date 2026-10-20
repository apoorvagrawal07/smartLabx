
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # One projection produces Q, K, and V.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        # True means the position is blocked.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        attended = weights @ v

        # Join the attention heads.
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        return self.output_projection(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm architecture.
        x = x + self.dropout(
            self.attention(self.norm1(x))
        )

        x = x + self.dropout(
            self.mlp(self.norm2(x))
        )

        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_model = 64

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    block = TransformerBlock(
        d_model=d_model,
        num_heads=4,
    )

    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in block.parameters()
            if p.requires_grad),
    )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
    assert torch.count_nonzero(block.attention.qkv.weight.grad) > 0

    block.eval()
    prefix_length = seq_len // 2
    with torch.no_grad():
        reference = block(x)
        changed_input = x.detach().clone()
        changed_input[:, prefix_length:] = torch.randn_like(
            changed_input[:, prefix_length:]
        )
        changed_output = block(changed_input)

    assert torch.allclose(
        reference[:, :prefix_length],
        changed_output[:, :prefix_length],
        atol=1e-6,
    )
    assert not torch.allclose(
        reference[:, prefix_length:],
        changed_output[:, prefix_length:],
    )

    print("Forward and backward passes succeeded!")
    print("Causal masking verified: future tokens do not affect earlier outputs.")
