
import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    """
    q: [batch, heads, sequence, head_dim]
    k: [batch, heads, sequence, head_dim]
    v: [batch, heads, sequence, head_dim]
    """
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))

    weights = F.softmax(scores, dim=-1)
    output = weights @ v

    return output, weights

def causal_attention(q, k, v):
    sequence_length = q.size(-2)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))

    # Future positions are not visible.
    mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=q.device,
        ),
        diagonal=1,
    )

    scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights



if __name__ == "__main__":
    torch.manual_seed(42)

    batch = 1
    heads = 2
    sequence_length = 4
    head_dim = 4

    q = torch.randn(batch, heads, sequence_length, head_dim)
    k = torch.randn(batch, heads, sequence_length, head_dim)
    v = torch.randn(batch, heads, sequence_length, head_dim)

    output, weights = causal_attention(q, k, v)

    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
    print("Weights for first head:")
    print(weights[0, 0])

    assert output.shape == (1, 2, 4, 4)
    assert weights.shape == (1, 2, 4, 4)
    assert torch.allclose(
        weights.triu(diagonal=1),
        torch.zeros_like(weights).triu(diagonal=1),
    )

    print("Causal masking works!")
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
        atol=1e-6,
    )

    print("All checks passed!")
