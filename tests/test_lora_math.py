"""The aggregation trap that RoLoRA fixes.

This is the canonical reference test: averaging A and B separately is NOT the same
as averaging the products. Asserts the inequality with tiny tensors and then asserts
that when A is shared across clients (FFA-LoRA / RoLoRA odd round) the equality holds.
"""

from __future__ import annotations

import torch


def test_naive_lora_average_is_biased() -> None:
    torch.manual_seed(0)
    N = 4
    in_dim, r, out_dim = 8, 2, 4
    As = [torch.randn(in_dim, r) for _ in range(N)]
    Bs = [torch.randn(r, out_dim) for _ in range(N)]

    correct = sum(A @ B for A, B in zip(As, Bs, strict=True)) / N
    naive = (sum(As) / N) @ (sum(Bs) / N)

    diff = (correct - naive).abs().max().item()
    assert diff > 1e-3, f"expected the bias to be observable; got {diff}"


def test_shared_a_makes_aggregation_exact() -> None:
    """When A is identical across clients (the RoLoRA invariant on odd rounds), the
    naive separate average equals the average of products."""
    torch.manual_seed(0)
    N = 4
    in_dim, r, out_dim = 8, 2, 4
    A_shared = torch.randn(in_dim, r)
    Bs = [torch.randn(r, out_dim) for _ in range(N)]

    correct = sum(A_shared @ B for B in Bs) / N
    naive = A_shared @ (sum(Bs) / N)

    assert torch.allclose(correct, naive, atol=1e-6)
