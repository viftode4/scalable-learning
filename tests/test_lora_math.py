"""The aggregation trap that RoLoRA fixes.

This is the canonical reference test: averaging A and B separately is NOT the same
as averaging the products. Asserts the inequality with tiny tensors and then asserts
that when A is shared across clients (FFA-LoRA / RoLoRA odd round) the equality holds.
"""

from __future__ import annotations

import torch


def test_naive_lora_average_is_biased() -> None:
    torch.manual_seed(0)
    num_clients = 4
    in_dim, r, out_dim = 8, 2, 4
    a_factors = [torch.randn(in_dim, r) for _ in range(num_clients)]
    b_factors = [torch.randn(r, out_dim) for _ in range(num_clients)]

    correct = sum(a_factor @ b_factor for a_factor, b_factor in zip(a_factors, b_factors, strict=True)) / num_clients
    naive = (sum(a_factors) / num_clients) @ (sum(b_factors) / num_clients)

    diff = (correct - naive).abs().max().item()
    assert diff > 1e-3, f"expected the bias to be observable; got {diff}"


def test_shared_a_makes_aggregation_exact() -> None:
    """When A is identical across clients (the RoLoRA invariant on odd rounds), the
    naive separate average equals the average of products."""
    torch.manual_seed(0)
    num_clients = 4
    in_dim, r, out_dim = 8, 2, 4
    a_shared = torch.randn(in_dim, r)
    b_factors = [torch.randn(r, out_dim) for _ in range(num_clients)]

    correct = sum(a_shared @ b_factor for b_factor in b_factors) / num_clients
    naive = a_shared @ (sum(b_factors) / num_clients)

    assert torch.allclose(correct, naive, atol=1e-6)
