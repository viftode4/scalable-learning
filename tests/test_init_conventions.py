"""LoRA initialisation convention asserts.

FFA-LoRA *requires* A non-zero and B zero. If you accidentally make B non-zero, the
adapter starts away from the identity and FFA-LoRA's randomness amplifies — the
deep-research plan flags this as a real bug source.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import mnist_fig2  # noqa: E402


def test_lora_b_is_zero_at_init() -> None:
    model = mnist_fig2.MLP(rank=1)
    for b_param in model.adapter_params("B"):
        assert torch.equal(b_param, torch.zeros_like(b_param)), "FFA-LoRA convention: B must be 0"


def test_lora_a_is_non_zero_at_init() -> None:
    torch.manual_seed(0)
    model = mnist_fig2.MLP(rank=1)
    for a_param in model.adapter_params("A"):
        assert a_param.abs().max().item() > 0.0, "A must be non-zero (Kaiming uniform)"
