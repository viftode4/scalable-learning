"""Models and adapter-factor utilities for the toy experiments.

Includes both the paper's Section 4.2 model (`PaperToyModel`) and the legacy
LoRA-on-frozen-base MLP kept only so the original `notebooks/mnist_fig2.py`
behaviour stays byte-identical.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ("lora", "ffa_lora", "rolora")

InitKind = Literal["kaiming", "orthogonal", "zeros"]


def init_factor(param: torch.Tensor, kind: InitKind) -> None:
    """Initialize an adapter factor in-place.

    ``kaiming`` matches the existing PaperToyModel default. ``orthogonal``
    is the RoLoRA-improvement variant motivated by FFA-LoRA's Appendix A.8
    ablation. ``zeros`` is the standard LoRA-on-base B-init convention,
    kept for the legacy MLP path only.
    """
    if kind == "kaiming":
        nn.init.kaiming_uniform_(param, a=5**0.5)
    elif kind == "orthogonal":
        nn.init.orthogonal_(param)
    elif kind == "zeros":
        nn.init.zeros_(param)
    else:
        raise ValueError(f"unknown init kind: {kind}")


class LoRALinear(nn.Module):
    """Linear layer with frozen base weight and a rank-``r`` LoRA correction.

    Forward: y = x W^T + (x A) B  (no bias on the LoRA term; matches the paper's setup).
    Init: A ~ Kaiming uniform (paper default), B = 0  → adapter starts at zero.
    """

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=True)
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + x @ self.A @ self.B


class MLP(nn.Module):
    """LoRA-on-frozen-base 2-layer MLP (legacy; use :class:`PaperToyModel` for §4.2)."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.fc1 = LoRALinear(784, 256, rank)
        self.fc2 = LoRALinear(256, 10, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def adapter_params(self, factor: str) -> list[nn.Parameter]:
        if factor == "A":
            return [self.fc1.A, self.fc2.A]
        if factor == "B":
            return [self.fc1.B, self.fc2.B]
        raise ValueError(factor)


class PaperToyModel(nn.Module):
    """Section 4.2, Eq. (10): f(x) = ReLU(x A B) W_out.

    A in R^{d x r}, B in R^{r x d}, W_out in R^{d x c}. W_out is fixed
    throughout training (registered as a buffer).

    Both A and B are Kaiming-initialized by default: the standard LoRA
    convention of B=0 would zero the forward (there is no additive base
    weight in this model), so no gradient signal would flow.
    """

    def __init__(
        self,
        rank: int = 16,
        in_dim: int = 784,
        num_classes: int = 10,
        *,
        init_a: InitKind = "kaiming",
        init_b: InitKind = "kaiming",
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, in_dim))
        init_factor(self.A, init_a)
        init_factor(self.B, init_b)
        w_out = torch.empty(in_dim, num_classes)
        nn.init.kaiming_uniform_(w_out, a=5**0.5)
        self.register_buffer("W_out", w_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return F.relu(x @ self.A @ self.B) @ self.W_out

    def adapter_params(self, factor: str) -> list[nn.Parameter]:
        if factor == "A":
            return [self.A]
        if factor == "B":
            return [self.B]
        raise ValueError(factor)


def adapter_params(model: nn.Module, factor: str) -> list[nn.Parameter]:
    """Module-level accessor; equivalent to ``model.adapter_params(factor)``."""
    return model.adapter_params(factor)  # type: ignore[no-any-return,attr-defined]


def set_factor_trainable(model: nn.Module, factor: str, *, trainable: bool) -> None:
    for p in adapter_params(model, factor):
        p.requires_grad = trainable
