"""`MethodConfig` dataclass + PRESETS registry.

Every variant in the comparison plot is fully described by a `MethodConfig`.
Adding a new variant means adding one entry to `PRESETS` and (if it needs new
knobs) extending the dataclass; the federated loop in `rounds.run_method`
dispatches on the dataclass and does not need per-variant special cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Alternation = Literal["lora", "ffa_lora", "rolora", "centralized"]


@dataclass(frozen=True)
class MethodConfig:
    name: str
    alternation: Alternation
    lr_a: float
    lr_b: float
    init_a: str = "kaiming"
    init_b: str = "kaiming"
    prox_mu: float = 0.0
    server_momentum: float = 0.0
    server_lr: float = 1.0

    def __post_init__(self) -> None:
        if self.alternation not in ("lora", "ffa_lora", "rolora", "centralized"):
            raise ValueError(f"unknown alternation: {self.alternation}")
        if self.lr_a <= 0 or self.lr_b <= 0:
            raise ValueError(f"lr_a/lr_b must be > 0; got {self.lr_a}, {self.lr_b}")
        if self.prox_mu < 0:
            raise ValueError(f"prox_mu must be >= 0; got {self.prox_mu}")
        if not 0.0 <= self.server_momentum < 1.0:
            raise ValueError(
                f"server_momentum must be in [0, 1); got {self.server_momentum}"
            )
        if self.server_lr <= 0:
            raise ValueError(f"server_lr must be > 0; got {self.server_lr}")
        if self.server_momentum == 0.0 and self.server_lr != 1.0:
            raise ValueError(
                "server_lr != 1.0 requires server_momentum > 0 "
                "(soft-averaging without momentum is a separate concept)"
            )
        if self.init_a not in ("kaiming", "orthogonal", "zeros"):
            raise ValueError(f"unknown init_a: {self.init_a}")
        if self.init_b not in ("kaiming", "orthogonal", "zeros"):
            raise ValueError(f"unknown init_b: {self.init_b}")


# Default toy lr. Picked to match the existing `--lr 0.02` default in
# `notebooks/mnist_fig2.py`. LoRA+ variants scale lr_b relative to this.
_BASE_LR = 0.02
# λ=16 is the paper-suggested ratio for RoBERTa-scale models where a frozen W₀
# stabilises the forward and the LoRA path is a small perturbation. On the toy
# model (no W₀; B is the entire forward signal) λ=16 overshoots and oscillates;
# a λ∈{2,4} sweep at 10c×1lpc/100 rounds put λ=2 ahead (0.8481 vs 0.8153 vs
# base RoLoRA's 0.8444).
_LORA_PLUS_LAMBDA = 2.0
_DEFAULT_PROX = 0.01
_DEFAULT_MOM = 0.9
# η_s = 1 − β: amplitude-preserving FedAvgM (steady-state server step equals
# one round of plain averaging; β only smooths direction).
_DEFAULT_SERVER_LR = 1.0 - _DEFAULT_MOM

PRESETS: dict[str, MethodConfig] = {
    "base_lora": MethodConfig(
        name="LoRA",
        alternation="lora",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR,
    ),
    "base_ffa_lora": MethodConfig(
        name="FFA-LoRA",
        alternation="ffa_lora",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR,
    ),
    "base_rolora": MethodConfig(
        name="RoLoRA",
        alternation="rolora",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR,
    ),
    "rolora_plus_lr": MethodConfig(
        name=f"RoLoRA + LoRA+ (λ={int(_LORA_PLUS_LAMBDA)})",
        alternation="rolora",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR * _LORA_PLUS_LAMBDA,
    ),
    "rolora_orth_a": MethodConfig(
        name="RoLoRA + orthogonal-A init",
        alternation="rolora",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR,
        init_a="orthogonal",
    ),
    "rolora_prox": MethodConfig(
        name=f"RoLoRA + FedProx (μ={_DEFAULT_PROX})",
        alternation="rolora",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR,
        prox_mu=_DEFAULT_PROX,
    ),
    "rolora_mom": MethodConfig(
        name=f"RoLoRA + server momentum (β={_DEFAULT_MOM}, η_s={_DEFAULT_SERVER_LR})",
        alternation="rolora",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR,
        server_momentum=_DEFAULT_MOM,
        server_lr=_DEFAULT_SERVER_LR,
    ),
    "rolora_kitchen_sink": MethodConfig(
        name="RoLoRA + all (LoRA+, orth-A, FedProx, momentum)",
        alternation="rolora",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR * _LORA_PLUS_LAMBDA,
        init_a="orthogonal",
        prox_mu=_DEFAULT_PROX,
        server_momentum=_DEFAULT_MOM,
        server_lr=_DEFAULT_SERVER_LR,
    ),
    "centralized": MethodConfig(
        name="Centralized (non-federated ceiling)",
        alternation="centralized",
        lr_a=_BASE_LR,
        lr_b=_BASE_LR,
    ),
}


def preset(name: str) -> MethodConfig:
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown preset '{name}'; known: {known}")
    return PRESETS[name]
