"""Client-side training and evaluation primitives.

`local_train` is extended over the pre-refactor version with:
- ``lr_a`` / ``lr_b``: per-factor learning rates (LoRA+ asymmetric lr). When
  unset, ``lr`` is used for both → existing behaviour preserved.
- ``prox_mu`` / ``prox_anchor``: FedProx-style proximal penalty applied only to
  the factor(s) being trained this round. ``prox_mu=0`` (the default) → no-op.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _proximal_penalty(
    active_params: dict[str, list[nn.Parameter]],
    anchor: dict[str, list[torch.Tensor]],
    mu: float,
) -> torch.Tensor:
    """Compute (mu/2) * sum_factor sum_param ||p - anchor_p||^2."""
    total: torch.Tensor | None = None
    for factor, params in active_params.items():
        anchor_list = anchor.get(factor)
        if anchor_list is None:
            continue
        for p, a in zip(params, anchor_list, strict=True):
            term = (p - a).pow(2).sum()
            total = term if total is None else total + term
    if total is None:
        # No active factor with an anchor; return zero scalar so caller can
        # safely `.backward()` without a branch.
        return torch.zeros((), device=next(iter(next(iter(active_params.values())))).device)
    return 0.5 * mu * total


def local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    lr: float | None = None,
    lr_a: float | None = None,
    lr_b: float | None = None,
    device: torch.device,
    grad_clip: float = 1.0,
    steps: int | None = None,
    epochs: int | None = None,
    prox_mu: float = 0.0,
    prox_anchor: dict[str, list[torch.Tensor]] | None = None,
) -> None:
    """Run local SGD on `model`.

    Exactly one of ``steps`` or ``epochs`` must be set.

    ``lr_a`` / ``lr_b`` override ``lr`` per factor. Use either the single-``lr``
    form (legacy) or the asymmetric form, not both. The optimizer is
    constructed with separate param groups so LoRA+ asymmetric rates take
    effect with no scheduler magic.
    """
    if (steps is None) == (epochs is None):
        raise ValueError("local_train: pass exactly one of `steps` or `epochs`")

    if lr is None and (lr_a is None or lr_b is None):
        raise ValueError("local_train: pass either `lr` or both `lr_a` and `lr_b`")
    if lr is not None and (lr_a is not None or lr_b is not None):
        raise ValueError(
            "local_train: pass either `lr` (legacy) or asymmetric `lr_a`/`lr_b`, not both"
        )

    if lr is not None:
        eff_lr_a = eff_lr_b = lr
    else:
        eff_lr_a = lr_a  # type: ignore[assignment]
        eff_lr_b = lr_b  # type: ignore[assignment]

    # Build param groups by inspecting the model's adapter_params. Only
    # trainable params end up in the optimizer.
    active_params: dict[str, list[nn.Parameter]] = {}
    param_groups: list[dict] = []
    for factor, factor_lr in (("A", eff_lr_a), ("B", eff_lr_b)):
        params = [p for p in model.adapter_params(factor) if p.requires_grad]  # type: ignore[attr-defined]
        if params:
            active_params[factor] = params
            param_groups.append({"params": params, "lr": factor_lr})

    # Fall back to ALL trainable params (covers the lora-base MLP case where
    # base weights happen to be frozen but other params might exist).
    if not param_groups:
        trainable = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable, "lr": eff_lr_a}]

    trainable_flat: list[nn.Parameter] = []
    for g in param_groups:
        trainable_flat.extend(g["params"])
    opt = torch.optim.SGD(param_groups)

    use_prox = prox_mu > 0.0 and prox_anchor is not None and len(active_params) > 0

    model.train()

    def step(x: torch.Tensor, y: torch.Tensor) -> None:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        if use_prox:
            loss = loss + _proximal_penalty(active_params, prox_anchor, prox_mu)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_flat, grad_clip)
        opt.step()

    if epochs is not None:
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                step(x, y)
        return

    seen = 0
    while seen < steps:  # type: ignore[operator]
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            step(x, y)
            seen += 1
            if seen >= steps:  # type: ignore[operator]
                break


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses, correct, total = [], 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(F.cross_entropy(out, y, reduction="sum").item())
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return sum(losses) / total, correct / total


__all__ = ["evaluate", "local_train"]
