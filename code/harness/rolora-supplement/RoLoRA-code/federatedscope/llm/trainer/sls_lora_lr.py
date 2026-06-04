"""SLS RoLoRA optimizer helpers.

Small dependency-light helpers live here so they can be unit-tested without
importing the full FederatedScope trainer stack.
"""

from __future__ import annotations

import os


def read_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def lora_lr_optimizer_target(model, base_lr: float):
    """Return model or param groups for optional LoRA A/B learning rates.

    Default behaviour is unchanged. When either ``SLS_LORA_LR_A`` or
    ``SLS_LORA_LR_B`` is set, build PyTorch optimizer param groups for currently
    trainable parameters:

    - LoRA-A parameters use ``SLS_LORA_LR_A`` if provided, else ``base_lr``.
    - LoRA-B parameters use ``SLS_LORA_LR_B`` if provided, else ``base_lr``.
    - Classifier / other trainable parameters keep ``base_lr``.

    RoLoRA's alternating freeze structure is preserved because the trainer sets
    ``requires_grad`` for the current phase before this helper runs.
    """
    lr_a = read_float_env("SLS_LORA_LR_A")
    lr_b = read_float_env("SLS_LORA_LR_B")
    if lr_a is None and lr_b is None:
        return model, None

    groups = {
        "lora_A": {"params": [], "lr": base_lr if lr_a is None else lr_a},
        "lora_B": {"params": [], "lr": base_lr if lr_b is None else lr_b},
        "base": {"params": [], "lr": base_lr},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            groups["lora_A"]["params"].append(param)
        elif "lora_B" in name:
            groups["lora_B"]["params"].append(param)
        else:
            groups["base"]["params"].append(param)

    param_groups = []
    summary = {}
    for group_name, group in groups.items():
        if group["params"]:
            param_groups.append(group)
            summary[group_name] = {
                "lr": group["lr"],
                "params": len(group["params"]),
            }
    if not param_groups:
        raise RuntimeError(
            "SLS_LORA_LR_A/B requested but no trainable parameters were found"
        )
    return param_groups, summary
