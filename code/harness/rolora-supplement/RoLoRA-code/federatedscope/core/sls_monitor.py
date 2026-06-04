"""Lightweight RoLoRA diagnostics for local supplement experiments.

The helpers are env-gated by ``SLS_MONITOR=1`` so normal reproduction runs keep
exactly the same training path unless monitoring is explicitly requested.
"""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

import torch

_MONITOR_TRUE = {"1", "true", "yes", "on"}
_CATEGORIES = ("lora_A", "lora_B", "classifier")


def monitor_enabled() -> bool:
    return os.environ.get("SLS_MONITOR", "").lower() in _MONITOR_TRUE


def category_for_name(name: str) -> str | None:
    for category in _CATEGORIES:
        if category in name:
            return category
    return None


def tensor_l2(tensor: Any) -> float:
    if tensor is None:
        return 0.0
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    return float(tensor.detach().float().norm().item())


def tensor_numel(tensor: Any) -> int:
    if tensor is None:
        return 0
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    return int(tensor.numel())


def _empty_squares() -> dict[str, float]:
    return {category: 0.0 for category in _CATEGORIES}


def _empty_counts() -> dict[str, int]:
    return {category: 0 for category in _CATEGORIES}


def clone_relevant_state(named_tensors: Iterable[tuple[str, Any]]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in named_tensors:
        if category_for_name(name) is None:
            continue
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        state[name] = tensor.detach().cpu().clone()
    return state


def state_from_mapping(mapping: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return clone_relevant_state(mapping.items())


def model_stats(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, float | int]:
    norm_sq = _empty_squares()
    counts = _empty_counts()
    trainable_counts = _empty_counts()
    total_trainable = 0
    total_params = 0
    state: dict[str, torch.Tensor] = {}

    for name, param in named_parameters:
        category = category_for_name(name)
        if category is None:
            continue
        detached = param.detach()
        norm_sq[category] += tensor_l2(detached) ** 2
        numel = tensor_numel(detached)
        counts[category] += numel
        total_params += numel
        if param.requires_grad:
            trainable_counts[category] += numel
            total_trainable += numel
        state[name] = detached

    stats: dict[str, float | int] = {
        "total_param_count": total_params,
        "total_trainable_params": total_trainable,
        "lora_product_norm": lora_product_norm(state),
    }
    for category in _CATEGORIES:
        stats[f"{category}_norm"] = math.sqrt(norm_sq[category])
        stats[f"{category}_params"] = counts[category]
        stats[f"{category}_trainable_params"] = trainable_counts[category]
    return stats


def state_norm_stats(state: Mapping[str, Any], prefix: str = "") -> dict[str, float | int]:
    norm_sq = _empty_squares()
    counts = _empty_counts()
    relevant: dict[str, Any] = {}
    for name, tensor in state.items():
        category = category_for_name(name)
        if category is None:
            continue
        norm_sq[category] += tensor_l2(tensor) ** 2
        counts[category] += tensor_numel(tensor)
        relevant[name] = tensor

    stats: dict[str, float | int] = {}
    for category in _CATEGORIES:
        stats[f"{prefix}{category}_norm"] = math.sqrt(norm_sq[category])
        stats[f"{prefix}{category}_params"] = counts[category]
    stats[f"{prefix}lora_product_norm"] = lora_product_norm(relevant)
    return stats


def diff_state_stats(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    prefix: str = "update_",
) -> dict[str, float | int]:
    diff_sq = _empty_squares()
    counts = _empty_counts()
    for name, before_tensor in before.items():
        category = category_for_name(name)
        if category is None or name not in after:
            continue
        after_tensor = after[name]
        if not torch.is_tensor(after_tensor):
            after_tensor = torch.as_tensor(after_tensor)
        delta = after_tensor.detach().cpu().float() - before_tensor.detach().cpu().float()
        diff_sq[category] += tensor_l2(delta) ** 2
        counts[category] += tensor_numel(delta)

    stats: dict[str, float | int] = {}
    for category in _CATEGORIES:
        stats[f"{prefix}{category}_norm"] = math.sqrt(diff_sq[category])
        stats[f"{prefix}{category}_params"] = counts[category]
    return stats


def lora_product_norm(state: Mapping[str, Any]) -> float:
    """Compute aggregate Frobenius norm of effective LoRA factors B@A."""
    norm_sq = 0.0
    for name, a_tensor in state.items():
        if "lora_A" not in name:
            continue
        b_name = name.replace("lora_A", "lora_B")
        b_tensor = state.get(b_name)
        if b_tensor is None:
            continue
        if not torch.is_tensor(a_tensor):
            a_tensor = torch.as_tensor(a_tensor)
        if not torch.is_tensor(b_tensor):
            b_tensor = torch.as_tensor(b_tensor)
        a = a_tensor.detach().float()
        b = b_tensor.detach().float()
        if a.ndim != 2 or b.ndim != 2 or b.shape[-1] != a.shape[0]:
            continue
        norm_sq += tensor_l2(torch.matmul(b, a)) ** 2
    return math.sqrt(norm_sq)


def client_drift_stats(
    global_state: Mapping[str, Any],
    models: Iterable[tuple[int, Mapping[str, Any]]],
) -> dict[str, float | int]:
    by_category: dict[str, list[float]] = {category: [] for category in _CATEGORIES}
    sample_sizes: list[int] = []
    for sample_size, local_state in models:
        sample_sizes.append(int(sample_size))
        diff = diff_state_stats(global_state, local_state, prefix="")
        for category in _CATEGORIES:
            by_category[category].append(float(diff.get(f"{category}_norm", 0.0)))

    stats: dict[str, float | int] = {
        "client_count": len(sample_sizes),
        "sample_size_total": sum(sample_sizes),
    }
    for category, values in by_category.items():
        if values:
            stats[f"client_delta_{category}_mean"] = statistics.fmean(values)
            stats[f"client_delta_{category}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            stats[f"client_delta_{category}_max"] = max(values)
            stats[f"client_delta_{category}_min"] = min(values)
        else:
            stats[f"client_delta_{category}_mean"] = 0.0
            stats[f"client_delta_{category}_std"] = 0.0
            stats[f"client_delta_{category}_max"] = 0.0
            stats[f"client_delta_{category}_min"] = 0.0
    return stats


def emit(kind: str, payload: Mapping[str, Any], prefix: str = "sls-monitor") -> None:
    if not monitor_enabled():
        return
    clean: dict[str, Any] = {"kind": kind}
    for key, value in payload.items():
        if isinstance(value, float):
            clean[key] = round(value, 8)
        elif isinstance(value, (str, int, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = value
    print(f"[{prefix}] {clean}", flush=True)


def wandb_log(payload: Mapping[str, Any], *, step: int, namespace: str) -> None:
    if not monitor_enabled():
        return
    try:
        import wandb
        if wandb.run is None:
            return
        clean = {
            f"{namespace}/{key}": value
            for key, value in payload.items()
            if isinstance(value, (int, float, bool))
        }
        clean[f"{namespace}/round"] = payload.get("round", step)
        if clean:
            try:
                wandb.define_metric(f"{namespace}/round")
                wandb.define_metric(f"{namespace}/*",
                                    step_metric=f"{namespace}/round")
            except Exception:
                pass
            wandb.log(clean)
    except Exception:
        # Monitoring must never affect training.
        return
