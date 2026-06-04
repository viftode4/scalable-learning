"""Product-preserving LoRA factor gauge fixes for RoLoRA experiments.

LoRA adapters are invariant under small-rank reparameterizations:

    B @ A == (B @ R) @ A_orth

when ``A = R @ A_orth``.  This module uses that gauge freedom to keep LoRA-A
row-orthogonal after server aggregation without changing the effective adapter.
It is deliberately env-gated so reproduction baselines stay unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

_DISABLED = {"", "0", "false", "no", "off", "none", "default", "disabled"}
_ORTHOGONAL_A = {
    "orthogonal_a",
    "orthogonal-a",
    "orth_a",
    "ortho_a",
    "qr",
    "qr_a",
    "gauge_fix",
    "gauge-fixed",
}
_ENV_KEYS = (
    "SLS_LORA_GAUGE",
    "SLS_LORA_GAUGE_FIX",
    "SLS_GAUGE_FIX",
)


def normalise_lora_gauge(value: Any) -> str | None:
    if value is None:
        return None
    compact = str(value).strip().lower()
    if compact in _DISABLED:
        return None
    if compact in _ORTHOGONAL_A:
        return "orthogonal_a"
    raise ValueError(f"unknown SLS LoRA gauge fix: {value!r}")


def lora_gauge_from_env(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    for key in _ENV_KEYS:
        if key in env:
            return normalise_lora_gauge(env.get(key))
    return None


def iter_lora_ab_pairs(
    state: Mapping[str, Any],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for a_key in sorted(state):
        if "lora_A" not in a_key:
            continue
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key in state:
            pairs.append((a_key, b_key))
    return pairs


def _can_gauge(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        torch.is_floating_point(a)
        and torch.is_floating_point(b)
        and a.ndim == 2
        and b.ndim == 2
        and a.shape[0] <= a.shape[1]
        and b.shape[1] == a.shape[0]
    )


def apply_orthogonal_a_gauge(
    state: MutableMapping[str, Any],
) -> dict[str, float | int | str]:
    """Row-orthogonalize every LoRA-A while preserving every ``B @ A`` product.

    For each LoRA pair, compute ``A.T = Q R``.  Then ``A = R.T Q.T`` and:

    ``B @ A = (B @ R.T) @ Q.T``.

    The updated state stores ``A <- Q.T`` and ``B <- B @ R.T``.
    """
    pair_count = 0
    skipped_count = 0
    max_product_rel_error = 0.0
    max_a_gram_error = 0.0

    for a_key, b_key in iter_lora_ab_pairs(state):
        a_raw = state[a_key]
        b_raw = state[b_key]
        if not torch.is_tensor(a_raw) or not torch.is_tensor(b_raw):
            skipped_count += 1
            continue
        if not _can_gauge(a_raw, b_raw):
            skipped_count += 1
            continue

        # CPU float32 QR is robust across local laptops, login nodes and MPS.
        a = a_raw.detach().float().cpu()
        b = b_raw.detach().float().cpu()
        product_before = b @ a

        q, r = torch.linalg.qr(a.T, mode="reduced")
        a_new = q.T.contiguous()
        b_new = (b @ r.T).contiguous()

        product_after = b_new @ a_new
        denom = product_before.norm().clamp_min(1e-12)
        rel_error = float((product_after - product_before).norm().item() / denom.item())
        gram = a_new @ a_new.T
        eye = torch.eye(gram.shape[0], dtype=gram.dtype)
        gram_error = float((gram - eye).abs().max().item())

        state[a_key] = a_new.to(device=a_raw.device, dtype=a_raw.dtype)
        state[b_key] = b_new.to(device=b_raw.device, dtype=b_raw.dtype)

        pair_count += 1
        max_product_rel_error = max(max_product_rel_error, rel_error)
        max_a_gram_error = max(max_a_gram_error, gram_error)

    return {
        "gauge_mode": "orthogonal_a",
        "gauge_pair_count": pair_count,
        "gauge_skipped_count": skipped_count,
        "gauge_max_product_rel_error": max_product_rel_error,
        "gauge_max_a_gram_error": max_a_gram_error,
    }


def apply_lora_gauge_from_env(
    state: MutableMapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, float | int | str]:
    mode = lora_gauge_from_env(environ)
    if mode is None:
        return {}
    if mode == "orthogonal_a":
        return apply_orthogonal_a_gauge(state)
    raise ValueError(f"unsupported SLS LoRA gauge mode: {mode!r}")
