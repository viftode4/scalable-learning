"""Function-preserving LoRA basis transport for RoLoRA experiments.

In alternating federated LoRA, an A-round replaces the basis ``A_old`` with
the aggregated ``A_new`` while the coefficients ``B`` stay frozen, so the
effective adapter jumps from ``B @ A_old`` to ``B @ A_new``.  This module
re-expresses the coefficients in the new basis with the least-squares
transport

    B' = argmin ||B' @ A_new - B @ A_old||_F  =  B @ A_old @ pinv(A_new)

so the learned function survives the basis change as well as the new basis
can represent it.  The transport matrix ``A_old @ pinv(A_new)`` is r x r and
deterministic given the old/new global factors, so applying it adds no
communication.  Like the gauge fix, it is env-gated (``SLS_LORA_TRANSPORT``)
so reproduction baselines stay unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

_DISABLED = {"", "0", "false", "no", "off", "none", "default", "disabled"}
_LS = {
    "ls",
    "least_squares",
    "least-squares",
    "lstsq",
    "transport",
    "basis_transport",
    "basis-transport",
    "bt",
    "function_preserving",
    "fp",
}
_ENV_KEYS = (
    "SLS_LORA_TRANSPORT",
    "SLS_LORA_BASIS_TRANSPORT",
)
_BETA_ENV_KEY = "SLS_LORA_TRANSPORT_BETA"
# Reject transports whose r x r matrix exploded; this signals a degenerate
# or near-rank-deficient A_new where "preserving the function" would mean
# amplifying noise instead.
_MAX_TRANSPORT_NORM = 1e6


def normalise_lora_transport(value: Any) -> str | None:
    if value is None:
        return None
    compact = str(value).strip().lower()
    if compact in _DISABLED:
        return None
    if compact in _LS:
        return "ls"
    raise ValueError(f"unknown SLS LoRA transport mode: {value!r}")


def lora_transport_from_env(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    env = os.environ if environ is None else environ
    for key in _ENV_KEYS:
        if key in env:
            return normalise_lora_transport(env.get(key))
    return None


def transport_beta_from_env(
    environ: Mapping[str, str] | None = None,
) -> float:
    env = os.environ if environ is None else environ
    raw = env.get(_BETA_ENV_KEY)
    if raw is None or str(raw).strip() == "":
        return 1.0
    beta = float(raw)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(
            f"{_BETA_ENV_KEY} must be in [0, 1]; got {raw!r}"
        )
    return beta


def iter_lora_ab_pairs(state: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for a_key in sorted(state):
        if "lora_A" not in a_key:
            continue
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key in state:
            pairs.append((a_key, b_key))
    return pairs


def _is_lora_pair(a: Any, b: Any) -> bool:
    return (
        torch.is_tensor(a)
        and torch.is_tensor(b)
        and torch.is_floating_point(a)
        and torch.is_floating_point(b)
        and a.ndim == 2
        and b.ndim == 2
        and a.shape[0] <= a.shape[1]
        and b.shape[1] == a.shape[0]
    )


def compute_transport_matrix(
    a_old: torch.Tensor,
    a_new: torch.Tensor,
) -> torch.Tensor | None:
    """Least-squares ``M`` with ``M @ A_new ~= A_old``; ``None`` on failure."""
    try:
        if int(torch.linalg.matrix_rank(a_new).item()) < a_new.shape[0]:
            # A rank-deficient new basis cannot carry the old function;
            # transporting onto it would silently collapse B directions.
            return None
        solution = torch.linalg.lstsq(a_new.T, a_old.T).solution
    except Exception:
        return None
    if solution.shape != (a_new.shape[0], a_old.shape[0]):
        return None
    transport = solution.T
    if not torch.isfinite(transport).all():
        return None
    if float(transport.norm().item()) > _MAX_TRANSPORT_NORM:
        return None
    return transport


def apply_basis_transport(
    old_state: Mapping[str, Any],
    new_state: MutableMapping[str, Any],
    beta: float = 1.0,
) -> dict[str, Any]:
    """Transport every LoRA-B in ``new_state`` across its A basis change.

    For each LoRA pair, compares ``A`` in ``old_state`` (pre-aggregation
    global factors) with ``A`` in ``new_state`` (post-aggregation).  When the
    basis moved, replaces ``B`` with ``B @ ((1 - beta) I + beta M)`` where
    ``M = A_old @ pinv(A_new)``.  ``A`` itself is never modified.
    """
    pair_count = 0
    unchanged_count = 0
    skipped_count = 0
    fallback_count = 0
    max_function_rel_error = 0.0
    max_basis_rel_change = 0.0
    transported_keys: list[tuple[str, str]] = []

    for a_key, b_key in iter_lora_ab_pairs(new_state):
        a_new_raw = new_state[a_key]
        b_raw = new_state[b_key]
        a_old_raw = old_state.get(a_key)
        if a_old_raw is None or not _is_lora_pair(a_new_raw, b_raw) or \
                not torch.is_tensor(a_old_raw) or \
                a_old_raw.shape != a_new_raw.shape:
            skipped_count += 1
            continue

        # CPU float32 keeps the small lstsq robust across MPS/CUDA/CPU runs.
        a_old = a_old_raw.detach().float().cpu()
        a_new = a_new_raw.detach().float().cpu()
        if torch.equal(a_old, a_new):
            unchanged_count += 1
            continue
        b = b_raw.detach().float().cpu()

        transport = compute_transport_matrix(a_old, a_new)
        if transport is None:
            fallback_count += 1
            continue

        rank = transport.shape[0]
        blend = (1.0 - beta) * torch.eye(rank) + beta * transport
        b_new = (b @ blend).contiguous()

        function_before = b @ a_old
        function_after = b_new @ a_new
        denom = function_before.norm().clamp_min(1e-12)
        rel_error = float(
            (function_after - function_before).norm().item() / denom.item()
        )
        basis_denom = a_old.norm().clamp_min(1e-12)
        basis_change = float(
            (a_new - a_old).norm().item() / basis_denom.item()
        )

        new_state[b_key] = b_new.to(
            device=b_raw.device, dtype=b_raw.dtype
        )
        pair_count += 1
        transported_keys.append((a_key, b_key))
        max_function_rel_error = max(max_function_rel_error, rel_error)
        max_basis_rel_change = max(max_basis_rel_change, basis_change)

    return {
        "transport_mode": "ls",
        "transport_beta": beta,
        "transport_pair_count": pair_count,
        "transport_unchanged_count": unchanged_count,
        "transport_skipped_count": skipped_count,
        "transport_fallback_count": fallback_count,
        "transport_max_function_rel_error": max_function_rel_error,
        "transport_max_basis_rel_change": max_basis_rel_change,
        "transport_applied_keys": transported_keys,
    }


def apply_lora_transport_from_env(
    old_state: Mapping[str, Any],
    new_state: MutableMapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    mode = lora_transport_from_env(environ)
    if mode is None:
        return {}
    if mode == "ls":
        return apply_basis_transport(
            old_state, new_state, beta=transport_beta_from_env(environ)
        )
    raise ValueError(f"unsupported SLS LoRA transport mode: {mode!r}")
