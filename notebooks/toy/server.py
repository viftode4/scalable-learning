"""Server-side aggregation primitives.

`broadcast`, `average_factor`, `assert_factor_identical` are lifted verbatim
from the pre-refactor `mnist_fig2.py`. `ServerMomentum` is the new
RoLoRA-improvement primitive: it keeps a per-factor Polyak buffer and
applies ``factor ← factor + β · (avg − factor)`` on each aggregation,
matching the FedAvgM convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def broadcast(server: nn.Module, clients: list[nn.Module], factor: str) -> None:
    """Copy ``factor`` (A or B) from server into every client in-place."""
    src = server.adapter_params(factor)  # type: ignore[attr-defined]
    for client in clients:
        for cp, sp in zip(client.adapter_params(factor), src, strict=True):  # type: ignore[attr-defined]
            cp.data.copy_(sp.data)


def average_factor(server: nn.Module, clients: list[nn.Module], factor: str) -> None:
    """Set server's ``factor`` to the per-tensor mean across clients."""
    for s_i, params in enumerate(server.adapter_params(factor)):  # type: ignore[attr-defined]
        stack = torch.stack(
            [client.adapter_params(factor)[s_i].data for client in clients]  # type: ignore[attr-defined]
        )
        params.data.copy_(stack.mean(dim=0))


def assert_factor_identical(server: nn.Module, clients: list[nn.Module], factor: str) -> None:
    src = server.adapter_params(factor)  # type: ignore[attr-defined]
    for s_i, sp in enumerate(src):
        if torch.isnan(sp.data).any():
            raise RuntimeError(
                f"server {factor}[{s_i}] contains NaN — training diverged. "
                "Lower the learning rate or strengthen gradient clipping."
            )
    for client in clients:
        for cp, sp in zip(client.adapter_params(factor), src, strict=True):  # type: ignore[attr-defined]
            assert torch.equal(cp.data, sp.data), f"client {factor} drifted from server"


class ServerMomentum:
    """FedAvgM/FedOpt-style server momentum, applied per adapter factor.

    Update rule:

        Δ_t = avg(clients) − server_t
        v_t = β · v_{t-1} + Δ_t        (v_0 = 0)
        server_{t+1} = server_t + η_s · v_t

    β smooths direction; η_s scales magnitude. The canonical
    amplitude-preserving choice is η_s = 1 − β: steady-state ‖η_s · v_t‖ →
    ‖Δ‖, same magnitude as plain `average_factor`, but the direction is an
    EMA over the last ~1/(1−β) rounds.

    With β = 0 and η_s = 1, this collapses to plain `average_factor`
    bit-identically; that equivalence is checked in
    `tests/test_toy_components.py`.
    """

    def __init__(self, beta: float, server_lr: float = 1.0) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"server_momentum beta must be in [0, 1); got {beta}")
        if server_lr <= 0.0:
            raise ValueError(f"server_lr must be > 0; got {server_lr}")
        if beta == 0.0 and server_lr != 1.0:
            raise ValueError(
                "server_lr != 1.0 is only meaningful with β > 0 (soft-averaging "
                "without momentum is a separate concept; not supported here)"
            )
        self.beta = beta
        self.server_lr = server_lr
        # Keyed by factor; populated lazily on first step() for that factor.
        self._buffers: dict[str, list[torch.Tensor]] = {}

    def step(self, server: nn.Module, clients: list[nn.Module], factor: str) -> None:
        # Special-case beta=0 so the test's bit-identical promise holds — going
        # through sp + (avg - sp) would introduce sub-ULP rounding even though
        # the math is equivalent. (server_lr is required to be 1.0 here.)
        if self.beta == 0.0:
            average_factor(server, clients, factor)
            return

        server_params = server.adapter_params(factor)  # type: ignore[attr-defined]
        buf = self._buffers.get(factor)
        if buf is None:
            buf = [torch.zeros_like(sp.data) for sp in server_params]
            self._buffers[factor] = buf

        for s_i, sp in enumerate(server_params):
            stack = torch.stack(
                [client.adapter_params(factor)[s_i].data for client in clients]  # type: ignore[attr-defined]
            )
            avg = stack.mean(dim=0)
            delta = avg - sp.data
            buf[s_i].mul_(self.beta).add_(delta)
            sp.data.add_(buf[s_i], alpha=self.server_lr)
