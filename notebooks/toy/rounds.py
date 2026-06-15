"""Federated and centralized training loops.

`run_method` dispatches on either a string method name (legacy API used by
`tests/test_mnist_fig2.py` and the pre-refactor `mnist_fig2.py`) or a
`MethodConfig` (new API used by `mnist_fig2_compare.py` and `sweep.py`). The
legacy path produces bit-identical curves to the pre-refactor implementation
for a given seed.

`run_centralized` is the non-federated ceiling: it trains one model on the
union of all client data with a matched gradient budget.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .client import evaluate, local_train
from .config import MethodConfig, preset
from .model import METHODS, PaperToyModel, set_factor_trainable
from .server import (
    ServerMomentum,
    assert_factor_identical,
    average_factor,
    broadcast,
)


def checkpoint_indices(rounds: int, log_every: int) -> list[int]:
    """Round indices at which `run_method` / `run_centralized` evaluate and log.

    Always [0, log_every, 2*log_every, ..., rounds]. The final round is
    always included even if it isn't a multiple of `log_every`.
    Round 0 is the pre-training evaluation.
    """
    if log_every < 1:
        raise ValueError(f"log_every must be >= 1; got {log_every}")
    idx = list(range(0, rounds + 1, log_every))
    if idx[-1] != rounds:
        idx.append(rounds)
    return idx


def _resolve_config(method_or_config: str | MethodConfig) -> MethodConfig:
    if isinstance(method_or_config, MethodConfig):
        return method_or_config
    assert method_or_config in METHODS, (
        f"unknown method string: {method_or_config!r}; "
        f"expected one of {METHODS} or a MethodConfig"
    )
    return preset(f"base_{method_or_config}")


def _snapshot_factor(model, factor: str) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.adapter_params(factor)]


def run_method(
    method_or_config: str | MethodConfig,
    *,
    train_sets: list[Subset],
    test_loader: DataLoader,
    rank: int,
    rounds: int,
    lr: float | None = None,
    batch_size: int,
    seed: int,
    device: torch.device,
    grad_clip: float = 1.0,
    local_steps: int | None = None,
    local_epochs: int | None = None,
    model_factory=None,
    log_every: int | None = None,
) -> tuple[list[float], list[float]]:
    """Run one method through `rounds` of FedAvg-style aggregation.

    Two calling conventions:

    1. **Legacy string API** (existing tests / `mnist_fig2.py` CLI):
       ``run_method("rolora", ..., lr=0.02, ...)``. Uses ``base_<method>``
       preset; both lr_a and lr_b are set to ``lr``. Output is bit-identical
       to the pre-refactor implementation for a given seed.

    2. **MethodConfig API** (comparison study): ``run_method(cfg, ..., lr=None, ...)``.
       Per-factor lr / init / prox / momentum come from the config. ``lr`` is
       ignored (must be omitted or None to keep precedence unambiguous).

    Pass either ``local_steps`` or ``local_epochs``, not both. ``model_factory``
    overrides the default ``PaperToyModel`` factory (used by the legacy MLP
    path in `tests/test_mnist_fig2.py`).
    """
    if (local_steps is None) == (local_epochs is None):
        raise ValueError("run_method: pass exactly one of local_steps / local_epochs")

    cfg = _resolve_config(method_or_config)
    if isinstance(method_or_config, str):
        # Legacy path: caller passed `lr`; both factors get it.
        if lr is None:
            raise ValueError(
                "run_method: legacy string API requires `lr`; pass a MethodConfig "
                "to use per-factor lr_a/lr_b instead"
            )
        cfg = MethodConfig(
            name=cfg.name,
            alternation=cfg.alternation,
            lr_a=lr,
            lr_b=lr,
            init_a=cfg.init_a,
            init_b=cfg.init_b,
            prox_mu=cfg.prox_mu,
            server_momentum=cfg.server_momentum,
        )
    else:
        if lr is not None:
            raise ValueError(
                "run_method: `lr` is ignored when passing a MethodConfig; "
                "use cfg.lr_a / cfg.lr_b instead"
            )

    if cfg.alternation == "centralized":
        raise ValueError(
            "run_method: pass alternation='centralized' configs to run_centralized, "
            "not run_method"
        )

    if model_factory is None:
        model_factory = PaperToyModel

    torch.manual_seed(seed)
    # Honour init_a / init_b only for PaperToyModel; the legacy MLP path keeps
    # its own (Kaiming-A, zero-B) convention.
    if model_factory is PaperToyModel:
        server = model_factory(rank, init_a=cfg.init_a, init_b=cfg.init_b).to(device)
    else:
        server = model_factory(rank).to(device)

    clients = [copy.deepcopy(server) for _ in train_sets]
    client_loaders = [
        DataLoader(s, batch_size=batch_size, shuffle=True, drop_last=False)
        for s in train_sets
    ]

    momentum = (
        ServerMomentum(cfg.server_momentum, server_lr=cfg.server_lr)
        if cfg.server_momentum > 0
        else None
    )

    log_step = log_every if log_every is not None else max(1, rounds // 10)
    eval_rounds = set(checkpoint_indices(rounds, log_step))

    # Round-0 (pre-training) eval — anchors the curve at the initial loss/acc.
    loss0, acc0 = evaluate(server, test_loader, device)
    losses: list[float] = [loss0]
    accs: list[float] = [acc0]
    print(f"  [{cfg.name[:24]:24}] round   0/{rounds}  loss={loss0:.4f}  acc={acc0:.4f}")

    for r in range(rounds):
        if cfg.alternation == "lora":
            for client in clients:
                set_factor_trainable(client, "A", trainable=True)
                set_factor_trainable(client, "B", trainable=True)
            active_factors: tuple[str, ...] = ("A", "B")
        elif cfg.alternation == "ffa_lora":
            for client in clients:
                set_factor_trainable(client, "A", trainable=False)
                set_factor_trainable(client, "B", trainable=True)
            assert_factor_identical(server, clients, "A")
            active_factors = ("B",)
        else:  # rolora
            train_b = r % 2 == 0  # even rounds train B, odd rounds train A
            for client in clients:
                set_factor_trainable(client, "A", trainable=not train_b)
                set_factor_trainable(client, "B", trainable=train_b)
            frozen = "A" if train_b else "B"
            assert_factor_identical(server, clients, frozen)
            active_factors = ("B",) if train_b else ("A",)

        prox_anchor: dict[str, list[torch.Tensor]] | None = None
        if cfg.prox_mu > 0:
            prox_anchor = {f: _snapshot_factor(server, f) for f in active_factors}

        for client, loader in zip(clients, client_loaders, strict=True):
            local_train(
                client,
                loader,
                lr_a=cfg.lr_a,
                lr_b=cfg.lr_b,
                device=device,
                grad_clip=grad_clip,
                steps=local_steps,
                epochs=local_epochs,
                prox_mu=cfg.prox_mu,
                prox_anchor=prox_anchor,
            )

        for f in active_factors:
            if momentum is not None:
                momentum.step(server, clients, f)
            else:
                average_factor(server, clients, f)
            broadcast(server, clients, f)

        if (r + 1) in eval_rounds:
            loss, acc = evaluate(server, test_loader, device)
            losses.append(loss)
            accs.append(acc)
            print(f"  [{cfg.name[:24]:24}] round {r + 1:3d}/{rounds}  loss={loss:.4f}  acc={acc:.4f}")

    return losses, accs


def run_centralized(
    train_sets: Iterable[Subset],
    test_loader: DataLoader,
    *,
    rank: int,
    rounds: int,
    local_steps: int | None = None,
    local_epochs: int | None = None,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    grad_clip: float = 1.0,
    model_factory=None,
    log_label: str = "centralized",
    log_every: int | None = None,
) -> tuple[list[float], list[float]]:
    """Train one model on the union of all client data; record a curve.

    Per "round" the model does ``local_epochs`` epochs (or ``local_steps`` steps)
    over the full union, then we evaluate. The x-axis matches the federated
    runs' round count so curves can be overlayed, but per-round compute is
    just ``local_epochs`` / ``local_steps`` worth of work — not multiplied by
    ``num_clients``. Centralized usually converges well before the federated
    budget is exhausted, so this still functions as a ceiling.

    Returns ``(losses, accs)`` of length ``rounds``.
    """
    if (local_steps is None) == (local_epochs is None):
        raise ValueError("run_centralized: pass exactly one of local_steps / local_epochs")
    if model_factory is None:
        model_factory = PaperToyModel

    torch.manual_seed(seed)
    model = model_factory(rank).to(device)

    union = ConcatDataset(list(train_sets))
    loader = DataLoader(union, batch_size=batch_size, shuffle=True, drop_last=False)

    log_step = log_every if log_every is not None else max(1, rounds // 10)
    eval_rounds = set(checkpoint_indices(rounds, log_step))

    # Round-0 (pre-training) eval — anchors the curve at the initial loss/acc.
    loss0, acc0 = evaluate(model, test_loader, device)
    losses: list[float] = [loss0]
    accs: list[float] = [acc0]
    print(f"  [{log_label[:24]:24}] round   0/{rounds}  loss={loss0:.4f}  acc={acc0:.4f}")

    # Per-round budget mirrors federated total sample-passes per round.
    # Epochs: federated does num_clients × local_epochs epochs over ~1/num_clients
    # of the data each = local_epochs epochs over the union. Don't multiply.
    # Steps: federated does num_clients × local_steps optimizer steps per round;
    # centralized loader has the same batch_size, so num_clients × local_steps
    # is the correct match here.
    for r in range(rounds):
        if local_epochs is not None:
            local_train(
                model,
                loader,
                lr=lr,
                device=device,
                grad_clip=grad_clip,
                epochs=local_epochs,
            )
        else:
            local_train(
                model,
                loader,
                lr=lr,
                device=device,
                grad_clip=grad_clip,
                steps=local_steps,
            )

        if (r + 1) in eval_rounds:
            loss, acc = evaluate(model, test_loader, device)
            losses.append(loss)
            accs.append(acc)
            print(f"  [{log_label[:24]:24}] round {r + 1:3d}/{rounds}  loss={loss:.4f}  acc={acc:.4f}")

    return losses, accs
