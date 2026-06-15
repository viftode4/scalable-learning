"""Reproduce Figure 2 of the RoLoRA paper on MNIST.

Three methods on the same 2-layer MLP under federated averaging:
- ``lora``      — standard LoRA, average A and B separately each round (the math bug).
- ``ffa_lora``  — freeze A at init, only train and average B.
- ``rolora``    — alternate: odd rounds train B (A frozen+shared), even rounds train A
                  (B frozen+shared). Exact aggregation in each round.

Runs all three sequentially on a laptop CPU in a few minutes. Asserts exactness
invariants during training (frozen factor is bit-identical across all clients before
local steps). Saves loss/accuracy curves to ``results/mnist_fig2.png``.

This script is the cheapest sanity check that the RoLoRA mechanism does what the paper
claims; per the deep-research plan, it must pass before any GPU work.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

METHODS = ("lora", "ffa_lora", "rolora")
INIT_VARIANTS = ("default", "orthogonal_a", "svd_compensated")
SYNC_POLICIES = ("full", "active_only")
DATA_SPLITS = ("iid", "label_shard")


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

    def lora_product(self) -> torch.Tensor:
        """Return the input-to-output LoRA matrix used by ``x @ A @ B``."""
        return self.A @ self.B

    def effective_weight(self) -> torch.Tensor:
        """Return the full input-to-output matrix seen by the forward pass."""
        return self.base.weight.T + self.lora_product()


class MLP(nn.Module):
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


def set_factor_trainable(model: MLP, factor: str, *, trainable: bool) -> None:
    for p in model.adapter_params(factor):
        p.requires_grad = trainable


def phase_for_round(round_id: int, pattern: str | None = None) -> str:
    """Return the toy RoLoRA phase for ``round_id``.

    Default matches the paper-style B/A alternation. A compact pattern such as
    ``BBA`` repeats exactly and lets the toy mirror the supplement
    ``SLS_PHASE_PATTERN`` switch.
    """
    if round_id < 0:
        raise ValueError("round_id must be non-negative")
    if pattern is None or pattern.strip() == "":
        return "B" if round_id % 2 == 0 else "A"
    compact = pattern.replace(",", " ").replace(";", " ").split()
    compact = ("".join(compact) if compact else pattern).upper()
    if not compact or set(compact) - {"A", "B"}:
        raise ValueError(f"phase pattern must contain only A/B phases: {pattern!r}")
    return compact[round_id % len(compact)]


def _normalise_init_variant(init_variant: str) -> str:
    value = init_variant.strip().lower().replace("-", "_")
    if value in {"", "default", "none", "off"}:
        return "default"
    if value in {"orthogonal", "orthogonal_a", "orthogonal_lora_a"}:
        return "orthogonal_a"
    if value in {"svd", "svd_compensated", "pissa", "pissa_compensated"}:
        return "svd_compensated"
    raise ValueError(f"unknown toy LoRA init variant: {init_variant!r}")


def apply_lora_initialization_to_layer(layer: LoRALinear, init_variant: str) -> None:
    """Apply a toy analogue of the GLUE LoRA initialization switches."""
    variant = _normalise_init_variant(init_variant)
    if variant == "default":
        return
    if variant == "orthogonal_a":
        target_fro = layer.A.detach().float().norm().clamp_min(1e-12)
        q, _ = torch.linalg.qr(torch.randn_like(layer.A), mode="reduced")
        scale = target_fro / (layer.A.shape[1] ** 0.5)
        layer.A.data.copy_((q * scale).to(layer.A))
        layer.B.data.zero_()
        return

    # ``svd_compensated`` starts from an identical function but moves the
    # top-r principal component into the trainable adapter. The toy LoRA uses
    # input-to-output matrices: effective W = base.weight.T + A @ B.
    before = layer.effective_weight().detach().float()
    u, s, vh = torch.linalg.svd(before.cpu(), full_matrices=False)
    rank = min(layer.A.shape[1], s.shape[0])
    sqrt_s = torch.sqrt(s[:rank])
    a_new = u[:, :rank] * sqrt_s.unsqueeze(0)
    b_new = sqrt_s.unsqueeze(1) * vh[:rank, :]
    layer.A.data.zero_()
    layer.B.data.zero_()
    layer.A.data[:, :rank].copy_(a_new.to(layer.A))
    layer.B.data[:rank, :].copy_(b_new.to(layer.B))
    layer.base.weight.data.copy_((before - layer.lora_product().detach().float()).T.to(layer.base.weight))


def apply_lora_initialization(model: MLP, init_variant: str) -> None:
    for layer in (model.fc1, model.fc2):
        apply_lora_initialization_to_layer(layer, init_variant)


def transport_layer_coefficients(layer: LoRALinear, old_a: torch.Tensor) -> dict[str, float | bool]:
    """Re-express ``B`` so ``A_new @ B_new`` preserves ``A_old @ B``."""
    a_old = old_a.detach().float().cpu()
    a_new = layer.A.detach().float().cpu()
    if torch.equal(a_old, a_new):
        return {"transported": False, "fallback": False, "rel_error": 0.0}
    if int(torch.linalg.matrix_rank(a_new).item()) < a_new.shape[1]:
        return {"transported": False, "fallback": True, "rel_error": 0.0}
    try:
        transform = torch.linalg.lstsq(a_new, a_old).solution
    except RuntimeError:
        return {"transported": False, "fallback": True, "rel_error": 0.0}
    b_old = layer.B.detach().float().cpu()
    product_before = a_old @ b_old
    b_new = transform @ b_old
    product_after = a_new @ b_new
    denom = product_before.norm().clamp_min(1e-12)
    rel_error = float((product_after - product_before).norm().item() / denom.item())
    layer.B.data.copy_(b_new.to(layer.B))
    return {"transported": True, "fallback": False, "rel_error": rel_error}


def transport_coefficients(server: MLP, old_a_params: list[torch.Tensor]) -> dict[str, float | int]:
    transported = 0
    fallback = 0
    max_rel_error = 0.0
    for layer, old_a in zip((server.fc1, server.fc2), old_a_params, strict=True):
        stats = transport_layer_coefficients(layer, old_a)
        transported += int(bool(stats["transported"]))
        fallback += int(bool(stats["fallback"]))
        max_rel_error = max(max_rel_error, float(stats["rel_error"]))
    return {
        "transported_layers": transported,
        "fallback_layers": fallback,
        "max_rel_error": max_rel_error,
    }


def orthogonal_gauge_layer(layer: LoRALinear) -> None:
    """QR-gauge the toy LoRA basis while preserving ``A @ B``."""
    q, r = torch.linalg.qr(layer.A.detach().float().cpu(), mode="reduced")
    b_new = r @ layer.B.detach().float().cpu()
    layer.A.data.copy_(q.to(layer.A))
    layer.B.data.copy_(b_new.to(layer.B))


def orthogonal_gauge(model: MLP) -> None:
    for layer in (model.fc1, model.fc2):
        orthogonal_gauge_layer(layer)


def broadcast(server: MLP, clients: list[MLP], factor: str) -> None:
    """Copy ``factor`` (A or B) from server into every client in-place."""
    src = server.adapter_params(factor)
    for client in clients:
        for cp, sp in zip(client.adapter_params(factor), src, strict=True):
            cp.data.copy_(sp.data)


def average_factor(server: MLP, clients: list[MLP], factor: str) -> None:
    """Set server's ``factor`` to the per-tensor mean across clients."""
    for s_i, params in enumerate(server.adapter_params(factor)):
        stack = torch.stack([client.adapter_params(factor)[s_i].data for client in clients])
        params.data.copy_(stack.mean(dim=0))


def assert_factor_identical(server: MLP, clients: list[MLP], factor: str) -> None:
    src = server.adapter_params(factor)
    for s_i, sp in enumerate(src):
        if torch.isnan(sp.data).any():
            raise RuntimeError(
                f"server {factor}[{s_i}] contains NaN — training diverged. "
                "Lower the learning rate or strengthen gradient clipping."
            )
    for client in clients:
        for cp, sp in zip(client.adapter_params(factor), src, strict=True):
            assert torch.equal(cp.data, sp.data), f"client {factor} drifted from server"


def iid_split(dataset, num_clients: int, rng: np.random.Generator) -> list[Subset]:
    n = len(dataset)
    idx = rng.permutation(n)
    chunks = np.array_split(idx, num_clients)
    return [Subset(dataset, c.tolist()) for c in chunks]


def dataset_label_at(dataset, index: int) -> int:
    """Return an integer class label from a Dataset or nested Subset."""
    if isinstance(dataset, Subset):
        return dataset_label_at(dataset.dataset, int(dataset.indices[index]))
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "labels", None)
    value = labels[index] if labels is not None else dataset[index][1]
    if isinstance(value, torch.Tensor):
        value = value.item()
    return int(value)


def label_shard_split(
    dataset,
    num_clients: int,
    rng: np.random.Generator,
    labels_per_client: int = 1,
) -> list[Subset]:
    """Non-IID toy split: each client receives examples from a few labels only."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if labels_per_client <= 0:
        raise ValueError("labels_per_client must be positive")

    by_label: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        by_label.setdefault(dataset_label_at(dataset, index), []).append(index)

    labels = sorted(by_label)
    if num_clients * labels_per_client < len(labels):
        raise ValueError(
            "label_shard split needs num_clients * labels_per_client >= number of labels"
        )

    label_order = rng.permutation(labels).tolist()
    client_to_labels: list[list[int]] = [[] for _ in range(num_clients)]
    label_to_clients: dict[int, list[int]] = {label: [] for label in labels}
    for client_idx in range(num_clients):
        for offset in range(labels_per_client):
            label = int(label_order[(client_idx * labels_per_client + offset) % len(labels)])
            client_to_labels[client_idx].append(label)
            label_to_clients[label].append(client_idx)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for label, indices in by_label.items():
        owners = label_to_clients[label]
        shuffled = rng.permutation(indices)
        chunks = np.array_split(shuffled, len(owners))
        for owner, chunk in zip(owners, chunks, strict=True):
            client_indices[owner].extend(chunk.tolist())

    for indices in client_indices:
        rng.shuffle(indices)
    return [Subset(dataset, indices) for indices in client_indices]


def split_clients(
    dataset,
    num_clients: int,
    rng: np.random.Generator,
    split: str = "iid",
    labels_per_client: int = 1,
) -> list[Subset]:
    split = split.strip().lower().replace("-", "_")
    if split == "iid":
        return iid_split(dataset, num_clients, rng)
    if split in {"label_shard", "label_skew", "one_label"}:
        return label_shard_split(dataset, num_clients, rng, labels_per_client)
    raise ValueError(f"unknown split {split!r}; expected one of {DATA_SPLITS}")


def sample_participants(
    clients: list[MLP],
    participation_rate: float,
    rng: np.random.Generator,
) -> list[MLP]:
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")
    if participation_rate >= 1:
        return clients
    count = max(1, int(participation_rate * len(clients)))
    indices = sorted(rng.choice(len(clients), size=count, replace=False).tolist())
    return [clients[i] for i in indices]


def local_train(
    model: MLP,
    loader: DataLoader,
    steps: int,
    lr: float,
    device: torch.device,
    grad_clip: float = 1.0,
    drift_mu: float = 0.0,
) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=lr)
    # Factor-wise drift correction (FedProx-style). Anchor each trainable
    # parameter to its global value at the start of the round (the just-
    # broadcast server factor) and penalize local drift via (mu/2)||w - w_t||^2.
    # In RoLoRA only one factor is trainable per round, so this corrects client
    # drift on exactly the alternating factor at zero extra communication -- the
    # within-factor heterogeneity gap RoLoRA's exact aggregation leaves open.
    anchors = [p.detach().clone() for p in trainable] if drift_mu > 0 else None
    model.train()
    seen = 0
    while seen < steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if anchors is not None:
                prox = sum(((p - a) ** 2).sum() for p, a in zip(trainable, anchors))
                loss = loss + 0.5 * drift_mu * prox
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            seen += 1
            if seen >= steps:
                break


@torch.no_grad()
def evaluate(model: MLP, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses, correct, total = [], 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(F.cross_entropy(out, y, reduction="sum").item())
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return sum(losses) / total, correct / total


def run_method(
    method: str,
    *,
    train_sets: list[Subset],
    test_loader: DataLoader,
    rank: int,
    rounds: int,
    local_steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    grad_clip: float = 1.0,
    init_variant: str = "default",
    phase_pattern: str | None = None,
    participation_rate: float = 1.0,
    sync_policy: str = "full",
    transport: bool = False,
    gauge: bool = False,
    drift_mu: float = 0.0,
) -> tuple[list[float], list[float]]:
    assert method in METHODS
    init_variant = _normalise_init_variant(init_variant)
    if sync_policy not in SYNC_POLICIES:
        raise ValueError(f"sync_policy must be one of {SYNC_POLICIES}")
    torch.manual_seed(seed)
    server = MLP(rank).to(device)
    apply_lora_initialization(server, init_variant)
    # Sanity: FFA-LoRA convention is B=0 (zero adapter at init).
    if init_variant != "svd_compensated":
        for b_param in server.adapter_params("B"):
            assert torch.equal(b_param, torch.zeros_like(b_param))

    clients: list[MLP] = [copy.deepcopy(server) for _ in train_sets]
    client_loaders = [
        DataLoader(s, batch_size=batch_size, shuffle=True, drop_last=False) for s in train_sets
    ]
    client_to_loader = {id(client): loader for client, loader in zip(clients, client_loaders, strict=True)}
    participant_rng = np.random.default_rng(seed + 104729)

    losses: list[float] = []
    accs: list[float] = []
    for r in range(rounds):
        participating_clients = sample_participants(clients, participation_rate, participant_rng)
        if method == "lora":
            for client in participating_clients:
                set_factor_trainable(client, "A", trainable=True)
                set_factor_trainable(client, "B", trainable=True)
            active_factors = ("A", "B")
        elif method == "ffa_lora":
            for client in participating_clients:
                set_factor_trainable(client, "A", trainable=False)
                set_factor_trainable(client, "B", trainable=True)
            active_factors = ("B",)
        else:  # rolora
            phase = phase_for_round(r, phase_pattern)
            train_b = phase == "B"
            for client in participating_clients:
                set_factor_trainable(client, "A", trainable=not train_b)
                set_factor_trainable(client, "B", trainable=train_b)
            frozen = "A" if train_b else "B"
            active_factors = ("B",) if train_b else ("A",)

        if sync_policy == "full":
            for factor in ("A", "B"):
                broadcast(server, participating_clients, factor)
        else:
            for factor in active_factors:
                broadcast(server, participating_clients, factor)
        if method == "ffa_lora" and sync_policy == "full":
            assert_factor_identical(server, participating_clients, "A")
        if method == "rolora" and sync_policy == "full":
            assert_factor_identical(server, participating_clients, frozen)

        old_a_params = [p.detach().clone() for p in server.adapter_params("A")]
        for client in participating_clients:
            loader = client_to_loader[id(client)]
            local_train(
                client, loader, steps=local_steps, lr=lr, device=device,
                grad_clip=grad_clip, drift_mu=drift_mu,
            )

        for f in active_factors:
            average_factor(server, participating_clients, f)
        if method == "rolora" and "A" in active_factors and transport:
            transport_coefficients(server, old_a_params)
        if gauge and "A" in active_factors:
            orthogonal_gauge(server)
        for f in active_factors:
            broadcast(server, participating_clients, f)
        if method == "rolora" and "A" in active_factors and (transport or gauge):
            broadcast(server, participating_clients, "B")

        loss, acc = evaluate(server, test_loader, device)
        losses.append(loss)
        accs.append(acc)
        if (r + 1) % max(1, rounds // 10) == 0:
            print(f"  [{method:8}] round {r + 1:3d}/{rounds}  loss={loss:.4f}  acc={acc:.4f}")

    return losses, accs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local-steps", type=int, default=20)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results/mnist_fig2.png"))
    p.add_argument("--subset", type=int, default=0, help="if >0, use first N train examples")
    p.add_argument("--split", choices=DATA_SPLITS, default="iid")
    p.add_argument("--labels-per-client", type=int, default=1)
    p.add_argument("--init", choices=INIT_VARIANTS, default="default")
    p.add_argument("--phase-pattern", default=None, help="optional RoLoRA phase pattern, e.g. BBA")
    p.add_argument("--participation-rate", type=float, default=1.0)
    p.add_argument("--sync-policy", choices=SYNC_POLICIES, default="full")
    p.add_argument("--transport", action="store_true", help="transport B after RoLoRA A-rounds")
    p.add_argument("--gauge", action="store_true", help="QR-gauge A after A-rounds")
    p.add_argument("--drift-mu", type=float, default=0.0, help="FedProx-style factor-wise drift correction strength")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, clients={args.clients}, rounds={args.rounds}, rank={args.rank}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    if args.subset > 0:
        train_ds = Subset(train_ds, list(range(args.subset)))

    rng = np.random.default_rng(args.seed)
    train_sets = split_clients(
        train_ds,
        args.clients,
        rng,
        split=args.split,
        labels_per_client=args.labels_per_client,
    )
    test_loader = DataLoader(test_ds, batch_size=256)

    curves: dict[str, tuple[list[float], list[float]]] = {}
    for method in METHODS:
        if method != "rolora" and (
            args.init != "default"
            or args.phase_pattern
            or args.transport
            or args.gauge
        ):
            continue
        print(f"-- {method} --")
        curves[method] = run_method(
            method,
            train_sets=train_sets,
            test_loader=test_loader,
            rank=args.rank,
            rounds=args.rounds,
            local_steps=args.local_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            grad_clip=args.grad_clip,
            init_variant=args.init,
            phase_pattern=args.phase_pattern,
            participation_rate=args.participation_rate,
            sync_policy=args.sync_policy,
            transport=args.transport,
            gauge=args.gauge,
            drift_mu=args.drift_mu,
        )

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for method, (loss, acc) in curves.items():
        ax_loss.plot(loss, label=method)
        ax_acc.plot(acc, label=method)
    ax_loss.set_xlabel("communication round")
    ax_loss.set_ylabel("test cross-entropy")
    ax_loss.legend()
    ax_acc.set_xlabel("communication round")
    ax_acc.set_ylabel("test accuracy")
    ax_acc.legend()
    fig.suptitle(
        f"RoLoRA Fig. 2 reproduction — MNIST, {args.clients} clients, rank {args.rank}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")

    print("final accuracies:")
    for method, (_, acc) in curves.items():
        print(f"  {method:8} {acc[-1]:.4f}")


if __name__ == "__main__":
    main()
