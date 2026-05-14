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


def broadcast(server: MLP, clients: list[MLP], factor: str) -> None:
    """Copy ``factor`` (A or B) from server into every client in-place."""
    src = server.adapter_params(factor)
    for client in clients:
        for cp, sp in zip(client.adapter_params(factor), src, strict=True):
            cp.data.copy_(sp.data)


def average_factor(server: MLP, clients: list[MLP], factor: str) -> None:
    """Set server's ``factor`` to the per-tensor mean across clients."""
    n = len(clients)
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


def local_train(
    model: MLP,
    loader: DataLoader,
    steps: int,
    lr: float,
    device: torch.device,
    grad_clip: float = 1.0,
) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=lr)
    model.train()
    seen = 0
    while seen < steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
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
) -> tuple[list[float], list[float]]:
    assert method in METHODS
    torch.manual_seed(seed)
    server = MLP(rank).to(device)
    # Sanity: FFA-LoRA convention is B=0 (zero adapter at init).
    for B in server.adapter_params("B"):
        assert torch.equal(B, torch.zeros_like(B))

    clients: list[MLP] = [copy.deepcopy(server) for _ in train_sets]
    client_loaders = [
        DataLoader(s, batch_size=batch_size, shuffle=True, drop_last=False) for s in train_sets
    ]

    losses: list[float] = []
    accs: list[float] = []
    for r in range(rounds):
        if method == "lora":
            for client in clients:
                set_factor_trainable(client, "A", trainable=True)
                set_factor_trainable(client, "B", trainable=True)
            active_factors = ("A", "B")
        elif method == "ffa_lora":
            for client in clients:
                set_factor_trainable(client, "A", trainable=False)
                set_factor_trainable(client, "B", trainable=True)
            assert_factor_identical(server, clients, "A")
            active_factors = ("B",)
        else:  # rolora
            train_B = r % 2 == 0  # even rounds train B, odd rounds train A
            for client in clients:
                set_factor_trainable(client, "A", trainable=not train_B)
                set_factor_trainable(client, "B", trainable=train_B)
            frozen = "A" if train_B else "B"
            assert_factor_identical(server, clients, frozen)
            active_factors = ("B",) if train_B else ("A",)

        for client, loader in zip(clients, client_loaders, strict=True):
            local_train(client, loader, steps=local_steps, lr=lr, device=device, grad_clip=grad_clip)

        for f in active_factors:
            average_factor(server, clients, f)
            broadcast(server, clients, f)

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
    train_sets = iid_split(train_ds, args.clients, rng)
    test_loader = DataLoader(test_ds, batch_size=256)

    curves: dict[str, tuple[list[float], list[float]]] = {}
    for method in METHODS:
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
