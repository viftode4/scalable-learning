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
    """LoRA-on-frozen-base 2-layer MLP (kept for legacy tests, NOT the paper model).

    Use :class:`PaperToyModel` instead for Section 4.2 reproductions.
    """

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


class PaperToyModel(nn.Module):
    """Section 4.2, Eq. (10): f(x) = ReLU(x A B) W_out.

    A in R^{d x r}, B in R^{r x d}, W_out in R^{d x c}.
    Only A and B are tunable; W_out is fixed (registered as a buffer so it is
    not in `.parameters()` and is never updated by the optimizer).

    Init: A and B both Kaiming-uniform. The standard LoRA-on-base convention
    (A Kaiming, B=0) does NOT apply here — there is no additive base weight,
    so B=0 would make the forward identically zero and no gradient signal
    would flow. Paper does not pin the init explicitly.
    """

    def __init__(self, rank: int = 16, in_dim: int = 784, num_classes: int = 10) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.kaiming_uniform_(self.B, a=5**0.5)
        # W_out is fixed throughout training. Buffer (not Parameter) so it
        # doesn't appear in .parameters() and can't be picked up by any
        # optimizer that filters by requires_grad.
        w_out = torch.empty(in_dim, num_classes)
        nn.init.kaiming_uniform_(w_out, a=5**0.5)
        self.register_buffer("W_out", w_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return F.relu(x @ self.A @ self.B) @ self.W_out

    def adapter_params(self, factor: str) -> list[nn.Parameter]:
        if factor == "A":
            return [self.A]
        if factor == "B":
            return [self.B]
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


def label_split(
    dataset,
    num_clients: int,
    labels_per_client: int,
    rng: np.random.Generator,
    num_classes: int = 10,
) -> list[Subset]:
    """Paper Section 4.2 non-IID split: each client owns the samples of a disjoint
    set of class labels (no overlap).

    Two paper-faithful settings:
        - 5 clients, 2 labels each   (covers 10 classes, balanced)
        - 10 clients, 1 label each   (covers 10 classes, balanced)

    Raises if num_clients * labels_per_client != num_classes.
    """
    if num_clients * labels_per_client != num_classes:
        raise ValueError(
            f"label_split partitions all {num_classes} labels exactly; "
            f"need num_clients * labels_per_client == {num_classes}, "
            f"got {num_clients} * {labels_per_client} = "
            f"{num_clients * labels_per_client}"
        )
    targets = np.array([int(dataset[i][1]) for i in range(len(dataset))])
    order = rng.permutation(num_classes)
    subsets = []
    for c in range(num_clients):
        own = order[c * labels_per_client : (c + 1) * labels_per_client]
        idx = np.where(np.isin(targets, own))[0]
        subsets.append(Subset(dataset, idx.tolist()))
    return subsets


def local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    lr: float,
    device: torch.device,
    grad_clip: float = 1.0,
    steps: int | None = None,
    epochs: int | None = None,
) -> None:
    """Run local SGD on `model`. Exactly one of `steps` or `epochs` must be set.

    The paper's Section 4.2 uses 5 full epochs per round. `steps` is kept for the
    legacy unit-test path that ran a tiny fixed iteration count.
    """
    if (steps is None) == (epochs is None):
        raise ValueError("local_train: pass exactly one of `steps` or `epochs`")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=lr)
    model.train()
    if epochs is not None:
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                opt.step()
        return
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
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    grad_clip: float = 1.0,
    local_steps: int | None = None,
    local_epochs: int | None = None,
    model_factory=None,
) -> tuple[list[float], list[float]]:
    """Run one method (lora/ffa_lora/rolora) through `rounds` of FedAvg.

    Pass either `local_epochs` (paper convention) or `local_steps` (legacy / tests).
    `model_factory(rank)` returns the per-client model; defaults to
    :class:`PaperToyModel` (paper-faithful Section 4.2 architecture).
    """
    assert method in METHODS
    if (local_steps is None) == (local_epochs is None):
        raise ValueError("run_method: pass exactly one of local_steps / local_epochs")
    if model_factory is None:
        model_factory = PaperToyModel
    torch.manual_seed(seed)
    server = model_factory(rank).to(device)

    clients = [copy.deepcopy(server) for _ in train_sets]
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
            train_b = r % 2 == 0  # even rounds train B, odd rounds train A
            for client in clients:
                set_factor_trainable(client, "A", trainable=not train_b)
                set_factor_trainable(client, "B", trainable=train_b)
            frozen = "A" if train_b else "B"
            assert_factor_identical(server, clients, frozen)
            active_factors = ("B",) if train_b else ("A",)

        for client, loader in zip(clients, client_loaders, strict=True):
            local_train(
                client,
                loader,
                lr=lr,
                device=device,
                grad_clip=grad_clip,
                steps=local_steps,
                epochs=local_epochs,
            )

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
    # Paper Section 4.2 defaults. The two paper settings are:
    #   --clients 5  --labels-per-client 2   (Fig. 2 left panel)
    #   --clients 10 --labels-per-client 1   (Fig. 2 right panel)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--labels-per-client", type=int, default=2,
                   help="non-IID label partition; ignored when --split iid")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=5,
                   help="paper §4.2 uses 5 local epochs per round")
    p.add_argument("--local-steps", type=int, default=None,
                   help="override --local-epochs with a fixed step count (legacy / debug)")
    p.add_argument("--rank", type=int, default=16,
                   help="paper §4.2 uses r=16")
    p.add_argument("--lr", type=float, default=0.02,
                   help="paper does not pin the lr for the §4.2 toy experiment")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=("label", "iid"), default="label",
                   help="paper §4.2 uses label-non-IID; iid is a sanity baseline")
    p.add_argument("--model", choices=("paper", "mlp"), default="paper",
                   help="`paper` = PaperToyModel (Eq. 10); `mlp` = legacy LoRA-on-base 2-layer MLP")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=None,
                   help="defaults to results/mnist_fig2_<config>.png")
    p.add_argument("--subset", type=int, default=0, help="if >0, use first N train examples")
    args = p.parse_args()

    if args.out is None:
        tag = f"c{args.clients}_{args.split}_r{args.rank}_{args.model}"
        args.out = Path(f"results/mnist_fig2_{tag}.png")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(
        f"device={device}, clients={args.clients}, split={args.split}, "
        f"rounds={args.rounds}, rank={args.rank}, model={args.model}, "
        f"local_epochs={args.local_epochs}, lr={args.lr}"
    )

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    if args.subset > 0:
        train_ds = Subset(train_ds, list(range(args.subset)))

    rng = np.random.default_rng(args.seed)
    if args.split == "label":
        train_sets = label_split(train_ds, args.clients, args.labels_per_client, rng)
    else:
        train_sets = iid_split(train_ds, args.clients, rng)
    print("per-client sample counts:", [len(s) for s in train_sets])
    test_loader = DataLoader(test_ds, batch_size=256)

    model_factory = PaperToyModel if args.model == "paper" else MLP

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
            local_epochs=None if args.local_steps is not None else args.local_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            grad_clip=args.grad_clip,
            model_factory=model_factory,
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
        f"RoLoRA Fig. 2 reproduction — MNIST, {args.clients} clients ({args.split}), rank {args.rank}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")

    print("final accuracies:")
    for method, (_, acc) in curves.items():
        print(f"  {method:8} {acc[-1]:.4f}")


if __name__ == "__main__":
    main()
