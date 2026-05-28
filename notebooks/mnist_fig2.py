"""Reproduce Figure 2 of the RoLoRA paper on MNIST.

Three methods on the §4.2 toy model under federated averaging:
- ``lora``      — standard LoRA, average A and B separately each round (the math bug).
- ``ffa_lora``  — freeze A at init, only train and average B.
- ``rolora``    — alternate: odd rounds train B (A frozen+shared), even rounds train A
                  (B frozen+shared).

Runs all three sequentially on a laptop CPU/MPS/GPU in a few minutes. Saves
loss/accuracy curves to ``results/mnist_fig2_<config>.png``.

This file is a thin CLI wrapper. All reusable building blocks live in
``notebooks/toy/``; ``mnist_fig2_compare.py`` and ``toy/sweep.py`` are the
entry points for the improvement-comparison study.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from toy import (
    METHODS,
    MLP,
    LoRALinear,
    PaperToyModel,
    assert_factor_identical,
    average_factor,
    broadcast,
    evaluate,
    iid_split,
    label_split,
    local_train,
    pick_device,
    run_method,
    set_factor_trainable,
)
from toy.plotting import plot_curves

__all__ = [
    "METHODS",
    "MLP",
    "LoRALinear",
    "PaperToyModel",
    "assert_factor_identical",
    "average_factor",
    "broadcast",
    "evaluate",
    "iid_split",
    "label_split",
    "local_train",
    "run_method",
    "set_factor_trainable",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # Paper §4.2 settings:
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
    p.add_argument("--rank", type=int, default=16, help="paper §4.2 uses r=16")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=("label", "iid"), default="label")
    p.add_argument("--model", choices=("paper", "mlp"), default="paper")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--subset", type=int, default=0, help="if >0, use first N train examples")
    args = p.parse_args()

    if args.out is None:
        tag = f"c{args.clients}_{args.split}_r{args.rank}_{args.model}"
        args.out = Path(f"results/mnist_fig2_{tag}.png")
    args.data_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
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

    plot_curves(
        curves,
        args.out,
        title=(
            f"RoLoRA Fig. 2 reproduction — MNIST, {args.clients} clients "
            f"({args.split}), rank {args.rank}"
        ),
    )
    print(f"saved {args.out}")
    print("final accuracies:")
    for method, (_, acc) in curves.items():
        print(f"  {method:8} {acc[-1]:.4f}")


if __name__ == "__main__":
    main()
