"""Comparison entry point: base RoLoRA vs improvement variants vs centralized ceiling.

Runs each requested variant once at the same (clients, split, rounds, rank, seed)
and overlays the curves on a single (loss, accuracy) plot. The centralized
ceiling — same model, same total gradient budget, no federated averaging — is
drawn as a dashed reference line.

Usage:
    uv run python notebooks/mnist_fig2_compare.py --clients 10 --labels-per-client 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from toy import (
    PRESETS,
    MethodConfig,
    PaperToyModel,
    iid_split,
    label_split,
    pick_device,
    run_centralized,
    run_method,
)
from toy.plotting import plot_curves
from toy.rounds import checkpoint_indices

DEFAULT_VARIANTS = (
    "base_lora",
    "base_ffa_lora",
    "base_rolora",
    "rolora_plus_lr",
    "rolora_orth_a",
    "rolora_prox",
    "rolora_mom",
    "rolora_kitchen_sink",
)


def _parse_variants(raw: str) -> list[str]:
    names = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        known = ", ".join(sorted(PRESETS))
        raise SystemExit(f"unknown variants: {unknown}; known: {known}")
    # `centralized` is always run separately as the ceiling (unless
    # --skip-ceiling). Drop it from the federated-loop list so we don't try
    # to FedAvg it.
    return [n for n in names if n != "centralized"]


def _make_splits(
    train_ds,
    *,
    clients: int,
    labels_per_client: int,
    split: str,
    seed: int,
) -> list[Subset]:
    rng = np.random.default_rng(seed)
    if split == "label":
        return label_split(train_ds, clients, labels_per_client, rng)
    if split == "iid":
        return iid_split(train_ds, clients, rng)
    raise ValueError(f"unknown split: {split}")


def run_one(
    *,
    variants: list[str],
    clients: int,
    labels_per_client: int,
    split: str,
    rounds: int,
    rank: int,
    local_epochs: int,
    batch_size: int,
    seed: int,
    grad_clip: float,
    device: torch.device,
    data_dir: Path,
    subset: int,
    skip_ceiling: bool = False,
    log_every: int | None = None,
) -> tuple[dict[str, tuple[list[float], list[float]]], tuple[list[float], list[float]] | None, dict]:
    """Run all requested federated variants + (optionally) the centralized ceiling.

    Returns (curves_dict, ceiling_curve_or_None, summary_dict).
    """
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    if subset > 0:
        train_ds = Subset(train_ds, list(range(subset)))

    train_sets = _make_splits(
        train_ds,
        clients=clients,
        labels_per_client=labels_per_client,
        split=split,
        seed=seed,
    )
    print(
        f"split={split}, clients={clients}, labels_per_client={labels_per_client}; "
        f"per-client sample counts: {[len(s) for s in train_sets]}"
    )
    test_loader = DataLoader(test_ds, batch_size=256)

    curves: dict[str, tuple[list[float], list[float]]] = {}
    summary: dict = {
        "config": {
            "clients": clients,
            "labels_per_client": labels_per_client,
            "split": split,
            "rounds": rounds,
            "rank": rank,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "grad_clip": grad_clip,
        },
        "variants": {},
    }

    for variant in variants:
        cfg: MethodConfig = PRESETS[variant]
        print(f"-- {variant} ({cfg.name}) --")
        losses, accs = run_method(
            cfg,
            train_sets=train_sets,
            test_loader=test_loader,
            rank=rank,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            grad_clip=grad_clip,
            model_factory=PaperToyModel,
            log_every=log_every,
        )
        curves[cfg.name] = (losses, accs)
        summary["variants"][variant] = {
            "name": cfg.name,
            "final_loss": losses[-1],
            "final_acc": accs[-1],
            "best_acc": max(accs),
        }

    ceiling: tuple[list[float], list[float]] | None = None
    if not skip_ceiling:
        ceiling_cfg = PRESETS["centralized"]
        print(f"-- centralized ({ceiling_cfg.name}) --")
        ceiling = run_centralized(
            train_sets,
            test_loader=test_loader,
            rank=rank,
            rounds=rounds,
            local_epochs=local_epochs,
            lr=ceiling_cfg.lr_a,
            batch_size=batch_size,
            seed=seed,
            device=device,
            grad_clip=grad_clip,
            model_factory=PaperToyModel,
            log_every=log_every,
        )
        summary["variants"]["centralized"] = {
            "name": ceiling_cfg.name,
            "final_loss": ceiling[0][-1],
            "final_acc": ceiling[1][-1],
            "best_acc": max(ceiling[1]),
        }

    return curves, ceiling, summary


def _aggregate(arrs: list[list[float]]) -> tuple[list[float], list[float]]:
    """Per-round mean and SEM (std/sqrt(N)) across a list of equal-length curves."""
    arr = np.asarray(arrs, dtype=float)  # shape: (N, rounds)
    mean = arr.mean(axis=0)
    if arr.shape[0] <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean.tolist(), sem.tolist()




def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--labels-per-client", type=int, default=1)
    p.add_argument("--split", choices=("label", "iid"), default="label")
    p.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="comma-separated PRESETS keys",
    )
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="run each variant N times with seeds [seed, seed+1, ..., seed+N-1] "
        "and plot mean ± SEM",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="terminal log + JSON checkpoint cadence (rounds). "
        "Default: max(1, rounds // 10).",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--skip-ceiling", action="store_true")
    args = p.parse_args()

    if args.num_seeds < 1:
        raise SystemExit(f"--num-seeds must be >= 1; got {args.num_seeds}")

    variants = _parse_variants(args.variants)
    seeds = [args.seed + i for i in range(args.num_seeds)]
    log_every = args.log_every if args.log_every is not None else max(1, args.rounds // 10)
    if log_every < 1:
        raise SystemExit(f"--log-every must be >= 1; got {log_every}")

    if args.out is None:
        tag = (
            f"c{args.clients}_l{args.labels_per_client}_{args.split}_r{args.rank}"
            f"_s{args.seed}_n{args.num_seeds}"
        )
        args.out = Path(f"results/mnist_fig2_compare_{tag}.png")
    args.data_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(
        f"device={device}, variants={variants}, rounds={args.rounds}, "
        f"rank={args.rank}, local_epochs={args.local_epochs}, seeds={seeds}"
    )

    # Per-variant accumulators across seeds. Indexed by cfg.name (display label),
    # matching what plot_curves wants.
    losses_by_label: dict[str, list[list[float]]] = {}
    accs_by_label: dict[str, list[list[float]]] = {}
    ceiling_losses: list[list[float]] = []
    ceiling_accs: list[list[float]] = []
    # Per-variant (by preset key) per-seed scalar summaries for the JSON.
    per_seed_summary: dict[str, list[dict]] = {}

    for seed in seeds:
        print(f"\n========== seed {seed} ==========")
        curves, ceiling, summary = run_one(
            variants=variants,
            clients=args.clients,
            labels_per_client=args.labels_per_client,
            split=args.split,
            rounds=args.rounds,
            rank=args.rank,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            seed=seed,
            grad_clip=args.grad_clip,
            device=device,
            data_dir=args.data_dir,
            subset=args.subset,
            skip_ceiling=args.skip_ceiling,
            log_every=log_every,
        )
        for label, (loss, acc) in curves.items():
            losses_by_label.setdefault(label, []).append(loss)
            accs_by_label.setdefault(label, []).append(acc)
        if ceiling is not None:
            ceiling_losses.append(ceiling[0])
            ceiling_accs.append(ceiling[1])
        for vkey, info in summary["variants"].items():
            per_seed_summary.setdefault(vkey, []).append(
                {
                    "seed": seed,
                    "final_loss": info["final_loss"],
                    "final_acc": info["final_acc"],
                    "best_acc": info["best_acc"],
                }
            )

    # run_method and run_centralized already return sparse curves aligned with
    # checkpoint_indices(rounds, log_every). Just aggregate.
    cp_idx = checkpoint_indices(args.rounds, log_every)

    mean_curves: dict[str, tuple[list[float], list[float]]] = {}
    sem_curves: dict[str, tuple[list[float], list[float]]] = {}
    per_variant_per_seed_sparse: dict[str, list[dict]] = {}
    for label in losses_by_label:
        sparse_losses = losses_by_label[label]
        sparse_accs = accs_by_label[label]
        loss_mean, loss_sem = _aggregate(sparse_losses)
        acc_mean, acc_sem = _aggregate(sparse_accs)
        mean_curves[label] = (loss_mean, acc_mean)
        sem_curves[label] = (loss_sem, acc_sem)
        per_variant_per_seed_sparse[label] = [
            {"seed": seeds[i], "losses": sparse_losses[i], "accs": sparse_accs[i]}
            for i in range(len(sparse_losses))
        ]

    ceiling_arg = None
    ceiling_sem_arg = None
    ceiling_per_seed_sparse: list[dict] | None = None
    if ceiling_losses:
        c_loss_mean, c_loss_sem = _aggregate(ceiling_losses)
        c_acc_mean, c_acc_sem = _aggregate(ceiling_accs)
        ceiling_arg = ("Centralized (ceiling)", c_loss_mean, c_acc_mean)
        ceiling_sem_arg = (c_loss_sem, c_acc_sem)
        ceiling_per_seed_sparse = [
            {"seed": seeds[i], "losses": ceiling_losses[i], "accs": ceiling_accs[i]}
            for i in range(len(ceiling_losses))
        ]

    suffix = f", {args.num_seeds} seeds (mean ± SEM)" if args.num_seeds > 1 else ""
    plot_curves(
        mean_curves,
        args.out,
        xs=cp_idx,
        title=(
            f"RoLoRA improvements — MNIST, {args.clients} clients "
            f"(split={args.split}, lpc={args.labels_per_client}), rank {args.rank}{suffix}"
        ),
        ceiling=ceiling_arg,
        curve_sems=sem_curves if args.num_seeds > 1 else None,
        ceiling_sem=ceiling_sem_arg if args.num_seeds > 1 else None,
    )
    print(f"saved {args.out}")

    agg_summary: dict = {
        "config": {
            "clients": args.clients,
            "labels_per_client": args.labels_per_client,
            "split": args.split,
            "rounds": args.rounds,
            "rank": args.rank,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "grad_clip": args.grad_clip,
        },
        "seeds": seeds,
        "num_seeds": args.num_seeds,
        "checkpoint_rounds": cp_idx,
        "variants": {},
    }
    # Build a quick lookup from preset key → display label so we can pull the
    # right sparse arrays out of per_variant_per_seed_sparse.
    label_for_key = {k: PRESETS[k].name for k in PRESETS}
    for vkey, rows in per_seed_summary.items():
        finals = [r["final_acc"] for r in rows]
        bests = [r["best_acc"] for r in rows]
        n = len(finals)
        if vkey == "centralized":
            sparse_rows = ceiling_per_seed_sparse or []
            loss_mean, acc_mean = (ceiling_arg[1], ceiling_arg[2]) if ceiling_arg else ([], [])
            loss_sem, acc_sem = ceiling_sem_arg if ceiling_sem_arg else ([], [])
        else:
            label = label_for_key[vkey]
            sparse_rows = per_variant_per_seed_sparse.get(label, [])
            loss_mean, acc_mean = mean_curves.get(label, ([], []))
            loss_sem, acc_sem = sem_curves.get(label, ([], []))
        # Merge per-seed scalars into the sparse rows so each row is self-contained.
        merged_rows = []
        for s_row, scalar_row in zip(sparse_rows, rows, strict=False):
            merged = dict(s_row)
            merged.update(
                {
                    "final_loss": scalar_row["final_loss"],
                    "final_acc": scalar_row["final_acc"],
                    "best_acc": scalar_row["best_acc"],
                }
            )
            merged_rows.append(merged)
        agg_summary["variants"][vkey] = {
            "name": PRESETS[vkey].name,
            "per_seed": merged_rows,
            "loss_mean": loss_mean,
            "loss_sem": loss_sem,
            "acc_mean": acc_mean,
            "acc_sem": acc_sem,
            "final_acc_mean": float(np.mean(finals)),
            "final_acc_sem": float(np.std(finals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "best_acc_mean": float(np.mean(bests)),
            "best_acc_sem": float(np.std(bests, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        }
    json_out = args.out.with_suffix(".json")
    json_out.write_text(json.dumps(agg_summary, indent=2))
    print(f"saved {json_out}")

    print(f"\nfinal accuracies (mean ± SEM across {args.num_seeds} seed(s)):")
    for vkey, info in agg_summary["variants"].items():
        print(
            f"  {vkey:24} {info['final_acc_mean']:.4f} ± {info['final_acc_sem']:.4f}  "
            f"(best {info['best_acc_mean']:.4f} ± {info['best_acc_sem']:.4f})"
        )


if __name__ == "__main__":
    main()
