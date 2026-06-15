#!/usr/bin/env python3
"""Run fast MNIST toy experiments for RoLoRA improvement triage.

This is the direct CPU loop for deciding which ideas deserve RoBERTa runs.
It uses `notebooks/mnist_fig2.py` as a library, runs a small bank of variants,
and writes both machine-readable JSON and a short Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))
import mnist_fig2  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    method: str = "rolora"
    init: str = "default"
    phase_pattern: str | None = None
    participation_rate: float = 1.0
    sync_policy: str = "full"
    transport: bool = False
    gauge: bool = False
    drift_mu: float = 0.0
    note: str = ""


VARIANTS: dict[str, Variant] = {
    "lora": Variant(
        name="lora",
        method="lora",
        note="standard LoRA; averages A and B separately",
    ),
    "ffa": Variant(
        name="ffa",
        method="ffa_lora",
        note="FFA-LoRA; freezes initial A and trains B",
    ),
    "rolora": Variant(
        name="rolora",
        note="vanilla RoLoRA B/A alternation",
    ),
    "orth": Variant(
        name="orth",
        init="orthogonal_a",
        note="orthogonal-A initialization; proposal axis 1",
    ),
    "orth_bba": Variant(
        name="orth_bba",
        init="orthogonal_a",
        phase_pattern="BBA",
        note="orthogonal-A plus BBA phase pattern; tests init x schedule interaction",
    ),
    "orth_bbba": Variant(
        name="orth_bbba",
        init="orthogonal_a",
        phase_pattern="BBBA",
        note="orthogonal-A plus BBBA phase pattern; longer B warm-up before A motion",
    ),
    "orth_bbbba": Variant(
        name="orth_bbbba",
        init="orthogonal_a",
        phase_pattern="BBBBA",
        note="orthogonal-A plus BBBBA phase pattern; aggressive B warm-up",
    ),
    "orth_transport": Variant(
        name="orth_transport",
        init="orthogonal_a",
        transport=True,
        note="basis transport after A-rounds",
    ),
    "partial": Variant(
        name="partial",
        init="orthogonal_a",
        participation_rate=0.5,
        note="ordinary partial participation with full sampled-client sync",
    ),
    "partial_stale": Variant(
        name="partial_stale",
        init="orthogonal_a",
        participation_rate=0.5,
        sync_policy="active_only",
        note="stress test: sampled clients receive only active factor, exposing stale frozen factors",
    ),
    "partial_stale_transport": Variant(
        name="partial_stale_transport",
        init="orthogonal_a",
        participation_rate=0.5,
        sync_policy="active_only",
        transport=True,
        note="basis transport under the stale-factor stress test",
    ),
    "svd": Variant(
        name="svd",
        init="svd_compensated",
        note="SVD-compensated init; proposal axis 1 variant",
    ),
    "svd_bba": Variant(
        name="svd_bba",
        init="svd_compensated",
        phase_pattern="BBA",
        note="SVD-compensated init plus BBA; tests whether compensated bases need slower A updates",
    ),
    # --- factor-wise drift correction (new idea, 2026-06-15) -----------------
    # RoLoRA makes aggregation exact but still FedAvgs the trained factor across
    # heterogeneous clients, so within-factor client drift is untouched -- the
    # residual heterogeneity gap the paper never addresses. These variants add a
    # FedProx-style proximal anchor to the single alternating factor each round.
    "rolora_fedprox_lo": Variant(
        name="rolora_fedprox_lo",
        drift_mu=0.02,
        note="RoLoRA + factor-wise FedProx drift correction (mu=0.02)",
    ),
    "rolora_fedprox": Variant(
        name="rolora_fedprox",
        drift_mu=0.1,
        note="RoLoRA + factor-wise FedProx drift correction (mu=0.1)",
    ),
    "rolora_fedprox_hi": Variant(
        name="rolora_fedprox_hi",
        drift_mu=0.5,
        note="RoLoRA + factor-wise FedProx drift correction (mu=0.5)",
    ),
    "orth_fedprox": Variant(
        name="orth_fedprox",
        init="orthogonal_a",
        drift_mu=0.1,
        note="orthogonal-A + factor-wise FedProx; stacks best init with drift correction",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="rolora,orth,orth_bba,orth_transport,partial,partial_stale,partial_stale_transport",
        help=f"Comma-separated variant keys. Available: {', '.join(sorted(VARIANTS))}",
    )
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated integer seeds.")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--local-steps", type=int, default=10)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset", type=int, default=4000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--split", choices=mnist_fig2.DATA_SPLITS, default="iid")
    parser.add_argument("--labels-per-client", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("evidence/toy_improvement_bank_latest"),
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_variant_keys(value: str) -> list[str]:
    keys = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(keys) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variant key(s): {unknown}. Available: {sorted(VARIANTS)}")
    return keys


def make_loaders(
    *,
    data_dir: Path,
    subset: int,
    test_subset: int,
    clients: int,
    seed: int,
    split: str,
    labels_per_client: int,
) -> tuple[list[Subset], DataLoader]:
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    if subset > 0:
        train_ds = Subset(train_ds, list(range(subset)))
    if test_subset > 0:
        test_ds = Subset(test_ds, list(range(test_subset)))
    rng = np.random.default_rng(seed)
    train_sets = mnist_fig2.split_clients(
        train_ds,
        clients,
        rng,
        split=split,
        labels_per_client=labels_per_client,
    )
    return train_sets, DataLoader(test_ds, batch_size=256)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0}


def write_markdown(outdir: Path, result: dict) -> None:
    lines = [
        "# Toy improvement bank",
        "",
        "Fast MNIST toy triage for RoLoRA improvement candidates. This is not a GLUE claim;",
        "it is a cheap filter for which ideas deserve RoBERTa-base / RoBERTa-Large runs.",
        "",
        "## Configuration",
        "",
        f"- clients: {result['config']['clients']}",
        f"- rounds: {result['config']['rounds']}",
        f"- local steps: {result['config']['local_steps']}",
        f"- rank: {result['config']['rank']}",
        f"- split: {result['config']['split']}",
        f"- labels per client: {result['config']['labels_per_client']}",
        f"- train subset: {result['config']['subset']}",
        f"- test subset: {result['config']['test_subset']}",
        f"- seeds: {', '.join(map(str, result['config']['seeds']))}",
        "",
        "## Results",
        "",
        "| Variant | Final acc mean ± std | Best acc mean ± std | Note |",
        "|---|---:|---:|---|",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['name']} | "
            f"{row['final_acc_mean'] * 100:.2f} ± {row['final_acc_std'] * 100:.2f} | "
            f"{row['best_acc_mean'] * 100:.2f} ± {row['best_acc_std'] * 100:.2f} | "
            f"{row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Promote a candidate to RoBERTa only if it beats vanilla RoLoRA in this toy",
            "bank by at least about 1 pp on mean final or best accuracy, or if it is a",
            "mechanism-specific stress test such as partial stale-basis transport.",
            "",
        ]
    )
    (outdir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    variant_keys = parse_variant_keys(args.variants)
    seeds = parse_csv_ints(args.seeds)
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs: list[dict] = []
    for seed in seeds:
        train_sets, test_loader = make_loaders(
            data_dir=args.data_dir,
            subset=args.subset,
            test_subset=args.test_subset,
            clients=args.clients,
            seed=seed,
            split=args.split,
            labels_per_client=args.labels_per_client,
        )
        for key in variant_keys:
            variant = VARIANTS[key]
            print(f"-- {variant.name} seed={seed} --", flush=True)
            losses, accs = mnist_fig2.run_method(
                variant.method,
                train_sets=train_sets,
                test_loader=test_loader,
                rank=args.rank,
                rounds=args.rounds,
                local_steps=args.local_steps,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=seed,
                device=device,
                grad_clip=args.grad_clip,
                init_variant=variant.init,
                phase_pattern=variant.phase_pattern,
                participation_rate=variant.participation_rate,
                sync_policy=variant.sync_policy,
                transport=variant.transport,
                gauge=variant.gauge,
                drift_mu=variant.drift_mu,
            )
            runs.append(
                {
                    "variant": asdict(variant),
                    "seed": seed,
                    "final_loss": losses[-1],
                    "final_acc": accs[-1],
                    "best_acc": max(accs),
                    "losses": losses,
                    "accs": accs,
                }
            )

    summary = []
    for key in variant_keys:
        variant = VARIANTS[key]
        selected = [run for run in runs if run["variant"]["name"] == variant.name]
        final_acc = summarize([float(run["final_acc"]) for run in selected])
        best_acc = summarize([float(run["best_acc"]) for run in selected])
        summary.append(
            {
                "name": variant.name,
                "final_acc_mean": final_acc["mean"],
                "final_acc_std": final_acc["std"],
                "best_acc_mean": best_acc["mean"],
                "best_acc_std": best_acc["std"],
                "note": variant.note,
            }
        )

    result = {
        "config": {
            "clients": args.clients,
            "rounds": args.rounds,
            "local_steps": args.local_steps,
            "rank": args.rank,
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "split": args.split,
            "labels_per_client": args.labels_per_client,
            "subset": args.subset,
            "test_subset": args.test_subset,
            "seeds": seeds,
            "device": str(device),
        },
        "summary": summary,
        "runs": runs,
    }
    (args.outdir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    write_markdown(args.outdir, result)
    print(f"wrote {args.outdir / 'results.json'}")
    print(f"wrote {args.outdir / 'summary.md'}")


if __name__ == "__main__":
    main()
