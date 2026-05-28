"""Sweep orchestrator for the (clients, labels_per_client) × seeds stress grid.

For each cell, runs the requested variants + centralized ceiling via
``mnist_fig2_compare.run_one``, writes one PNG per (clients, labels_per_client,
seed) cell, and appends rows to a single aggregated CSV
(`results/mnist_fig2_sweep_<tag>.csv`) for downstream analysis.

Default grid hits the paper-faithful settings (5c×2, 10c×1) plus two stress
points (20c×1 = 2 owners/class, 50c×1 = 5 owners/class).

Usage:
    uv run python -m toy.sweep --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Importing `mnist_fig2_compare` requires `notebooks/` on sys.path. When this
# module is invoked as `python -m toy.sweep`, sys.path[0] is the cwd, so we
# add the package's parent (= `notebooks/`) explicitly.
_NOTEBOOKS = Path(__file__).resolve().parent.parent
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

import mnist_fig2_compare  # noqa: E402

from toy import pick_device  # noqa: E402
from toy.config import PRESETS  # noqa: E402
from toy.plotting import plot_curves  # noqa: E402
from toy.rounds import checkpoint_indices  # noqa: E402


def _parse_grid(raw: str) -> list[tuple[int, int]]:
    """Parse ``--grid '5,2 10,1 20,1 50,1'`` into [(5,2), (10,1), (20,1), (50,1)]."""
    pairs = []
    for tok in raw.split():
        c, lpc = tok.split(",")
        pairs.append((int(c), int(lpc)))
    return pairs


def _parse_ints(raw: str) -> list[int]:
    return [int(s) for s in raw.split(",") if s.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--grid",
        type=str,
        default="5,2 10,1 20,1 50,1",
        help="space-separated 'clients,labels_per_client' pairs",
    )
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument(
        "--variants",
        type=str,
        default=",".join(mnist_fig2_compare.DEFAULT_VARIANTS),
    )
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--split", choices=("label", "iid"), default="label")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("results/sweep"))
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--skip-ceiling", action="store_true")
    args = p.parse_args()

    grid = _parse_grid(args.grid)
    seeds = _parse_ints(args.seeds)
    variants = mnist_fig2_compare._parse_variants(args.variants)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.csv is None:
        args.csv = args.out_dir / "summary.csv"

    device = pick_device()
    print(
        f"device={device}, grid={grid}, seeds={seeds}, variants={variants}, "
        f"rounds={args.rounds}, rank={args.rank}"
    )

    # Aggregate rows in memory, dump once at the end (cheap; sweep is small).
    rows: list[dict] = []
    fieldnames = [
        "variant",
        "clients",
        "labels_per_client",
        "seed",
        "rounds",
        "rank",
        "final_loss",
        "final_acc",
        "best_acc",
        "auc_acc",
    ]

    for clients, lpc in grid:
        for seed in seeds:
            print(f"\n=== cell: clients={clients}, lpc={lpc}, seed={seed} ===")
            curves, ceiling, summary = mnist_fig2_compare.run_one(
                variants=variants,
                clients=clients,
                labels_per_client=lpc,
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
            )

            cell_tag = f"c{clients}_l{lpc}_s{seed}"
            cell_png = args.out_dir / f"{cell_tag}.png"
            ceiling_arg = (
                ("Centralized (ceiling)", ceiling[0], ceiling[1]) if ceiling is not None else None
            )
            cp_idx = checkpoint_indices(args.rounds, max(1, args.rounds // 10))
            plot_curves(
                curves,
                cell_png,
                xs=cp_idx,
                title=f"clients={clients}, lpc={lpc}, seed={seed}",
                ceiling=ceiling_arg,
            )
            print(f"saved {cell_png}")

            for vname, info in summary["variants"].items():
                losses_accs = ceiling if vname == "centralized" else curves[PRESETS[vname].name]
                accs = losses_accs[1]
                rows.append(
                    {
                        "variant": vname,
                        "clients": clients,
                        "labels_per_client": lpc,
                        "seed": seed,
                        "rounds": args.rounds,
                        "rank": args.rank,
                        "final_loss": info["final_loss"],
                        "final_acc": info["final_acc"],
                        "best_acc": info["best_acc"],
                        "auc_acc": sum(accs) / len(accs),
                    }
                )

    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
