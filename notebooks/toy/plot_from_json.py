"""Re-render the comparison plot from a saved `mnist_fig2_compare` JSON.

Reads the sparse per-checkpoint arrays written by
`notebooks/mnist_fig2_compare.py` and produces the same two-panel PNG without
re-running training. Useful for tweaking the plot (title, output path) after
the fact, or for plotting old results on a new machine.

Usage:
    uv run python notebooks/toy/plot_from_json.py \\
        results/mnist_fig2_compare_c10_l1_label_r16_s0_n3.json \\
        --out results/replot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importing the `toy` package requires `notebooks/` on sys.path when this is
# invoked as a script.
_NOTEBOOKS = Path(__file__).resolve().parent.parent
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

from toy.plotting import plot_curves  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json", type=Path, help="path to a *_compare_*.json")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument(
        "--no-bands",
        action="store_true",
        help="hide ±SEM shaded bands even when the JSON has multi-seed data",
    )
    args = p.parse_args()

    data = json.loads(args.json.read_text())
    cfg = data["config"]
    cp_rounds = data["checkpoint_rounds"]
    num_seeds = data.get("num_seeds", 1)

    if args.out is None:
        args.out = args.json.with_suffix(".replot.png")

    curves: dict[str, tuple[list[float], list[float]]] = {}
    sems: dict[str, tuple[list[float], list[float]]] = {}
    ceiling = None
    ceiling_sem = None
    for vkey, info in data["variants"].items():
        loss_mean = info["loss_mean"]
        acc_mean = info["acc_mean"]
        loss_sem = info["loss_sem"]
        acc_sem = info["acc_sem"]
        if vkey == "centralized":
            ceiling = ("Centralized (ceiling)", loss_mean, acc_mean)
            ceiling_sem = (loss_sem, acc_sem)
        else:
            curves[info["name"]] = (loss_mean, acc_mean)
            sems[info["name"]] = (loss_sem, acc_sem)

    show_bands = num_seeds > 1 and not args.no_bands
    suffix = f", {num_seeds} seeds (mean ± SEM)" if show_bands else ""
    title = args.title or (
        f"RoLoRA improvements — MNIST, {cfg['clients']} clients "
        f"(split={cfg['split']}, lpc={cfg['labels_per_client']}), "
        f"rank {cfg['rank']}{suffix}"
    )

    plot_curves(
        curves,
        args.out,
        xs=cp_rounds,
        title=title,
        ceiling=ceiling,
        curve_sems=sems if show_bands else None,
        ceiling_sem=ceiling_sem if show_bands else None,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
