"""Re-render the comparison plot from one or more saved `mnist_fig2_compare` JSONs.

Reads the sparse per-checkpoint arrays written by
`notebooks/mnist_fig2_compare.py` and produces the two-panel PNG without
re-running training. Multiple JSON files can be supplied — their variants are
merged onto a single figure. Useful for stitching together a baselines run +
one-off variant runs that were generated separately.

Usage:
    # single-file replot
    uv run python notebooks/toy/plot_from_json.py \\
        results/foo.json --out results/replot.png

    # merge several runs into one figure
    uv run python notebooks/toy/plot_from_json.py \\
        results_extra/baselines_n5_r100_log5.json \\
        results_extra/orth_a_n5_r100_log5.json \\
        --out results_extra/baselines_plus_orth_a.png

All JSONs must share the same `checkpoint_rounds`; if any disagree the script
exits with an error. The ceiling is taken from the first file that has one.
On duplicate variant keys, the first file wins (later files' duplicates are
skipped with a warning).
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
    p.add_argument(
        "json",
        type=Path,
        nargs="+",
        help="one or more *_compare_*.json files; variants are merged",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument(
        "--no-bands",
        action="store_true",
        help="hide ±SEM shaded bands even when the JSONs have multi-seed data",
    )
    args = p.parse_args()

    datasets = [(path, json.loads(path.read_text())) for path in args.json]

    cp_rounds = datasets[0][1]["checkpoint_rounds"]
    for path, data in datasets[1:]:
        if data["checkpoint_rounds"] != cp_rounds:
            raise SystemExit(
                f"checkpoint_rounds mismatch:\n"
                f"  {datasets[0][0]} → {cp_rounds}\n"
                f"  {path} → {data['checkpoint_rounds']}\n"
                "all JSONs must share the same x-axis."
            )

    # num_seeds for the suffix: use min across files (the most-conservative
    # claim about how many seeds back the curves).
    num_seeds = min(d.get("num_seeds", 1) for _, d in datasets)

    curves: dict[str, tuple[list[float], list[float]]] = {}
    sems: dict[str, tuple[list[float], list[float]]] = {}
    ceiling = None
    ceiling_sem = None
    seen_variant_keys: set[str] = set()

    for path, data in datasets:
        for vkey, info in data["variants"].items():
            loss_mean = info["loss_mean"]
            acc_mean = info["acc_mean"]
            loss_sem = info["loss_sem"]
            acc_sem = info["acc_sem"]
            if vkey == "centralized":
                if ceiling is None:
                    ceiling = ("Centralized (ceiling)", loss_mean, acc_mean)
                    ceiling_sem = (loss_sem, acc_sem)
                continue
            if vkey in seen_variant_keys:
                print(f"  skipping duplicate variant '{vkey}' from {path}")
                continue
            seen_variant_keys.add(vkey)
            curves[info["name"]] = (loss_mean, acc_mean)
            sems[info["name"]] = (loss_sem, acc_sem)

    if args.out is None:
        args.out = datasets[0][0].with_suffix(".replot.png")

    cfg = datasets[0][1]["config"]
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
