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

    # use raw std instead of SEM for the bands
    uv run python notebooks/toy/plot_from_json.py results/foo.json --band std

All JSONs must share the same `checkpoint_rounds`; if any disagree the script
exits with an error. The ceiling is taken from the first file that has one.
On duplicate variant keys, the first file wins (later files' duplicates are
skipped with a warning).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Importing the `toy` package requires `notebooks/` on sys.path when this is
# invoked as a script.
_NOTEBOOKS = Path(__file__).resolve().parent.parent
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from toy.plotting import plot_curves  # noqa: E402


def _plot_single_panel(
    curves: dict[str, tuple[list[float], list[float]]],
    sems: dict[str, tuple[list[float], list[float]]] | None,
    ceiling: tuple[str, list[float], list[float]] | None,
    ceiling_sem: tuple[list[float], list[float]] | None,
    xs: list[int],
    component: int,  # 0 = loss, 1 = accuracy
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    """Render one panel (loss or accuracy) to `out`. plotting.py untouched."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")
    for label, pair in curves.items():
        series = pair[component]
        (line,) = ax.plot(xs, series, label=label)
        if sems is not None and label in sems:
            band = sems[label][component]
            ax.fill_between(
                xs,
                [m - s for m, s in zip(series, band, strict=True)],
                [m + s for m, s in zip(series, band, strict=True)],
                color=line.get_color(),
                alpha=0.2,
                linewidth=0,
            )
    if ceiling is not None:
        c_series = ceiling[1 + component]
        ax.plot(xs, c_series, label=ceiling[0], linestyle="--", color="black", linewidth=1.2)
        if ceiling_sem is not None:
            c_band = ceiling_sem[component]
            ax.fill_between(
                xs,
                [m - s for m, s in zip(c_series, c_band, strict=True)],
                [m + s for m, s in zip(c_series, c_band, strict=True)],
                color="black",
                alpha=0.15,
                linewidth=0,
            )
    ax.set_xlabel("communication round")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    # constrained_layout sizes title space automatically; wrap=True still
    # protects against absurdly long single-line custom titles.
    fig.suptitle(title, fontsize=10, wrap=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

# Two-tailed t-critical values at α=0.05, indexed by N (df = N-1). Covers the
# realistic seed-count range; for N > 30 we use the large-sample normal
# approximation (1.96).
_T_CRIT_95: dict[int, float] = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    12: 2.201,
    15: 2.145,
    20: 2.093,
    25: 2.064,
    30: 2.045,
}


def _t_crit_95(n: int) -> float:
    if n <= 1:
        return 0.0
    if n in _T_CRIT_95:
        return _T_CRIT_95[n]
    if n > 30:
        return 1.96
    # Linear interpolation between nearest table entries for unlisted N.
    keys = sorted(_T_CRIT_95)
    lo = max(k for k in keys if k <= n)
    hi = min(k for k in keys if k >= n)
    if lo == hi:
        return _T_CRIT_95[lo]
    t_lo, t_hi = _T_CRIT_95[lo], _T_CRIT_95[hi]
    return t_lo + (t_hi - t_lo) * (n - lo) / (hi - lo)


def _band_scale(band: str, n: int) -> float:
    """Multiplier applied to the *stored* SEM to convert it to the chosen band.

    The JSON stores ``sem = std / sqrt(n)``, so:
      - SEM → SEM:   scale = 1
      - SEM → std:   scale = sqrt(n)
      - SEM → 95% CI: scale = t_crit(df=n-1, 0.975)
    """
    if n <= 1:
        return 0.0
    if band == "sem":
        return 1.0
    if band == "std":
        return math.sqrt(n)
    if band == "ci95":
        return _t_crit_95(n)
    raise ValueError(f"unknown --band: {band!r}")


def _band_label(band: str) -> str:
    return {"sem": "SEM", "std": "std", "ci95": "95% CI"}[band]


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
        "--band",
        choices=("sem", "std", "ci95"),
        default="sem",
        help="shaded-band style: 'sem' (std/sqrt(N), default), 'std' (raw std), "
        "or 'ci95' (95%% confidence interval via t-distribution)",
    )
    p.add_argument(
        "--no-bands",
        action="store_true",
        help="hide shaded bands even when the JSONs have multi-seed data",
    )
    p.add_argument(
        "--panels",
        choices=("both", "loss", "acc"),
        default="both",
        help="which panel(s) to render: 'both' (loss + accuracy, default), "
        "'loss' only, or 'acc' only",
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

    # Min seeds across files — used for the title suffix (most-conservative
    # claim) and for show_bands gating. Band scaling itself uses per-variant N
    # since each variant came from a specific file.
    num_seeds = min(d.get("num_seeds", 1) for _, d in datasets)

    curves: dict[str, tuple[list[float], list[float]]] = {}
    sems: dict[str, tuple[list[float], list[float]]] = {}
    ceiling = None
    ceiling_sem = None
    seen_variant_keys: set[str] = set()

    def _scaled(sem_list: list[float], n: int) -> list[float]:
        s = _band_scale(args.band, n)
        return [x * s for x in sem_list]

    for path, data in datasets:
        file_n = data.get("num_seeds", 1)
        for vkey, info in data["variants"].items():
            loss_mean = info["loss_mean"]
            acc_mean = info["acc_mean"]
            loss_band = _scaled(info["loss_sem"], file_n)
            acc_band = _scaled(info["acc_sem"], file_n)
            if vkey == "centralized":
                if ceiling is None:
                    ceiling = ("Centralized (ceiling)", loss_mean, acc_mean)
                    ceiling_sem = (loss_band, acc_band)
                continue
            if vkey in seen_variant_keys:
                print(f"  skipping duplicate variant '{vkey}' from {path}")
                continue
            seen_variant_keys.add(vkey)
            curves[info["name"]] = (loss_mean, acc_mean)
            sems[info["name"]] = (loss_band, acc_band)

    if args.out is None:
        args.out = datasets[0][0].with_suffix(".replot.png")

    cfg = datasets[0][1]["config"]
    show_bands = num_seeds > 1 and not args.no_bands
    band_label = _band_label(args.band)
    if args.title is not None:
        title = args.title
    elif args.panels == "both":
        suffix = f", {num_seeds} seeds (mean ± {band_label})" if show_bands else ""
        title = (
            f"RoLoRA improvements — MNIST, {cfg['clients']} clients "
            f"(split={cfg['split']}, lpc={cfg['labels_per_client']}), "
            f"rank {cfg['rank']}{suffix}"
        )
    else:
        # Single-panel auto-title: split onto two lines so it fits the 6-inch
        # figure width and stays informative.
        what = "accuracy" if args.panels == "acc" else "loss"
        suffix = f"  ({num_seeds} seeds, ± {band_label})" if show_bands else ""
        title = (
            f"MNIST test {what} — {cfg['clients']} clients (lpc={cfg['labels_per_client']}), "
            f"rank {cfg['rank']}\n"
            f"RoLoRA improvements{suffix}"
        )

    if args.panels == "both":
        plot_curves(
            curves,
            args.out,
            xs=cp_rounds,
            title=title,
            ceiling=ceiling,
            curve_sems=sems if show_bands else None,
            ceiling_sem=ceiling_sem if show_bands else None,
        )
    else:
        component, ylabel = (
            (0, "test cross-entropy") if args.panels == "loss" else (1, "test accuracy")
        )
        _plot_single_panel(
            curves,
            sems=sems if show_bands else None,
            ceiling=ceiling,
            ceiling_sem=ceiling_sem if show_bands else None,
            xs=cp_rounds,
            component=component,
            ylabel=ylabel,
            title=title,
            out=args.out,
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
