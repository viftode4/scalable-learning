#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Plot per-run supplement diagnostics extracted from FederatedScope logs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PHASE_COLORS = {"A": "#ff7f0e", "B": "#1f77b4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_dir",
        type=Path,
        help="Directory containing server_metrics.csv, client_*_metrics.csv, phase_markers.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <metrics_dir>/figures.",
    )
    parser.add_argument(
        "--title",
        default="RoLoRA supplement diagnostics",
        help="Figure title prefix.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def as_int(value: str | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def percent(value: float) -> float:
    return value * 100.0


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def aggregate(rows: list[dict[str, str]], metric: str) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(metric))
        if math.isnan(value):
            continue
        grouped[as_int(row.get("round"))].append(value)
    out: dict[int, dict[str, float]] = {}
    for round_id, values in grouped.items():
        vals = finite(values)
        out[round_id] = {
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": float(len(vals)),
        }
    return out


def phase_by_round(rows: list[dict[str, str]]) -> dict[int, str]:
    phases: dict[int, str] = {}
    for row in rows:
        phases[as_int(row.get("round"))] = row.get("phase", "")
    return phases


def metadata(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row.get("key", ""): row.get("value", "") for row in rows}


def inferred_phase(round_id: int, meta: dict[str, str]) -> str:
    if meta.get("mode") != "rolora":
        return ""
    pattern = meta.get("sls_phase_pattern", "")
    if pattern and pattern != "default":
        compact = "".join(pattern.replace(",", " ").replace(";", " ").split()).upper()
        if compact and set(compact) <= {"A", "B"}:
            return compact[round_id % len(compact)]
    warmup = meta.get("sls_b_warmup_rounds", "")
    if warmup and warmup != "default":
        try:
            warmup_rounds = int(warmup)
        except ValueError:
            warmup_rounds = 0
        if warmup_rounds > 0:
            if round_id < warmup_rounds:
                return "B"
            return "A" if (round_id - warmup_rounds) % 2 == 0 else "B"
    return "B" if round_id % 2 == 0 else "A"


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return paths


def annotate_phases(ax: plt.Axes, phases: dict[int, str], y: float, scale: float = 1.0) -> None:
    for round_id, phase in sorted(phases.items()):
        color = PHASE_COLORS.get(phase, "#777777")
        ax.text(
            round_id,
            y,
            phase,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": color, "edgecolor": "none"},
        )
        ax.axvspan(round_id - 0.5, round_id + 0.5, color=color, alpha=0.05 * scale)


def rows_by_kind(rows: list[dict[str, str]], kind: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("kind") == kind]


def series(rows: list[dict[str, str]], metric: str) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("round") in (None, "") or row.get(metric) in (None, ""):
            continue
        value = as_float(row.get(metric))
        if not math.isnan(value):
            grouped[as_int(row.get("round"))].append(value)
    pairs = [
        (round_id, statistics.fmean(values))
        for round_id, values in grouped.items()
        if values
    ]
    pairs.sort(key=lambda item: item[0])
    return [round_id for round_id, _ in pairs], [value for _, value in pairs]


def legend_if_present(ax: plt.Axes, **kwargs: object) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(**kwargs)


def plot_line_if_present(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    x, y = series(rows, metric)
    if x:
        ax.plot(x, y, marker="o", linewidth=1.8, color=color, linestyle=linestyle, label=label)


def plot_internal_monitors(
    internal_rows: list[dict[str, str]],
    agg_rows: list[dict[str, str]],
    phases: dict[int, str],
    output_dir: Path,
    title: str,
) -> None:
    if not internal_rows and not agg_rows:
        return

    fit_start = rows_by_kind(internal_rows, "fit_start")
    fit_end = rows_by_kind(internal_rows, "fit_end")
    pre = rows_by_kind(internal_rows, "global_pre_train")
    post = rows_by_kind(internal_rows, "client_post_train")
    drift = rows_by_kind(agg_rows, "aggregate_client_drift")
    result = rows_by_kind(agg_rows, "aggregate_result")
    if not (fit_start or fit_end or pre or post or drift or result):
        return

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.0), sharex=True)
    ax_norm, ax_local, ax_agg = axes

    norm_rows = pre or fit_start
    local_rows = post or fit_end
    local_prefix = "client #1" if post else "mean local"

    plot_line_if_present(ax_norm, norm_rows, "lora_A_norm", label="||A|| before train", color="#ff7f0e")
    plot_line_if_present(ax_norm, norm_rows, "lora_B_norm", label="||B|| before train", color="#1f77b4")
    plot_line_if_present(ax_norm, norm_rows, "lora_product_norm", label="||BA|| before train", color="#9467bd")
    plot_line_if_present(ax_norm, norm_rows, "classifier_norm", label="classifier before train", color="#2ca02c")
    ax_norm.set_title(f"{title}: internal factor norms")
    ax_norm.set_ylabel("L2 / Frobenius norm")
    ax_norm.grid(True, alpha=0.25)
    legend_if_present(ax_norm, frameon=False, fontsize=8)

    plot_line_if_present(ax_local, local_rows, "update_lora_A_norm", label=f"{local_prefix} ΔA", color="#ff7f0e")
    plot_line_if_present(ax_local, local_rows, "update_lora_B_norm", label=f"{local_prefix} ΔB", color="#1f77b4")
    plot_line_if_present(ax_local, local_rows, "update_classifier_norm", label=f"{local_prefix} Δclassifier", color="#2ca02c")
    ax_local.set_ylabel("Local update norm")
    ax_local.grid(True, alpha=0.25)
    legend_if_present(ax_local, frameon=False, fontsize=8)

    plot_line_if_present(ax_agg, drift, "client_delta_lora_A_mean", label="mean client ΔA", color="#ff7f0e")
    plot_line_if_present(ax_agg, drift, "client_delta_lora_B_mean", label="mean client ΔB", color="#1f77b4")
    plot_line_if_present(ax_agg, drift, "client_delta_classifier_mean", label="mean client Δclassifier", color="#2ca02c")
    plot_line_if_present(ax_agg, result, "aggregate_update_lora_A_norm", label="aggregate ΔA", color="#ff7f0e", linestyle="--")
    plot_line_if_present(ax_agg, result, "aggregate_update_lora_B_norm", label="aggregate ΔB", color="#1f77b4", linestyle="--")
    plot_line_if_present(ax_agg, result, "aggregate_update_classifier_norm", label="aggregate Δclassifier", color="#2ca02c", linestyle="--")
    ax_agg.set_xlabel("Communication round")
    ax_agg.set_ylabel("Aggregation/update norm")
    ax_agg.grid(True, alpha=0.25)
    legend_if_present(ax_agg, frameon=False, fontsize=8, ncols=2)

    all_rounds = set(phases)
    for rows in (fit_start, fit_end, pre, post, drift, result):
        all_rounds |= {as_int(row.get("round")) for row in rows if row.get("round") not in (None, "")}
    for ax in axes:
        ylims = ax.get_ylim()
        y = ylims[0] + (ylims[1] - ylims[0]) * 0.06
        annotate_phases(ax, phases, y, scale=0.6)
    max_round = max(all_rounds, default=0)
    ax_agg.set_xlim(-0.5, max_round + 0.5)
    save_figure(fig, output_dir, "supplement_internal_monitors")


def plot(metrics_dir: Path, output_dir: Path, title: str) -> None:
    server = read_csv(metrics_dir / "server_metrics.csv")
    client_eval = read_csv(metrics_dir / "client_eval_metrics.csv")
    client_train = read_csv(metrics_dir / "client_train_metrics.csv")
    phase_rows = read_csv(metrics_dir / "phase_markers.csv")
    internal_rows = read_csv(metrics_dir / "internal_monitor.csv")
    agg_rows = read_csv(metrics_dir / "aggregation_monitor.csv")
    meta = metadata(read_csv(metrics_dir / "run_metadata.csv"))

    phases = phase_by_round(phase_rows)
    eval_test = aggregate(client_eval, "test_acc")
    train_acc = aggregate(client_train, "train_acc")

    server_rounds = [as_int(row.get("round")) for row in server]
    server_test = [percent(as_float(row.get("test_acc"))) for row in server]
    server_val = [percent(as_float(row.get("val_acc"))) for row in server]
    server_std = [percent(as_float(row.get("test_acc_std"))) for row in server]

    eval_rounds = sorted(eval_test)
    eval_mean = [percent(eval_test[round_id]["mean"]) for round_id in eval_rounds]
    eval_std = [percent(eval_test[round_id]["std"]) for round_id in eval_rounds]
    train_rounds = sorted(train_acc)
    train_mean = [percent(train_acc[round_id]["mean"]) for round_id in train_rounds]
    for round_id in set(server_rounds) | set(eval_rounds) | set(train_rounds):
        phases.setdefault(round_id, inferred_phase(round_id, meta))

    fig, (ax_curve, ax_spread) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.1]},
    )

    if server_rounds:
        ax_curve.plot(
            server_rounds,
            server_test,
            marker="o",
            linewidth=2.2,
            color="#1f77b4",
            label="server weighted test",
        )
        ax_curve.plot(
            server_rounds,
            server_val,
            marker="s",
            linewidth=1.8,
            color="#2ca02c",
            label="server weighted val",
        )
    if eval_rounds:
        ax_curve.plot(
            eval_rounds,
            eval_mean,
            marker=".",
            linewidth=1.4,
            linestyle="--",
            color="#9467bd",
            label="client test mean (may lead server aggregate)",
        )
        lower = [mean - std for mean, std in zip(eval_mean, eval_std, strict=True)]
        upper = [mean + std for mean, std in zip(eval_mean, eval_std, strict=True)]
        ax_curve.fill_between(eval_rounds, lower, upper, color="#9467bd", alpha=0.10)
    if train_rounds:
        ax_curve.plot(
            train_rounds,
            train_mean,
            marker="x",
            linewidth=1.2,
            color="#8c564b",
            alpha=0.85,
            label="client train mean",
        )

    all_curve_values = finite(server_test + server_val + eval_mean + train_mean)
    phase_y = (min(all_curve_values) - 1.5) if all_curve_values else 0.0
    annotate_phases(ax_curve, phases, phase_y)
    ax_curve.set_title(title)
    ax_curve.set_ylabel("Accuracy (%)")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend(frameon=False, fontsize=8)

    if server_rounds:
        ax_spread.plot(
            server_rounds,
            server_std,
            marker="o",
            linewidth=2.0,
            color="#d62728",
            label="server test acc std across clients",
        )
    if eval_rounds:
        ax_spread.plot(
            eval_rounds,
            eval_std,
            marker=".",
            linewidth=1.4,
            linestyle="--",
            color="#9467bd",
            label="raw client eval std",
        )
    spread_values = finite(server_std + eval_std)
    spread_phase_y = (min(spread_values) - 0.5) if spread_values else 0.0
    annotate_phases(ax_spread, phases, spread_phase_y, scale=0.7)
    ax_spread.set_xlabel("Communication round")
    ax_spread.set_ylabel("Client spread (pp)")
    ax_spread.grid(True, alpha=0.25)
    legend_if_present(ax_spread, frameon=False, fontsize=8)

    max_round = max(set(server_rounds) | set(eval_rounds) | set(train_rounds) | set(phases), default=0)
    ax_spread.set_xlim(-0.5, max_round + 0.5)
    save_figure(fig, output_dir, "supplement_diagnostics_curve")
    plot_internal_monitors(internal_rows, agg_rows, phases, output_dir, title)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.metrics_dir / "figures"
    plot(args.metrics_dir, output_dir, args.title)
    print(f"wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
