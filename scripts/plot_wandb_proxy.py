#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Create draft-report proxy plots from the exported W&B QNLI C50 bundle.

The input bundle is deliberately treated as RoBERTa-base proxy evidence, not as
RoBERTa-Large Table-1 reproduction evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHOD_LABELS = {
    "rolora": "RoLoRA",
    "lora": "LoRA",
    "ffa_lora": "FFA-LoRA",
}
METHOD_COLORS = {
    "rolora": "#1f77b4",
    "lora": "#d62728",
    "ffa_lora": "#ff7f0e",
}
GROUP_STYLES = {
    ("rolora", "1e-2"): ("#1f77b4", "-"),
    ("rolora", "5e-3"): ("#17becf", "-."),
    ("lora", "2e-2"): ("#d62728", "--"),
    ("ffa_lora", "2e-2"): ("#ff7f0e", ":"),
}
SELECTED_GROUPS = [
    ("rolora", "1e-2"),
    ("rolora", "5e-3"),
    ("lora", "2e-2"),
    ("ffa_lora", "2e-2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("evidence/wandb_qnli_c50_r4_20260603"),
        help="Directory containing runs_summary.csv and server_history.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <evidence-dir>/figures.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
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


def lr_value(lr: str) -> float:
    return float(lr)


def percent(value: float) -> float:
    return value * 100.0


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    vals = [value for value in values if not math.isnan(value)]
    if not vals:
        return math.nan, math.nan
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return mean, 0.0
    ci95 = 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))
    return mean, ci95


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return paths


def group_summary(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["lr"])].append(row)
    return grouped


def plot_convergence(history_rows: list[dict[str, str]], output_dir: Path) -> None:
    by_group_round: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    present_groups = set()
    for row in history_rows:
        key = (row["method"], row["lr"])
        if key not in SELECTED_GROUPS:
            continue
        present_groups.add(key)
        by_group_round[(*key, as_int(row["round"]))].append(percent(as_float(row["server_test_acc"])))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method, lr in SELECTED_GROUPS:
        if (method, lr) not in present_groups:
            continue
        rounds = sorted(
            round_num
            for m, group_lr, round_num in by_group_round
            if m == method and group_lr == lr
        )
        means: list[float] = []
        cis: list[float] = []
        for round_num in rounds:
            mean, ci95 = mean_ci(by_group_round[(method, lr, round_num)])
            means.append(mean)
            cis.append(ci95)
        label = f"{METHOD_LABELS[method]} lr={lr}"
        color, linestyle = GROUP_STYLES.get((method, lr), (METHOD_COLORS[method], "-"))
        ax.plot(rounds, means, label=label, color=color, linewidth=2.2, linestyle=linestyle)
        if len(rounds) > 1:
            lower = [mean - ci for mean, ci in zip(means, cis, strict=True)]
            upper = [mean + ci for mean, ci in zip(means, cis, strict=True)]
            ax.fill_between(rounds, lower, upper, color=color, alpha=0.13)

    ax.set_title("QNLI C50/r4 RoBERTa-base proxy convergence (W&B export)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Server test accuracy (%)")
    ax.set_ylim(45, 90)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, output_dir, "proxy_a_server_accuracy_convergence")


def plot_final_best(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped = group_summary(summary_rows)
    items = sorted(grouped.items(), key=lambda item: (item[0][0], lr_value(item[0][1])))
    labels: list[str] = []
    best_means: list[float] = []
    best_cis: list[float] = []
    final_means: list[float] = []
    final_cis: list[float] = []
    colors: list[str] = []
    for (method, lr), rows in items:
        labels.append(f"{METHOD_LABELS[method]}\n{lr}\nn={len(rows)}")
        best_mean, best_ci = mean_ci(percent(as_float(row["best_server_test_acc"])) for row in rows)
        final_mean, final_ci = mean_ci(percent(as_float(row["final_server_test_acc"])) for row in rows)
        best_means.append(best_mean)
        best_cis.append(best_ci)
        final_means.append(final_mean)
        final_cis.append(final_ci)
        colors.append(METHOD_COLORS[method])

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.bar([value - 0.18 for value in x], final_means, width=0.36, yerr=final_cis, label="Final", color=colors, alpha=0.62, capsize=3)
    ax.bar([value + 0.18 for value in x], best_means, width=0.36, yerr=best_cis, label="Best", color=colors, alpha=0.95, capsize=3)
    ax.set_title("Final and best accuracy by method/LR (RoBERTa-base proxy)")
    ax.set_ylabel("Server test accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(45, 90)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "proxy_b_final_best_by_method_lr")


def plot_rolora_lr_sweep(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    rolora_rows = [row for row in summary_rows if row["method"] == "rolora"]
    by_lr: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rolora_rows:
        by_lr[row["lr"]].append(row)

    lrs = sorted(by_lr, key=lr_value)
    x = [lr_value(lr) for lr in lrs]
    best_mean: list[float] = []
    best_ci: list[float] = []
    final_mean: list[float] = []
    final_ci: list[float] = []
    for lr in lrs:
        final_m, final_c = mean_ci(percent(as_float(row["final_server_test_acc"])) for row in by_lr[lr])
        best_m, best_c = mean_ci(percent(as_float(row["best_server_test_acc"])) for row in by_lr[lr])
        final_mean.append(final_m)
        final_ci.append(final_c)
        best_mean.append(best_m)
        best_ci.append(best_c)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(x, final_mean, yerr=final_ci, marker="o", label="Final", linewidth=2.0, capsize=3)
    ax.errorbar(x, best_mean, yerr=best_ci, marker="s", label="Best", linewidth=2.0, capsize=3)
    for lr in lrs:
        for row in by_lr[lr]:
            ax.scatter(
                lr_value(lr),
                percent(as_float(row["best_server_test_acc"])),
                color="#1f77b4",
                alpha=0.35,
                s=28,
            )
    ax.set_xscale("log")
    ax.set_title("RoLoRA LR sweep selects 1e-2 / 5e-3 as proxy controls")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Server test accuracy (%)")
    ax.set_ylim(45, 90)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "proxy_c_rolora_lr_sweep")


def plot_seed_trajectories(history_rows: list[dict[str, str]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for ax, lr in zip(axes, ["1e-2", "5e-3"], strict=True):
        rows = [row for row in history_rows if row["method"] == "rolora" and row["lr"] == lr]
        by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_seed[as_int(row["seed"])].append(row)
        for seed, seed_rows in sorted(by_seed.items()):
            ordered = sorted(seed_rows, key=lambda row: as_int(row["round"]))
            ax.plot(
                [as_int(row["round"]) for row in ordered],
                [percent(as_float(row["server_test_acc"])) for row in ordered],
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=f"seed {seed}",
            )
        ax.set_title(f"RoLoRA lr={lr}")
        ax.set_xlabel("Communication round")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Server test accuracy (%)")
    axes[0].set_ylim(45, 90)
    fig.suptitle("Per-seed proxy trajectories avoid single-run overclaiming", y=1.03)
    save_figure(fig, output_dir, "proxy_d_rolora_per_seed_trajectories")


def plot_walltime(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    ordered = sorted(summary_rows, key=lambda row: row["created_at"])
    runtimes = [as_float(row["runtime_seconds"]) / 3600.0 for row in ordered]
    labels = [row["name"].replace("_", "\n") for row in ordered]
    colors = [METHOD_COLORS[row["method"]] for row in ordered]

    fig, ax = plt.subplots(figsize=(13.0, 4.2))
    ax.bar(range(len(ordered)), runtimes, color=colors, alpha=0.85)
    ax.axhline(4.0, color="black", linestyle="--", linewidth=1.0, label="4h gpu-a100-small cap")
    ax.set_title("W&B runs marked crashed after completing rounds 0-19")
    ax.set_ylabel("Runtime (hours)")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylim(0, 4.2)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "proxy_e_walltime_crash_audit")


def write_summary(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped = group_summary(summary_rows)
    lines = [
        "# W&B proxy plot summary",
        "",
        "Scope: QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds. This is proxy evidence, not RoBERTa-Large Table-1 reproduction.",
        "",
        "| Method | LR | n | final acc mean ± CI95 | best acc mean ± CI95 | final round(s) | state(s) |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for (method, lr), rows in sorted(grouped.items(), key=lambda item: (item[0][0], lr_value(item[0][1]))):
        final_m, final_ci = mean_ci(percent(as_float(row["final_server_test_acc"])) for row in rows)
        best_m, best_ci = mean_ci(percent(as_float(row["best_server_test_acc"])) for row in rows)
        rounds = ",".join(str(as_int(row["final_round"])) for row in rows)
        states = ",".join(sorted({row["state"] for row in rows}))
        lines.append(
            f"| {METHOD_LABELS[method]} | `{lr}` | {len(rows)} | "
            f"{final_m:.2f} ± {final_ci:.2f} | {best_m:.2f} ± {best_ci:.2f} | {rounds} | {states} |"
        )
    lines.extend(
        [
            "",
            "## Figure files",
            "",
            "- `proxy_a_server_accuracy_convergence.{png,pdf}` — selected mean ± CI95 convergence curves.",
            "- `proxy_b_final_best_by_method_lr.{png,pdf}` — final/best accuracy summary by method/LR.",
            "- `proxy_c_rolora_lr_sweep.{png,pdf}` — RoLoRA LR sweep used to choose the proxy control.",
            "- `proxy_d_rolora_per_seed_trajectories.{png,pdf}` — per-seed RoLoRA trajectories for lr=1e-2 and 5e-3.",
            "- `proxy_e_walltime_crash_audit.{png,pdf}` — runtime/status audit explaining crashed labels.",
        ]
    )
    (output_dir / "proxy_plot_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    evidence_dir: Path = args.evidence_dir
    output_dir: Path = args.output_dir or evidence_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(evidence_dir / "runs_summary.csv")
    history_rows = read_csv(evidence_dir / "server_history.csv")
    if len(summary_rows) != 28:
        raise SystemExit(f"Expected 28 run summaries, found {len(summary_rows)}")
    if len(history_rows) != 560:
        raise SystemExit(f"Expected 560 server history rows, found {len(history_rows)}")

    plot_convergence(history_rows, output_dir)
    plot_final_best(summary_rows, output_dir)
    plot_rolora_lr_sweep(summary_rows, output_dir)
    plot_seed_trajectories(history_rows, output_dir)
    plot_walltime(summary_rows, output_dir)
    write_summary(summary_rows, output_dir)
    print(f"Wrote W&B proxy figures to {output_dir}")


if __name__ == "__main__":
    main()
