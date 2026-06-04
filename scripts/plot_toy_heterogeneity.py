#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Create draft-report plots from Daniel's toy heterogeneity result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "base_lora": "#d62728",
    "base_ffa_lora": "#ff7f0e",
    "base_rolora": "#1f77b4",
    "centralized": "#2ca02c",
    "rolora_orth_a": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("evidence/toy_heterogeneity_20260603"),
        help="Directory containing baselines and orthogonal-A JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <evidence-dir>/figures.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def ci95_from_sem(sem: float) -> float:
    return 1.96 * sem


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_baseline_curves(baselines: dict, output_dir: Path) -> None:
    rounds = baselines["checkpoint_rounds"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for key in ["base_lora", "base_ffa_lora", "base_rolora", "centralized"]:
        variant = baselines["variants"][key]
        label = variant["name"].replace(" (non-federated ceiling)", "")
        acc_mean = [value * 100.0 for value in variant["acc_mean"]]
        acc_ci = [ci95_from_sem(value) * 100.0 for value in variant["acc_sem"]]
        loss_mean = variant["loss_mean"]
        loss_ci = [ci95_from_sem(value) for value in variant["loss_sem"]]
        color = COLORS[key]
        axes[0].plot(rounds, acc_mean, label=label, color=color, linewidth=2.0)
        axes[0].fill_between(
            rounds,
            [mean - ci for mean, ci in zip(acc_mean, acc_ci, strict=True)],
            [mean + ci for mean, ci in zip(acc_mean, acc_ci, strict=True)],
            color=color,
            alpha=0.12,
        )
        axes[1].plot(rounds, loss_mean, label=label, color=color, linewidth=2.0)
        axes[1].fill_between(
            rounds,
            [mean - ci for mean, ci in zip(loss_mean, loss_ci, strict=True)],
            [mean + ci for mean, ci in zip(loss_mean, loss_ci, strict=True)],
            color=color,
            alpha=0.12,
        )
    axes[0].set_title("Toy heterogeneity accuracy")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(0, 102)
    axes[1].set_title("Toy heterogeneity loss")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.set_xlabel("Communication round")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("10 clients, one label per client: RoLoRA closes much of the heterogeneity gap", y=1.03)
    save_figure(fig, output_dir, "toy_g_heterogeneous_baselines_curves")


def plot_baseline_final_bar(baselines: dict, output_dir: Path) -> None:
    keys = ["base_lora", "base_ffa_lora", "base_rolora", "centralized"]
    labels = [baselines["variants"][key]["name"].replace(" (non-federated ceiling)", "") for key in keys]
    means = [baselines["variants"][key]["final_acc_mean"] * 100.0 for key in keys]
    cis = [ci95_from_sem(baselines["variants"][key]["final_acc_sem"]) * 100.0 for key in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(labels, means, yerr=cis, color=[COLORS[key] for key in keys], capsize=4, alpha=0.9)
    ax.set_title("Final toy accuracy, mean ± CI95 over 5 seeds")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_ylim(50, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, output_dir, "toy_g_heterogeneous_baselines_final_acc")


def plot_orthogonal_delta(baselines: dict, orthogonal: dict, output_dir: Path) -> None:
    base = baselines["variants"]["base_rolora"]
    orth = orthogonal["variants"]["rolora_orth_a"]
    base_by_seed = {entry["seed"]: entry for entry in base["per_seed"]}
    orth_by_seed = {entry["seed"]: entry for entry in orth["per_seed"]}
    seeds = sorted(set(base_by_seed) & set(orth_by_seed))
    base_acc = [base_by_seed[seed]["final_acc"] * 100.0 for seed in seeds]
    orth_acc = [orth_by_seed[seed]["final_acc"] * 100.0 for seed in seeds]
    deltas = [orth_value - base_value for base_value, orth_value in zip(base_acc, orth_acc, strict=True)]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for i, seed in enumerate(seeds):
        axes[0].plot([0, 1], [base_acc[i], orth_acc[i]], marker="o", color="#555555", alpha=0.8)
        axes[0].text(1.03, orth_acc[i], f"s{seed}", va="center", fontsize=8)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Vanilla\nRoLoRA", "Orthogonal-A\nRoLoRA"])
    axes[0].set_ylabel("Final test accuracy (%)")
    axes[0].set_title("Paired seed outcomes")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar([str(seed) for seed in seeds], deltas, color="#9467bd", alpha=0.9)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Per-seed delta")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Orthogonal-A minus vanilla (pp)")
    axes[1].grid(True, axis="y", alpha=0.25)
    mean_delta = sum(deltas) / len(deltas)
    fig.suptitle(f"Orthogonal-A improves toy RoLoRA by {mean_delta:.2f} percentage points on average", y=1.03)
    save_figure(fig, output_dir, "toy_h_orthogonal_a_paired_delta")


def write_summary(baselines: dict, orthogonal: dict, output_dir: Path) -> None:
    lines = [
        "# Toy heterogeneity plot summary",
        "",
        "Scope: MNIST toy model, 10 clients, one label per client, label split, 100 rounds, rank 16, 5 seeds.",
        "Source: `origin/fix-rolora:results_extra/{baselines_n5_r100_log5.json,orth_a_n5_r100_log5.json}` copied into this evidence directory.",
        "",
        "| Variant | Final acc mean ± CI95 | Best acc mean ± CI95 |",
        "|---|---:|---:|",
    ]
    for key in ["base_lora", "base_ffa_lora", "base_rolora", "centralized"]:
        variant = baselines["variants"][key]
        lines.append(
            f"| {variant['name']} | "
            f"{variant['final_acc_mean'] * 100.0:.2f} ± {ci95_from_sem(variant['final_acc_sem']) * 100.0:.2f} | "
            f"{variant['best_acc_mean'] * 100.0:.2f} ± {ci95_from_sem(variant['best_acc_sem']) * 100.0:.2f} |"
        )
    orth = orthogonal["variants"]["rolora_orth_a"]
    base = baselines["variants"]["base_rolora"]
    delta = (orth["final_acc_mean"] - base["final_acc_mean"]) * 100.0
    lines.extend(
        [
            f"| {orth['name']} | {orth['final_acc_mean'] * 100.0:.2f} ± {ci95_from_sem(orth['final_acc_sem']) * 100.0:.2f} | {orth['best_acc_mean'] * 100.0:.2f} ± {ci95_from_sem(orth['best_acc_sem']) * 100.0:.2f} |",
            "",
            f"Orthogonal-A final-accuracy gain over vanilla RoLoRA: **{delta:.2f} percentage points**.",
            "",
            "## Figure files",
            "",
            "- `toy_g_heterogeneous_baselines_curves.{png,pdf}` — loss/accuracy curves with CI95 bands.",
            "- `toy_g_heterogeneous_baselines_final_acc.{png,pdf}` — final accuracy summary.",
            "- `toy_h_orthogonal_a_paired_delta.{png,pdf}` — paired seed deltas for orthogonal-A.",
        ]
    )
    (output_dir / "toy_plot_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    evidence_dir: Path = args.evidence_dir
    output_dir: Path = args.output_dir or evidence_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = load_json(evidence_dir / "baselines_n5_r100_log5.json")
    orthogonal = load_json(evidence_dir / "orth_a_n5_r100_log5.json")
    if baselines["num_seeds"] != 5 or orthogonal["num_seeds"] != 5:
        raise SystemExit("Expected 5 seeds in both toy result files")

    plot_baseline_curves(baselines, output_dir)
    plot_baseline_final_bar(baselines, output_dir)
    plot_orthogonal_delta(baselines, orthogonal, output_dir)
    write_summary(baselines, orthogonal, output_dir)
    print(f"Wrote toy heterogeneity figures to {output_dir}")


if __name__ == "__main__":
    main()
