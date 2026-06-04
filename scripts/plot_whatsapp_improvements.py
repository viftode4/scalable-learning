#!/usr/bin/env python3
# ruff: noqa: E402
"""Create WhatsApp-friendly RoLoRA improvement comparison plots."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("evidence/whatsapp_20260604")
OLD = Path("evidence/wandb_qnli_c50_r4_20260603")
ORTH = Path("evidence/improvement_diagnostics_20260603/proxy_orth_a_seed0_partial/server_metrics.csv")
BBA0 = Path("evidence/improvement_diagnostics_20260603/proxy_phase_bba_orth_a_seed0_partial/server_metrics.csv")
BBA1 = Path("evidence/improvement_diagnostics_20260604/proxy_phase_bba_orth_a_seed1_partial/server_metrics.csv")
AGG0 = Path("evidence/improvement_diagnostics_20260603/proxy_phase_bba_orth_a_seed0_partial/aggregation_monitor.csv")


def pct(x: float) -> float:
    return 100.0 * float(x)


def mean_ci(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.fmean(vals), 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_server(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["test_pct"] = df["test_acc"].astype(float).map(pct)
    df["val_pct"] = df["val_acc"].astype(float).map(pct)
    return df


def plot_main_curve() -> None:
    hist = pd.read_csv(OLD / "server_history.csv")
    old = hist[(hist["method"] == "rolora") & (hist["lr"].astype(float) == 0.01)].copy()
    old["test_pct"] = old["server_test_acc"].astype(float).map(pct)

    old_rounds = sorted(old["round"].unique())
    old_mean, old_ci = [], []
    for r in old_rounds:
        m, c = mean_ci(old.loc[old["round"] == r, "test_pct"].tolist())
        old_mean.append(m)
        old_ci.append(c)

    orth = load_server(ORTH)
    bba0 = load_server(BBA0)
    bba1 = load_server(BBA1)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(old_rounds, old_mean, color="#1f77b4", linewidth=2.6)
    ax.fill_between(
        old_rounds,
        [m - c for m, c in zip(old_mean, old_ci, strict=True)],
        [m + c for m, c in zip(old_mean, old_ci, strict=True)],
        color="#1f77b4",
        alpha=0.14,
    )
    if not orth.empty:
        ax.plot(orth["round"], orth["test_pct"], color="#9467bd", linestyle="--", marker="o", markersize=3.5, linewidth=2.1)
    if not bba0.empty:
        ax.plot(bba0["round"], bba0["test_pct"], color="#2ca02c", marker="o", markersize=3.8, linewidth=3.0)
    # Keep the WhatsApp plot clean: only show the live replication once it has
    # enough rounds to be interpretable.
    if len(bba1) >= 5:
        ax.plot(bba1["round"], bba1["test_pct"], color="#2ca02c", linestyle=":", marker="x", markersize=4, linewidth=2.0)

    ax.axhline(88.5411, color="#2ca02c", linestyle=":", alpha=0.35)
    old_final = old_mean[-1]
    orth_final = float(orth["test_pct"].iloc[-1]) if not orth.empty else math.nan
    bba0_final = float(bba0["test_pct"].iloc[-1]) if not bba0.empty else math.nan

    def label_at_end(y: float, text_y: float, text: str, color: str, *, weight: str = "normal") -> None:
        if math.isnan(y):
            return
        ax.annotate(
            text,
            xy=(19, y),
            xytext=(20.45, text_y),
            textcoords="data",
            color=color,
            fontsize=9.5,
            weight=weight,
            va="center",
            ha="left",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.1, "alpha": 0.9},
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": color, "alpha": 0.88},
            clip_on=False,
        )

    label_at_end(bba0_final, 89.7, f"Orth-A+BBA {bba0_final:.2f}%", "#2ca02c", weight="bold")
    label_at_end(old_final, 85.6, f"Old RoLoRA mean {old_final:.2f}% ± CI", "#1f77b4")
    label_at_end(orth_final, 82.0, f"Orth-A+AB {orth_final:.2f}%", "#9467bd")
    if len(bba1) >= 5:
        label_at_end(float(bba1["test_pct"].iloc[-1]), 79.0, "BBA seed1 live", "#2ca02c")

    if not math.isnan(orth_final) and not math.isnan(bba0_final):
        ax.text(
            8.5,
            90.0,
            f"+{bba0_final - orth_final:.2f} pp vs Orth-A+AB",
            color="#176b17",
            fontsize=11,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#2ca02c", "alpha": 0.9},
        )
    ax.set_title("RoLoRA proxy: old baseline vs phase-controlled improvement", fontsize=14, weight="bold")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Server test accuracy (%)")
    ax.set_ylim(47, 92)
    ax.set_xlim(-0.5, 24.2)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.0,
        -0.18,
        "Scope: QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds proxy — not RoBERTa-Large Table 1.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555",
    )
    save(fig, "01_main_curve_old_vs_improvements")


def plot_final_bars() -> None:
    summ = pd.read_csv(OLD / "runs_summary.csv")
    old_rows = summ[(summ["method"] == "rolora") & (summ["lr"].astype(float) == 0.01)]
    old_mean, old_ci = mean_ci(old_rows["final_server_test_acc"].astype(float).map(pct).tolist())
    old_seed0 = pct(float(old_rows[old_rows["seed"] == 0]["final_server_test_acc"].iloc[0]))
    orth = load_server(ORTH)
    bba0 = load_server(BBA0)

    labels = [
        "Old RoLoRA\nseed0",
        "Old RoLoRA\nmean 3 seeds",
        "Orth-A + AB\nseed0",
        "Orth-A + BBA\nseed0",
    ]
    vals = [
        old_seed0,
        old_mean,
        float(orth["test_pct"].iloc[-1]),
        float(bba0["test_pct"].iloc[-1]),
    ]
    errs = [0.0, old_ci, 0.0, 0.0]
    colors = ["#aec7e8", "#1f77b4", "#9467bd", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    bars = ax.bar(labels, vals, yerr=errs, color=colors, capsize=4, alpha=0.92)
    for bar, value in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.45, f"{value:.2f}%", ha="center", va="bottom", fontsize=10, weight="bold")
    gain_vs_orth = vals[-1] - vals[-2]
    gain_vs_old0 = vals[-1] - vals[0]
    ax.text(
        0.5,
        0.94,
        f"BBA+Orth gain: +{gain_vs_orth:.2f} pp vs Orth-A+AB; +{gain_vs_old0:.2f} pp vs old seed0",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        weight="bold",
        color="#176b17",
    )
    ax.set_title("Final server test accuracy: old vs improvement", fontsize=14, weight="bold")
    ax.set_ylabel("Final server test accuracy (%)")
    ax.set_ylim(78, 91)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.0,
        -0.16,
        "Single-seed improvement bars are preliminary; replication seed 1 is currently running.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555",
    )
    save(fig, "02_final_accuracy_bars")


def plot_phase_mechanics() -> None:
    agg = pd.read_csv(AGG0)
    if agg.empty:
        return
    result = agg[agg["kind"] == "aggregate_result"].copy()
    if result.empty:
        return
    result["round"] = result["round"].astype(int)
    result["dA"] = result["aggregate_update_lora_A_norm"].astype(float)
    result["dB"] = result["aggregate_update_lora_B_norm"].astype(float)
    phases = ["B" if r % 3 in (0, 1) else "A" for r in result["round"]]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(result["round"], result["dA"], marker="o", linewidth=2.2, label="aggregate ΔA norm", color="#ff7f0e")
    ax.plot(result["round"], result["dB"], marker="s", linewidth=2.2, label="aggregate ΔB norm", color="#1f77b4")
    for r, phase in zip(result["round"], phases, strict=True):
        ax.text(r, max(result["dA"].max(), result["dB"].max()) * 1.03, phase, ha="center", fontsize=8, color="#444")
    ax.set_title("Mechanics check: BBA updates only the active LoRA factor", fontsize=14, weight="bold")
    ax.set_xlabel("Communication round (phase label on top)")
    ax.set_ylabel("Aggregate update norm")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.0,
        -0.18,
        "B rounds: ΔA≈0, ΔB>0. A rounds: ΔA>0, ΔB≈0. This supports the claim that the gain is not a logging artifact.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555",
    )
    save(fig, "03_phase_mechanics_bba")


def write_summary() -> None:
    summ = pd.read_csv(OLD / "runs_summary.csv")
    old_rows = summ[(summ["method"] == "rolora") & (summ["lr"].astype(float) == 0.01)]
    old_mean, old_ci = mean_ci(old_rows["final_server_test_acc"].astype(float).map(pct).tolist())
    old_seed0 = pct(float(old_rows[old_rows["seed"] == 0]["final_server_test_acc"].iloc[0]))
    orth = load_server(ORTH)
    bba0 = load_server(BBA0)
    bba1 = load_server(BBA1)
    lines = [
        "# WhatsApp plot bundle — 2026-06-04",
        "",
        "Scope: QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds proxy.",
        "Not paper-scale RoBERTa-Large Table 1.",
        "",
        f"- Old RoLoRA lr=1e-2 seed0 final: {old_seed0:.2f}%.",
        f"- Old RoLoRA lr=1e-2 3-seed mean final: {old_mean:.2f}% ± {old_ci:.2f} CI95.",
        f"- Orth-A + default AB seed0 final: {float(orth['test_pct'].iloc[-1]):.2f}%.",
        f"- Orth-A + BBA seed0 final: {float(bba0['test_pct'].iloc[-1]):.2f}%.",
        f"- Gain vs Orth-A+AB seed0: {float(bba0['test_pct'].iloc[-1]) - float(orth['test_pct'].iloc[-1]):+.2f} pp.",
        f"- Gain vs old RoLoRA seed0: {float(bba0['test_pct'].iloc[-1]) - old_seed0:+.2f} pp.",
    ]
    if not bba1.empty:
        lines.append(f"- BBA+Orth seed1 currently parsed through round {int(bba1['round'].max())}; latest test {float(bba1['test_pct'].iloc[-1]):.2f}%.")
    lines += [
        "",
        "Files:",
        "- `01_main_curve_old_vs_improvements.png`",
        "- `02_final_accuracy_bars.png`",
        "- `03_phase_mechanics_bba.png`",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_main_curve()
    plot_final_bars()
    plot_phase_mechanics()
    write_summary()
    print(f"wrote WhatsApp plots to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
