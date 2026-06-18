#!/usr/bin/env python3
"""Bar chart for the QNLI proxy improvement table (report Table 1 -> figure).

Values are the final-round server test accuracy, mean +/- 95% CI (1.96 s/sqrt(n)),
taken verbatim from runs/REGISTRY.md section 1. Regenerate after any new seed:

    code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python \
        scripts/plot_lm_proxy_improvements.py

Writes report/figures/lm_proxy_improvements.{pdf,png}. PDF is vector for Overleaf
\\includegraphics; PNG is for preview.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "report" / "figures"

# (label, mean, ci95, n) — from runs/REGISTRY.md section 1, lr 1e-2, final round.
ARMS = [
    ("RoLoRA\n(baseline)", 85.10, 2.97, 3),
    ("SVD-\ncompensated", 85.70, 0.82, 3),
    ("Adaptive refresh\n+ orth. $A$", 86.67, 0.43, 3),
    ("Orthogonal $A$\n+ BBA", 87.60, 1.43, 3),
]
BASELINE = 85.10

labels = [a[0] for a in ARMS]
means = [a[1] for a in ARMS]
ci95 = [a[2] for a in ARMS]
ns = [a[3] for a in ARMS]

# baseline grey; improvement arms in a steady palette; BBA (headline) accented.
colors = ["#9e9e9e", "#7baFd4", "#6fb07f", "#c1666b"]

fig, ax = plt.subplots(figsize=(6.4, 3.9))
x = range(len(ARMS))
ax.bar(
    x, means, yerr=ci95, capsize=6, color=colors, edgecolor="black", linewidth=0.6,
    error_kw=dict(ecolor="#2b2b2b", lw=1.3, capthick=1.3), zorder=3, width=0.66,
)
ax.axhline(BASELINE, ls="--", lw=1.0, color="#9e9e9e", zorder=1)
ax.text(len(ARMS) - 0.45, BASELINE + 0.12, "baseline", fontsize=8,
        color="#6b6b6b", ha="right", va="bottom")

# value labels above each error-bar cap
for xi, m, c, n in zip(x, means, ci95, ns):
    ax.text(xi, m + c + 0.22, f"{m:.2f}\n$\\pm${c:.2f} (n={n})",
            ha="center", va="bottom", fontsize=8.5)

ax.set_ylim(80, 92)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Final-round test accuracy (%)", fontsize=10)
ax.set_title("QNLI proxy improvements (RoBERTa-base, 50 clients, rank 4, lr $10^{-2}$)",
             fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=9)
ax.grid(axis="y", ls=":", lw=0.5, color="#cccccc", zorder=0)
fig.tight_layout()

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    p = OUT / f"lm_proxy_improvements.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {p.relative_to(REPO)}")
