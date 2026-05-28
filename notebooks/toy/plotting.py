"""Curve-plotting helpers.

`plot_curves` takes a dict of ``{label: (losses, accs)}`` and renders the
standard two-panel (loss, accuracy) figure. With ``curve_sems`` supplied,
shaded ±SEM bands are drawn around each curve (and around the ceiling if
``ceiling_sem`` is also supplied). The centralized ceiling, when supplied,
is drawn as a dashed reference line on both panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(
    curves: dict[str, tuple[list[float], list[float]]],
    out: Path,
    *,
    title: str,
    xs: list[int] | None = None,
    ceiling: tuple[str, list[float], list[float]] | None = None,
    curve_sems: dict[str, tuple[list[float], list[float]]] | None = None,
    ceiling_sem: tuple[list[float], list[float]] | None = None,
) -> None:
    """Render a (loss, accuracy) two-panel plot to ``out``.

    ``curves`` maps ``label -> (loss_curve, acc_curve)``. Each curve is the
    mean across seeds (or a single run when N=1).

    ``xs`` (optional) is the shared x-axis (round indices). When None, falls
    back to ``range(len(curve))``. Use this when the saved curves are sparse
    checkpoints rather than per-round samples.

    ``curve_sems`` (optional) maps the same labels to
    ``(loss_sem, acc_sem)`` — the standard error of the mean per round.
    When supplied, a ±SEM shaded band is drawn around each curve.

    ``ceiling`` is an optional ``(label, losses, accs)`` triple drawn as a
    dashed line on both panels — intended for the centralized baseline.
    ``ceiling_sem`` mirrors ``curve_sems`` for the ceiling.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for label, (loss, acc) in curves.items():
        x_curve = xs if xs is not None else list(range(len(loss)))
        (loss_line,) = ax_loss.plot(x_curve, loss, label=label)
        (acc_line,) = ax_acc.plot(x_curve, acc, label=label)
        if curve_sems is not None and label in curve_sems:
            loss_sem, acc_sem = curve_sems[label]
            ax_loss.fill_between(
                x_curve,
                [m - s for m, s in zip(loss, loss_sem, strict=True)],
                [m + s for m, s in zip(loss, loss_sem, strict=True)],
                color=loss_line.get_color(),
                alpha=0.2,
                linewidth=0,
            )
            ax_acc.fill_between(
                x_curve,
                [m - s for m, s in zip(acc, acc_sem, strict=True)],
                [m + s for m, s in zip(acc, acc_sem, strict=True)],
                color=acc_line.get_color(),
                alpha=0.2,
                linewidth=0,
            )
    if ceiling is not None:
        c_label, c_loss, c_acc = ceiling
        x_ceil = xs if xs is not None else list(range(len(c_loss)))
        ax_loss.plot(x_ceil, c_loss, label=c_label, linestyle="--", color="black", linewidth=1.2)
        ax_acc.plot(x_ceil, c_acc, label=c_label, linestyle="--", color="black", linewidth=1.2)
        if ceiling_sem is not None:
            c_loss_sem, c_acc_sem = ceiling_sem
            ax_loss.fill_between(
                x_ceil,
                [m - s for m, s in zip(c_loss, c_loss_sem, strict=True)],
                [m + s for m, s in zip(c_loss, c_loss_sem, strict=True)],
                color="black",
                alpha=0.15,
                linewidth=0,
            )
            ax_acc.fill_between(
                x_ceil,
                [m - s for m, s in zip(c_acc, c_acc_sem, strict=True)],
                [m + s for m, s in zip(c_acc, c_acc_sem, strict=True)],
                color="black",
                alpha=0.15,
                linewidth=0,
            )
    ax_loss.set_xlabel("communication round")
    ax_loss.set_ylabel("test cross-entropy")
    ax_loss.legend(fontsize=8)
    ax_acc.set_xlabel("communication round")
    ax_acc.set_ylabel("test accuracy")
    ax_acc.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
