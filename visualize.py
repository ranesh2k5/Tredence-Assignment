"""
visualize.py — All visualization for the self-pruning project
=============================================================
Produces a single publication-quality figure with 4 rows:
  Row 0 — Accuracy learning curves + λ schedule
  Row 1 — Gate value histograms (bimodal separation)
  Row 2 — Before / After pruning accuracy comparison
  Row 3 — Accuracy vs Sparsity scatter + MAC reduction bars
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch

# ── Aesthetics ────────────────────────────────────────────────────────────────
BG      = "#0D1117"
PANEL   = "#161B22"
GRID    = "#21262D"
SPINE   = "#30363D"
FG      = "white"
MUTED   = "#8B949E"
PALETTE = ["#58A6FF", "#F78166", "#56D364"]   # blue / red-orange / green


def _style_ax(ax: plt.Axes, *, ylim=None, xlim=None) -> None:
    """Apply consistent dark-theme styling to an axes."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)


def _legend(ax: plt.Axes, **kw) -> None:
    ax.legend(
        facecolor=PANEL, labelcolor=FG, edgecolor=SPINE,
        fontsize=9, **kw
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results: list[dict], save_path: str = "pruning_results.png") -> None:
    """
    Full 4-row summary figure.

    Args:
        results   : list of experiment result dicts from train.run_experiment()
        save_path : output file path (.png)
    """
    n   = len(results)
    fig = plt.figure(figsize=(7 * n, 22), facecolor=BG)
    fig.suptitle(
        "Self-Pruning CNN  ·  CIFAR-10  ·  Learnable Gate Sparsity",
        fontsize=20, fontweight="bold", color=FG, y=0.995,
    )

    gs = gridspec.GridSpec(4, n, figure=fig, hspace=0.52, wspace=0.30)

    # ── Row 0: learning curves + λ schedule ──────────────────────────────────
    for i, res in enumerate(results):
        ax  = fig.add_subplot(gs[0, i])
        ax2 = ax.twinx()

        epochs = range(1, len(res["train_accs"]) + 1)
        ax.plot(epochs, res["train_accs"],  color=PALETTE[i], lw=2,
                label="Train acc")
        ax.plot(epochs, res["test_accs"],   color=PALETTE[i], lw=2.5,
                ls="--", label="Test acc", alpha=0.85)
        ax.fill_between(epochs, res["train_accs"], res["test_accs"],
                        alpha=0.07, color=PALETTE[i])

        # fine-tune tail
        n_train = len(res["train_accs"])
        if res["finetune_accs"]:
            ft_epochs = range(n_train + 1, n_train + len(res["finetune_accs"]) + 1)
            ax.plot(ft_epochs, res["finetune_accs"], color=PALETTE[i],
                    lw=2.5, ls=":", label="Fine-tune")
            ax.axvline(x=n_train + 0.5, color=SPINE, lw=1.2, ls="--")
            ax.text(n_train + 0.7, 5, "prune↓", color=MUTED, fontsize=8, va="bottom")

        ax2.plot(epochs, res["lam_history"], color="#FFD700", lw=1.2,
                 ls="-.", alpha=0.7, label="λ (eff)")
        ax2.set_ylabel("λ value", color="#FFD700", fontsize=9)
        ax2.tick_params(colors="#FFD700", labelsize=8)
        ax2.set_facecolor(PANEL)

        _style_ax(ax, ylim=(0, 75))
        ax.set_title(f"λ = {res['lam']:.0e}   (final: {res['test_acc']:.1f}%)",
                     fontsize=13, pad=6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")

        handles  = [
            Line2D([0],[0], color=PALETTE[i], lw=2,   label="Train"),
            Line2D([0],[0], color=PALETTE[i], lw=2.5, ls="--", label="Test"),
            Line2D([0],[0], color=PALETTE[i], lw=2.5, ls=":",  label="Fine-tune"),
            Line2D([0],[0], color="#FFD700",  lw=1.2, ls="-.", label="λ schedule"),
        ]
        ax.legend(handles=handles, facecolor=PANEL, labelcolor=FG,
                  edgecolor=SPINE, fontsize=8, loc="upper left")

    # ── Row 1: gate value histograms (bimodal) ────────────────────────────────
    for i, res in enumerate(results):
        ax    = fig.add_subplot(gs[1, i])
        gates = res["all_gates"]

        pruned = gates[gates <  0.01]
        active = gates[gates >= 0.01]

        # Two separate histograms with a visual gap
        ax.hist(active, bins=80, range=(0.01, 1.0), color=PALETTE[i],
                alpha=0.85, label=f"Active ({len(active):,})", zorder=3)
        ax.hist(pruned, bins=30, range=(0.0, 0.01),
                color="white", alpha=0.9,
                label=f"Pruned ({len(pruned):,})", zorder=4)

        ax.axvline(x=0.01, color="#FFD700", lw=1.5, ls="--",
                   label="τ = 0.01", zorder=5)

        # Annotate bimodal stats
        pct = 100.0 * len(pruned) / max(len(gates), 1)
        ax.text(0.55, 0.93,
                f"Pruned: {pct:.1f}%\nActive: {100-pct:.1f}%",
                transform=ax.transAxes, color=FG, fontsize=10,
                va="top", ha="left",
                bbox=dict(facecolor=SPINE, alpha=0.75, pad=4, edgecolor="none"))

        _style_ax(ax, xlim=(0, 1))
        ax.set_title(f"Gate Distribution  λ={res['lam']:.0e}", fontsize=12, pad=6)
        ax.set_xlabel("Gate value  σ(g)")
        ax.set_ylabel("Count")
        _legend(ax, loc="upper center")

    # ── Row 2: Before / After pruning accuracy bars ───────────────────────────
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[2, i])

        stages = ["Pre-prune\n(soft gates)",
                  "Post-threshold\n(hard mask)",
                  "After fine-tune"]
        values = [
            res["acc_before_prune"],
            res["acc_after_threshold"],
            res["test_acc"],
        ]
        colors = ["#8B949E", PALETTE[i], "#56D364"]

        bars = ax.bar(stages, values, color=colors, edgecolor="none",
                      width=0.55, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    color=FG, fontsize=11, fontweight="bold")

        # Delta annotations
        drop = res["acc_before_prune"] - res["acc_after_threshold"]
        gain = res["test_acc"]         - res["acc_after_threshold"]
        ax.annotate("", xy=(1, values[1]), xytext=(0, values[0]),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
        ax.annotate("", xy=(2, values[2]), xytext=(1, values[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#56D364", lw=1.2))
        ax.text(0.5,  max(values) * 0.55, f"Δ={-drop:+.1f}%",
                ha="center", color=MUTED,    fontsize=10)
        ax.text(1.5,  max(values) * 0.55, f"Δ={gain:+.1f}%",
                ha="center", color="#56D364", fontsize=10)

        _style_ax(ax, ylim=(0, max(values) * 1.18))
        ax.set_title(f"Pruning Stages  λ={res['lam']:.0e}", fontsize=12, pad=6)
        ax.set_ylabel("Test Accuracy (%)")
        ax.tick_params(axis="x", labelsize=8.5)

    # ── Row 3: Acc vs Sparsity scatter + MAC reduction ────────────────────────
    ax_sc  = fig.add_subplot(gs[3, 0])
    ax_mac = fig.add_subplot(gs[3, 1])
    ax_sp  = fig.add_subplot(gs[3, 2]) if n >= 3 else None

    # Scatter: accuracy vs sparsity
    for i, res in enumerate(results):
        ax_sc.scatter(res["sparsity"], res["test_acc"],
                      color=PALETTE[i], s=200, zorder=5,
                      edgecolors=FG, linewidths=0.7,
                      label=f"λ={res['lam']:.0e}")
        ax_sc.annotate(
            f"λ={res['lam']:.0e}",
            (res["sparsity"], res["test_acc"]),
            textcoords="offset points", xytext=(8, 5),
            color=PALETTE[i], fontsize=9, fontweight="bold",
        )
    _style_ax(ax_sc, xlim=(0, 100), ylim=(20, 80))
    ax_sc.set_title("Accuracy vs Sparsity Trade-off", fontsize=12)
    ax_sc.set_xlabel("Sparsity (% gates < τ)")
    ax_sc.set_ylabel("Test Accuracy (%)")
    _legend(ax_sc)

    # Bar: MAC reduction
    lam_labels = [f"λ={r['lam']:.0e}" for r in results]
    mac_vals   = [r["mac_reduction"] for r in results]
    bars = ax_mac.bar(lam_labels, mac_vals, color=PALETTE[:n],
                      edgecolor="none", width=0.5, zorder=3)
    for bar, val in zip(bars, mac_vals):
        ax_mac.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    color=FG, fontsize=11, fontweight="bold")
    _style_ax(ax_mac, ylim=(0, 100))
    ax_mac.set_title("Theoretical MAC Reduction\n(linear layers)", fontsize=12)
    ax_mac.set_ylabel("MAC Reduction (%)")

    # Sparsity soft vs hard comparison (if 3 experiments)
    if ax_sp is not None:
        x = np.arange(n)
        w = 0.35
        soft_vals = [r["sparsity_soft"] for r in results]
        hard_vals = [r["sparsity"]      for r in results]
        ax_sp.bar(x - w/2, soft_vals, w, color="#8B949E", label="Soft gates",
                  edgecolor="none", zorder=3)
        ax_sp.bar(x + w/2, hard_vals, w, color=PALETTE[1], label="Hard mask",
                  edgecolor="none", zorder=3)
        ax_sp.set_xticks(x)
        ax_sp.set_xticklabels(lam_labels)
        _style_ax(ax_sp, ylim=(0, 100))
        ax_sp.set_title("Sparsity: Soft vs Hard Threshold", fontsize=12)
        ax_sp.set_ylabel("Sparsity (%)")
        _legend(ax_sp)

    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓  Saved figure → {save_path}")
