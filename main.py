"""
main.py — Entry point for the Self-Pruning CNN project
=======================================================
Run:  python main.py

Trains three experiments (low / medium / high λ), applies hard pruning,
fine-tunes each model, then produces a summary table and visualization.
"""

from __future__ import annotations
import os
import torch

from utils     import set_seed, get_cifar10_loaders
from train     import TrainConfig, run_experiment, print_summary_table
from visualize import plot_results


# ══════════════════════════════════════════════════════════════════════════════
# Global settings
# ══════════════════════════════════════════════════════════════════════════════

SEED       = 42
BATCH_SIZE = 128
EPOCHS     = 20          # training epochs per run
FT_EPOCHS  = 5           # fine-tune epochs after hard pruning
DATA_DIR   = "./data"
OUT_IMG    = "pruning_results.png"

# λ values to sweep — calibrated for LOG-PENALTY sparsity loss (range ≈ 0–5).
# The log penalty -log(1 - σ(g) + ε) ≈ 1.9 at init (σ≈0.85), so:
#   λ=0.01 → sparsity ≈ 1.9% of total loss  → light pruning  (~10–25% sparse)
#   λ=0.10 → sparsity ≈ 19%  of total loss  → medium pruning (~40–60% sparse)
#   λ=0.40 → sparsity ≈ 76%  of total loss  → aggressive     (~70–90% sparse)
LAMBDA_SWEEP = [0.01, 0.10, 0.40]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Reproducibility ──────────────────────────────────────────────────────
    set_seed(SEED)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("  Loading CIFAR-10 …")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )
    print(
        f"  Train : {len(train_loader.dataset):,} samples  "
        f"({len(train_loader)} batches)\n"
        f"  Test  : {len(test_loader.dataset):,} samples  "
        f"({len(test_loader)} batches)"
    )

    # ── Experiments ──────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for lam in LAMBDA_SWEEP:
        cfg = TrainConfig(
            target_lam      = lam,
            epochs          = EPOCHS,
            finetune_epochs = FT_EPOCHS,
            lr              = 3e-4,
            weight_decay    = 1e-4,
            label_smoothing = 0.1,
            dropout_p       = 0.3,
            batch_size      = BATCH_SIZE,
            prune_tau       = 0.10,   # raised from 1e-2 → 0.10 to match log-penalty gate range
            warmup_frac     = 0.10,   # 2 epochs warmup (was 0.30 = 6 epochs, too slow)
        )
        result = run_experiment(
            cfg          = cfg,
            train_loader = train_loader,
            test_loader  = test_loader,
            device       = device,
            verbose      = True,
        )
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Visualization ─────────────────────────────────────────────────────────
    plot_results(all_results, save_path=OUT_IMG)

    print("  Done.\n")


if __name__ == "__main__":
    main()
