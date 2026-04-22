"""
train.py — Training, fine-tuning, and experiment orchestration
==============================================================
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model import SelfPruningCNN
from utils import (
    compute_loss,
    compute_sparsity,
    accuracy,
    get_lambda,
    estimate_flop_reduction,
    model_summary,
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    All hyperparameters for a single experiment run.

    Attributes:
        target_lam      : final sparsity coefficient λ
        epochs          : total training epochs
        finetune_epochs : additional fine-tune epochs after hard pruning
        lr              : initial Adam learning rate
        weight_decay    : L2 regularisation on weights (not gates)
        label_smoothing : cross-entropy label smoothing factor
        dropout_p       : dropout probability in classifier head
        batch_size      : mini-batch size
        prune_tau       : gate threshold for hard pruning (post-training)
        warmup_frac     : fraction of epochs for λ warm-up ramp
        seed            : random seed (set once externally)
    """
    target_lam      : float = 0.10   # calibrated for normalized sparsity loss in [0,1]
    epochs          : int   = 20
    finetune_epochs : int   = 5
    lr              : float = 3e-4
    weight_decay    : float = 1e-4
    label_smoothing : float = 0.1
    dropout_p       : float = 0.3
    batch_size      : int   = 128
    prune_tau       : float = 0.10   # raised from 1e-2; log-penalty gates cluster 0.02–0.15 before zeroing
    warmup_frac     : float = 0.10   # 2-epoch warmup; 30% was too slow for 20 epochs


# ══════════════════════════════════════════════════════════════════════════════
# Single-epoch helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model    : SelfPruningCNN,
    loader   : DataLoader,
    optimizer: torch.optim.Optimizer,
    lam      : float,
    cfg      : TrainConfig,
    device   : torch.device,
) -> tuple[float, float, float]:
    """
    One full pass over the training set.

    Returns:
        (mean_total_loss, mean_ce_loss, mean_accuracy_%)
    """
    model.train()
    tot_loss = ce_loss_sum = correct = n = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)        # faster than zero_grad()
        logits = model(images)
        loss, ce, _ = compute_loss(
            logits, labels, model, lam, cfg.label_smoothing
        )
        loss.backward()

        # Gradient clipping — improves stability during λ ramp
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs           = labels.size(0)
        tot_loss    += loss.item() * bs
        ce_loss_sum += ce.item()   * bs
        correct     += (logits.detach().argmax(1) == labels).sum().item()
        n           += bs

    return tot_loss / n, ce_loss_sum / n, 100.0 * correct / n


@torch.no_grad()
def evaluate(
    model : SelfPruningCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on `loader`.

    Returns:
        (mean_cross_entropy_loss, accuracy_%)
    """
    model.eval()
    ce_sum = correct = n = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)
        logits  = model(images)
        ce_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        n       += labels.size(0)

    return ce_sum / n, 100.0 * correct / n


# ══════════════════════════════════════════════════════════════════════════════
# Full experiment: train → hard-prune → fine-tune
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    cfg         : TrainConfig,
    train_loader: DataLoader,
    test_loader : DataLoader,
    device      : torch.device,
    verbose     : bool = True,
) -> dict:
    """
    Three-phase pipeline for one λ configuration:

        Phase 1 — Soft-gate training
        ─────────────────────────────
        Train with curriculum λ schedule (ramp from 0 → target_lam).
        Cosine annealing LR + weight decay + label smoothing + dropout.

        Phase 2 — Hard threshold
        ─────────────────────────
        All gates < tau are forced to exactly 0.  The hard_mask buffer is
        registered on each PrunableLinear; subsequent forward passes skip
        those connections.

        Phase 3 — Fine-tuning
        ──────────────────────
        Brief re-training with λ=0 and a reduced LR.  The hard mask stays in
        place; only the surviving weights (and remaining gate scores) are updated.
        This compensates for any accuracy drop from hard pruning.

    Returns a result dict with training curves, final metrics, and gate arrays.
    """
    model     = SelfPruningCNN(dropout_p=cfg.dropout_p).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * 0.05,   # decay to 5% of initial LR
    )

    # ── History buffers ──────────────────────────────────────────────────────
    train_accs   : list[float] = []
    test_accs    : list[float] = []
    train_losses : list[float] = []
    lam_history  : list[float] = []

    # ── Phase 1: Soft-gate training ──────────────────────────────────────────
    if verbose:
        _header(cfg.target_lam)

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        lam = get_lambda(epoch, cfg.epochs, cfg.target_lam, cfg.warmup_frac)
        lam_history.append(lam)

        tr_loss, _, tr_acc = train_epoch(
            model, train_loader, optimizer, lam, cfg, device
        )
        _, ts_acc = evaluate(model, test_loader, device)
        scheduler.step()

        train_accs.append(tr_acc)
        test_accs.append(ts_acc)
        train_losses.append(tr_loss)

        if verbose:
            sp       = compute_sparsity(model, cfg.prune_tau)
            mean_g   = model.all_gates_flat().mean().item()   # watch this fall!
            lr_now   = scheduler.get_last_lr()[0]
            print(
                f"  {epoch:>3}  loss={tr_loss:.4f}  "
                f"tr={tr_acc:.1f}%  ts={ts_acc:.1f}%  "
                f"gate_mean={mean_g:.3f}  sparse={sp:.1f}%  "
                f"λ={lam:.2f}  lr={lr_now:.2e}"
            )

    acc_before_prune = test_accs[-1]
    sparsity_soft    = compute_sparsity(model, cfg.prune_tau)

    # ── Phase 2: Hard threshold ───────────────────────────────────────────────
    prune_report = model.apply_hard_threshold(cfg.prune_tau)
    _, acc_after_threshold = evaluate(model, test_loader, device)

    if verbose:
        total_pruned = sum(prune_report.values())
        print(f"\n  ── Hard threshold (τ={cfg.prune_tau:.0e}) ──────────────")
        for name, n in prune_report.items():
            print(f"     {name}: {n:,} weights pruned")
        print(f"     Total pruned       : {total_pruned:,}")
        print(f"     Accuracy before    : {acc_before_prune:.2f}%")
        print(f"     Accuracy after     : {acc_after_threshold:.2f}%")
        drop = acc_before_prune - acc_after_threshold
        print(f"     Drop from pruning  : {drop:+.2f}%")

    # ── Phase 3: Fine-tune ────────────────────────────────────────────────────
    finetune_accs: list[float] = []
    if cfg.finetune_epochs > 0:
        ft_optimizer = Adam(
            model.parameters(),
            lr=cfg.lr * 0.1,     # reduced LR for gentle adaptation
            weight_decay=cfg.weight_decay,
        )
        ft_scheduler = CosineAnnealingLR(
            ft_optimizer,
            T_max=cfg.finetune_epochs,
            eta_min=cfg.lr * 0.005,
        )
        if verbose:
            print(f"\n  ── Fine-tuning ({cfg.finetune_epochs} epochs, λ=0) ──────────")

        for ft_ep in range(1, cfg.finetune_epochs + 1):
            tr_loss, _, tr_acc = train_epoch(
                model, train_loader, ft_optimizer, lam=0.0, cfg=cfg, device=device
            )
            _, ts_acc = evaluate(model, test_loader, device)
            ft_scheduler.step()
            finetune_accs.append(ts_acc)

            if verbose:
                print(f"  FT{ft_ep:>2}  loss={tr_loss:.4f}  tr={tr_acc:.1f}%  ts={ts_acc:.1f}%")

    # ── Final metrics ─────────────────────────────────────────────────────────
    final_acc = finetune_accs[-1] if finetune_accs else acc_after_threshold
    sparsity  = compute_sparsity(model, cfg.prune_tau)
    flops     = estimate_flop_reduction(model, cfg.prune_tau)

    if verbose:
        print(f"\n  ══ Final Results  λ={cfg.target_lam:.0e} ══════════════════")
        print(f"     Test accuracy    : {final_acc:.2f}%")
        print(f"     Sparsity         : {sparsity:.1f}%")
        print(f"     MAC reduction    : {flops['mac_reduction_pct']:.1f}%")
        print(f"     Fine-tune gain   : {final_acc - acc_after_threshold:+.2f}%\n")
        model_summary(model)

    all_gates = model.all_gates_flat().cpu().numpy()

    return {
        "lam"              : cfg.target_lam,
        "test_acc"         : final_acc,
        "acc_before_prune" : acc_before_prune,
        "acc_after_threshold": acc_after_threshold,
        "sparsity"         : sparsity,
        "sparsity_soft"    : sparsity_soft,
        "mac_reduction"    : flops["mac_reduction_pct"],
        "train_accs"       : train_accs,
        "test_accs"        : test_accs,
        "finetune_accs"    : finetune_accs,
        "train_losses"     : train_losses,
        "lam_history"      : lam_history,
        "all_gates"        : all_gates,
        "prune_report"     : prune_report,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results: list[dict]) -> None:
    """Pretty comparison table across all λ experiments."""
    sep = "─" * 80
    print(f"\n{sep}")
    print(
        f"  {'Lambda':>8}  {'Test Acc':>9}  {'Pre-Prune':>10}  "
        f"{'Post-Hard':>10}  {'Sparsity':>9}  {'MAC ↓':>7}"
    )
    print(sep)
    for r in results:
        print(
            f"  {r['lam']:>8.0e}  {r['test_acc']:>8.2f}%  "
            f"{r['acc_before_prune']:>9.2f}%  "
            f"{r['acc_after_threshold']:>9.2f}%  "
            f"{r['sparsity']:>8.1f}%  {r['mac_reduction']:>6.1f}%"
        )
    print(f"{sep}\n")


def _header(lam: float) -> None:
    print(f"\n{'═'*60}")
    print(f"  Experiment  λ = {lam:.0e}")
    print(f"{'═'*60}")
    print(f"  {'Ep':>3}  {'Loss':>8}  {'Train':>6}  {'Test':>6}  "
          f"{'Sparse':>7}  {'λ-eff':>8}  {'LR':>9}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*9}")
