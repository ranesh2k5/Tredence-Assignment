"""
utils.py — Data loading, loss functions, metrics, and reproducibility
=====================================================================
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SelfPruningCNN


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Pin all relevant RNG sources for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN (slight speed cost — acceptable for research)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 data loaders
# ══════════════════════════════════════════════════════════════════════════════

# Standard CIFAR-10 normalisation statistics
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(
    data_dir   : str = "./data",
    batch_size : int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for CIFAR-10.

    Training augmentations applied:
      • RandomCrop(32, padding=4)   — translation invariance
      • RandomHorizontalFlip        — left/right symmetry
      • ColorJitter                 — mild photometric robustness

    Test set uses only normalisation (no augmentation → reproducible metrics).
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# Lambda scheduling  (curriculum sparsity)
# ══════════════════════════════════════════════════════════════════════════════

def get_lambda(
    epoch      : int,
    total_epochs: int,
    target_lam : float,
    warmup_frac: float = 0.10,
) -> float:
    """
    Linearly ramp λ from 0 → target_lam over the first `warmup_frac` of training,
    then hold constant.

    Why curriculum sparsity?
    ─────────────────────────
    Applying the full sparsity penalty from epoch 1 forces the network to prune
    before it has learned useful representations, degrading accuracy.  Starting
    with λ=0 lets the network first build a good feature space; then gradually
    increasing λ encourages pruning of the *least useful* connections.

    Why 10% warmup (not 30%)?
    ──────────────────────────
    With 20 epochs total, 30% warmup = 6 epochs of λ≈0.  By that point the
    network has already committed its gate scores.  10% warmup = 2 epochs,
    just enough for the CNN to learn basic edge detectors before pruning starts.

    Args:
        epoch        : current epoch (1-indexed)
        total_epochs : total number of training epochs
        target_lam   : final target value of λ  (on normalized 0-1 scale)
        warmup_frac  : fraction of epochs used for linear ramp (default 0.10)

    Returns:
        Effective λ for the current epoch.
    """
    warmup_end = max(1, int(total_epochs * warmup_frac))
    if epoch <= warmup_end:
        return target_lam * (epoch / warmup_end)
    return target_lam


# ══════════════════════════════════════════════════════════════════════════════
# Loss function
# ══════════════════════════════════════════════════════════════════════════════

def sparsity_loss(model: SelfPruningCNN) -> torch.Tensor:
    """
    Log-based sparsity penalty that strongly pushes gates toward zero.

    Loss = mean( -log(1 - σ(g) + ε) )

    Why log-penalty instead of mean(σ(g))?
    ────────────────────────────────────────
    The old loss mean(σ(g)) has gradient σ(g)·(1-σ(g))/N — this is the
    sigmoid derivative, which *vanishes* at both extremes.  Gates near 1.0
    (where all gates start, σ≈0.85) receive gradient ≈ 0.13/N — extremely
    small.  With N≈1.3M weights, this is ~1e-7 per gate per step: effectively
    zero.  Gates stall and never collapse.

    The log penalty fixes both problems:

    1.  ∂/∂g [ -log(1 - σ(g) + ε) ] = σ(g) / (1 - σ(g) + ε)

        For σ(g) = 0.85 → gradient ≈ 0.85/0.15 ≈ 5.7   (43× larger!)
        For σ(g) = 0.50 → gradient ≈ 0.50/0.50 ≈ 1.0
        For σ(g) = 0.10 → gradient ≈ 0.10/0.90 ≈ 0.11  (self-stops once pruned)

        This is a *monotonically increasing* function of σ(g): the larger the
        gate, the harder it is pushed down.  Exactly the opposite of the old
        loss.  Gates at 0.85 feel strong pressure; gates near 0 feel almost
        none (they're already pruned — leave them alone).

    2.  Bimodality is natural: large gates are pushed aggressively until they
        cross the decision boundary; small gates are left stable.  The result
        is a clean 0/1 distribution rather than a smeared middle cluster.

    Stability:
        ε = 1e-6 prevents log(0) when σ(g) → 1.0 exactly.
        Output is normalized by N (mean not sum), keeping the loss in a
        predictable range so the same λ values work across architectures.

    Rule of thumb for λ with this loss (output range ≈ 0–5):
        λ = 0.01 → light pruning
        λ = 0.10 → medium pruning
        λ = 0.40 → aggressive pruning
    """
    EPS = 1e-6
    gate_scores = model.all_gate_scores()
    all_gates = torch.cat([gs.sigmoid().reshape(-1) for gs in gate_scores])
    # -log(1 - gate + ε): high value when gate→1, low when gate→0
    penalty = -torch.log(1.0 - all_gates + EPS)
    return penalty.mean()  # normalized; gradient ∝ gate/(1-gate)


def compute_loss(
    logits  : torch.Tensor,
    targets : torch.Tensor,
    model   : SelfPruningCNN,
    lam     : float,
    label_smoothing: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss = LabelSmoothedCE  +  λ × SparsityLoss

    Label smoothing (ε=0.1) prevents over-confident softmax outputs and
    improves calibration, typically adding +0.5-1% top-1 accuracy.

    Returns:
        (total_loss, ce_loss, sparsity_loss)  — all scalar tensors
    """
    ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    sp = sparsity_loss(model)
    return ce + lam * sp, ce, sp


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy as a percentage."""
    correct = (logits.argmax(dim=1) == targets).sum().item()
    return 100.0 * correct / targets.size(0)


def compute_sparsity(model: SelfPruningCNN, tau: float = 0.10) -> float:
    """
    Percentage of gates whose sigmoid value is below `tau`.
    A gate < tau contributes effectively zero signal → pruned connection.

    Threshold raised from 1e-2 → 0.10:
        The log-penalty pushes gates toward 0 but they often settle in the
        0.02–0.15 range before snapping fully to zero.  tau=0.01 would
        falsely report 0% sparsity for those intermediate-value gates.
        tau=0.10 gives an accurate real-time picture of effectively-dead gates.
        Use the same value for apply_hard_threshold() (see main.py).
    """
    all_gates = model.all_gates_flat()
    return 100.0 * float((all_gates < tau).float().mean().item())


# ══════════════════════════════════════════════════════════════════════════════
# FLOP / parameter reduction estimator
# ══════════════════════════════════════════════════════════════════════════════

def estimate_flop_reduction(model: SelfPruningCNN, tau: float = 1e-2) -> dict[str, float]:
    """
    Estimate the theoretical reduction in multiply-accumulate (MAC) operations
    for the prunable linear layers due to gate sparsity.

    For a linear layer of shape (out, in), each active (non-pruned) weight
    requires 1 MAC per sample.  Pruned weights can be skipped entirely on
    hardware that supports sparse computation.

    Note: This is a *theoretical* estimate.  Actual speedup depends on the
    sparsity pattern and hardware support (e.g. block sparsity, sparse BLAS).

    Returns a dict with keys:
        dense_macs_linear   : MACs in all linear layers if fully dense
        sparse_macs_linear  : effective MACs after pruning
        mac_reduction_pct   : percentage reduction
        param_reduction_pct : percentage of linear weights pruned
    """
    layers = model.prunable_layers()

    dense_macs = sum(l.in_features * l.out_features for l in layers)
    active     = sum(
        int((l.get_gates() >= tau).sum().item()) for l in layers
    )
    sparse_macs = active

    return {
        "dense_macs_linear"  : dense_macs,
        "sparse_macs_linear" : sparse_macs,
        "mac_reduction_pct"  : 100.0 * (1.0 - sparse_macs / max(dense_macs, 1)),
        "param_reduction_pct": 100.0 * (1.0 - active / max(dense_macs, 1)),
    }


def model_summary(model: SelfPruningCNN) -> None:
    """Print a concise parameter and architecture summary."""
    stats = model.param_stats()
    flops = estimate_flop_reduction(model)
    sep   = "─" * 52
    print(f"\n  {sep}")
    print(f"  Model Parameter Summary")
    print(f"  {sep}")
    print(f"  Total parameters      : {stats['total_params']:>10,}")
    print(f"  Prunable weights      : {stats['prunable_weights']:>10,}")
    print(f"  Active weights        : {stats['active_weights']:>10,}")
    print(f"  Pruned weights        : {stats['pruned_weights']:>10,}")
    print(f"  Linear MAC reduction  : {flops['mac_reduction_pct']:>9.1f}%")
    print(f"  {sep}\n")
