"""
model.py — Self-Pruning Neural Network architecture
====================================================
Defines:
  • PrunableLinear  – learnable-gate fully-connected layer
  • SelfPruningCNN  – lightweight CNN backbone + PrunableLinear head
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# PrunableLinear
# ══════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Fully-connected layer with per-weight learnable soft gates.

    Each scalar weight  w_ij  is paired with a gate score  g_ij.
    Forward pass:
        gate   = sigmoid(g_ij)          ∈ (0, 1)
        w_eff  = w_ij  ×  gate_ij      ← soft mask
        output = X @ W_eff.T + bias

    Why sigmoid?
    ─────────────
    Sigmoid maps any real number to (0, 1), giving a continuous,
    differentiable approximation to a binary on/off switch.  Gradients
    flow through the gate into g_ij, so the network learns *which*
    connections to keep—not just their magnitudes.

    Hard-pruning support
    ─────────────────────
    After training, call `apply_hard_threshold(tau)` to clamp gates
    below `tau` to exactly 0.  The layer then operates on a strictly
    sparse effective weight matrix.

    Args:
        in_features  : width of input vector
        out_features : width of output vector
        bias         : whether to include a bias term (default True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Trainable tensors ────────────────────────────────────────────────
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Optional hard mask (set by apply_hard_threshold, not trained)
        self.register_buffer("hard_mask", None)

        self._reset_parameters()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _reset_parameters(self) -> None:
        """
        Kaiming-uniform for weights (same as nn.Linear).
        Gate scores initialised so sigmoid ≈ 0.85 → network starts mostly open,
        ensuring rich gradients flow on epoch 1.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound  = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # logit(0.85) ≈ 1.735 → gates start near 0.85
        nn.init.constant_(self.gate_scores, 1.735)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates      = torch.sigmoid(self.gate_scores)          # (out, in)
        if self.hard_mask is not None:
            gates  = gates * self.hard_mask                   # zero out pruned
        eff_weight = self.weight * gates                      # soft masking
        return F.linear(x, eff_weight, self.bias)

    # ── Post-training hard pruning ────────────────────────────────────────────

    def apply_hard_threshold(self, tau: float = 1e-2) -> int:
        """
        Freeze all gates below `tau` to exactly 0 by storing a binary mask.
        Returns the number of connections pruned.

        This is a non-destructive operation: gate_scores are untouched, so
        fine-tuning with the mask in place can continue.
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            mask  = (gates >= tau).float()                    # 1 = keep, 0 = prune
            self.hard_mask = mask
        n_pruned = int((mask == 0).sum().item())
        return n_pruned

    def remove_hard_threshold(self) -> None:
        """Restore soft gating (undo hard threshold)."""
        self.hard_mask = None

    # ── Analysis helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Detached gate values for analysis (respects hard mask if set)."""
        g = torch.sigmoid(self.gate_scores)
        if self.hard_mask is not None:
            g = g * self.hard_mask
        return g

    @property
    def sparsity(self) -> float:
        """Fraction of gates currently below 1e-2."""
        g = self.get_gates()
        return float((g < 1e-2).float().mean().item())

    @property
    def total_weights(self) -> int:
        return self.in_features * self.out_features

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CNN Backbone  (no pruning — spatial feature extractor)
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv → BN → ReLU building block."""

    def __init__(
        self,
        in_ch : int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
# SelfPruningCNN  – main model
# ══════════════════════════════════════════════════════════════════════════════

class SelfPruningCNN(nn.Module):
    """
    Hybrid CNN + Prunable-Linear network for CIFAR-10.

    Architecture
    ─────────────
    Input  3×32×32
      │
      ├─ ConvBlock(3  → 32, 3×3)  \
      ├─ ConvBlock(32 → 32, 3×3)   ├─ stage 1
      └─ MaxPool2d(2×2) → 16×16  /
      │
      ├─ ConvBlock(32 → 64, 3×3)  \
      ├─ ConvBlock(64 → 64, 3×3)   ├─ stage 2
      └─ MaxPool2d(2×2) → 8×8    /
      │
      ├─ ConvBlock(64 → 128, 3×3) \
      ├─ ConvBlock(128→ 128, 3×3)  ├─ stage 3
      └─ AdaptiveAvgPool → 4×4   /
      │
      └─ Flatten → 128×4×4 = 2048
           │
           PrunableLinear(2048 → 512) + ReLU + Dropout
           PrunableLinear(512  → 256) + ReLU + Dropout
           PrunableLinear(256  →  10)

    Design rationale
    ─────────────────
    • CNN stages learn *where* (spatial) and *what* (channel) features matter;
      convolution+BN handles covariate shift without needing pruning gates there.
    • PrunableLinear layers prune the high-dimensional feature→decision mapping,
      which is where most parameters live and over-fitting occurs most.
    • Dropout is placed between prunable layers; it acts as a complementary
      regulariser, independent of gate sparsity.

    Args:
        dropout_p : dropout probability between prunable layers (default 0.3)
    """

    def __init__(self, dropout_p: float = 0.3) -> None:
        super().__init__()

        # ── CNN feature extractor ────────────────────────────────────────────
        self.features = nn.Sequential(
            # Stage 1 — 32×32 → 16×16
            ConvBlock(  3,  32), ConvBlock( 32,  32),
            nn.MaxPool2d(2, 2),
            # Stage 2 — 16×16 → 8×8
            ConvBlock( 32,  64), ConvBlock( 64,  64),
            nn.MaxPool2d(2, 2),
            # Stage 3 — 8×8 → 4×4
            ConvBlock( 64, 128), ConvBlock(128, 128),
            nn.AdaptiveAvgPool2d(4),
        )
        # 128 channels × 4×4 spatial = 2 048 features after flatten

        # ── Prunable classifier head ─────────────────────────────────────────
        self.fc1     = PrunableLinear(2_048, 512)
        self.drop1   = nn.Dropout(p=dropout_p)
        self.fc2     = PrunableLinear(512,   256)
        self.drop2   = nn.Dropout(p=dropout_p)
        self.fc3     = PrunableLinear(256,    10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                     # (B, 128, 4, 4)
        x = x.flatten(1)                         # (B, 2048)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.fc3(x)                       # (B, 10)  raw logits

    # ── Convenience helpers ──────────────────────────────────────────────────

    def prunable_layers(self) -> list[PrunableLinear]:
        """Return all PrunableLinear modules in order."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def all_gate_scores(self) -> list[torch.Tensor]:
        """Live (differentiable) gate-score tensors — used by sparsity loss."""
        return [layer.gate_scores for layer in self.prunable_layers()]

    @torch.no_grad()
    def all_gates_flat(self) -> torch.Tensor:
        """Concatenated, detached gate values across all prunable layers."""
        return torch.cat([layer.get_gates().view(-1) for layer in self.prunable_layers()])

    def apply_hard_threshold(self, tau: float = 1e-2) -> dict[str, int]:
        """
        Apply hard threshold to every PrunableLinear layer.
        Returns a dict {layer_name: n_pruned}.
        """
        report: dict[str, int] = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                report[name] = module.apply_hard_threshold(tau)
        return report

    def remove_hard_threshold(self) -> None:
        """Restore soft gating in every PrunableLinear layer."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                module.remove_hard_threshold()

    def param_stats(self) -> dict[str, int]:
        """
        Count total vs prunable vs active parameters.
        'Active' = gates above threshold (i.e. effectively non-zero).
        """
        total_prunable = sum(l.total_weights for l in self.prunable_layers())
        total_active   = sum(
            int((l.get_gates() >= 1e-2).sum().item())
            for l in self.prunable_layers()
        )
        total_params   = sum(p.numel() for p in self.parameters())
        return {
            "total_params"   : total_params,
            "prunable_weights": total_prunable,
            "active_weights"  : total_active,
            "pruned_weights"  : total_prunable - total_active,
        }
