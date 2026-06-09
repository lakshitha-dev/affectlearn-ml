"""Focal loss for DAiSEE class imbalance.

DAiSEE engagement labels are heavily skewed (very-low: ~0.7 %, very-high: ~45 %).
Plain cross-entropy lets the model ignore rare classes.
Focal loss (Lin et al., 2017) down-weights easy examples so the model focuses on
hard, under-represented ones.  gamma=2, alpha=inverse-class-frequency is the
recommended configuration from the training guide.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        gamma: Focusing parameter (0 = standard CE, 2 recommended).
        alpha: Per-class weight tensor (shape [num_classes]).
               Pass the output of DAiSEEDataset.class_weights() here.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha      # registered as a plain attribute (not a parameter)
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)  # UNWEIGHTED ce
        pt   = torch.exp(-ce)                          # true-class probability
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:                     # per-sample alpha multiplier
            at   = self.alpha.to(logits.device).gather(0, targets)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
