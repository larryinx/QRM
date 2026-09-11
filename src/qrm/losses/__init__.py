"""QRM Loss Functions.

This module provides loss functions for QRM training:
- qrm_tree_loss: Weighted cross-entropy loss over candidates
- qrm_diversity_loss: Energy score based diversity loss (optional ablation)

Also re-exports TRM loss utilities for convenience.
"""

from qrm.losses.qrm_tree import qrm_diversity_loss, qrm_tree_loss
from qrm.losses.trm import IGNORE_LABEL_ID, stablemax_cross_entropy

__all__ = [
    # QRM losses
    "qrm_tree_loss",
    "qrm_diversity_loss",
    # TRM utilities
    "IGNORE_LABEL_ID",
    "stablemax_cross_entropy",
]
