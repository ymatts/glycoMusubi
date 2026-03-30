"""Loss functions for KGE training."""

from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.losses.bce_loss import BCEWithLogitsKGELoss
from glycoMusubi.losses.composite_loss import CompositeLoss
from glycoMusubi.losses.cmca_loss import CMCALoss

__all__ = [
    "MarginRankingLoss",
    "BCEWithLogitsKGELoss",
    "CompositeLoss",
    "CMCALoss",
]
