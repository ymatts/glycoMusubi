"""Multi-task composite loss for GlycoKG-Net training.

Combines link-prediction loss, structural contrastive loss, hyperbolic
regularization, and L2 regularization into a single differentiable objective.

References
----------
- Section 4.1 of ``docs/architecture/model_architecture_design.md``
- Section 4.2.5 of ``docs/design/algorithm_design.md``
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """Multi-task composite loss.

    ``L_total = L_link + lambda_struct * L_struct + lambda_hyp * L_hyp + lambda_reg * L_reg + lambda_node * L_node``

    * **L_link** -- any existing link-prediction loss (``MarginRankingLoss``
      or ``BCEWithLogitsKGELoss``).
    * **L_struct** -- InfoNCE-style contrastive loss on glycan structural
      embeddings encouraging glycans that share motifs / substructures to
      have similar representations.
    * **L_hyp** -- Riemannian gradient penalty for embeddings near the
      Poincare ball boundary, discouraging degenerate solutions where
      all points collapse to the boundary.
    * **L_reg** -- L2 regularization on all embedding tensors.
    * **L_node** -- Cross-entropy loss for node classification (optional).

    Parameters
    ----------
    link_loss : nn.Module
        An existing loss instance whose ``forward(pos_scores, neg_scores)``
        returns a scalar loss.
    lambda_struct : float
        Weight for the structural contrastive term (default 0.1).
    lambda_hyp : float
        Weight for the hyperbolic regularization term (default 0.01).
    lambda_reg : float
        Weight for the L2 regularization term (default 0.01).
    lambda_node : float
        Weight for the node classification loss term (default 0.0).
        When 0.0, the term is skipped entirely.
    struct_temperature : float
        Temperature for the InfoNCE contrastive loss (default 0.07).
    curvature : float
        Curvature of the Poincare ball for hyperbolic regularization
        (default 1.0).
    """

    def __init__(
        self,
        link_loss: nn.Module,
        lambda_struct: float = 0.1,
        lambda_hyp: float = 0.01,
        lambda_reg: float = 0.01,
        lambda_node: float = 0.0,
        struct_temperature: float = 0.07,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.link_loss = link_loss
        self.lambda_struct = lambda_struct
        self.lambda_hyp = lambda_hyp
        self.lambda_reg = lambda_reg
        self.lambda_node = lambda_node
        self.struct_temperature = struct_temperature
        self.curvature = curvature

    # ------------------------------------------------------------------
    def structural_contrastive_loss(
        self,
        glycan_embeddings: torch.Tensor,
        positive_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE-style contrastive loss for glycan structural similarity.

        For each positive pair (i, j) the loss treats *j* as the positive
        and all other glycans in the batch as negatives (and vice-versa in
        a symmetric formulation).

        Parameters
        ----------
        glycan_embeddings : torch.Tensor
            ``[N_glycan, d]`` -- L2-normalized internally.
        positive_pairs : torch.Tensor
            ``[P, 2]`` -- index pairs into ``glycan_embeddings`` that denote
            glycans sharing motifs or substructures.

        Returns
        -------
        torch.Tensor
            Scalar contrastive loss.
        """
        if positive_pairs.numel() == 0:
            return glycan_embeddings.new_tensor(0.0)

        # L2 normalize embeddings
        z = F.normalize(glycan_embeddings, dim=-1)  # [N, d]

        # Full similarity matrix [N, N]
        sim = torch.mm(z, z.t()) / self.struct_temperature

        idx_i = positive_pairs[:, 0]  # [P]
        idx_j = positive_pairs[:, 1]  # [P]

        # Direction i -> j: for anchor i, positive is j
        logits_ij = sim[idx_i]                    # [P, N]
        loss_ij = F.cross_entropy(logits_ij, idx_j)

        # Direction j -> i: for anchor j, positive is i
        logits_ji = sim[idx_j]                    # [P, N]
        loss_ji = F.cross_entropy(logits_ji, idx_i)

        return (loss_ij + loss_ji) / 2.0

    # ------------------------------------------------------------------
    def hyperbolic_regularization(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian gradient penalty for embeddings near the Poincare ball boundary.

        As points approach the boundary of the Poincare ball (||x|| -> 1/sqrt(c)),
        gradients in the ambient space become vanishingly small while Riemannian
        gradients explode.  This penalty discourages embeddings from drifting to
        the boundary by penalising high conformal factors.

        ``L_hyp = mean(lambda_x^2)`` where ``lambda_x = 2 / (1 - c * ||x||^2)``

        Parameters
        ----------
        embeddings : torch.Tensor
            Embedding tensor ``[N, d]`` (points that live in or are mapped into
            the Poincare ball).

        Returns
        -------
        torch.Tensor
            Scalar regularization penalty.
        """
        c = self.curvature
        eps = 1e-5
        x_sqnorm = (embeddings * embeddings).sum(dim=-1)
        # Clamp so we stay inside the open ball.
        x_sqnorm = x_sqnorm.clamp(max=1.0 / c - eps)
        # Conformal factor: lambda_x = 2 / (1 - c * ||x||^2)
        lam = 2.0 / (1.0 - c * x_sqnorm).clamp(min=eps)
        return (lam ** 2).mean()

    # ------------------------------------------------------------------
    @staticmethod
    def _l2_regularization(
        all_embeddings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute L2 norm regularization over a dictionary of embeddings.

        Parameters
        ----------
        all_embeddings : dict[str, torch.Tensor]
            Mapping from name to embedding tensor (arbitrary shape).

        Returns
        -------
        torch.Tensor
            Scalar sum of L2 norms.
        """
        reg = sum(e.norm(2) for e in all_embeddings.values())
        # sum() returns int 0 if dict is empty; ensure tensor output
        if not isinstance(reg, torch.Tensor):
            return torch.tensor(0.0)
        return reg

    # ------------------------------------------------------------------
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        glycan_embeddings: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
        all_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        hyperbolic_embeddings: Optional[torch.Tensor] = None,
        node_logits: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the composite loss.

        ``L_total = L_link + lambda_struct * L_struct + lambda_hyp * L_hyp + lambda_reg * L_reg + lambda_node * L_node``

        Parameters
        ----------
        pos_scores : torch.Tensor
            Scores for positive triples (passed to ``link_loss``).
        neg_scores : torch.Tensor
            Scores for negative triples (passed to ``link_loss``).
        glycan_embeddings : torch.Tensor, optional
            ``[N_glycan, d]`` for the structural contrastive term.
        positive_pairs : torch.Tensor, optional
            ``[P, 2]`` positive glycan pairs for the contrastive term.
        all_embeddings : dict[str, torch.Tensor], optional
            Named embedding tensors for L2 regularization.
        hyperbolic_embeddings : torch.Tensor, optional
            ``[N, d]`` embeddings that are mapped into the Poincare ball,
            for hyperbolic boundary regularization.
        node_logits : torch.Tensor, optional
            ``[N, C]`` logits from a node classification head.
        node_labels : torch.Tensor, optional
            ``[N]`` ground-truth class indices for the node classification
            loss.

        Returns
        -------
        torch.Tensor
            Scalar composite loss.
        """
        loss = self.link_loss(pos_scores, neg_scores)

        if glycan_embeddings is not None and positive_pairs is not None:
            loss = loss + self.lambda_struct * self.structural_contrastive_loss(
                glycan_embeddings, positive_pairs
            )

        if hyperbolic_embeddings is not None:
            loss = loss + self.lambda_hyp * self.hyperbolic_regularization(
                hyperbolic_embeddings
            )

        if all_embeddings is not None:
            loss = loss + self.lambda_reg * self._l2_regularization(
                all_embeddings
            )

        if (
            self.lambda_node > 0
            and node_logits is not None
            and node_labels is not None
        ):
            loss = loss + self.lambda_node * F.cross_entropy(
                node_logits, node_labels
            )

        return loss
