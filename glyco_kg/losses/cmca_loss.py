"""Cross-Modal Contrastive Alignment (CMCA) loss.

Implements the two-phase contrastive alignment objective from
Architecture Design Section 4.2:

* **Phase 1 — Intra-modal contrastive**: pull positive pairs (shared
  motifs, co-glycosylation, etc.) together while pushing random
  negatives apart within the same modality.
* **Phase 2 — Cross-modal alignment**: align modality-specific
  embeddings (e.g. glycan-tree, ESM2-protein) with their corresponding
  KG embeddings via symmetric InfoNCE.

References
----------
- Section 4.2 of ``docs/architecture/model_architecture_design.md``
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMCALoss(nn.Module):
    """Cross-Modal Contrastive Alignment loss.

    Parameters
    ----------
    temperature : float
        Temperature scaling for the InfoNCE logits (default 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Phase 1: Intra-modal contrastive
    # ------------------------------------------------------------------

    def intra_modal_loss(
        self,
        embeddings: torch.Tensor,
        positive_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss within a single modality.

        For every positive pair ``(i, j)`` the loss treats ``j`` as the
        positive and all other embeddings in the batch as negatives (and
        symmetrically ``i`` as the positive for anchor ``j``).

        Parameters
        ----------
        embeddings : torch.Tensor
            Shape ``[N, d]`` — embeddings for a single modality.
        positive_pairs : torch.Tensor
            Shape ``[P, 2]`` — indices into ``embeddings`` that form
            positive pairs (e.g. glycans sharing motifs, proteins
            glycosylated by the same glycan).

        Returns
        -------
        torch.Tensor
            Scalar InfoNCE loss.
        """
        if positive_pairs.numel() == 0:
            return embeddings.new_tensor(0.0)

        z = F.normalize(embeddings, dim=-1)  # [N, d]
        sim = torch.mm(z, z.t()) / self.temperature  # [N, N]

        idx_i = positive_pairs[:, 0]  # [P]
        idx_j = positive_pairs[:, 1]  # [P]

        # Direction i -> j
        loss_ij = F.cross_entropy(sim[idx_i], idx_j)
        # Direction j -> i
        loss_ji = F.cross_entropy(sim[idx_j], idx_i)

        return (loss_ij + loss_ji) / 2.0

    # ------------------------------------------------------------------
    # Phase 2: Cross-modal alignment
    # ------------------------------------------------------------------

    def cross_modal_loss(
        self,
        modal_embeddings: torch.Tensor,
        kg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric InfoNCE loss to align modality-specific and KG embeddings.

        Each row ``i`` in ``modal_embeddings`` is paired with row ``i``
        in ``kg_embeddings`` as its positive; all other rows serve as
        negatives.

        Parameters
        ----------
        modal_embeddings : torch.Tensor
            Shape ``[N, d]`` — modality-specific embeddings (e.g.
            glycan-tree or ESM2-protein).
        kg_embeddings : torch.Tensor
            Shape ``[N, d]`` — corresponding KG embeddings for the same
            entities.

        Returns
        -------
        torch.Tensor
            Scalar symmetric InfoNCE loss.
        """
        if modal_embeddings.shape[0] == 0:
            return modal_embeddings.new_tensor(0.0)

        z_modal = F.normalize(modal_embeddings, dim=-1)  # [N, d]
        z_kg = F.normalize(kg_embeddings, dim=-1)  # [N, d]

        # Cross-similarity: [N, N]
        logits = torch.mm(z_modal, z_kg.t()) / self.temperature

        # Targets: row i should match column i
        targets = torch.arange(
            logits.shape[0], device=logits.device, dtype=torch.long
        )

        # Symmetric: modal -> KG and KG -> modal
        loss_m2k = F.cross_entropy(logits, targets)
        loss_k2m = F.cross_entropy(logits.t(), targets)

        return (loss_m2k + loss_k2m) / 2.0

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        modal_embeddings: Optional[torch.Tensor] = None,
        kg_embeddings: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute CMCA losses.

        Parameters
        ----------
        modal_embeddings : torch.Tensor or None
            Shape ``[N, d]`` — modality-specific embeddings.  Required
            for both intra-modal and cross-modal terms.
        kg_embeddings : torch.Tensor or None
            Shape ``[N, d]`` — KG embeddings.  Required for the
            cross-modal term.
        positive_pairs : torch.Tensor or None
            Shape ``[P, 2]`` — positive pairs for the intra-modal term.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys ``'intra_modal_loss'`` and
            ``'cross_modal_loss'``, each a scalar tensor.  Losses that
            cannot be computed (missing inputs) are returned as 0.
        """
        device = (
            modal_embeddings.device
            if modal_embeddings is not None
            else (kg_embeddings.device if kg_embeddings is not None else "cpu")
        )
        zero = torch.tensor(0.0, device=device)

        # Intra-modal
        if modal_embeddings is not None and positive_pairs is not None:
            intra = self.intra_modal_loss(modal_embeddings, positive_pairs)
        else:
            intra = zero

        # Cross-modal
        if modal_embeddings is not None and kg_embeddings is not None:
            cross = self.cross_modal_loss(modal_embeddings, kg_embeddings)
        else:
            cross = zero

        return {
            "intra_modal_loss": intra,
            "cross_modal_loss": cross,
        }
