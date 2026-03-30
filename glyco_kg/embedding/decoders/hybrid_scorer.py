"""Hybrid link prediction scorer combining multiple scoring paradigms.

Combines TransE (translational), DistMult (bilinear), RotatE (rotational),
a neural MLP scorer, and Poincare hyperbolic distance with per-relation
learnable weights (softmax-normalized).

The combiner weights are predicted from a dedicated relation embedding
(not tied to any sub-scorer) for better decoupling.

References
----------
- Section 3.6.1 of ``docs/architecture/model_architecture_design.md``
- Section 4.2.4 of ``docs/design/algorithm_design.md``
"""

from __future__ import annotations

import torch
import torch.nn as nn

from glycoMusubi.embedding.decoders.distmult import DistMultDecoder
from glycoMusubi.embedding.decoders.rotate import RotatEDecoder
from glycoMusubi.embedding.models.poincare import PoincareDistance


class HybridLinkScorer(nn.Module):
    """Composite link prediction scorer.

    ``score(h, r, t) = w1(r)*TransE + w2(r)*DistMult + w3(r)*RotatE
                      + w4(r)*Neural + w5(r)*Poincare``

    Per-relation learnable weights are obtained by applying a learned
    linear layer to a dedicated relation embedding (decoupled from
    sub-scorers) and normalizing via softmax so they sum to 1.

    Parameters
    ----------
    embedding_dim : int
        Dimension of entity embeddings (default 256).
    num_relations : int
        Number of relation types in the knowledge graph (default 12).
    neural_hidden_dim : int
        Hidden dimension of the neural scorer MLP (default 512).
    dropout : float
        Dropout probability for the neural scorer (default 0.1).
    curvature : float
        Curvature of the Poincare ball for the hyperbolic scorer (default 1.0).
    p_norm : int
        Norm order for the TransE scorer (default 2).
    """

    NUM_SUB_SCORERS = 5

    def __init__(
        self,
        embedding_dim: int = 256,
        num_relations: int = 12,
        neural_hidden_dim: int = 512,
        dropout: float = 0.1,
        curvature: float = 1.0,
        p_norm: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.complex_dim = embedding_dim // 2
        self.p_norm = p_norm

        # ---- Sub-scorers (reuse existing decoders) ----
        self.distmult = DistMultDecoder()
        self.rotate = RotatEDecoder()
        self.poincare = PoincareDistance(curvature=curvature)

        # ---- Relation embeddings (one per sub-scorer that needs them) ----
        # TransE: full-dim translation vector
        self.rel_embed_transe = nn.Embedding(num_relations, embedding_dim)
        # DistMult: full-dim bilinear relation diagonal
        self.rel_embed_distmult = nn.Embedding(num_relations, embedding_dim)
        # RotatE: half-dim phase-angle relation embeddings
        self.rel_embed_rotate = nn.Embedding(num_relations, self.complex_dim)
        # Poincare: full-dim tangent-space translations
        self.rel_embed_poincare = nn.Embedding(num_relations, embedding_dim)

        # ---- Neural scorer MLP: [h || r || t] -> scalar ----
        # Uses the TransE relation embedding for the concatenation.
        self.neural_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 3, neural_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(neural_hidden_dim, 1),
        )

        # ---- Per-sub-scorer scale normalization ----
        # Each sub-scorer outputs on a different scale (e.g., TransE: [-25,0],
        # DistMult: [-50,+50], Poincaré: [-5,0]).  Learnable log-scale and
        # bias parameters normalise them to a comparable range before the
        # weighted combination.
        self.score_log_scale = nn.Parameter(torch.zeros(self.NUM_SUB_SCORERS))
        self.score_bias = nn.Parameter(torch.zeros(self.NUM_SUB_SCORERS))

        # ---- Per-relation weight predictor (decoupled) ----
        # Dedicated relation embedding for the combiner weights,
        # not tied to any sub-scorer's relation embeddings.
        self.rel_embed_weights = nn.Embedding(num_relations, embedding_dim)
        self.weight_net = nn.Linear(embedding_dim, self.NUM_SUB_SCORERS)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Apply sensible initialization."""
        nn.init.xavier_uniform_(self.rel_embed_transe.weight)
        nn.init.xavier_uniform_(self.rel_embed_distmult.weight)
        nn.init.xavier_uniform_(self.rel_embed_rotate.weight)
        nn.init.xavier_uniform_(self.rel_embed_poincare.weight)
        nn.init.xavier_uniform_(self.rel_embed_weights.weight)
        nn.init.zeros_(self.weight_net.bias)
        nn.init.xavier_uniform_(self.weight_net.weight)

    # ------------------------------------------------------------------
    def forward(
        self,
        head: torch.Tensor,
        relation_idx: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of triples.

        Parameters
        ----------
        head : torch.Tensor
            Head entity embeddings ``[batch, embedding_dim]``.
        relation_idx : torch.Tensor
            Integer relation-type indices ``[batch]``.
        tail : torch.Tensor
            Tail entity embeddings ``[batch, embedding_dim]``.

        Returns
        -------
        torch.Tensor
            Plausibility scores ``[batch]``.
        """
        # -- Relation embeddings --
        r_transe = self.rel_embed_transe(relation_idx)       # [B, d]
        r_dm = self.rel_embed_distmult(relation_idx)         # [B, d]
        r_rot = self.rel_embed_rotate(relation_idx)          # [B, d/2]
        r_poincare = self.rel_embed_poincare(relation_idx)   # [B, d]

        # -- Sub-scorer outputs (all: higher = more plausible) --
        # TransE: negative L2 distance after translation
        score_transe = -(head + r_transe - tail).norm(
            p=self.p_norm, dim=-1
        )                                                     # [B]

        score_dm = self.distmult(head, r_dm, tail)            # [B]
        score_rot = self.rotate(head, r_rot, tail)            # [B]

        # Neural scorer uses the TransE relation embedding (strongest
        # baseline on this KG) for the concatenation.
        score_nn = self.neural_scorer(
            torch.cat([head, r_transe, tail], dim=-1)
        ).squeeze(-1)                                         # [B]

        score_hyp = self.poincare(head, r_poincare, tail)     # [B]

        # -- Normalise sub-scorer outputs to comparable scales --
        raw_scores = torch.stack(
            [score_transe, score_dm, score_rot, score_nn, score_hyp], dim=-1
        )                                                       # [B, 5]
        # score_i' = score_i / exp(log_scale_i) + bias_i
        norm_scores = raw_scores / self.score_log_scale.exp() + self.score_bias

        # -- Per-relation adaptive weights (decoupled) --
        r_w = self.rel_embed_weights(relation_idx)            # [B, d]
        weights = torch.softmax(self.weight_net(r_w), dim=-1) # [B, 5]

        score = (weights * norm_scores).sum(dim=-1)           # [B]
        return score
