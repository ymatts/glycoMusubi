"""Protein encoder for knowledge graph embedding.

Provides three encoding strategies:

* **learnable** (default) -- plain ``nn.Embedding`` lookup, suitable when
  pre-trained protein language model embeddings are not available.
* **esm2** -- loads pre-computed ESM-2 per-protein embeddings from a
  cache directory and projects them through a two-layer MLP
  (1280 -> 512 -> 256).  Falls back to learnable embeddings for
  proteins whose cache file is missing.
* **esm2_site_aware** -- like ``esm2`` but incorporates glycosylation
  site context.  For each known site position *p*, a local window of
  per-residue ESM-2 embeddings ``[p-w : p+w]`` is extracted and
  combined with a positional encoding and the global mean to produce
  a site-context vector.  Site contexts are aggregated and merged with
  the sequence-level embedding.  Falls back to standard ``esm2``
  pooling when no site positions are available.

The ``esm2`` / ``esm2_site_aware`` modes expect ``.pt`` files in
*cache_path*, keyed by integer node index.  For ``esm2`` each file
contains a 1-D tensor ``[1280]``.  For ``esm2_site_aware`` each file
should contain a 2-D tensor ``[L, 1280]`` (per-residue embeddings).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _positional_encoding(position: int, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding for a single integer position.

    Parameters
    ----------
    position : int
        Residue position (0-indexed).
    dim : int
        Encoding dimensionality.

    Returns
    -------
    torch.Tensor
        Shape ``[dim]``.
    """
    pe = torch.zeros(dim)
    for i in range(0, dim, 2):
        denom = math.exp(i * (-math.log(10000.0) / dim))
        pe[i] = math.sin(position * denom)
        if i + 1 < dim:
            pe[i + 1] = math.cos(position * denom)
    return pe


class ProteinEncoder(nn.Module):
    """Encode protein nodes into fixed-dimensional embedding vectors.

    Parameters
    ----------
    num_proteins : int
        Number of protein entities in the KG (for learnable table).
    output_dim : int
        Dimensionality of the output embedding vector.
    method : str
        ``"learnable"``, ``"esm2"``, or ``"esm2_site_aware"``.
    esm2_dim : int
        Dimensionality of the pre-computed ESM-2 embeddings (default 1280).
    cache_path : str or Path, optional
        Directory containing pre-computed ESM-2 ``.pt`` files.
        Only used when *method* is ``"esm2"`` or ``"esm2_site_aware"``.
    dropout : float
        Dropout probability in the projection MLP.
    site_positions_map : dict mapping int to list, optional
        Protein index -> list of glycosylation site data.  Entries can be
        plain ``int`` positions (legacy format) or ``(position, residue,
        site_type)`` tuples for enriched site data.
        Only used when *method* is ``"esm2_site_aware"``.
    site_window : int
        Half-window size for site context extraction (default 15).
    """

    # Site type vocabulary for site_type_emb: N-linked=0, O-linked=1, C-linked=2, unknown=3
    SITE_TYPE_VOCAB = {"N-linked": 0, "O-linked": 1, "C-linked": 2}
    SITE_TYPE_UNKNOWN = 3
    NUM_SITE_TYPES = 4

    VALID_METHODS = ("learnable", "esm2", "esm2_site_aware")

    def __init__(
        self,
        num_proteins: int,
        output_dim: int = 256,
        method: str = "learnable",
        esm2_dim: int = 1280,
        cache_path: Optional[Union[str, Path]] = None,
        dropout: float = 0.1,
        site_positions_map: Optional[Dict[int, List]] = None,
        site_window: int = 15,
    ) -> None:
        super().__init__()
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method {method!r}; choose from {self.VALID_METHODS}"
            )

        self.num_proteins = num_proteins
        self.output_dim = output_dim
        self.method = method
        self.esm2_dim = esm2_dim
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.site_window = site_window
        self.site_positions_map = site_positions_map or {}

        # Learnable embedding (used by "learnable" and as fallback for "esm2"/"esm2_site_aware")
        self.embedding = nn.Embedding(num_proteins, output_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # ESM-2 projection MLP: 1280 -> 512 -> 256 (matches architecture doc)
        if method in ("esm2", "esm2_site_aware"):
            mid_dim = 512
            self.projection = nn.Sequential(
                nn.Linear(esm2_dim, mid_dim),      # 1280 -> 512
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(mid_dim),
                nn.Linear(mid_dim, output_dim),    # 512 -> 256
            )

        # Site-aware components
        if method == "esm2_site_aware":
            pe_dim = 64  # positional encoding dimensionality
            site_type_emb_dim = 32

            # Site type embedding: N-linked=0, O-linked=1, C-linked=2, unknown=3
            self.site_type_emb = nn.Embedding(self.NUM_SITE_TYPES, site_type_emb_dim)

            # MLP_site: [window_mean || global_mean || positional_encoding || site_type_emb] -> output_dim
            site_input_dim = esm2_dim + esm2_dim + pe_dim + site_type_emb_dim
            self.mlp_site = nn.Sequential(
                nn.Linear(site_input_dim, mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(mid_dim),
                nn.Linear(mid_dim, output_dim),
            )

            # Site count encoding: scalar -> small embedding
            self._site_count_dim = 32
            self.site_count_mlp = nn.Sequential(
                nn.Linear(1, self._site_count_dim),
                nn.GELU(),
            )

            # MLP_merge: [sequence_embedding || AGG(site_contexts) || site_count_encoding] -> output_dim
            merge_input_dim = output_dim + output_dim + self._site_count_dim
            self.mlp_merge = nn.Sequential(
                nn.Linear(merge_input_dim, mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(mid_dim),
                nn.Linear(mid_dim, output_dim),
            )

            self._pe_dim = pe_dim

        # Runtime cache: index -> tensor on correct device
        self._esm2_cache: Dict[int, torch.Tensor] = {}
        # Track which indices have no cached ESM-2 file (avoid repeated I/O)
        self._missing_indices: Set[int] = set()

    # ------------------------------------------------------------------
    # ESM-2 cache helpers
    # ------------------------------------------------------------------

    def _load_esm2_embedding(
        self, idx: int, per_residue: bool = False
    ) -> Optional[torch.Tensor]:
        """Try to load an ESM-2 embedding from the cache directory.

        Parameters
        ----------
        idx : int
            Protein index.
        per_residue : bool
            If True, return per-residue embeddings ``[L, esm2_dim]``
            without mean-pooling.

        Returns
        -------
        torch.Tensor or None
            Returns ``None`` if the file does not exist or cannot be loaded.
        """
        if self.cache_path is None:
            return None
        if idx in self._missing_indices:
            return None

        filepath = self.cache_path / f"{idx}.pt"
        if not filepath.exists():
            self._missing_indices.add(idx)
            return None

        try:
            tensor = torch.load(filepath, map_location="cpu", weights_only=True)
            if tensor.dtype == torch.float16:
                tensor = tensor.float()
            if per_residue:
                if tensor.dim() == 1:
                    # Only sequence-level available; expand to [1, D]
                    if tensor.shape[0] != self.esm2_dim:
                        self._missing_indices.add(idx)
                        return None
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() == 2:
                    if tensor.shape[1] != self.esm2_dim:
                        self._missing_indices.add(idx)
                        return None
                else:
                    self._missing_indices.add(idx)
                    return None
                return tensor
            else:
                if tensor.dim() == 2:
                    tensor = tensor.mean(dim=0)  # mean-pool per-residue -> sequence-level
                if tensor.shape[0] != self.esm2_dim:
                    logger.warning(
                        "ESM-2 cache file %s has unexpected shape %s; expected dim %d. "
                        "Falling back to learnable embedding.",
                        filepath,
                        tensor.shape,
                        self.esm2_dim,
                    )
                    self._missing_indices.add(idx)
                    return None
                return tensor
        except Exception:
            logger.debug("Failed to load ESM-2 cache for index %d", idx, exc_info=True)
            self._missing_indices.add(idx)
            return None

    def _get_esm2_embeddings(
        self, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ESM-2 embeddings and a boolean mask of available entries.

        Returns
        -------
        esm2_embs : torch.Tensor
            Shape ``[B, esm2_dim]``.  Pre-computed embeddings (zeros where missing).
        has_esm2 : torch.Tensor
            Shape ``[B]``.  Boolean mask -- ``True`` where a cached embedding was found.
        """
        device = indices.device
        batch_size = indices.numel()
        esm2_embs = torch.zeros(batch_size, self.esm2_dim)
        has_esm2 = torch.zeros(batch_size, dtype=torch.bool)

        for i, idx_t in enumerate(indices.view(-1)):
            idx = idx_t.item()
            if idx in self._esm2_cache:
                emb = self._esm2_cache[idx]
                # Cache may contain per-residue [L, D]; mean-pool for standard mode
                if emb.dim() == 2:
                    esm2_embs[i] = emb.mean(dim=0)
                else:
                    esm2_embs[i] = emb
                has_esm2[i] = True
            else:
                emb = self._load_esm2_embedding(idx)
                if emb is not None:
                    self._esm2_cache[idx] = emb
                    esm2_embs[i] = emb
                    has_esm2[i] = True

        return esm2_embs.to(device), has_esm2.to(device)

    def _get_per_residue_embedding(self, idx: int) -> Optional[torch.Tensor]:
        """Return per-residue ESM-2 embedding ``[L, esm2_dim]`` for a protein.

        Uses the cache if the stored tensor is 2-D; otherwise attempts
        to reload from disk with ``per_residue=True``.
        """
        if idx in self._esm2_cache:
            cached = self._esm2_cache[idx]
            if cached.dim() == 2:
                return cached
        # Not in cache or cached as 1-D: try loading as per-residue
        emb = self._load_esm2_embedding(idx, per_residue=True)
        if emb is not None:
            self._esm2_cache[idx] = emb  # overwrite with richer 2-D version
        return emb

    def clear_cache(self) -> None:
        """Clear the internal ESM-2 embedding cache."""
        self._esm2_cache.clear()
        self._missing_indices.clear()

    # ------------------------------------------------------------------
    # Site-aware helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_site_entry(entry) -> Tuple[int, int]:
        """Parse a site_positions_map entry into (position, site_type_idx).

        Accepts either plain ``int`` (legacy format, returns unknown type)
        or ``(position, residue, site_type)`` tuple (enriched format).
        """
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            pos, _residue, stype = entry[0], entry[1], entry[2]
            type_idx = ProteinEncoder.SITE_TYPE_VOCAB.get(
                str(stype), ProteinEncoder.SITE_TYPE_UNKNOWN
            )
            return int(pos), type_idx
        return int(entry), ProteinEncoder.SITE_TYPE_UNKNOWN

    def _compute_site_context(
        self,
        per_residue: torch.Tensor,
        site_position: int,
        device: torch.device,
        site_type_idx: int = 3,
    ) -> torch.Tensor:
        """Compute site-context vector for a single glycosylation site.

        Parameters
        ----------
        per_residue : torch.Tensor
            Shape ``[L, esm2_dim]``.
        site_position : int
            0-indexed residue position of the glycosylation site.
        device : torch.device
            Target device.
        site_type_idx : int
            Site type index (0=N-linked, 1=O-linked, 2=C-linked, 3=unknown).

        Returns
        -------
        torch.Tensor
            Shape ``[output_dim]``.
        """
        seq_len = per_residue.size(0)
        w = self.site_window

        # Clamp window to sequence boundaries
        win_start = max(0, site_position - w)
        win_end = min(seq_len, site_position + w + 1)

        window_embs = per_residue[win_start:win_end]  # [W', esm2_dim]
        window_mean = window_embs.mean(dim=0)  # [esm2_dim]
        global_mean = per_residue.mean(dim=0)  # [esm2_dim]
        pe = _positional_encoding(site_position, self._pe_dim).to(device)  # [pe_dim]

        # Site type embedding
        st_idx = torch.tensor([site_type_idx], dtype=torch.long, device=device)
        st_emb = self.site_type_emb(st_idx).squeeze(0)  # [site_type_emb_dim]

        site_input = torch.cat([window_mean, global_mean, pe, st_emb], dim=0)
        return self.mlp_site(site_input.unsqueeze(0).to(device)).squeeze(0)  # [output_dim]

    def _forward_site_aware(self, indices: torch.Tensor) -> torch.Tensor:
        """Site-aware forward pass with batched processing.

        Proteins are classified into three groups for efficient batched
        computation:

        1. **Site-aware** -- per-residue ESM-2 + known glycosylation sites
           (processed individually due to variable site counts).
        2. **ESM-2 only** -- pre-computed ESM-2 without site information
           (batched through the projection MLP).
        3. **Learnable** -- no ESM-2 available (batched embedding lookup).
        """
        device = indices.device
        flat = indices.view(-1)
        batch_size = flat.numel()
        output = torch.zeros(batch_size, self.output_dim, device=device)

        # Classify proteins into groups
        site_aware_indices: List[Tuple[int, int, torch.Tensor, List[int]]] = []
        esm2_batch_positions: List[int] = []
        esm2_batch_embs: List[torch.Tensor] = []
        learnable_positions: List[int] = []
        learnable_idx: List[torch.Tensor] = []

        for i, idx_t in enumerate(flat):
            idx = idx_t.item()
            raw_sites = self.site_positions_map.get(idx, [])

            if raw_sites:
                per_res = self._get_per_residue_embedding(idx)
                if per_res is not None:
                    # Parse site entries (supports both int and tuple formats)
                    parsed = [self._parse_site_entry(s) for s in raw_sites]
                    site_aware_indices.append((i, idx, per_res, parsed))
                    continue

            # No sites or no per-residue -> try ESM-2 sequence-level
            emb_1d = self._load_esm2_embedding(idx, per_residue=False)
            if emb_1d is not None:
                esm2_batch_positions.append(i)
                esm2_batch_embs.append(emb_1d)
            else:
                learnable_positions.append(i)
                learnable_idx.append(idx_t)

        # Group 1: Site-aware (sequential due to variable site counts)
        for i, idx, per_res, parsed_sites in site_aware_indices:
            global_mean = per_res.mean(dim=0).to(device)
            seq_emb = self.projection(global_mean.unsqueeze(0)).squeeze(0)

            site_contexts = [
                self._compute_site_context(per_res, p, device, site_type_idx=st)
                for p, st in parsed_sites
                if 0 <= p < per_res.size(0)
            ]

            if site_contexts:
                agg_sites = torch.stack(site_contexts, dim=0).mean(dim=0)
                count_tensor = torch.tensor(
                    [float(len(site_contexts))], device=device
                )
                count_enc = self.site_count_mlp(count_tensor)
                merge_input = torch.cat([seq_emb, agg_sites, count_enc], dim=0)
                output[i] = self.mlp_merge(merge_input.unsqueeze(0)).squeeze(0)
            else:
                output[i] = seq_emb

        # Group 2: ESM-2 only (batched projection)
        if esm2_batch_embs:
            esm2_stack = torch.stack(esm2_batch_embs, dim=0).to(device)
            projected = self.projection(esm2_stack)
            pos_tensor = torch.tensor(esm2_batch_positions, dtype=torch.long)
            output[pos_tensor] = projected

        # Group 3: Learnable only (batched embedding lookup)
        if learnable_idx:
            learn_indices = torch.stack(learnable_idx).to(device)
            learn_embs = self.embedding(learn_indices)
            pos_tensor = torch.tensor(learnable_positions, dtype=torch.long)
            output[pos_tensor] = learn_embs

        return output.view(*indices.shape, self.output_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute protein embeddings.

        Parameters
        ----------
        indices : torch.Tensor
            Long tensor of protein indices, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, output_dim]``.
        """
        if self.method == "learnable":
            return self.embedding(indices)

        if self.method == "esm2_site_aware":
            return self._forward_site_aware(indices)

        # ESM-2 mode: use pre-computed embeddings where available,
        # fall back to learnable for missing entries.
        flat = indices.view(-1)
        esm2_embs, has_esm2 = self._get_esm2_embeddings(flat)

        # Project ESM-2 embeddings
        projected = self.projection(esm2_embs)  # [B, output_dim]

        # Learnable fallback for missing ESM-2 entries
        learnable_embs = self.embedding(flat)  # [B, output_dim]

        # Combine: use projected ESM-2 where available, learnable otherwise
        mask = has_esm2.unsqueeze(-1).float()  # [B, 1]
        output = mask * projected + (1.0 - mask) * learnable_embs

        return output.view(*indices.shape, self.output_dim)
