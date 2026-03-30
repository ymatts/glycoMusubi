"""Type-constrained negative sampler for heterogeneous KG link prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMA_DIR = _PROJECT_ROOT / "schemas"


def _load_type_constraints(schema_dir: Path) -> Dict[str, Dict[str, list]]:
    """Build per-relation valid source/target type lists from schemas.

    Returns
    -------
    constraints : dict
        ``{relation: {"source_types": [...], "target_types": [...]}}``
    """
    constraints: Dict[str, Dict[str, list]] = {}

    # relation_config.yaml
    rc_path = schema_dir / "relation_config.yaml"
    if rc_path.exists():
        with open(rc_path, "r") as f:
            rc = yaml.safe_load(f) or {}
        for rel, spec in rc.get("relation_types", {}).items():
            src = spec.get("source_type")
            tgt = spec.get("target_type")
            constraints[rel] = {
                "source_types": src if isinstance(src, list) else [src],
                "target_types": tgt if isinstance(tgt, list) else [tgt],
            }

    # edge_schema.yaml (supplement for relations not in relation_config)
    es_path = schema_dir / "edge_schema.yaml"
    if es_path.exists():
        with open(es_path, "r") as f:
            es = yaml.safe_load(f) or {}
        for rel, spec in es.get("edge_types", {}).items():
            if rel in constraints:
                continue
            src = spec.get("source_type")
            tgt = spec.get("target_type")
            if src is not None and tgt is not None:
                constraints[rel] = {
                    "source_types": src if isinstance(src, list) else [src],
                    "target_types": tgt if isinstance(tgt, list) else [tgt],
                }

    return constraints


class TypeConstrainedNegativeSampler:
    """Generate negative triples respecting entity type constraints.

    For a positive triple ``(src_type, relation, dst_type)`` with indices
    ``(h, r, t)``, negative samples are produced by corrupting either the
    head or tail while constraining replacements to valid entity types as
    defined in ``relation_config.yaml`` / ``edge_schema.yaml``.

    For example, for ``has_glycan`` (protein -> glycan), corrupted tails
    are drawn exclusively from glycan nodes.

    Parameters
    ----------
    node_type_offsets : dict[str, tuple[int, int]]
        ``{node_type: (start_index, count)}`` — contiguous index ranges
        for each node type.  This is used for fast vectorised sampling.
    schema_dir : str or Path or None
        Path to schema directory.
    num_negatives : int
        Number of negative samples per positive triple.
    corrupt_head_prob : float
        Probability of corrupting the head entity (vs. the tail).
    """

    def __init__(
        self,
        node_type_offsets: Dict[str, Tuple[int, int]],
        schema_dir: Optional[Union[str, Path]] = None,
        num_negatives: int = 256,
        corrupt_head_prob: float = 0.5,
    ) -> None:
        self.num_negatives = num_negatives
        self.corrupt_head_prob = corrupt_head_prob

        sd = Path(schema_dir) if schema_dir else _DEFAULT_SCHEMA_DIR
        self._constraints = _load_type_constraints(sd)

        # Pre-compute flat index tensors per node type for O(1) sampling
        self._type_indices: Dict[str, torch.Tensor] = {}
        for ntype, (start, count) in node_type_offsets.items():
            self._type_indices[ntype] = torch.arange(start, start + count)

        # Pre-compute merged index tensors for multi-type constraints
        self._valid_head_cache: Dict[str, torch.Tensor] = {}
        self._valid_tail_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_valid_indices(self, types: List[str]) -> torch.Tensor:
        """Return concatenated index tensor for the given node types."""
        parts = [self._type_indices[t] for t in types if t in self._type_indices]
        if not parts:
            raise ValueError(
                f"No node indices found for types {types}. "
                f"Available types: {list(self._type_indices.keys())}"
            )
        return torch.cat(parts)

    def _valid_heads_for(self, relation: str) -> torch.Tensor:
        if relation not in self._valid_head_cache:
            c = self._constraints.get(relation)
            if c is None:
                # Unconstrained — use all nodes
                self._valid_head_cache[relation] = torch.cat(
                    list(self._type_indices.values())
                )
            else:
                self._valid_head_cache[relation] = self._get_valid_indices(
                    c["source_types"]
                )
        return self._valid_head_cache[relation]

    def _valid_tails_for(self, relation: str) -> torch.Tensor:
        if relation not in self._valid_tail_cache:
            c = self._constraints.get(relation)
            if c is None:
                self._valid_tail_cache[relation] = torch.cat(
                    list(self._type_indices.values())
                )
            else:
                self._valid_tail_cache[relation] = self._get_valid_indices(
                    c["target_types"]
                )
        return self._valid_tail_cache[relation]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        head: torch.Tensor,
        relation: List[str],
        tail: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate type-constrained negative samples for a batch of triples.

        Parameters
        ----------
        head : Tensor, shape ``(B,)``
            Head entity indices.
        relation : list[str], length ``B``
            Relation names.
        tail : Tensor, shape ``(B,)``
            Tail entity indices.
        generator : torch.Generator or None
            Optional RNG for reproducibility.

        Returns
        -------
        neg_head : Tensor, shape ``(B, num_negatives)``
            Corrupted head indices (or original head if tail was corrupted).
        neg_tail : Tensor, shape ``(B, num_negatives)``
            Corrupted tail indices (or original tail if head was corrupted).
        """
        B = head.size(0)
        K = self.num_negatives

        neg_head = head.unsqueeze(1).expand(B, K).clone()
        neg_tail = tail.unsqueeze(1).expand(B, K).clone()

        # Decide which samples corrupt head vs tail
        corrupt_mask = torch.rand(B, K, generator=generator) < self.corrupt_head_prob

        for i in range(B):
            rel = relation[i]
            n_corrupt_head = int(corrupt_mask[i].sum().item())
            n_corrupt_tail = K - n_corrupt_head

            if n_corrupt_head > 0:
                valid = self._valid_heads_for(rel)
                idx = torch.randint(
                    len(valid), (n_corrupt_head,), generator=generator
                )
                neg_head[i, corrupt_mask[i]] = valid[idx]

            if n_corrupt_tail > 0:
                valid = self._valid_tails_for(rel)
                idx = torch.randint(
                    len(valid), (n_corrupt_tail,), generator=generator
                )
                neg_tail[i, ~corrupt_mask[i]] = valid[idx]

        return neg_head, neg_tail

    def sample_flat(
        self,
        head: torch.Tensor,
        relation: List[str],
        tail: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate negative triples as a flat ``(N, 3)`` index tensor.

        Columns are ``[head_idx, relation_idx_placeholder, tail_idx]``.
        The relation column is set to ``-1`` (callers should map relations
        to integer indices externally).

        Parameters
        ----------
        head, relation, tail, generator
            Same as :meth:`sample`.

        Returns
        -------
        neg_triples : Tensor, shape ``(B * num_negatives, 3)``
        """
        neg_h, neg_t = self.sample(head, relation, tail, generator)
        B, K = neg_h.shape
        triples = torch.stack(
            [
                neg_h.reshape(-1),
                torch.full((B * K,), -1, dtype=torch.long),
                neg_t.reshape(-1),
            ],
            dim=1,
        )
        return triples


class FunctionPoolRestrictor:
    """Restrict negative sampling pool by glycan function for has_glycan.

    For ``has_glycan`` edges (and any other relations in *target_relations*),
    the negative tail pool is restricted to glycans that have at least one
    function label (N-linked, O-linked, etc.).  This removes the ~95% of
    glycans that never appear in ``has_glycan`` triples, eliminating trivially
    easy negatives.

    Parameters
    ----------
    glycan_function_indices : dict[str, list[int]]
        ``{function_term: [glycan_local_indices]}`` — glycan indices grouped
        by function label.
    target_relations : set[str]
        Relation names to restrict (default ``{"has_glycan"}``).
    """

    def __init__(
        self,
        glycan_function_indices: Dict[str, List[int]],
        target_relations: Optional[Set[str]] = None,
    ) -> None:
        all_indices: Set[int] = set()
        for indices in glycan_function_indices.values():
            all_indices.update(indices)
        self._all_func_glycans = torch.tensor(
            sorted(all_indices), dtype=torch.long
        )
        self._target_relations = target_relations or {"has_glycan"}
        logger.info(
            "FunctionPoolRestrictor: %d function-bearing glycans for relations %s",
            len(self._all_func_glycans),
            self._target_relations,
        )

    def __call__(
        self, edge_type: Tuple[str, str, str]
    ) -> Optional[torch.Tensor]:
        """Return restricted index tensor or ``None`` for unrestricted."""
        _, rel_name, _ = edge_type
        if rel_name in self._target_relations:
            return self._all_func_glycans
        return None
