"""GlycoKGNet: Unified multi-modal heterogeneous KG neural network.

Four-stage architecture:
  Stage 1: Modality-specific encoders (GlycanEncoder, ProteinEncoder, TextEncoder)
  Stage 2: BioHGT layers (heterogeneous graph transformer with bio-priors)
  Stage 3: Cross-Modal Fusion (gated cross-attention)
  Stage 4: Task-specific decoders (link scoring via forward-compatible interface)

Compatible with the existing Trainer interface (BaseKGEModel):
  - forward(data) -> Dict[str, Tensor]
  - score(head, relation, tail) -> Tensor
  - score_triples(data, head_type, head_idx, relation_idx, tail_type, tail_idx) -> Tensor

Phase 2 components (GlycanTreeEncoder, BioHGTLayer, HybridLinkScorer) are
imported optionally and fall back to Phase 1 equivalents when unavailable.

Reference: Section 6.3 of docs/architecture/model_architecture_design.md
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.encoders import GlycanEncoder, ProteinEncoder, TextEncoder
from glycoMusubi.embedding.models.base import BaseKGEModel
from glycoMusubi.embedding.models.cross_modal_fusion import CrossModalFusion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Phase 2 imports -- fall back gracefully when not yet available
# ---------------------------------------------------------------------------

try:
    from glycoMusubi.embedding.encoders.glycan_tree_encoder import GlycanTreeEncoder

    _HAS_TREE_ENCODER = True
except ImportError:
    _HAS_TREE_ENCODER = False
    GlycanTreeEncoder = None  # type: ignore[assignment,misc]

try:
    from glycoMusubi.embedding.models.biohgt import BioHGTLayer

    _HAS_BIOHGT = True
except ImportError:
    _HAS_BIOHGT = False
    BioHGTLayer = None  # type: ignore[assignment,misc]

try:
    from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer

    _HAS_HYBRID_SCORER = True
except ImportError:
    _HAS_HYBRID_SCORER = False
    HybridLinkScorer = None  # type: ignore[assignment,misc]

try:
    from glycoMusubi.embedding.decoders.node_classifier import NodeClassifier

    _HAS_NODE_CLASSIFIER = True
except ImportError:
    _HAS_NODE_CLASSIFIER = False
    NodeClassifier = None  # type: ignore[assignment,misc]

try:
    from glycoMusubi.embedding.decoders.graph_level_decoder import GraphLevelDecoder

    _HAS_GRAPH_DECODER = True
except ImportError:
    _HAS_GRAPH_DECODER = False
    GraphLevelDecoder = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# GlycoKGNet
# ---------------------------------------------------------------------------


class GlycoKGNet(BaseKGEModel):
    """Unified glycoMusubi Neural Network.

    Parameters
    ----------
    num_nodes_dict : Dict[str, int]
        Mapping from node type name to node count.
    num_relations : int
        Total number of distinct relation types.
    embedding_dim : int
        Shared embedding dimensionality (default 256).
    glycan_encoder_type : str
        Glycan encoding strategy: ``"learnable"``, ``"wurcs_features"``,
        ``"hybrid"``, or ``"tree_mpnn"`` (requires GlycanTreeEncoder).
    protein_encoder_type : str
        Protein encoding strategy: ``"learnable"`` or ``"esm2"``.
    num_hgt_layers : int
        Number of BioHGT layers (used only when BioHGTLayer is available).
    num_hgt_heads : int
        Number of attention heads per BioHGT layer.
    use_bio_prior : bool
        Whether to use biosynthetic pathway priors in BioHGT.
    use_cross_modal_fusion : bool
        Whether to apply cross-modal gated attention fusion (Stage 3).
    num_fusion_heads : int
        Number of attention heads in the cross-modal fusion module.
    decoder_type : str
        Decoder for link scoring: ``"hybrid"`` (requires HybridLinkScorer),
        ``"distmult"`` (built-in fallback).
    dropout : float
        Dropout probability used across sub-modules.
    esm2_cache_path : str or None
        Path to pre-computed ESM-2 embeddings for ``protein_encoder_type="esm2"``.
    wurcs_map : Dict[int, str] or None
        Mapping from glycan index to WURCS string.
    text_node_types : list[str] or None
        Node types that use the text encoder (default: disease, pathway).
    edge_types : list[tuple[str, str, str]] or None
        Canonical ``(src_type, relation, dst_type)`` triples for BioHGT.
        If ``None``, uses BioHGTLayer defaults.
    node_classifier : NodeClassifier or None
        Optional pre-constructed node classification decoder.
    graph_decoder : GraphLevelDecoder or None
        Optional pre-constructed graph-level prediction decoder.
    """

    # Node types expected to have glycan structures
    _GLYCAN_NODE_TYPE = "glycan"
    # Node types expected to have protein sequences
    _PROTEIN_NODE_TYPE = "protein"
    # Default text node types
    _DEFAULT_TEXT_NODE_TYPES = ("disease", "pathway")

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 256,
        # Encoder config
        glycan_encoder_type: str = "learnable",
        protein_encoder_type: str = "learnable",
        # BioHGT config
        num_hgt_layers: int = 4,
        num_hgt_heads: int = 8,
        use_bio_prior: bool = True,
        hgt_max_edges_per_type: int = 0,
        # Fusion config
        use_cross_modal_fusion: bool = True,
        num_fusion_heads: int = 4,
        # Decoder config
        decoder_type: str = "hybrid",
        # General
        dropout: float = 0.1,
        # Data paths / mappings
        esm2_cache_path: Optional[str] = None,
        wurcs_map: Optional[Dict[int, str]] = None,
        text_node_types: Optional[List[str]] = None,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        # Site-aware protein encoder
        site_positions_map: Optional[Dict[int, List]] = None,
        # Glycan function features
        function_feature_map: Optional[Dict[int, torch.Tensor]] = None,
        # Phase 4: optional downstream decoders
        node_classifier: Optional[nn.Module] = None,
        graph_decoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(num_nodes_dict, num_relations, embedding_dim)

        self._glycan_encoder_type = glycan_encoder_type
        self._protein_encoder_type = protein_encoder_type
        self._use_cross_modal_fusion = use_cross_modal_fusion
        self._use_bio_prior = use_bio_prior
        self._hgt_max_edges_per_type = hgt_max_edges_per_type
        self._text_node_types = tuple(text_node_types or self._DEFAULT_TEXT_NODE_TYPES)

        # Track which node types have modality-specific encoders
        self._modality_node_types: Set[str] = set()

        # ----------------------------------------------------------------
        # Stage 1: Modality-specific encoders
        # ----------------------------------------------------------------
        self._init_glycan_encoder(
            glycan_encoder_type, embedding_dim, wurcs_map, dropout,
            function_feature_map=function_feature_map,
        )
        self._init_protein_encoder(
            protein_encoder_type, embedding_dim, esm2_cache_path, dropout,
            site_positions_map=site_positions_map,
        )
        self._init_text_encoders(embedding_dim)

        # ----------------------------------------------------------------
        # Stage 2: BioHGT layers (used as components, not as a full model)
        # ----------------------------------------------------------------
        self._use_biohgt = False
        if _HAS_BIOHGT and num_hgt_layers > 0:
            node_type_names = sorted(num_nodes_dict.keys())
            self.hgt_layers = nn.ModuleList([
                BioHGTLayer(
                    in_dim=embedding_dim,
                    out_dim=embedding_dim,
                    num_heads=num_hgt_heads,
                    node_types=node_type_names,
                    edge_types=edge_types,
                    use_bio_prior=use_bio_prior,
                    dropout=dropout,
                )
                for _ in range(num_hgt_layers)
            ])
            self._use_biohgt = True
            logger.info(
                "Stage 2: BioHGT enabled (%d layers, %d heads)",
                num_hgt_layers, num_hgt_heads,
            )
        else:
            if not _HAS_BIOHGT and num_hgt_layers > 0:
                logger.warning(
                    "BioHGTLayer not available (glycoMusubi.embedding.models.biohgt "
                    "not found). Falling back to Phase 1 embedding-only mode."
                )

        # ----------------------------------------------------------------
        # Stage 3: Cross-modal fusion
        # ----------------------------------------------------------------
        self._fusion: Optional[CrossModalFusion] = None
        if use_cross_modal_fusion and self._modality_node_types:
            self._fusion = CrossModalFusion(
                embed_dim=embedding_dim,
                num_heads=num_fusion_heads,
                dropout=dropout,
            )
            logger.info(
                "Stage 3: Cross-modal fusion enabled for node types: %s",
                sorted(self._modality_node_types),
            )

        # ----------------------------------------------------------------
        # Stage 4: Decoder
        # ----------------------------------------------------------------
        self._init_decoder(decoder_type, embedding_dim, num_relations)

        # ----------------------------------------------------------------
        # Phase 4: Optional downstream decoders
        # ----------------------------------------------------------------
        self.node_classifier = node_classifier
        self.graph_decoder = graph_decoder

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_glycan_encoder(
        self,
        encoder_type: str,
        embed_dim: int,
        wurcs_map: Optional[Dict[int, str]],
        dropout: float,
        function_feature_map: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        """Initialise the glycan encoder (Phase 1 or Phase 2)."""
        num_glycans = self.num_nodes_dict.get(self._GLYCAN_NODE_TYPE, 0)
        if num_glycans == 0:
            return

        if encoder_type == "tree_mpnn":
            if _HAS_TREE_ENCODER:
                self.glycan_tree_encoder = GlycanTreeEncoder(output_dim=embed_dim, hidden_dim=embed_dim)
                self._modality_node_types.add(self._GLYCAN_NODE_TYPE)
                # Function feature gated fusion for tree_mpnn mode
                if function_feature_map:
                    self._tree_function_feature_map = function_feature_map
                    func_dim = len(GlycanEncoder.FUNCTION_CATEGORIES)
                    self._tree_function_proj = nn.Sequential(
                        nn.Linear(func_dim, embed_dim // 4),
                        nn.GELU(),
                        nn.Linear(embed_dim // 4, embed_dim),
                    )
                    self._tree_function_gate = nn.Sequential(
                        nn.Linear(embed_dim * 2, embed_dim),
                        nn.Sigmoid(),
                    )
                    logger.info(
                        "Glycan encoder: GlycanTreeEncoder (tree_mpnn) + function gate (%d glycans)",
                        len(function_feature_map),
                    )
                else:
                    logger.info("Glycan encoder: GlycanTreeEncoder (tree_mpnn)")
            else:
                logger.warning(
                    "GlycanTreeEncoder not available; falling back to 'hybrid' glycan encoder."
                )
                encoder_type = "hybrid"

        if encoder_type != "tree_mpnn":
            # Use Phase 1 GlycanEncoder
            self.glycan_encoder = GlycanEncoder(
                num_glycans=num_glycans,
                output_dim=embed_dim,
                method=encoder_type,
                wurcs_map=wurcs_map,
                function_feature_map=function_feature_map,
            )
            if encoder_type in ("wurcs_features", "hybrid"):
                self._modality_node_types.add(self._GLYCAN_NODE_TYPE)
            logger.info("Glycan encoder: GlycanEncoder (method=%s)", encoder_type)

    def _init_protein_encoder(
        self,
        encoder_type: str,
        embed_dim: int,
        esm2_cache_path: Optional[str],
        dropout: float,
        site_positions_map: Optional[Dict[int, List[int]]] = None,
    ) -> None:
        """Initialise the protein encoder."""
        num_proteins = self.num_nodes_dict.get(self._PROTEIN_NODE_TYPE, 0)
        if num_proteins == 0:
            return

        self.protein_encoder = ProteinEncoder(
            num_proteins=num_proteins,
            output_dim=embed_dim,
            method=encoder_type,
            cache_path=esm2_cache_path,
            dropout=dropout,
            site_positions_map=site_positions_map,
        )
        if encoder_type in ("esm2", "esm2_site_aware"):
            self._modality_node_types.add(self._PROTEIN_NODE_TYPE)
        logger.info("Protein encoder: ProteinEncoder (method=%s)", encoder_type)

    def _init_text_encoders(self, embed_dim: int) -> None:
        """Initialise text encoders for applicable node types."""
        self.text_encoders = nn.ModuleDict()
        for ntype in self._text_node_types:
            num_nodes = self.num_nodes_dict.get(ntype, 0)
            if num_nodes > 0:
                self.text_encoders[ntype] = TextEncoder(
                    num_entities=num_nodes,
                    output_dim=embed_dim,
                )
                logger.info(
                    "Text encoder for '%s': TextEncoder (%d entities)",
                    ntype, num_nodes,
                )

    def _init_decoder(
        self,
        decoder_type: str,
        embed_dim: int,
        num_relations: int,
    ) -> None:
        """Initialise the link-scoring decoder."""
        self._decoder_has_own_relations = False

        if decoder_type == "hybrid" and _HAS_HYBRID_SCORER:
            self.decoder = HybridLinkScorer(
                embedding_dim=embed_dim,
                num_relations=num_relations,
            )
            self._decoder_type = "hybrid"
            self._decoder_has_own_relations = True
            logger.info("Decoder: HybridLinkScorer")
        else:
            if decoder_type == "hybrid" and not _HAS_HYBRID_SCORER:
                logger.warning(
                    "HybridLinkScorer not available; falling back to DistMult scorer."
                )
            self._decoder_type = "distmult_fallback"
            logger.info("Decoder: DistMult fallback (using base relation embeddings)")

    def _apply_tree_function_gate(
        self, base_emb: torch.Tensor, num_glycans: int
    ) -> torch.Tensor:
        """Apply gated fusion with function features for tree_mpnn mode."""
        func_dim = len(GlycanEncoder.FUNCTION_CATEGORIES)
        device = base_emb.device
        func_feats = torch.zeros(num_glycans, func_dim, device=device)
        for idx, feat in self._tree_function_feature_map.items():
            if idx < num_glycans:
                func_feats[idx] = feat.to(device)
        func_emb = self._tree_function_proj(func_feats)
        gate_input = torch.cat([base_emb, func_emb], dim=-1)
        gate = self._tree_function_gate(gate_input)
        return gate * base_emb + (1 - gate) * func_emb

    # ------------------------------------------------------------------
    # Stage 1: Compute initial node embeddings from encoders
    # ------------------------------------------------------------------

    def _compute_initial_embeddings(
        self, data: HeteroData
    ) -> Dict[str, torch.Tensor]:
        """Produce initial per-type node embeddings from modality encoders.

        For node types without a dedicated encoder, the base-class learnable
        ``nn.Embedding`` tables (from ``BaseKGEModel``) are used.
        """
        emb_dict: Dict[str, torch.Tensor] = {}
        device = next(self.parameters()).device

        for node_type, num_nodes in self.num_nodes_dict.items():
            idx = torch.arange(num_nodes, device=device)

            if node_type == self._GLYCAN_NODE_TYPE:
                if (
                    self._glycan_encoder_type == "tree_mpnn"
                    and hasattr(self, "glycan_tree_encoder")
                ):
                    # GlycanTreeEncoder.forward() expects List[GlycanTree]
                    trees = getattr(data[node_type], "trees", None)
                    if trees is not None:
                        glycan_emb = self.glycan_tree_encoder(trees)
                        # Apply function feature gate if available
                        if hasattr(self, "_tree_function_proj"):
                            glycan_emb = self._apply_tree_function_gate(glycan_emb, num_nodes)
                        emb_dict[node_type] = glycan_emb
                        continue
                    # Fall through to Phase 1 learnable if no tree data
                if hasattr(self, "glycan_encoder"):
                    emb_dict[node_type] = self.glycan_encoder(idx)
                    continue

            if node_type == self._PROTEIN_NODE_TYPE and hasattr(self, "protein_encoder"):
                emb_dict[node_type] = self.protein_encoder(idx)
                continue

            if node_type in self.text_encoders:
                emb_dict[node_type] = self.text_encoders[node_type](idx)
                continue

            # Fallback: use base-class learnable embedding
            if node_type in self.node_embeddings:
                emb_dict[node_type] = self.node_embeddings[node_type](idx)

        return emb_dict

    # ------------------------------------------------------------------
    # Stage 2: BioHGT message passing
    # ------------------------------------------------------------------

    def _run_biohgt(
        self,
        emb_dict: Dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """Run BioHGT message passing over the heterogeneous graph."""
        if not self._use_biohgt:
            return emb_dict

        # Build edge_index_dict from the HeteroData
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for edge_type in data.edge_types:
            edge_store = data[edge_type]
            if hasattr(edge_store, "edge_index"):
                edge_index_dict[edge_type] = edge_store.edge_index

        # Subsample large edge types to bound GPU memory
        max_e = self._hgt_max_edges_per_type
        if max_e > 0:
            for etype, ei in edge_index_dict.items():
                if ei.size(1) > max_e:
                    perm = torch.randperm(ei.size(1), device=ei.device)[:max_e]
                    edge_index_dict[etype] = ei[:, perm]

        # Run through stacked BioHGT layers
        x_dict = emb_dict
        for layer in self.hgt_layers:
            x_dict = layer(x_dict, edge_index_dict)

        return x_dict

    # ------------------------------------------------------------------
    # Stage 3: Cross-modal fusion
    # ------------------------------------------------------------------

    def _run_fusion(
        self,
        emb_dict: Dict[str, torch.Tensor],
        initial_emb_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply cross-modal gated attention fusion."""
        if self._fusion is None:
            return emb_dict

        fused_dict = dict(emb_dict)
        for ntype in self._modality_node_types:
            if ntype in emb_dict and ntype in initial_emb_dict:
                fused_dict[ntype] = self._fusion(
                    h_kg=emb_dict[ntype],
                    h_modality=initial_emb_dict[ntype],
                )
        return fused_dict

    # ------------------------------------------------------------------
    # Public encode interface
    # ------------------------------------------------------------------

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Run all encoder stages and return per-type node embeddings.

        Stage 1: Modality-specific encoders
        Stage 2: BioHGT message passing
        Stage 3: Cross-modal fusion

        Parameters
        ----------
        data : HeteroData
            PyG heterogeneous graph.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``{node_type: Tensor[num_nodes, embedding_dim]}``
        """
        # Stage 1: initial embeddings
        initial_emb_dict = self._compute_initial_embeddings(data)

        # Stage 2: BioHGT message passing
        emb_dict = self._run_biohgt(initial_emb_dict, data)

        # Stage 3: cross-modal fusion
        emb_dict = self._run_fusion(emb_dict, initial_emb_dict)

        return emb_dict

    # ------------------------------------------------------------------
    # BaseKGEModel interface
    # ------------------------------------------------------------------

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Compute node embeddings. Alias for :meth:`encode`.

        This satisfies the ``BaseKGEModel.forward`` contract and is
        called by the Trainer during training.
        """
        return self.encode(data)

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Score (head, relation, tail) triples.

        When using HybridLinkScorer, ``relation`` should be integer
        relation indices ``[batch]``. When using the DistMult fallback,
        ``relation`` should be relation embedding vectors ``[batch, dim]``.

        The Trainer calls this method via ``score_triples()`` which
        passes relation embeddings from ``get_relation_embedding()``.
        We handle both cases transparently.

        Parameters
        ----------
        head : torch.Tensor
            Head entity embeddings ``[batch, dim]``.
        relation : torch.Tensor
            Relation embeddings ``[batch, dim]`` or indices ``[batch]``.
        tail : torch.Tensor
            Tail entity embeddings ``[batch, dim]``.

        Returns
        -------
        torch.Tensor
            Scalar scores ``[batch]``.
        """
        if self._decoder_type == "hybrid":
            # HybridLinkScorer.forward expects (head, relation_idx, tail)
            # where relation_idx is integer indices.
            if relation.dim() == 1 and relation.dtype in (torch.long, torch.int):
                return self.decoder(head, relation, tail)
            # If relation is an embedding vector, we cannot directly use
            # HybridLinkScorer (which needs indices for its own relation
            # embeddings). Fall back to DistMult scoring.
            return (head * relation * tail).sum(dim=-1)
        else:
            # DistMult fallback: <h, r, t>
            return (head * relation * tail).sum(dim=-1)

    def score_triples(
        self,
        data: HeteroData,
        head_type: str,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_type: str,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end scoring: embed nodes then score triples.

        Overrides the base class method to leverage HybridLinkScorer
        when available (passing relation indices directly instead of
        embedding vectors).

        Parameters
        ----------
        data : HeteroData
            Input heterogeneous graph.
        head_type : str
            Node type of head entities.
        head_idx : torch.Tensor
            Indices of head entities within their type ``[batch]``.
        relation_idx : torch.Tensor
            Relation type indices ``[batch]``.
        tail_type : str
            Node type of tail entities.
        tail_idx : torch.Tensor
            Indices of tail entities within their type ``[batch]``.

        Returns
        -------
        torch.Tensor
            Triple scores ``[batch]``.
        """
        emb_dict = self.forward(data)
        head_emb = emb_dict[head_type][head_idx]
        tail_emb = emb_dict[tail_type][tail_idx]

        if self._decoder_type == "hybrid":
            # Pass relation indices directly to HybridLinkScorer
            return self.decoder(head_emb, relation_idx, tail_emb)
        else:
            # DistMult fallback: look up relation embeddings then score
            rel_emb = self.relation_embeddings(relation_idx)
            return self.score(head_emb, rel_emb, tail_emb)

    # ------------------------------------------------------------------
    # Phase 4: Downstream task methods
    # ------------------------------------------------------------------

    def node_classify(
        self,
        data: HeteroData,
        task: str,
        node_type: str,
    ) -> torch.Tensor:
        """Classify nodes of a given type for a specific task.

        Runs the full encoder pipeline (Stages 1-3) then applies the
        node classification head.

        Parameters
        ----------
        data : HeteroData
            Input heterogeneous graph.
        task : str
            Classification task name (must match a key in the
            ``NodeClassifier.task_configs``).
        node_type : str
            Which node type to classify.

        Returns
        -------
        torch.Tensor
            Logits ``[num_nodes, num_classes]``.

        Raises
        ------
        RuntimeError
            If no ``node_classifier`` was provided at construction time.
        """
        if self.node_classifier is None:
            raise RuntimeError(
                "node_classify() requires a NodeClassifier. "
                "Pass node_classifier= when constructing GlycoKGNet."
            )
        emb_dict = self.encode(data)
        return self.node_classifier(emb_dict[node_type], task)

    def predict_graph(
        self,
        data: HeteroData,
        subgraph_nodes: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Make graph-level predictions using attentive readout.

        Runs the full encoder pipeline (Stages 1-3) then aggregates
        node embeddings with the graph-level decoder.

        Parameters
        ----------
        data : HeteroData
            Input heterogeneous graph.
        subgraph_nodes : dict[str, torch.Tensor], optional
            If provided, restricts which nodes are pooled.  Keys are
            node types and values are index tensors.  If ``None``, all
            node embeddings (concatenated across types) are used.

        Returns
        -------
        torch.Tensor
            ``(batch_size, num_classes)`` predictions.

        Raises
        ------
        RuntimeError
            If no ``graph_decoder`` was provided at construction time.
        """
        if self.graph_decoder is None:
            raise RuntimeError(
                "predict_graph() requires a GraphLevelDecoder. "
                "Pass graph_decoder= when constructing GlycoKGNet."
            )
        emb_dict = self.encode(data)

        if subgraph_nodes is not None:
            parts = [
                emb_dict[ntype][idx]
                for ntype, idx in subgraph_nodes.items()
                if ntype in emb_dict
            ]
        else:
            parts = list(emb_dict.values())

        if not parts:
            raise ValueError("No node embeddings found to pool over.")

        node_embeddings = torch.cat(parts, dim=0)
        return self.graph_decoder(node_embeddings)

    # ------------------------------------------------------------------
    # Inductive: encode novel entities from features
    # ------------------------------------------------------------------

    def encode_novel_entity(
        self,
        node_type: str,
        features: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute embedding for an unseen entity using feature encoders.

        For inductive inference on held-out or entirely new entities.
        Uses the model's feature encoders (WURCS, ESM-2) rather than
        learnable embedding lookup.

        Parameters
        ----------
        node_type : str
            Entity type (e.g., ``"glycan"``, ``"protein"``).
        features : dict
            Feature data keyed by feature type:
            - For glycan: ``{"wurcs": "<WURCS string>"}``
            - For protein: ``{"esm2": Tensor[1280]}`` or
              ``{"esm2_per_residue": Tensor[L, 1280], "site_positions": [int, ...]}``

        Returns
        -------
        torch.Tensor
            Embedding vector ``[1, embedding_dim]``.
        """
        device = next(self.parameters()).device

        if node_type == self._GLYCAN_NODE_TYPE and hasattr(self, "glycan_encoder"):
            encoder = self.glycan_encoder
            if hasattr(encoder, "wurcs_proj") and "wurcs" in features:
                from glycoMusubi.embedding.encoders.glycan_encoder import extract_wurcs_features
                wurcs_feat = extract_wurcs_features(features["wurcs"]).to(device)
                wurcs_emb = encoder.wurcs_proj(wurcs_feat.unsqueeze(0))  # [1, dim]

                if hasattr(encoder, "fusion_mlp"):
                    # Hybrid mode: use zero learnable + WURCS features
                    learnable_emb = torch.zeros(
                        1, self.embedding_dim, device=device
                    )
                    fused = torch.cat([learnable_emb, wurcs_emb], dim=-1)
                    return encoder.fusion_mlp(fused)
                return wurcs_emb

        if node_type == self._PROTEIN_NODE_TYPE and hasattr(self, "protein_encoder"):
            encoder = self.protein_encoder
            if hasattr(encoder, "projection"):
                if "esm2" in features:
                    esm2_emb = features["esm2"].to(device)
                    if esm2_emb.dim() == 1:
                        esm2_emb = esm2_emb.unsqueeze(0)
                    return encoder.projection(esm2_emb)

                if "esm2_per_residue" in features:
                    per_res = features["esm2_per_residue"].to(device)
                    global_mean = per_res.mean(dim=0, keepdim=True)
                    return encoder.projection(global_mean)

        # Fallback: zero embedding (unknown entity type or missing features)
        logger.warning(
            "Cannot encode novel entity of type '%s' - missing encoder or features. "
            "Returning zero embedding.",
            node_type,
        )
        return torch.zeros(1, self.embedding_dim, device=device)

    # ------------------------------------------------------------------
    # Convenience / inspection
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def stage_info(self) -> Dict[str, Any]:
        """Summary of which stages are active."""
        return {
            "glycan_encoder": self._glycan_encoder_type,
            "protein_encoder": self._protein_encoder_type,
            "biohgt_enabled": self._use_biohgt,
            "biohgt_layers": len(self.hgt_layers) if self._use_biohgt else 0,
            "cross_modal_fusion_enabled": self._fusion is not None,
            "modality_node_types": sorted(self._modality_node_types),
            "decoder_type": self._decoder_type,
        }

    def __repr__(self) -> str:
        info = self.stage_info
        lines = [
            f"GlycoKGNet(",
            f"  embedding_dim={self.embedding_dim},",
            f"  node_types={sorted(self.num_nodes_dict.keys())},",
            f"  num_relations={self.num_relations},",
            f"  glycan_encoder={info['glycan_encoder']},",
            f"  protein_encoder={info['protein_encoder']},",
            f"  biohgt={info['biohgt_enabled']} ({info['biohgt_layers']} layers),",
            f"  cross_modal_fusion={info['cross_modal_fusion_enabled']},",
            f"  decoder={info['decoder_type']},",
            f"  trainable_params={self.num_parameters:,}",
            f")",
        ]
        return "\n".join(lines)
