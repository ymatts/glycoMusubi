"""Modality-specific encoders for glycoMusubi node embedding.

Provides encoder modules that map heterogeneous node types into a
shared 256-dimensional embedding space:

* :class:`GlycanEncoder` -- glycan structures (learnable / WURCS features / hybrid)
* :class:`GlycanTreeEncoder` -- glycan tree structures via Tree-MPNN (Phase 2)
* :class:`ProteinEncoder` -- proteins (learnable / ESM-2 pre-computed)
* :class:`TextEncoder` -- disease names, pathway labels, etc. (hash-based learnable)
"""

from glycoMusubi.embedding.encoders.glycan_encoder import (
    GlycanEncoder,
    extract_wurcs_features,
)
from glycoMusubi.embedding.encoders.glycan_tree_encoder import GlycanTreeEncoder
from glycoMusubi.embedding.encoders.protein_encoder import ProteinEncoder
from glycoMusubi.embedding.encoders.text_encoder import TextEncoder
from glycoMusubi.embedding.encoders.wurcs_tree_parser import (
    GlycanTree,
    GlycosidicBond,
    MonosaccharideNode,
    glycan_tree_to_tensors,
    parse_wurcs_to_tree,
)

__all__ = [
    "GlycanEncoder",
    "GlycanTreeEncoder",
    "GlycanTree",
    "GlycosidicBond",
    "MonosaccharideNode",
    "ProteinEncoder",
    "TextEncoder",
    "extract_wurcs_features",
    "glycan_tree_to_tensors",
    "parse_wurcs_to_tree",
]
