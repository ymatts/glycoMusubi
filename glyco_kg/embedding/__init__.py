"""Embedding module for glycoMusubi knowledge graph.

Provides KGE models (TransE, DistMult, RotatE), link-prediction decoders,
and utilities for computing node/relation embeddings on heterogeneous graphs.
"""

from glycoMusubi.embedding.models.base import BaseKGEModel
from glycoMusubi.embedding.models.glycoMusubie import TransE, DistMult, RotatE

__all__ = [
    "BaseKGEModel",
    "TransE",
    "DistMult",
    "RotatE",
]
