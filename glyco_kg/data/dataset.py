"""GlycoKGDataset: PyG InMemoryDataset for glycoMusubi."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch_geometric.data import HeteroData, InMemoryDataset

from glycoMusubi.data.converter import KGConverter

logger = logging.getLogger(__name__)


class GlycoKGDataset(InMemoryDataset):
    """PyG :class:`InMemoryDataset` wrapper around the glycoMusubi knowledge graph.

    On first access the dataset converts the TSV/Parquet KG files into a
    :class:`HeteroData` object via :class:`KGConverter` and caches the
    processed result to disk.  Subsequent loads are instantaneous.

    Parameters
    ----------
    root : str or Path
        Root directory for the dataset.  Processed files will be stored
        under ``<root>/processed/``.
    kg_dir : str or Path
        Directory containing ``nodes.tsv`` (or ``.parquet``) and
        ``edges.tsv`` (or ``.parquet``).
    schema_dir : str or Path or None
        Directory with schema YAML files.  Defaults to ``<project>/schemas``.
    feature_dim : int
        Dimensionality for Xavier-initialised node features.
    transform, pre_transform, pre_filter
        Standard PyG dataset hooks.
    """

    def __init__(
        self,
        root: Union[str, Path],
        kg_dir: Union[str, Path] = "kg",
        schema_dir: Optional[Union[str, Path]] = None,
        feature_dim: int = 256,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self._kg_dir = Path(kg_dir)
        self._schema_dir = Path(schema_dir) if schema_dir else None
        self._feature_dim = feature_dim

        super().__init__(
            str(root),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    # InMemoryDataset interface
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self):
        # We don't manage raw downloads; the KG is built by build_kg.py
        return []

    @property
    def processed_file_names(self):
        return ["hetero_data.pt"]

    def download(self):
        # No download step — KG files are produced by the ETL pipeline.
        pass

    def process(self):
        converter = KGConverter(
            kg_dir=self._kg_dir,
            schema_dir=self._schema_dir,
        )
        data, node_mappings = converter.convert(feature_dim=self._feature_dim)

        if self.pre_filter is not None:
            data = self.pre_filter(data)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

        logger.info(
            "Processed GlycoKGDataset: %d node types, %d edge types",
            len(data.node_types),
            len(data.edge_types),
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def hetero_data(self) -> HeteroData:
        """Return the single :class:`HeteroData` graph."""
        return self[0]
