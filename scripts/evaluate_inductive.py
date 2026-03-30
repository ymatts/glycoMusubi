#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inductive (zero-shot) link prediction evaluation CLI.

Usage:
    python scripts/evaluate_inductive.py --experiment glycokgnet_inductive_r1
    python scripts/evaluate_inductive.py --experiment glycokgnet_inductive_r1 \\
        --relations has_glycan has_site

Loads a trained GlycoKGNet checkpoint, creates an inductive entity split,
and evaluates zero-shot link prediction performance on held-out entities.
Reports both transductive and inductive MRR side by side.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from glycoMusubi.utils.config import ExperimentConfig, load_experiment_config

logger = logging.getLogger(__name__)


def _build_edge_type_mapping(
    data,
) -> Tuple[Dict[Tuple[str, str, str], int], List[Tuple[str, str, str]]]:
    """Build stable edge type -> index mapping."""
    edge_types = sorted(data.edge_types, key=str)
    edge_type_to_idx = {et: i for i, et in enumerate(edge_types)}
    return edge_type_to_idx, edge_types


def _hetero_to_global_triples(
    data,
    edge_type_to_idx: Dict[Tuple[str, str, str], int],
    node_type_order: List[str],
    num_nodes_dict: Dict[str, int],
) -> torch.Tensor:
    """Convert heterogeneous edges to global (h, r, t) triples."""
    offsets = {}
    offset = 0
    for ntype in node_type_order:
        offsets[ntype] = offset
        offset += num_nodes_dict[ntype]

    rows = []
    for etype in data.edge_types:
        if etype not in edge_type_to_idx:
            continue
        src_type, rel, dst_type = etype
        ei = data[etype].edge_index
        rel_idx = edge_type_to_idx[etype]
        for col in range(ei.size(1)):
            h = offsets[src_type] + ei[0, col].item()
            t = offsets[dst_type] + ei[1, col].item()
            rows.append([h, rel_idx, t])

    if not rows:
        return torch.zeros(0, 3, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def run_inductive_evaluation(
    cfg: ExperimentConfig,
    relations: Optional[List[str]] = None,
) -> Dict:
    """Run inductive evaluation on a trained model.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.
    relations : list of str, optional
        Target relation names to evaluate.

    Returns
    -------
    dict
        Evaluation results.
    """
    from glycoMusubi.data.converter import KGConverter
    from glycoMusubi.data.inductive_splits import create_inductive_split
    from glycoMusubi.evaluation.inductive_adapter import InductiveAdapter
    from glycoMusubi.evaluation.inductive_evaluator import InductiveEvaluator

    device = cfg.resolve_device()
    run_dir = cfg.run_dir

    # 1. Load KG data
    logger.info("Loading KG data from %s", cfg.data.kg_dir)
    converter = KGConverter(
        node_file=str(Path(cfg.data.kg_dir) / cfg.data.node_file),
        edge_file=str(Path(cfg.data.kg_dir) / cfg.data.edge_file),
    )
    data = converter.to_hetero_data(feature_dim=cfg.embedding_dim)

    # 2. Build edge type mapping
    edge_type_to_idx, edge_types = _build_edge_type_mapping(data)
    num_relations = len(edge_types)
    node_type_order = sorted(data.node_types)
    num_nodes_dict = {
        ntype: data[ntype].num_nodes for ntype in node_type_order
    }

    # 3. Create inductive split
    logger.info(
        "Creating inductive split (holdout_ratio=%.2f, min_degree=%d)",
        cfg.data.holdout_ratio,
        cfg.data.inductive_min_degree,
    )
    split = create_inductive_split(
        data,
        holdout_ratio=cfg.data.holdout_ratio,
        min_degree=cfg.data.inductive_min_degree,
        holdout_node_types=["glycan", "protein"],
        target_relations=relations or ["has_glycan", "has_site"],
        seed=cfg.seed,
    )

    # 4. Build model
    logger.info("Building model: %s", cfg.model.name)
    from scripts.embedding_pipeline import _build_model
    from scripts.embedding_pipeline import _build_wurcs_map as _pipeline_build_wurcs_map

    wurcs_map = None
    if cfg.model.name.lower() == "glycokgnet" and cfg.model.glycan_encoder_type in (
        "wurcs_features",
        "hybrid",
    ):
        # Build node_mappings from converter for WURCS map
        node_mappings = getattr(converter, "node_mappings", {})
        if not node_mappings:
            # Build from nodes.tsv
            import pandas as pd
            nodes_path = Path(cfg.data.kg_dir) / cfg.data.node_file
            if nodes_path.exists():
                ndf = pd.read_csv(nodes_path, sep="\t", low_memory=False)
                glycan_rows = ndf[ndf.get("node_type", ndf.get("type", "")) == "glycan"]
                id_col = "node_id" if "node_id" in ndf.columns else ndf.columns[0]
                node_mappings = {
                    "glycan": {
                        str(row[id_col]).strip(): i
                        for i, (_, row) in enumerate(glycan_rows.iterrows())
                    }
                }
        wurcs_map = _pipeline_build_wurcs_map(node_mappings)

    model = _build_model(
        cfg,
        num_nodes_dict,
        num_relations,
        wurcs_map=wurcs_map,
        edge_types=edge_types,
    )

    # 5. Load checkpoint
    ckpt_path = run_dir / "last.pt"
    if ckpt_path.exists():
        logger.info("Loading checkpoint from %s", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning("No checkpoint found at %s, using random weights", ckpt_path)

    model = model.to(device)
    model.eval()

    # 6. Build global holdout entity set
    holdout_global: Set[int] = set()
    offsets = {}
    offset = 0
    for ntype in node_type_order:
        offsets[ntype] = offset
        offset += num_nodes_dict[ntype]

    for ntype, ids in split.holdout_entity_ids.items():
        for idx in ids:
            holdout_global.add(offsets[ntype] + idx)

    total_entities = sum(num_nodes_dict.values())

    # 7. Convert to global triples
    # Use training data for the model's forward pass
    train_data = split.train_data

    # Attach edge_type_idx to train data
    for etype in train_data.edge_types:
        if etype in edge_type_to_idx:
            num_edges = train_data[etype].edge_index.size(1)
            train_data[etype].edge_type_idx = torch.full(
                (num_edges,), edge_type_to_idx[etype], dtype=torch.long
            )

    # Build test triples from inductive triples
    from glycoMusubi.data.inductive_splits import inductive_triples_to_tensor

    test_triples = inductive_triples_to_tensor(
        split, edge_type_to_idx, offsets
    )

    # Build all triples for filtering
    all_triples = _hetero_to_global_triples(
        data, edge_type_to_idx, node_type_order, num_nodes_dict
    )

    # Also include training-only triples
    train_triples = _hetero_to_global_triples(
        train_data, edge_type_to_idx, node_type_order, num_nodes_dict
    )

    logger.info(
        "Test triples: %d, All triples: %d, Holdout entities: %d",
        test_triples.shape[0],
        all_triples.shape[0],
        len(holdout_global),
    )

    if test_triples.shape[0] == 0:
        logger.error("No inductive test triples found")
        return {"error": "No inductive test triples"}

    # 8. Create adapter and evaluator
    adapter = InductiveAdapter(
        model=model,
        train_data=train_data,
        holdout_entity_ids=split.holdout_entity_ids,
        edge_type_to_idx=edge_type_to_idx,
        node_type_order=node_type_order,
        device=device,
    )

    # Build relation name mapping
    relation_names = {
        idx: f"{s}::{r}::{d}" for (s, r, d), idx in edge_type_to_idx.items()
    }

    evaluator = InductiveEvaluator(
        target_relations=relations,
        batch_size=128,
    )

    # 9. Run evaluation
    logger.info("Running inductive evaluation...")
    result = evaluator.evaluate(
        model=adapter,
        test_triples=test_triples,
        all_triples=all_triples,
        num_entities=total_entities,
        holdout_global_ids=holdout_global,
        relation_names=relation_names,
    )

    # 10. Format and save results
    output = {
        "transductive": result.transductive.metrics,
        "inductive": result.inductive.metrics,
        "per_relation_inductive": result.per_relation_inductive,
        "holdout_stats": result.holdout_stats,
        "split_stats": split.stats,
    }

    output_path = run_dir / "inductive_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("INDUCTIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTransductive MRR: {result.transductive.metrics.get('mrr', 0.0):.4f}")
    print(f"  Hits@1: {result.transductive.metrics.get('hits@1', 0.0):.4f}")
    print(f"  Hits@10: {result.transductive.metrics.get('hits@10', 0.0):.4f}")
    print(f"  Triples: {result.transductive.num_triples}")
    print(f"\nInductive MRR: {result.inductive.metrics.get('mrr', 0.0):.4f}")
    print(f"  Hits@1: {result.inductive.metrics.get('hits@1', 0.0):.4f}")
    print(f"  Hits@10: {result.inductive.metrics.get('hits@10', 0.0):.4f}")
    print(f"  Triples: {result.inductive.num_triples}")
    print(f"\nPer-relation Inductive:")
    for rel, metrics in result.per_relation_inductive.items():
        print(f"  {rel}: MRR={metrics.get('mrr', 0.0):.4f}")
    print("=" * 60)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inductive (zero-shot) link prediction"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment config (without .yaml)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Config directory (default: configs)",
    )
    parser.add_argument(
        "--relations",
        nargs="+",
        default=None,
        help="Target relation names to evaluate (e.g., has_glycan has_site)",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Config overrides (e.g., model.inductive=true)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_experiment_config(
        config_dir=args.config_dir,
        experiment=args.experiment,
        overrides=args.overrides,
    )

    run_inductive_evaluation(cfg, relations=args.relations)


if __name__ == "__main__":
    main()
