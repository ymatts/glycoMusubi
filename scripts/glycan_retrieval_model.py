#!/usr/bin/env python3
"""Protein→Glycan retrieval model (independent of KG).

Specialized two-tower model for has_glycan prediction:
- Protein tower: ESM2 mean-pool → MLP
- Glycan tower: WURCS features (from KG cache) → MLP
- Scoring: dot product in shared embedding space
- Candidate restriction by function class (N-linked/O-linked)
- Hard negative sampling within same function class

Evaluation:
- MRR, Hits@k within function-class-restricted candidate set
- Compare with KG-based approach (R8: MRR=0.092)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_func

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(data_dir: str = "data_clean"):
    data_dir = Path(data_dir)

    # Protein→glycan edges
    edges = pd.read_csv(data_dir / "edges_glycan_protein.tsv", sep="\t")
    edges = edges[["glycan_id", "protein_id"]].drop_duplicates()

    # Glycan function labels
    func = pd.read_csv(data_dir / "glycan_function_labels.tsv", sep="\t")
    func = func[["glycan_id", "function_term"]].drop_duplicates()

    # ESM2 cache
    with open(data_dir / "esm2_cache" / "id_to_idx.json") as f:
        esm_id_to_idx = json.load(f)

    # WURCS features (from KG pipeline)
    wurcs_path = data_dir / "wurcs_features_cache.pt"
    if wurcs_path.exists():
        wurcs_cache = torch.load(wurcs_path, map_location="cpu", weights_only=False)
    else:
        wurcs_cache = None

    # Glycan structures for fallback features
    raw_dir = Path("data_raw")
    structs_path = raw_dir / "glytoucan_structures.tsv"
    wurcs_dict: Dict[str, str] = {}
    if structs_path.exists():
        glycan_structs = pd.read_csv(structs_path, sep="\t")
        # Handle different column names
        id_col = "glytoucan_ac" if "glytoucan_ac" in glycan_structs.columns else "glycan_id"
        str_col = "wurcs" if "wurcs" in glycan_structs.columns else "structure"
        glycan_structs = glycan_structs[[id_col, str_col]].dropna()
        wurcs_dict = dict(zip(glycan_structs[id_col], glycan_structs[str_col]))

    return edges, func, esm_id_to_idx, data_dir / "esm2_cache", wurcs_cache, wurcs_dict


def build_glycan_features(
    glycan_ids: List[str],
    wurcs_cache: Optional[dict],
    wurcs_dict: Dict[str, str],
    feature_dim: int = 128,
) -> torch.Tensor:
    """Build glycan feature matrix.

    Uses WURCS features from cache if available, else simple WURCS-derived features.
    """
    # Use structured cache: {features: [N, D], glycan_ids: [...], id_to_idx: {...}}
    if wurcs_cache is not None and "features" in wurcs_cache:
        cache_feats = wurcs_cache["features"]
        cache_id2idx = wurcs_cache.get("id_to_idx", {})
        feature_dim = cache_feats.shape[1]
        features = []
        for gid in glycan_ids:
            if gid in cache_id2idx:
                features.append(cache_feats[cache_id2idx[gid]])
            else:
                features.append(torch.zeros(feature_dim))
        return torch.stack(features)

    features = []
    for gid in glycan_ids:
        wurcs = wurcs_dict.get(gid, "")
        feat = _wurcs_to_features(wurcs, feature_dim)
        features.append(feat)

    return torch.stack(features)


def _wurcs_to_features(wurcs: str, dim: int = 128) -> torch.Tensor:
    """Extract simple features from WURCS string."""
    feat = torch.zeros(dim)
    if not wurcs:
        return feat

    # Length features
    feat[0] = len(wurcs) / 1000.0
    # Monosaccharide count (approximate from WURCS)
    feat[1] = wurcs.count("]") / 20.0
    # Linkage count
    feat[2] = wurcs.count("-") / 20.0
    # Has branching
    feat[3] = 1.0 if "|" in wurcs else 0.0
    # Common sugar indicators
    sugars = ["Glc", "Gal", "Man", "Fuc", "Sia", "GlcNAc", "GalNAc", "Xyl"]
    for i, sugar in enumerate(sugars):
        if i + 4 < dim:
            feat[i + 4] = 1.0 if sugar.lower() in wurcs.lower() else 0.0
    # Hash-based features (capture structure diversity)
    for i in range(min(dim - 20, 108)):
        char_idx = i % len(wurcs) if wurcs else 0
        feat[20 + i] = ord(wurcs[char_idx]) / 127.0 if char_idx < len(wurcs) else 0

    return feat


def _resolve_esm2(pid: str, id_to_idx: Dict[str, int]) -> Optional[int]:
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """Two-tower retrieval model for protein→glycan prediction."""

    def __init__(
        self,
        protein_input_dim: int = 1280,
        glycan_input_dim: int = 128,
        embed_dim: int = 256,
        hidden: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.protein_tower = nn.Sequential(
            nn.Linear(protein_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, embed_dim),
        )
        self.glycan_tower = nn.Sequential(
            nn.Linear(glycan_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, embed_dim),
        )
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_protein(self, x: torch.Tensor) -> torch.Tensor:
        return F_func.normalize(self.protein_tower(x), dim=-1)

    def encode_glycan(self, x: torch.Tensor) -> torch.Tensor:
        return F_func.normalize(self.glycan_tower(x), dim=-1)

    def forward(
        self, protein_feats: torch.Tensor, glycan_feats: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity matrix [B_p, B_g]."""
        p_emb = self.encode_protein(protein_feats)
        g_emb = self.encode_glycan(glycan_feats)
        return torch.matmul(p_emb, g_emb.T) / self.temperature.abs().clamp(min=0.01)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class RetrievalTrainer:
    def __init__(
        self,
        model: TwoTowerModel,
        protein_feats: Dict[str, torch.Tensor],
        glycan_feats: torch.Tensor,
        glycan_ids: List[str],
        train_edges: List[Tuple[str, str]],
        val_edges: List[Tuple[str, str]],
        test_edges: List[Tuple[str, str]],
        glycan_func: Dict[str, Set[str]],
        protein_func: Dict[str, Set[str]],
        device: str = "cpu",
        num_negatives: int = 64,
        lr: float = 1e-3,
    ):
        self.model = model.to(device)
        self.device = device
        self.protein_feats = protein_feats
        self.glycan_feats = glycan_feats.to(device)
        self.glycan_ids = glycan_ids
        self.glycan_id_to_idx = {g: i for i, g in enumerate(glycan_ids)}
        self.train_edges = train_edges
        self.val_edges = val_edges
        self.test_edges = test_edges
        self.glycan_func = glycan_func
        self.protein_func = protein_func
        self.num_negatives = num_negatives

        # Build function-class indexes
        self.func_glycan_idx: Dict[str, List[int]] = defaultdict(list)
        for gid, funcs in glycan_func.items():
            if gid in self.glycan_id_to_idx:
                idx = self.glycan_id_to_idx[gid]
                for func in funcs:
                    self.func_glycan_idx[func].append(idx)

        # Training positive pairs index
        self.train_positive = defaultdict(set)
        for pid, gid in train_edges:
            if gid in self.glycan_id_to_idx:
                self.train_positive[pid].add(self.glycan_id_to_idx[gid])

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    def _sample_negatives(self, pid: str, pos_idx: int) -> torch.Tensor:
        """Sample hard negatives from same function class."""
        funcs = self.protein_func.get(pid, set())
        candidates = set()
        for func in funcs:
            candidates.update(self.func_glycan_idx.get(func, []))

        if not candidates:
            candidates = set(range(len(self.glycan_ids)))

        # Remove positives
        positives = self.train_positive.get(pid, set())
        candidates -= positives

        candidates = list(candidates)
        if len(candidates) > self.num_negatives:
            neg_idx = np.random.choice(candidates, self.num_negatives, replace=False)
        else:
            neg_idx = np.array(candidates)

        return torch.tensor(neg_idx, dtype=torch.long)

    def train_epoch(self, batch_size: int = 128) -> float:
        self.model.train()
        perm = np.random.permutation(len(self.train_edges))
        total_loss = 0.0

        for start in range(0, len(perm), batch_size):
            end = min(start + batch_size, len(perm))
            batch_idx = perm[start:end]

            # Collect protein features and positive glycan indices
            p_feats = []
            pos_g_idx = []
            neg_g_idx = []
            valid = []

            for i in batch_idx:
                pid, gid = self.train_edges[i]
                if pid not in self.protein_feats or gid not in self.glycan_id_to_idx:
                    continue
                p_feats.append(self.protein_feats[pid])
                pos_idx = self.glycan_id_to_idx[gid]
                pos_g_idx.append(pos_idx)
                neg = self._sample_negatives(pid, pos_idx)
                neg_g_idx.append(neg)
                valid.append(True)

            if not p_feats:
                continue

            p_batch = torch.stack(p_feats).to(self.device)
            B = p_batch.shape[0]

            # Positive glycan features
            pos_g_batch = self.glycan_feats[torch.tensor(pos_g_idx)]

            # For each sample, compute InfoNCE loss
            self.optimizer.zero_grad()

            p_emb = self.model.encode_protein(p_batch)  # [B, D]

            batch_loss = torch.tensor(0.0, device=self.device)
            for j in range(B):
                # Positive
                pos_g = self.glycan_feats[pos_g_idx[j]:pos_g_idx[j] + 1]
                pos_emb = self.model.encode_glycan(pos_g)  # [1, D]

                # Negatives
                neg_indices = neg_g_idx[j].to(self.device)
                if len(neg_indices) == 0:
                    continue
                neg_g = self.glycan_feats[neg_indices]
                neg_emb = self.model.encode_glycan(neg_g)  # [K, D]

                # All candidates
                all_emb = torch.cat([pos_emb, neg_emb], dim=0)  # [1+K, D]
                logits = torch.matmul(p_emb[j:j + 1], all_emb.T) / self.model.temperature.abs().clamp(min=0.01)
                target = torch.zeros(1, dtype=torch.long, device=self.device)  # positive is at index 0
                batch_loss += F_func.cross_entropy(logits, target)

            batch_loss /= max(B, 1)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += batch_loss.item() * B

        return total_loss / max(len(self.train_edges), 1)

    @torch.no_grad()
    def evaluate(
        self,
        edges: List[Tuple[str, str]],
        restrict_by_func: bool = False,
    ) -> Dict:
        """Evaluate retrieval with MRR, Hits@k."""
        self.model.eval()

        # Pre-compute all glycan embeddings
        all_g_emb = self.model.encode_glycan(self.glycan_feats)  # [N_g, D]

        ranks = []
        mrrs = []
        per_func_ranks: Dict[str, List[float]] = defaultdict(list)

        for pid, gid in edges:
            if pid not in self.protein_feats or gid not in self.glycan_id_to_idx:
                continue

            p_feat = self.protein_feats[pid].unsqueeze(0).to(self.device)
            p_emb = self.model.encode_protein(p_feat)  # [1, D]

            # Scores against all glycans
            scores = torch.matmul(p_emb, all_g_emb.T).squeeze(0)  # [N_g]

            target_idx = self.glycan_id_to_idx[gid]

            if restrict_by_func:
                # Restrict to same function class
                funcs = self.protein_func.get(pid, set())
                valid_idx = set()
                for func in funcs:
                    valid_idx.update(self.func_glycan_idx.get(func, []))
                if not valid_idx:
                    valid_idx = set(range(len(self.glycan_ids)))
                # Ensure target is in valid set
                valid_idx.add(target_idx)
                valid_idx = sorted(valid_idx)

                sub_scores = scores[valid_idx]
                target_pos = valid_idx.index(target_idx)
                rank = (sub_scores > sub_scores[target_pos]).sum().item() + 1
            else:
                rank = (scores > scores[target_idx]).sum().item() + 1

            ranks.append(rank)
            mrrs.append(1.0 / rank)

            # Track by function
            funcs = self.glycan_func.get(gid, set())
            for func in funcs:
                per_func_ranks[func].append(rank)

        if not ranks:
            return {"mrr": 0, "hits@1": 0, "hits@3": 0, "hits@10": 0}

        ranks_arr = np.array(ranks)
        result = {
            "mrr": np.mean(mrrs),
            "hits@1": (ranks_arr <= 1).mean(),
            "hits@3": (ranks_arr <= 3).mean(),
            "hits@10": (ranks_arr <= 10).mean(),
            "mr": ranks_arr.mean(),
            "n_evaluated": len(ranks),
        }

        for func, func_ranks in per_func_ranks.items():
            fr = np.array(func_ranks)
            result[f"mrr_{func}"] = (1.0 / fr).mean()
            result[f"hits@10_{func}"] = (fr <= 10).mean()
            result[f"n_{func}"] = len(fr)

        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    edges, func_df, esm_id_to_idx, esm2_dir, wurcs_cache, wurcs_dict = load_data()

    # Build glycan function map
    glycan_func: Dict[str, Set[str]] = defaultdict(set)
    for _, row in func_df.iterrows():
        glycan_func[row["glycan_id"]].add(row["function_term"])

    # Build protein function map (from edges + glycan functions)
    protein_func: Dict[str, Set[str]] = defaultdict(set)
    for _, row in edges.iterrows():
        funcs = glycan_func.get(row["glycan_id"], set())
        protein_func[row["protein_id"]].update(funcs)

    # All glycans in has_glycan edges
    all_glycans = sorted(edges["glycan_id"].unique())
    logger.info("Glycans in has_glycan: %d", len(all_glycans))

    # Build glycan features
    glycan_feats = build_glycan_features(all_glycans, wurcs_cache, wurcs_dict)
    logger.info("Glycan features: %s", glycan_feats.shape)

    # Load protein ESM2 features
    protein_feats: Dict[str, torch.Tensor] = {}
    for pid in edges["protein_id"].unique():
        idx = _resolve_esm2(pid, esm_id_to_idx)
        if idx is None:
            continue
        pt_path = esm2_dir / f"{idx}.pt"
        if not pt_path.exists():
            continue
        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)
        protein_feats[pid] = emb
    logger.info("Protein features loaded: %d", len(protein_feats))

    # Split edges (protein-level, 80/10/10)
    all_edges = list(zip(edges["glycan_id"], edges["protein_id"]))
    # Swap to (protein_id, glycan_id) format
    all_edges = [(pid, gid) for gid, pid in all_edges]

    # Protein-level split
    all_prots = sorted(set(pid for pid, _ in all_edges))
    np.random.seed(42)
    np.random.shuffle(all_prots)
    n_tr = int(len(all_prots) * 0.8)
    n_va = int(len(all_prots) * 0.1)
    train_prots = set(all_prots[:n_tr])
    val_prots = set(all_prots[n_tr:n_tr + n_va])
    test_prots = set(all_prots[n_tr + n_va:])

    train_edges = [(p, g) for p, g in all_edges if p in train_prots]
    val_edges = [(p, g) for p, g in all_edges if p in val_prots]
    test_edges = [(p, g) for p, g in all_edges if p in test_prots]

    logger.info("Edges: train=%d, val=%d, test=%d", len(train_edges), len(val_edges), len(test_edges))

    # Train
    model = TwoTowerModel(
        protein_input_dim=1280,
        glycan_input_dim=glycan_feats.shape[1],
        embed_dim=256,
    )
    trainer = RetrievalTrainer(
        model=model,
        protein_feats=protein_feats,
        glycan_feats=glycan_feats,
        glycan_ids=all_glycans,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        glycan_func=glycan_func,
        protein_func=protein_func,
        device=device,
        num_negatives=64,
        lr=1e-3,
    )

    best_val_mrr = 0.0
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(100):
        loss = trainer.train_epoch(batch_size=128)

        if (epoch + 1) % 5 == 0:
            val_results = trainer.evaluate(val_edges, restrict_by_func=True)
            logger.info(
                "Epoch %d: loss=%.4f, val_MRR=%.4f, H@1=%.4f, H@10=%.4f",
                epoch + 1, loss, val_results["mrr"], val_results["hits@1"], val_results["hits@10"],
            )
            if val_results["mrr"] > best_val_mrr:
                best_val_mrr = val_results["mrr"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience // 5:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

    # Load best and evaluate
    model.load_state_dict(best_state)

    logger.info("\n=== Test Evaluation ===")

    # Full ranking
    test_full = trainer.evaluate(test_edges, restrict_by_func=False)
    logger.info("Full ranking: MRR=%.4f, H@1=%.4f, H@10=%.4f, MR=%.1f (n=%d)",
                test_full["mrr"], test_full["hits@1"], test_full["hits@10"],
                test_full["mr"], test_full["n_evaluated"])

    # Function-restricted ranking
    test_func = trainer.evaluate(test_edges, restrict_by_func=True)
    logger.info("Func-restricted: MRR=%.4f, H@1=%.4f, H@10=%.4f, MR=%.1f (n=%d)",
                test_func["mrr"], test_func["hits@1"], test_func["hits@10"],
                test_func["mr"], test_func["n_evaluated"])

    # Per-function breakdown
    for func in ["N-linked", "O-linked", "Glycosphingolipid", "Other"]:
        mrr_key = f"mrr_{func}"
        if mrr_key in test_func:
            logger.info("  %s: MRR=%.4f, H@10=%.4f (n=%d)",
                        func, test_func[mrr_key], test_func.get(f"hits@10_{func}", 0),
                        test_func.get(f"n_{func}", 0))

    # Save
    output_dir = Path("experiments_v2/glycan_retrieval")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    summary = {
        "full_ranking": {k: v for k, v in test_full.items() if isinstance(v, (int, float))},
        "func_restricted": {k: v for k, v in test_func.items() if isinstance(v, (int, float))},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
