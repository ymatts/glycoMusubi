"""Tests for cell-type conditioned glycan prediction v2."""

from __future__ import annotations

import numpy as np
import pytest

# Import functions from the script
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import torch.nn.functional as F

from celltype_conditioned_glycan_v2 import (
    build_enzyme_expression_matrix,
    build_within_cluster_enzyme_scores,
    compute_metrics,
    rank_within_cluster,
    CellTypeGlycanScorer,
)


class TestEnzymeExpressionMatrix:
    """Test enzyme expression matrix construction."""

    def test_shape(self, tmp_path):
        """Matrix shape should be (n_celltypes, n_genes)."""
        ts_file = tmp_path / "ts.tsv"
        ts_file.write_text(
            "cell_type\ttissue_general\tgene_symbol\tmean_expr\tfrac_expressing\tn_cells\tuniprot_id\tin_kg\n"
            "T cell\tblood\tFUT8\t2.5\t0.3\t100\t\t1\n"
            "T cell\tblood\tST6GAL1\t1.2\t0.2\t100\t\t1\n"
            "B cell\tblood\tFUT8\t3.0\t0.4\t80\t\t1\n"
            "B cell\tblood\tST6GAL1\t0.5\t0.1\t80\t\t1\n"
            "Macrophage\tliver\tFUT8\t1.0\t0.2\t50\t\t1\n"
        )
        gene_list = ["FUT8", "ST6GAL1", "MGAT5"]
        matrix, ct_index = build_enzyme_expression_matrix(ts_file, gene_list)

        assert matrix.shape == (3, 3), f"Expected (3, 3), got {matrix.shape}"
        assert len(ct_index) == 3
        # MGAT5 should be all zeros (not in TS data)
        assert matrix[:, 2].sum() == 0.0

    def test_values(self, tmp_path):
        """Expression values should be correctly placed."""
        ts_file = tmp_path / "ts.tsv"
        ts_file.write_text(
            "cell_type\ttissue_general\tgene_symbol\tmean_expr\tfrac_expressing\tn_cells\tuniprot_id\tin_kg\n"
            "T cell\tblood\tFUT8\t2.5\t0.3\t100\t\t1\n"
        )
        gene_list = ["FUT8"]
        matrix, ct_index = build_enzyme_expression_matrix(ts_file, gene_list)

        assert matrix.shape == (1, 1)
        assert abs(matrix[0, 0] - 2.5) < 1e-5


class TestWithinClusterScores:
    """Test within-cluster enzyme score computation."""

    def test_nonempty(self):
        """Clusters with enzyme-linked glycans should have non-zero scores."""
        enzyme_matrix = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        gene_list = ["GENE_A", "GENE_B"]
        gene_to_glycans = {
            "GENE_A": {"G00001AA", "G00002BB"},
            "GENE_B": {"G00002BB", "G00003CC"},
        }
        cluster_assignments = {"G00001AA": 0, "G00002BB": 0, "G00003CC": 1}
        glycan_local_idx = {"G00001AA": 0, "G00002BB": 1, "G00003CC": 2}

        scores = build_within_cluster_enzyme_scores(
            enzyme_matrix, gene_list, gene_to_glycans,
            cluster_assignments, glycan_local_idx, n_clusters=2,
        )

        assert 0 in scores, "Cluster 0 should have scores"
        assert scores[0].shape[0] == 2, "Should have 2 cell types"
        assert scores[0].shape[1] == 2, "Cluster 0 should have 2 glycans"
        assert (scores[0] > 0).any(), "Should have non-zero scores"

    def test_empty_cluster(self):
        """Clusters without enzyme-linked glycans should have zero scores or be absent."""
        enzyme_matrix = np.array([[1.0]], dtype=np.float32)
        gene_list = ["GENE_A"]
        gene_to_glycans = {"GENE_A": {"G00001AA"}}
        cluster_assignments = {"G00001AA": 0, "G00002BB": 1}
        glycan_local_idx = {"G00001AA": 0, "G00002BB": 1}

        scores = build_within_cluster_enzyme_scores(
            enzyme_matrix, gene_list, gene_to_glycans,
            cluster_assignments, glycan_local_idx, n_clusters=2,
        )

        # Cluster 1 (G00002BB) has no enzyme link, so scores should be zero
        if 1 in scores:
            assert (scores[1] == 0).all(), "Cluster 1 should have zero scores"


class TestRankWithinCluster:
    """Test ranking logic."""

    def test_best_rank(self):
        """Highest score should get rank 1."""
        scores = np.array([1.0, 3.0, 2.0])
        assert rank_within_cluster(scores, 1) == 1

    def test_worst_rank(self):
        """Lowest score should get last rank."""
        scores = np.array([3.0, 1.0, 2.0])
        assert rank_within_cluster(scores, 1) == 3

    def test_tied(self):
        """Tied scores should share rank."""
        scores = np.array([2.0, 2.0, 2.0])
        # All tied -> rank 1
        assert rank_within_cluster(scores, 0) == 1


class TestComputeMetrics:
    """Test metric computation."""

    def test_perfect(self):
        """All rank-1 should give MRR=1.0 and H@k=1.0."""
        ranks = np.array([1.0, 1.0, 1.0])
        m = compute_metrics(ranks)
        assert abs(m["mrr"] - 1.0) < 1e-6
        assert abs(m["hits@1"] - 1.0) < 1e-6
        assert abs(m["hits@10"] - 1.0) < 1e-6

    def test_simple(self):
        """Check MRR for known ranks."""
        ranks = np.array([1.0, 2.0, 5.0])
        m = compute_metrics(ranks)
        expected_mrr = np.mean([1.0, 0.5, 0.2])
        assert abs(m["mrr"] - expected_mrr) < 1e-6
        assert abs(m["hits@1"] - 1.0 / 3) < 1e-6
        assert abs(m["hits@5"] - 1.0) < 1e-6


class TestOracleBetterThanMarginal:
    """Test that oracle >= marginal (logical constraint)."""

    def test_oracle_mrr_at_least_marginal_mrr(self):
        """Oracle MRR (best cell type) should be >= marginal MRR on average."""
        np.random.seed(42)
        n_ct, n_glycans_c = 10, 20
        scores = np.random.rand(n_ct, n_glycans_c).astype(np.float32)
        marginal = scores.mean(axis=0)

        rr_marginal = []
        rr_oracle = []
        for true_j in range(n_glycans_c):
            rank_m = rank_within_cluster(marginal, true_j)
            rank_o = min(rank_within_cluster(scores[ct], true_j) for ct in range(n_ct))
            rr_marginal.append(1.0 / rank_m)
            rr_oracle.append(1.0 / rank_o)

        mrr_marginal = np.mean(rr_marginal)
        mrr_oracle = np.mean(rr_oracle)
        assert mrr_oracle >= mrr_marginal, (
            f"Oracle MRR {mrr_oracle:.4f} < marginal MRR {mrr_marginal:.4f}"
        )


class TestCellTypeGlycanScorer:
    """Test the learned model."""

    def test_forward_shape(self):
        """Output should be (B, K)."""
        import torch

        model = CellTypeGlycanScorer(enzyme_dim=10, glycan_dim=16, hidden=32, n_glycans=20)
        enzyme_vec = torch.randn(4, 10)
        glycan_embs = torch.randn(20, 16)
        glycan_idx = torch.arange(20)
        scores = model(enzyme_vec, glycan_embs, glycan_idx)
        assert scores.shape == (4, 20), f"Expected (4, 20), got {scores.shape}"

    def test_gradient_flow(self):
        """Loss should produce gradients."""
        import torch

        model = CellTypeGlycanScorer(enzyme_dim=10, glycan_dim=16, hidden=32, n_glycans=5)
        enzyme_vec = torch.randn(2, 10)
        glycan_embs = torch.randn(5, 16)
        glycan_idx = torch.arange(5)
        scores = model(enzyme_vec, glycan_embs, glycan_idx)
        loss = F.cross_entropy(scores, torch.tensor([0, 2]))
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
