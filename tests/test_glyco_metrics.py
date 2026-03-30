"""Tests for glyco-specific evaluation metrics.

Covers:
  - GSR: perfect negative correlation for ordered distance vs. similarity
  - GSR: near-zero correlation for random embeddings
  - CAS: known pairs rank higher than random pairs
  - Taxonomy hierarchical consistency: 1.0 when all predictions correct
  - All metrics return finite values
"""

from __future__ import annotations

import math

import pytest
import torch

from glycoMusubi.evaluation.glyco_metrics import (
    cross_modal_alignment_score,
    glycan_structure_recovery,
    taxonomy_hierarchical_consistency,
)


# ------------------------------------------------------------------
# GSR (Glycan Structure Recovery)
# ------------------------------------------------------------------


class TestGlycanStructureRecovery:
    """Tests for glycan_structure_recovery (Spearman rank correlation)."""

    def test_perfect_negative_correlation(self) -> None:
        """When high similarity => low distance, GSR should be close to -1.

        structural_similarities = [1, 2, 3, 4, 5]
        embedding_distances     = [5, 4, 3, 2, 1]
        These have perfectly opposite ranking, so rho = -1.
        """
        sim = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        gsr = glycan_structure_recovery(sim, dist)
        assert pytest.approx(gsr, abs=1e-6) == -1.0

    def test_perfect_positive_correlation(self) -> None:
        """When similarity and distance have same ranking, GSR = 1.0."""
        sim = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        gsr = glycan_structure_recovery(sim, dist)
        assert pytest.approx(gsr, abs=1e-6) == 1.0

    def test_random_embeddings_near_zero(self) -> None:
        """Random embeddings should produce GSR near 0."""
        torch.manual_seed(42)
        sim = torch.rand(200)
        dist = torch.rand(200)
        gsr = glycan_structure_recovery(sim, dist)
        assert -0.3 < gsr < 0.3, f"GSR for random data should be near 0, got {gsr}"

    def test_single_element_returns_zero(self) -> None:
        """With fewer than 2 elements, GSR returns 0."""
        sim = torch.tensor([1.0])
        dist = torch.tensor([2.0])
        gsr = glycan_structure_recovery(sim, dist)
        assert gsr == 0.0

    def test_gsr_returns_finite(self) -> None:
        """GSR always returns a finite value."""
        sim = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        dist = torch.tensor([2.0, 7.0, 1.0, 8.0, 2.0])
        gsr = glycan_structure_recovery(sim, dist)
        assert math.isfinite(gsr)

    def test_gsr_in_valid_range(self) -> None:
        """GSR should be in [-1, 1]."""
        torch.manual_seed(7)
        sim = torch.rand(50)
        dist = torch.rand(50)
        gsr = glycan_structure_recovery(sim, dist)
        assert -1.0 <= gsr <= 1.0


# ------------------------------------------------------------------
# CAS (Cross-modal Alignment Score)
# ------------------------------------------------------------------


class TestCrossModalAlignmentScore:
    """Tests for cross_modal_alignment_score (mean reciprocal rank)."""

    def test_perfect_alignment(self) -> None:
        """When glycan[i] is closest to protein[i], CAS = 1.0."""
        # Orthogonal embeddings: each glycan matches exactly one protein
        glycan_emb = torch.eye(4)
        protein_emb = torch.eye(4)
        known_pairs = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])

        cas = cross_modal_alignment_score(glycan_emb, protein_emb, known_pairs)
        assert pytest.approx(cas, abs=1e-6) == 1.0

    def test_known_pairs_rank_higher(self) -> None:
        """Known pairs should generally rank well (CAS > 0)."""
        torch.manual_seed(42)
        d = 32
        glycan_emb = torch.randn(5, d)
        protein_emb = torch.randn(8, d)

        # Make glycan 0 very similar to protein 2
        protein_emb[2] = glycan_emb[0] + 0.01 * torch.randn(d)
        known_pairs = torch.tensor([[0, 2]])

        cas = cross_modal_alignment_score(glycan_emb, protein_emb, known_pairs)
        assert cas > 0.0
        assert cas <= 1.0

    def test_empty_pairs_returns_zero(self) -> None:
        """Empty known_pairs returns 0."""
        glycan_emb = torch.randn(3, 8)
        protein_emb = torch.randn(5, 8)
        known_pairs = torch.zeros(0, 2, dtype=torch.long)

        cas = cross_modal_alignment_score(glycan_emb, protein_emb, known_pairs)
        assert cas == 0.0

    def test_cas_in_valid_range(self) -> None:
        """CAS (MRR) is in (0, 1]."""
        torch.manual_seed(99)
        glycan_emb = torch.randn(4, 16)
        protein_emb = torch.randn(6, 16)
        known_pairs = torch.tensor([[0, 1], [2, 3]])

        cas = cross_modal_alignment_score(glycan_emb, protein_emb, known_pairs)
        assert 0.0 < cas <= 1.0

    def test_cas_returns_finite(self) -> None:
        """CAS always returns a finite value."""
        glycan_emb = torch.randn(3, 8)
        protein_emb = torch.randn(5, 8)
        known_pairs = torch.tensor([[0, 0], [1, 2]])

        cas = cross_modal_alignment_score(glycan_emb, protein_emb, known_pairs)
        assert math.isfinite(cas)


# ------------------------------------------------------------------
# THC (Taxonomy Hierarchical Consistency)
# ------------------------------------------------------------------


class TestTaxonomyHierarchicalConsistency:
    """Tests for taxonomy_hierarchical_consistency."""

    def test_all_correct_returns_one(self) -> None:
        """When all predictions match labels, THC = 1.0."""
        predictions = {
            "kingdom": torch.tensor([0, 0, 1, 1]),
            "phylum": torch.tensor([0, 1, 2, 3]),
            "class": torch.tensor([0, 1, 2, 3]),
        }
        labels = {
            "kingdom": torch.tensor([0, 0, 1, 1]),
            "phylum": torch.tensor([0, 1, 2, 3]),
            "class": torch.tensor([0, 1, 2, 3]),
        }
        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert pytest.approx(thc, abs=1e-6) == 1.0

    def test_all_parent_correct_child_wrong(self) -> None:
        """When parent is correct but child is always wrong, THC = 0.0."""
        predictions = {
            "kingdom": torch.tensor([0, 0, 1, 1]),
            "phylum": torch.tensor([9, 9, 9, 9]),  # All wrong
        }
        labels = {
            "kingdom": torch.tensor([0, 0, 1, 1]),
            "phylum": torch.tensor([0, 1, 2, 3]),
        }
        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert pytest.approx(thc, abs=1e-6) == 0.0

    def test_single_level_returns_one(self) -> None:
        """With only one taxonomy level, consistency is trivially 1.0."""
        predictions = {"kingdom": torch.tensor([0, 1, 2])}
        labels = {"kingdom": torch.tensor([0, 1, 2])}
        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert thc == 1.0

    def test_partial_consistency(self) -> None:
        """Test with partial consistency between parent and child levels."""
        # 4 samples: parent correct for all, child correct for indices 0,1 only
        predictions = {
            "kingdom": torch.tensor([0, 0, 1, 1]),
            "phylum": torch.tensor([0, 1, 9, 9]),
        }
        labels = {
            "kingdom": torch.tensor([0, 0, 1, 1]),
            "phylum": torch.tensor([0, 1, 2, 3]),
        }
        # All 4 parent correct, 2 child correct => THC = 2/4 = 0.5
        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert pytest.approx(thc, abs=1e-6) == 0.5

    def test_parent_wrong_not_counted(self) -> None:
        """Instances where parent is wrong don't contribute to THC."""
        predictions = {
            "kingdom": torch.tensor([0, 9, 9, 1]),  # Only 0,3 correct
            "phylum": torch.tensor([0, 1, 2, 3]),  # 0,3 correct
        }
        labels = {
            "kingdom": torch.tensor([0, 0, 1, 1]),
            "phylum": torch.tensor([0, 1, 2, 3]),
        }
        # Parent correct for indices 0, 3 (2 total)
        # Child correct for 0, 3 => 2/2 = 1.0
        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert pytest.approx(thc, abs=1e-6) == 1.0

    def test_thc_returns_finite(self) -> None:
        """THC always returns a finite value."""
        predictions = {
            "a": torch.tensor([0, 1]),
            "b": torch.tensor([0, 1]),
        }
        labels = {
            "a": torch.tensor([0, 1]),
            "b": torch.tensor([0, 1]),
        }
        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert math.isfinite(thc)

    def test_three_level_hierarchy(self) -> None:
        """THC with 3 levels: consistency checked across both transitions."""
        predictions = {
            "kingdom": torch.tensor([0, 0]),
            "phylum": torch.tensor([0, 0]),
            "class": torch.tensor([0, 0]),
        }
        labels = {
            "kingdom": torch.tensor([0, 0]),
            "phylum": torch.tensor([0, 0]),
            "class": torch.tensor([0, 0]),
        }
        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert pytest.approx(thc, abs=1e-6) == 1.0
