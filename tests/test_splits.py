"""Unit tests for glycoMusubi.data.splits (random_link_split, relation_stratified_split).

Includes inverse-relation leak prevention tests.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from glycoMusubi.data.splits import (
    check_inverse_leak,
    random_link_split,
    relation_stratified_split,
)


# ---------------------------------------------------------------------------
# Fixture: a small HeteroData with known edge counts
# ---------------------------------------------------------------------------

@pytest.fixture
def split_data():
    """HeteroData with enough edges to test splitting meaningfully.

    Uses unique (src, dst) pairs so that edge-tuple overlap tests are valid.
    """
    data = HeteroData()

    # Nodes
    data["protein"].x = torch.randn(10, 16)
    data["protein"].num_nodes = 10
    data["glycan"].x = torch.randn(8, 16)
    data["glycan"].num_nodes = 8
    data["disease"].x = torch.randn(4, 16)
    data["disease"].num_nodes = 4

    # Build unique edge pairs for has_glycan: 10 proteins x 8 glycans = 80 possible
    # Use 60 unique edges
    all_pairs_hg = [(i, j) for i in range(10) for j in range(8)]
    hg_pairs = all_pairs_hg[:60]
    src_hg = torch.tensor([p[0] for p in hg_pairs])
    dst_hg = torch.tensor([p[1] for p in hg_pairs])
    data["protein", "has_glycan", "glycan"].edge_index = torch.stack([src_hg, dst_hg])

    # Build unique edge pairs for associated_with_disease: 10 x 4 = 40 possible
    # Use all 40
    all_pairs_ad = [(i, j) for i in range(10) for j in range(4)]
    src_ad = torch.tensor([p[0] for p in all_pairs_ad])
    dst_ad = torch.tensor([p[1] for p in all_pairs_ad])
    data["protein", "associated_with_disease", "disease"].edge_index = torch.stack([src_ad, dst_ad])

    return data


class TestRandomLinkSplit:
    """Tests for random_link_split."""

    def test_split_ratios(self, split_data):
        """Edges are split approximately according to val_ratio / test_ratio."""
        val_ratio = 0.1
        test_ratio = 0.2
        train, val, test = random_link_split(
            split_data, val_ratio=val_ratio, test_ratio=test_ratio, seed=42
        )

        for etype in split_data.edge_types:
            total = split_data[etype].edge_index.size(1)
            n_train = train[etype].edge_index.size(1)
            n_val = val[etype].edge_index.size(1)
            n_test = test[etype].edge_index.size(1)

            # Edges should not be lost or duplicated
            assert n_train + n_val + n_test == total

            # Check approximate ratios (allow +/- 2 edges for rounding)
            expected_test = int(total * test_ratio)
            expected_val = int(total * val_ratio)
            assert abs(n_test - expected_test) <= 1
            assert abs(n_val - expected_val) <= 1

    def test_no_edge_overlap(self, split_data):
        """Train, val, test edges must not overlap for each edge type."""
        train, val, test = random_link_split(split_data, seed=42)

        for etype in split_data.edge_types:
            train_set = _edge_set(train[etype].edge_index)
            val_set = _edge_set(val[etype].edge_index)
            test_set = _edge_set(test[etype].edge_index)

            assert train_set.isdisjoint(val_set), f"train/val overlap in {etype}"
            assert train_set.isdisjoint(test_set), f"train/test overlap in {etype}"
            assert val_set.isdisjoint(test_set), f"val/test overlap in {etype}"

    def test_node_features_shared(self, split_data):
        """Node features should be identical (shared) across splits."""
        train, val, test = random_link_split(split_data, seed=42)

        for ntype in split_data.node_types:
            assert torch.equal(train[ntype].x, val[ntype].x)
            assert torch.equal(train[ntype].x, test[ntype].x)

    def test_seed_reproducibility(self, split_data):
        """Same seed produces identical splits."""
        train1, val1, test1 = random_link_split(split_data, seed=123)
        train2, val2, test2 = random_link_split(split_data, seed=123)

        for etype in split_data.edge_types:
            assert torch.equal(train1[etype].edge_index, train2[etype].edge_index)
            assert torch.equal(val1[etype].edge_index, val2[etype].edge_index)
            assert torch.equal(test1[etype].edge_index, test2[etype].edge_index)

    def test_different_seeds_different_splits(self, split_data):
        """Different seeds produce different splits (with high probability)."""
        train1, _, _ = random_link_split(split_data, seed=1)
        train2, _, _ = random_link_split(split_data, seed=9999)

        any_different = False
        for etype in split_data.edge_types:
            ei1 = train1[etype].edge_index
            ei2 = train2[etype].edge_index
            if ei1.shape != ei2.shape or not torch.equal(ei1, ei2):
                any_different = True
                break
        assert any_different, "Different seeds should produce different splits"


class TestRelationStratifiedSplit:
    """Tests for relation_stratified_split."""

    def test_per_relation_ratios(self, split_data):
        """Each relation type is split individually with correct totals."""
        train, val, test = relation_stratified_split(
            split_data, val_ratio=0.1, test_ratio=0.2, seed=42
        )

        for etype in split_data.edge_types:
            total = split_data[etype].edge_index.size(1)
            n_train = train[etype].edge_index.size(1)
            n_val = val[etype].edge_index.size(1)
            n_test = test[etype].edge_index.size(1)

            assert n_train + n_val + n_test == total

    def test_no_edge_overlap(self, split_data):
        """Stratified split also has no overlap between train/val/test."""
        train, val, test = relation_stratified_split(split_data, seed=42)

        for etype in split_data.edge_types:
            train_set = _edge_set(train[etype].edge_index)
            val_set = _edge_set(val[etype].edge_index)
            test_set = _edge_set(test[etype].edge_index)

            assert train_set.isdisjoint(val_set)
            assert train_set.isdisjoint(test_set)
            assert val_set.isdisjoint(test_set)

    def test_seed_reproducibility(self, split_data):
        """Same seed produces identical stratified splits."""
        train1, val1, test1 = relation_stratified_split(split_data, seed=77)
        train2, val2, test2 = relation_stratified_split(split_data, seed=77)

        for etype in split_data.edge_types:
            assert torch.equal(train1[etype].edge_index, train2[etype].edge_index)
            assert torch.equal(val1[etype].edge_index, val2[etype].edge_index)
            assert torch.equal(test1[etype].edge_index, test2[etype].edge_index)

    def test_small_edge_type(self):
        """Edge types with very few edges should not crash."""
        data = HeteroData()
        data["a"].x = torch.randn(5, 8)
        data["a"].num_nodes = 5
        data["b"].x = torch.randn(3, 8)
        data["b"].num_nodes = 3

        # Only 3 edges — each split gets at least 1
        data["a", "rel", "b"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])

        train, val, test = relation_stratified_split(
            data, val_ratio=0.1, test_ratio=0.1, seed=42
        )
        etype = ("a", "rel", "b")
        n_train = train[etype].edge_index.size(1)
        n_val = val[etype].edge_index.size(1)
        n_test = test[etype].edge_index.size(1)

        assert n_train + n_val + n_test == 3
        assert n_train >= 1  # At least one training edge


# ---------------------------------------------------------------------------
# Inverse relation leak prevention tests
# ---------------------------------------------------------------------------

@pytest.fixture
def inverse_data():
    """HeteroData with inverse relations for leak testing.

    Creates (protein, has_glycan, glycan) and (glycan, glycan_of, protein)
    as inverse relations, with enough edges for meaningful split testing.
    """
    data = HeteroData()

    data["protein"].x = torch.randn(10, 16)
    data["protein"].num_nodes = 10
    data["glycan"].x = torch.randn(8, 16)
    data["glycan"].num_nodes = 8

    # has_glycan: protein -> glycan (40 edges)
    pairs_hg = [(i, j) for i in range(10) for j in range(4)]
    src_hg = torch.tensor([p[0] for p in pairs_hg])
    dst_hg = torch.tensor([p[1] for p in pairs_hg])
    data["protein", "has_glycan", "glycan"].edge_index = torch.stack([src_hg, dst_hg])

    # glycan_of: glycan -> protein (40 edges, inverse of has_glycan)
    # Deliberately create exact inverses: (j, i) for each (i, j) in has_glycan
    src_go = torch.tensor([p[1] for p in pairs_hg])
    dst_go = torch.tensor([p[0] for p in pairs_hg])
    data["glycan", "glycan_of", "protein"].edge_index = torch.stack([src_go, dst_go])

    return data


@pytest.fixture
def inverse_relation_map():
    """Bidirectional inverse relation mapping."""
    return {
        "has_glycan": "glycan_of",
        "glycan_of": "has_glycan",
    }


class TestInverseRelationLeakPrevention:
    """Tests for inverse-relation leak prevention in split functions."""

    def test_with_inverse_map_no_leaked_triples(self, inverse_data, inverse_relation_map):
        """With inverse_relation_map, no (t, r_inv, h) in test when (h, r, t) in train."""
        train, val, test = random_link_split(
            inverse_data,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=42,
            inverse_relation_map=inverse_relation_map,
        )

        leaks = check_inverse_leak(train, val, test, inverse_relation_map)
        assert leaks["val_leaks"] == 0, f"Found {leaks['val_leaks']} val leaks"
        assert leaks["test_leaks"] == 0, f"Found {leaks['test_leaks']} test leaks"

    def test_without_inverse_map_unchanged_behavior(self, inverse_data):
        """Without inverse_relation_map, behavior is unchanged (regression)."""
        train1, val1, test1 = random_link_split(
            inverse_data, val_ratio=0.1, test_ratio=0.2, seed=42,
            inverse_relation_map=None,
        )

        # Should still produce valid splits
        for etype in inverse_data.edge_types:
            total = inverse_data[etype].edge_index.size(1)
            n_train = train1[etype].edge_index.size(1)
            n_val = val1[etype].edge_index.size(1)
            n_test = test1[etype].edge_index.size(1)
            assert n_train + n_val + n_test == total

    def test_check_inverse_leak_identifies_leaks(self, inverse_relation_map):
        """check_inverse_leak() correctly identifies leaks when they exist."""
        # Manually construct train/val/test with a known leak:
        # train has (protein 0, has_glycan, glycan 0)
        # val has   (glycan 0, glycan_of, protein 0) -> this is an inverse leak
        train = HeteroData()
        val = HeteroData()
        test = HeteroData()

        for d in (train, val, test):
            d["protein"].x = torch.randn(5, 16)
            d["protein"].num_nodes = 5
            d["glycan"].x = torch.randn(5, 16)
            d["glycan"].num_nodes = 5

        # Train: protein 0 -> glycan 0 via has_glycan
        train["protein", "has_glycan", "glycan"].edge_index = torch.tensor([[0], [0]])
        # Give train an empty glycan_of to avoid missing edge type issues
        train["glycan", "glycan_of", "protein"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Val: glycan 0 -> protein 0 via glycan_of (this is the inverse leak!)
        val["glycan", "glycan_of", "protein"].edge_index = torch.tensor([[0], [0]])
        val["protein", "has_glycan", "glycan"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Test: empty
        test["glycan", "glycan_of", "protein"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        test["protein", "has_glycan", "glycan"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        leaks = check_inverse_leak(train, val, test, inverse_relation_map)
        assert leaks["val_leaks"] == 1
        assert leaks["test_leaks"] == 0

    def test_check_inverse_leak_returns_empty_when_no_leaks(self, inverse_relation_map):
        """check_inverse_leak() returns zero counts when no leaks exist."""
        train = HeteroData()
        val = HeteroData()
        test = HeteroData()

        for d in (train, val, test):
            d["protein"].x = torch.randn(5, 16)
            d["protein"].num_nodes = 5
            d["glycan"].x = torch.randn(5, 16)
            d["glycan"].num_nodes = 5

        # Train: protein 0 -> glycan 0
        train["protein", "has_glycan", "glycan"].edge_index = torch.tensor([[0], [0]])
        train["glycan", "glycan_of", "protein"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Val: protein 1 -> glycan 1 (not inverse of any train triple)
        val["protein", "has_glycan", "glycan"].edge_index = torch.tensor([[1], [1]])
        val["glycan", "glycan_of", "protein"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Test: glycan 2 -> protein 2 (not inverse of any train triple)
        test["glycan", "glycan_of", "protein"].edge_index = torch.tensor([[2], [2]])
        test["protein", "has_glycan", "glycan"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        leaks = check_inverse_leak(train, val, test, inverse_relation_map)
        assert leaks["val_leaks"] == 0
        assert leaks["test_leaks"] == 0

    def test_works_with_random_link_split(self, inverse_data, inverse_relation_map):
        """Inverse leak prevention works with random_link_split."""
        train, val, test = random_link_split(
            inverse_data,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=42,
            inverse_relation_map=inverse_relation_map,
        )

        # Verify all edges are accounted for (some may have moved from val/test to train)
        for etype in inverse_data.edge_types:
            total = inverse_data[etype].edge_index.size(1)
            n_train = train[etype].edge_index.size(1)
            n_val = val[etype].edge_index.size(1)
            n_test = test[etype].edge_index.size(1)
            assert n_train + n_val + n_test == total

        # Verify no leaks remain
        leaks = check_inverse_leak(train, val, test, inverse_relation_map)
        assert leaks["val_leaks"] == 0
        assert leaks["test_leaks"] == 0

    def test_works_with_relation_stratified_split(self, inverse_data, inverse_relation_map):
        """Inverse leak prevention works with relation_stratified_split."""
        train, val, test = relation_stratified_split(
            inverse_data,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=42,
            inverse_relation_map=inverse_relation_map,
        )

        # Verify all edges are accounted for
        for etype in inverse_data.edge_types:
            total = inverse_data[etype].edge_index.size(1)
            n_train = train[etype].edge_index.size(1)
            n_val = val[etype].edge_index.size(1)
            n_test = test[etype].edge_index.size(1)
            assert n_train + n_val + n_test == total

        # Verify no leaks remain
        leaks = check_inverse_leak(train, val, test, inverse_relation_map)
        assert leaks["val_leaks"] == 0
        assert leaks["test_leaks"] == 0

    def test_filtering_preserves_most_triples(self, inverse_data, inverse_relation_map):
        """Filtering should not remove too many triples (statistical check)."""
        train, val, test = random_link_split(
            inverse_data,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=42,
            inverse_relation_map=inverse_relation_map,
        )

        total_original = sum(
            inverse_data[etype].edge_index.size(1) for etype in inverse_data.edge_types
        )
        total_after = sum(
            train[etype].edge_index.size(1) + val[etype].edge_index.size(1) + test[etype].edge_index.size(1)
            for etype in inverse_data.edge_types
        )

        # All edges should be preserved (just moved between splits)
        assert total_after == total_original

        # Val+test should still have some edges (leak prevention shouldn't drain them completely)
        total_val = sum(val[etype].edge_index.size(1) for etype in val.edge_types)
        total_test = sum(test[etype].edge_index.size(1) for etype in test.edge_types)
        # With 80 total edges and 0.1/0.2 split, val+test should have at least a few edges
        assert total_val + total_test > 0, "Val+test should retain some edges after leak prevention"

    def test_no_inverse_map_split_data_preserved(self, split_data):
        """Without inverse_relation_map, split results should equal standard split."""
        train_no_inv, val_no_inv, test_no_inv = random_link_split(
            split_data, val_ratio=0.1, test_ratio=0.2, seed=42,
            inverse_relation_map=None,
        )
        train_plain, val_plain, test_plain = random_link_split(
            split_data, val_ratio=0.1, test_ratio=0.2, seed=42,
        )

        for etype in split_data.edge_types:
            assert torch.equal(
                train_no_inv[etype].edge_index, train_plain[etype].edge_index
            )
            assert torch.equal(
                val_no_inv[etype].edge_index, val_plain[etype].edge_index
            )
            assert torch.equal(
                test_no_inv[etype].edge_index, test_plain[etype].edge_index
            )


# ---------------------------------------------------------------------------
# Tests for _build_inverse_relation_map (from embedding_pipeline)
# ---------------------------------------------------------------------------

class TestBuildInverseRelationMap:
    """Tests for _build_inverse_relation_map reading edge_schema.yaml."""

    def test_returns_symmetric_pairs(self):
        """Inverse map must contain parent_of↔child_of and subsumes↔subsumed_by."""
        from scripts.embedding_pipeline import _build_inverse_relation_map

        inv_map = _build_inverse_relation_map()

        # Must find the key structural inverse pairs
        assert inv_map.get("parent_of") == "child_of"
        assert inv_map.get("child_of") == "parent_of"
        assert inv_map.get("subsumes") == "subsumed_by"
        assert inv_map.get("subsumed_by") == "subsumes"

    def test_map_is_bidirectional(self):
        """Every entry (a→b) must have a reverse entry (b→a)."""
        from scripts.embedding_pipeline import _build_inverse_relation_map

        inv_map = _build_inverse_relation_map()
        assert len(inv_map) > 0, "inverse relation map should not be empty"

        for rel, inv in inv_map.items():
            assert inv_map.get(inv) == rel, (
                f"Missing reverse: {inv}→{rel} (have {rel}→{inv})"
            )

    def test_non_empty(self):
        """Map should contain at least the declared inverse pairs."""
        from scripts.embedding_pipeline import _build_inverse_relation_map

        inv_map = _build_inverse_relation_map()
        # At minimum: parent_of↔child_of, subsumes↔subsumed_by = 4 entries
        assert len(inv_map) >= 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_set(edge_index: torch.Tensor) -> set:
    """Convert a [2, E] edge_index to a set of (src, dst) tuples."""
    return {(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.size(1))}
