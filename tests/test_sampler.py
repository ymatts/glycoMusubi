"""Unit tests for glycoMusubi.data.sampler."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from glycoMusubi.data.sampler import FunctionPoolRestrictor, TypeConstrainedNegativeSampler


# ---------------------------------------------------------------------------
# Fixture: sampler with known type offsets
# ---------------------------------------------------------------------------

@pytest.fixture
def type_offsets():
    """Node type offsets matching the mini KG (15 nodes):
      protein: [0, 4), enzyme: [4, 6), glycan: [6, 9),
      disease: [9, 11), variant: [11, 12), compound: [12, 13), site: [13, 15)
    """
    return {
        "protein": (0, 4),
        "enzyme": (4, 2),
        "glycan": (6, 3),
        "disease": (9, 2),
        "variant": (11, 1),
        "compound": (12, 1),
        "site": (13, 2),
    }


@pytest.fixture
def schema_dir():
    """Path to the project schemas directory."""
    return Path(__file__).resolve().parents[1] / "schemas"


@pytest.fixture
def sampler(type_offsets, schema_dir):
    """TypeConstrainedNegativeSampler with 16 negatives for fast testing."""
    return TypeConstrainedNegativeSampler(
        node_type_offsets=type_offsets,
        schema_dir=schema_dir,
        num_negatives=16,
        corrupt_head_prob=0.5,
    )


class TestTypeConstrainedNegativeSampler:
    """Tests for TypeConstrainedNegativeSampler."""

    def test_has_glycan_tail_constraint(self, sampler):
        """For has_glycan (protein->glycan), corrupted tails must be glycan nodes [6,9)."""
        # Batch of 4 has_glycan triples
        head = torch.tensor([0, 0, 1, 1])
        relation = ["has_glycan"] * 4
        tail = torch.tensor([6, 7, 6, 8])

        gen = torch.Generator().manual_seed(42)
        neg_head, neg_tail = sampler.sample(head, relation, tail, generator=gen)

        # For tails that were corrupted (different from original), they must be in [6, 9)
        for i in range(4):
            for j in range(16):
                t = neg_tail[i, j].item()
                if t != tail[i].item():
                    assert 6 <= t < 9, (
                        f"has_glycan negative tail {t} outside glycan range [6, 9)"
                    )

    def test_has_glycan_head_constraint(self, sampler):
        """For has_glycan, corrupted heads must be protein [0,4) or enzyme [4,6)."""
        head = torch.tensor([0, 1])
        relation = ["has_glycan"] * 2
        tail = torch.tensor([6, 7])

        gen = torch.Generator().manual_seed(42)
        neg_head, neg_tail = sampler.sample(head, relation, tail, generator=gen)

        for i in range(2):
            for j in range(16):
                h = neg_head[i, j].item()
                if h != head[i].item():
                    assert 0 <= h < 6, (
                        f"has_glycan negative head {h} outside protein/enzyme range [0, 6)"
                    )

    def test_inhibits_constraint(self, sampler):
        """For inhibits (compound->enzyme), heads must be compound, tails must be enzyme."""
        head = torch.tensor([12])     # compound range [12, 13)
        relation = ["inhibits"]
        tail = torch.tensor([4])      # enzyme range [4, 6)

        gen = torch.Generator().manual_seed(42)
        neg_head, neg_tail = sampler.sample(head, relation, tail, generator=gen)

        for j in range(16):
            h = neg_head[0, j].item()
            t = neg_tail[0, j].item()
            if h != head[0].item():
                assert 12 <= h < 13, f"inhibits head {h} not in compound [12, 13)"
            if t != tail[0].item():
                assert 4 <= t < 6, f"inhibits tail {t} not in enzyme [4, 6)"

    def test_num_negatives(self, type_offsets, schema_dir):
        """Output shape matches (B, num_negatives)."""
        for k in [1, 8, 64]:
            s = TypeConstrainedNegativeSampler(
                node_type_offsets=type_offsets,
                schema_dir=schema_dir,
                num_negatives=k,
            )
            head = torch.tensor([0, 1])
            relation = ["has_glycan", "has_glycan"]
            tail = torch.tensor([6, 7])

            neg_h, neg_t = s.sample(head, relation, tail)
            assert neg_h.shape == (2, k)
            assert neg_t.shape == (2, k)

    def test_positive_not_in_negatives_probabilistic(self, sampler):
        """Positive triples should rarely appear among negatives.

        With type-constrained sampling and small pools, collisions can happen.
        We test that not *all* negatives are the positive triple.
        """
        head = torch.tensor([0])
        relation = ["has_glycan"]
        tail = torch.tensor([6])

        gen = torch.Generator().manual_seed(42)
        neg_h, neg_t = sampler.sample(head, relation, tail, generator=gen)

        # Check that at least one negative differs from the positive
        positive_matches = (neg_h[0] == head[0]) & (neg_t[0] == tail[0])
        assert not positive_matches.all(), (
            "All negatives are identical to the positive — sampler is broken"
        )

    def test_sample_flat(self, sampler):
        """sample_flat returns (B*K, 3) tensor with -1 relation column."""
        head = torch.tensor([0, 12, 0])
        relation = ["has_glycan", "inhibits", "associated_with_disease"]
        tail = torch.tensor([6, 4, 9])

        triples = sampler.sample_flat(head, relation, tail)
        B, K = 3, 16
        assert triples.shape == (B * K, 3)
        assert (triples[:, 1] == -1).all()

    def test_seed_reproducibility(self, sampler):
        """Same generator seed produces identical negatives."""
        head = torch.tensor([0, 12])
        relation = ["has_glycan", "inhibits"]
        tail = torch.tensor([6, 4])

        gen1 = torch.Generator().manual_seed(99)
        nh1, nt1 = sampler.sample(head, relation, tail, generator=gen1)

        gen2 = torch.Generator().manual_seed(99)
        nh2, nt2 = sampler.sample(head, relation, tail, generator=gen2)

        assert torch.equal(nh1, nh2)
        assert torch.equal(nt1, nt2)

    def test_corrupt_head_prob_zero(self, type_offsets, schema_dir):
        """With corrupt_head_prob=0, only tails are corrupted."""
        s = TypeConstrainedNegativeSampler(
            node_type_offsets=type_offsets,
            schema_dir=schema_dir,
            num_negatives=32,
            corrupt_head_prob=0.0,
        )
        head = torch.tensor([0])
        relation = ["has_glycan"]
        tail = torch.tensor([6])

        gen = torch.Generator().manual_seed(42)
        neg_h, neg_t = s.sample(head, relation, tail, generator=gen)

        # Head should be unchanged (all equal to original)
        assert (neg_h[0] == 0).all()

    def test_corrupt_head_prob_one(self, type_offsets, schema_dir):
        """With corrupt_head_prob=1, only heads are corrupted."""
        s = TypeConstrainedNegativeSampler(
            node_type_offsets=type_offsets,
            schema_dir=schema_dir,
            num_negatives=32,
            corrupt_head_prob=1.0,
        )
        head = torch.tensor([0])
        relation = ["has_glycan"]
        tail = torch.tensor([6])

        gen = torch.Generator().manual_seed(42)
        neg_h, neg_t = s.sample(head, relation, tail, generator=gen)

        # Tail should be unchanged
        assert (neg_t[0] == 6).all()


class TestFunctionPoolRestrictor:
    """Tests for FunctionPoolRestrictor."""

    def test_has_glycan_returns_restricted_indices(self):
        """has_glycan should return the merged function-bearing glycan indices."""
        func_indices = {
            "N-linked": [0, 1, 2],
            "O-linked": [2, 3, 4],
        }
        restrictor = FunctionPoolRestrictor(func_indices)
        result = restrictor(("protein", "has_glycan", "glycan"))
        assert result is not None
        # Should contain unique sorted indices: {0,1,2,3,4}
        assert result.tolist() == [0, 1, 2, 3, 4]

    def test_other_relation_returns_none(self):
        """Non-target relations should return None (no restriction)."""
        func_indices = {
            "N-linked": [0, 1, 2],
        }
        restrictor = FunctionPoolRestrictor(func_indices)
        result = restrictor(("compound", "inhibits", "enzyme"))
        assert result is None

    def test_custom_target_relations(self):
        """Custom target_relations should be respected."""
        func_indices = {"N-linked": [10, 20]}
        restrictor = FunctionPoolRestrictor(
            func_indices, target_relations={"has_glycan", "catalyzes"}
        )
        assert restrictor(("enzyme", "catalyzes", "glycan")) is not None
        assert restrictor(("protein", "has_glycan", "glycan")) is not None
        assert restrictor(("protein", "has_site", "site")) is None

    def test_empty_function_indices(self):
        """Empty function indices should produce an empty tensor."""
        restrictor = FunctionPoolRestrictor({})
        result = restrictor(("protein", "has_glycan", "glycan"))
        assert result is not None
        assert result.numel() == 0
