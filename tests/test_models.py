"""Unit tests for KGE models (TransE, DistMult, RotatE) and the base class.

Tests cover:
  - BaseKGEModel cannot be instantiated directly (ABC)
  - TransE: forward shape, score shape, positives ranked higher
  - DistMult: forward shape, score symmetry
  - RotatE: forward shape, rotation vector norm
  - Decoder modules: TransEDecoder, DistMultDecoder, RotatEDecoder
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.base import BaseKGEModel
from glycoMusubi.embedding.models.glycoMusubie import TransE, DistMult, RotatE
from glycoMusubi.embedding.decoders.transe import TransEDecoder
from glycoMusubi.embedding.decoders.distmult import DistMultDecoder
from glycoMusubi.embedding.decoders.rotate import RotatEDecoder


# ======================================================================
# Fixtures
# ======================================================================

NUM_NODES_DICT = {"protein": 10, "glycan": 8, "disease": 5}
NUM_RELATIONS = 4
EMBEDDING_DIM = 32
BATCH_SIZE = 6


@pytest.fixture()
def num_nodes_dict():
    return NUM_NODES_DICT.copy()


@pytest.fixture()
def mini_hetero_data() -> HeteroData:
    """Minimal HeteroData for testing forward pass."""
    data = HeteroData()
    data["protein"].num_nodes = 10
    data["glycan"].num_nodes = 8
    data["disease"].num_nodes = 5
    return data


@pytest.fixture()
def transe_model(num_nodes_dict) -> TransE:
    return TransE(num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM)


@pytest.fixture()
def distmult_model(num_nodes_dict) -> DistMult:
    return DistMult(num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM)


@pytest.fixture()
def rotate_model(num_nodes_dict) -> RotatE:
    return RotatE(num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM)


# ======================================================================
# TestBaseKGEModel
# ======================================================================


class TestBaseKGEModel:
    """Tests for the abstract base class."""

    def test_abstract_methods(self, num_nodes_dict) -> None:
        """BaseKGEModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            BaseKGEModel(num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM)


# ======================================================================
# TestTransE
# ======================================================================


class TestTransE:
    """Tests for the TransE model."""

    def test_forward_shape(
        self, transe_model: TransE, mini_hetero_data: HeteroData
    ) -> None:
        """Forward returns embeddings with correct shapes per node type."""
        emb_dict = transe_model(mini_hetero_data)
        assert set(emb_dict.keys()) == {"protein", "glycan", "disease"}
        assert emb_dict["protein"].shape == (10, EMBEDDING_DIM)
        assert emb_dict["glycan"].shape == (8, EMBEDDING_DIM)
        assert emb_dict["disease"].shape == (5, EMBEDDING_DIM)

    def test_score_shape(self, transe_model: TransE) -> None:
        """Score returns [batch] tensor."""
        head = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        rel = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        tail = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        scores = transe_model.score(head, rel, tail)
        assert scores.shape == (BATCH_SIZE,)

    def test_score_positive_higher(self, transe_model: TransE) -> None:
        """Positive triple (h + r = t) should score higher than random."""
        torch.manual_seed(42)
        h = torch.randn(1, EMBEDDING_DIM)
        r = torch.randn(1, EMBEDDING_DIM)
        t_correct = h + r  # Perfect triple for TransE
        t_random = torch.randn(1, EMBEDDING_DIM)

        score_pos = transe_model.score(h, r, t_correct)
        score_neg = transe_model.score(h, r, t_random)

        # score_pos = -||h + r - t_correct|| = 0 (best possible)
        # score_neg = -||h + r - t_random|| < 0 (typically)
        assert score_pos.item() > score_neg.item()

    def test_embedding_dim(self, transe_model: TransE) -> None:
        """Model stores the correct embedding_dim."""
        assert transe_model.embedding_dim == EMBEDDING_DIM

    def test_p_norm(self, num_nodes_dict) -> None:
        """TransE respects p_norm parameter."""
        model_l1 = TransE(num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM, p_norm=1)
        model_l2 = TransE(num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM, p_norm=2)

        h = torch.randn(1, EMBEDDING_DIM)
        r = torch.randn(1, EMBEDDING_DIM)
        t = torch.randn(1, EMBEDDING_DIM)

        score_l1 = model_l1.score(h, r, t)
        score_l2 = model_l2.score(h, r, t)

        # L1 and L2 norms give different values for the same input
        assert not torch.allclose(score_l1, score_l2)

    def test_score_triples_end_to_end(
        self, transe_model: TransE, mini_hetero_data: HeteroData
    ) -> None:
        """score_triples performs end-to-end scoring."""
        head_idx = torch.tensor([0, 1, 2])
        rel_idx = torch.tensor([0, 1, 0])
        tail_idx = torch.tensor([0, 1, 2])

        scores = transe_model.score_triples(
            mini_hetero_data, "protein", head_idx, rel_idx, "glycan", tail_idx
        )
        assert scores.shape == (3,)

    def test_get_embeddings_detached(
        self, transe_model: TransE, mini_hetero_data: HeteroData
    ) -> None:
        """get_embeddings returns detached tensors."""
        emb_dict = transe_model.get_embeddings(mini_hetero_data)
        for v in emb_dict.values():
            assert not v.requires_grad


# ======================================================================
# TestDistMult
# ======================================================================


class TestDistMult:
    """Tests for the DistMult model."""

    def test_forward_shape(
        self, distmult_model: DistMult, mini_hetero_data: HeteroData
    ) -> None:
        """Forward returns embeddings with correct shapes."""
        emb_dict = distmult_model(mini_hetero_data)
        assert emb_dict["protein"].shape == (10, EMBEDDING_DIM)

    def test_score_shape(self, distmult_model: DistMult) -> None:
        """Score returns [batch] tensor."""
        head = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        rel = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        tail = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        scores = distmult_model.score(head, rel, tail)
        assert scores.shape == (BATCH_SIZE,)

    def test_score_symmetric(self, distmult_model: DistMult) -> None:
        """DistMult scoring is symmetric: score(h, r, t) == score(t, r, h)."""
        h = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        r = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        t = torch.randn(BATCH_SIZE, EMBEDDING_DIM)

        score_hrt = distmult_model.score(h, r, t)
        score_trh = distmult_model.score(t, r, h)

        assert torch.allclose(score_hrt, score_trh, atol=1e-6)


# ======================================================================
# TestRotatE
# ======================================================================


class TestRotatE:
    """Tests for the RotatE model."""

    def test_forward_shape(
        self, rotate_model: RotatE, mini_hetero_data: HeteroData
    ) -> None:
        """Forward returns embeddings with correct shapes."""
        emb_dict = rotate_model(mini_hetero_data)
        assert emb_dict["protein"].shape == (10, EMBEDDING_DIM)
        assert emb_dict["glycan"].shape == (8, EMBEDDING_DIM)

    def test_score_shape(self, rotate_model: RotatE) -> None:
        """Score returns [batch] tensor."""
        head = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        # Relation is phase angles with half dimension.
        rel = torch.randn(BATCH_SIZE, EMBEDDING_DIM // 2)
        tail = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        scores = rotate_model.score(head, rel, tail)
        assert scores.shape == (BATCH_SIZE,)

    def test_rotation_norm(self, rotate_model: RotatE) -> None:
        """Relation embeddings represent phases; the complex rotation has
        unit modulus.

        Given phase angles theta, the rotation r = exp(i*theta) satisfies
        |r| = 1 for all elements.
        """
        rel_idx = torch.arange(NUM_RELATIONS)
        phases = rotate_model.get_relation_embedding(rel_idx)

        # Construct complex rotation: polar(1, phase)
        r_complex = torch.polar(torch.ones_like(phases), phases)

        # Modulus should be 1.0 for all elements.
        modulus = r_complex.abs()
        assert torch.allclose(modulus, torch.ones_like(modulus), atol=1e-6)

    def test_even_dim_required(self, num_nodes_dict) -> None:
        """RotatE requires even embedding_dim."""
        with pytest.raises(ValueError, match="embedding_dim must be even"):
            RotatE(num_nodes_dict, NUM_RELATIONS, embedding_dim=33)

    def test_relation_embedding_dim(self, rotate_model: RotatE) -> None:
        """Relation embeddings have half the entity dimension."""
        rel_idx = torch.tensor([0])
        rel_emb = rotate_model.get_relation_embedding(rel_idx)
        assert rel_emb.shape == (1, EMBEDDING_DIM // 2)

    def test_score_perfect_rotation(self, rotate_model: RotatE) -> None:
        """When t = h * r (in complex space), score should be maximal (0)."""
        h_real = torch.randn(1, EMBEDDING_DIM)
        phases = torch.randn(1, EMBEDDING_DIM // 2)

        # Convert to complex
        h_c = torch.view_as_complex(h_real.view(-1, EMBEDDING_DIM // 2, 2))
        r_c = torch.polar(torch.ones_like(phases), phases)

        # Compute t = h * r
        t_c = h_c * r_c
        t_real = torch.view_as_real(t_c).view(1, EMBEDDING_DIM)

        score = rotate_model.score(h_real, phases, t_real)
        # Score = -||h*r - t|| = -||0|| = 0 (maximum possible)
        assert score.item() == pytest.approx(0.0, abs=1e-5)


# ======================================================================
# TestDecoders
# ======================================================================


class TestTransEDecoder:
    """Tests for the standalone TransE decoder."""

    def test_forward_shape(self) -> None:
        decoder = TransEDecoder(p_norm=2)
        h = torch.randn(4, 64)
        r = torch.randn(4, 64)
        t = torch.randn(4, 64)
        scores = decoder(h, r, t)
        assert scores.shape == (4,)

    def test_perfect_triple(self) -> None:
        """h + r = t gives score 0."""
        decoder = TransEDecoder()
        h = torch.randn(1, 64)
        r = torch.randn(1, 64)
        t = h + r
        score = decoder(h, r, t)
        assert score.item() == pytest.approx(0.0, abs=1e-5)


class TestDistMultDecoder:
    """Tests for the standalone DistMult decoder."""

    def test_forward_shape(self) -> None:
        decoder = DistMultDecoder()
        h = torch.randn(4, 64)
        r = torch.randn(4, 64)
        t = torch.randn(4, 64)
        scores = decoder(h, r, t)
        assert scores.shape == (4,)

    def test_symmetric(self) -> None:
        """DistMult is symmetric: score(h,r,t) == score(t,r,h)."""
        decoder = DistMultDecoder()
        h = torch.randn(4, 64)
        r = torch.randn(4, 64)
        t = torch.randn(4, 64)
        assert torch.allclose(decoder(h, r, t), decoder(t, r, h), atol=1e-6)


class TestRotatEDecoder:
    """Tests for the standalone RotatE decoder."""

    def test_forward_shape(self) -> None:
        decoder = RotatEDecoder()
        h = torch.randn(4, 64)
        r = torch.randn(4, 32)  # Phase angles, half dim
        t = torch.randn(4, 64)
        scores = decoder(h, r, t)
        assert scores.shape == (4,)

    def test_perfect_rotation(self) -> None:
        """When t = h * r in complex space, score should be 0."""
        decoder = RotatEDecoder()
        dim = 64
        h_real = torch.randn(1, dim)
        phases = torch.randn(1, dim // 2)

        h_c = torch.view_as_complex(h_real.view(-1, dim // 2, 2))
        r_c = torch.polar(torch.ones_like(phases), phases)
        t_c = h_c * r_c
        t_real = torch.view_as_real(t_c).view(1, dim)

        score = decoder(h_real, phases, t_real)
        assert score.item() == pytest.approx(0.0, abs=1e-5)
