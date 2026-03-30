"""Unit tests for HybridLinkScorer.

Tests cover:
  - forward output shape [batch]
  - per-relation weights sum to 1 (softmax normalization)
  - weight_logits shape [num_relations, 3]
  - DistMult component correctness
  - RotatE component correctness
  - Neural scorer MLP operation
  - gradient flow through all components
  - single relation type operation
  - all relation types operation
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer


# ======================================================================
# Constants
# ======================================================================

EMBEDDING_DIM = 32
NUM_RELATIONS = 6
BATCH_SIZE = 8


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def scorer() -> HybridLinkScorer:
    """Standard HybridLinkScorer with small dimensions for fast testing."""
    return HybridLinkScorer(
        embedding_dim=EMBEDDING_DIM,
        num_relations=NUM_RELATIONS,
        neural_hidden_dim=64,
        dropout=0.0,
    )


@pytest.fixture()
def batch_inputs():
    """Standard batch inputs for testing."""
    torch.manual_seed(42)
    head = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    tail = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    relation_idx = torch.randint(0, NUM_RELATIONS, (BATCH_SIZE,))
    return head, relation_idx, tail


# ======================================================================
# TestHybridLinkScorer
# ======================================================================


class TestHybridLinkScorerOutputShape:
    """Tests for forward output shape."""

    def test_forward_output_shape(
        self, scorer: HybridLinkScorer, batch_inputs
    ) -> None:
        """Forward returns [batch] tensor."""
        head, relation_idx, tail = batch_inputs
        scores = scorer(head, relation_idx, tail)
        assert scores.shape == (BATCH_SIZE,)

    def test_forward_output_shape_single(self, scorer: HybridLinkScorer) -> None:
        """Forward works with batch size 1."""
        head = torch.randn(1, EMBEDDING_DIM)
        tail = torch.randn(1, EMBEDDING_DIM)
        relation_idx = torch.tensor([0])
        scores = scorer(head, relation_idx, tail)
        assert scores.shape == (1,)

    def test_forward_output_dtype(
        self, scorer: HybridLinkScorer, batch_inputs
    ) -> None:
        """Forward returns float tensor."""
        head, relation_idx, tail = batch_inputs
        scores = scorer(head, relation_idx, tail)
        assert scores.dtype == torch.float32


class TestPerRelationWeights:
    """Tests for per-relation softmax weights."""

    def test_weights_sum_to_one(self, scorer: HybridLinkScorer) -> None:
        """Per-relation weights from weight_net sum to 1 after softmax."""
        for rel_id in range(NUM_RELATIONS):
            rel_idx = torch.tensor([rel_id])
            r_dm = scorer.rel_embed_distmult(rel_idx)
            weights = torch.softmax(scorer.weight_net(r_dm), dim=-1)
            assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_weights_all_positive(self, scorer: HybridLinkScorer) -> None:
        """Softmax outputs are all positive."""
        for rel_id in range(NUM_RELATIONS):
            rel_idx = torch.tensor([rel_id])
            r_dm = scorer.rel_embed_distmult(rel_idx)
            weights = torch.softmax(scorer.weight_net(r_dm), dim=-1)
            assert (weights > 0).all()

    def test_weight_logits_shape(self, scorer: HybridLinkScorer) -> None:
        """weight_net maps [*, embedding_dim] -> [*, NUM_SUB_SCORERS]."""
        # Feed all relation embeddings at once
        all_rel_idx = torch.arange(NUM_RELATIONS)
        r_dm = scorer.rel_embed_distmult(all_rel_idx)  # [num_relations, dim]
        logits = scorer.weight_net(r_dm)
        assert logits.shape == (NUM_RELATIONS, HybridLinkScorer.NUM_SUB_SCORERS)


class TestDistMultComponent:
    """Tests for the DistMult sub-scorer component."""

    def test_distmult_score_correctness(self, scorer: HybridLinkScorer) -> None:
        """DistMult component computes element-wise h*r*t then sum."""
        torch.manual_seed(0)
        head = torch.randn(1, EMBEDDING_DIM)
        tail = torch.randn(1, EMBEDDING_DIM)
        rel_idx = torch.tensor([0])

        r_dm = scorer.rel_embed_distmult(rel_idx)
        score_from_decoder = scorer.distmult(head, r_dm, tail)

        # Manual computation
        expected = (head * r_dm * tail).sum(dim=-1)
        assert torch.allclose(score_from_decoder, expected, atol=1e-6)

    def test_distmult_symmetric(self, scorer: HybridLinkScorer) -> None:
        """DistMult sub-score is symmetric: score(h,r,t) == score(t,r,h)."""
        torch.manual_seed(0)
        head = torch.randn(4, EMBEDDING_DIM)
        tail = torch.randn(4, EMBEDDING_DIM)
        rel_idx = torch.tensor([0, 1, 2, 3])

        r_dm = scorer.rel_embed_distmult(rel_idx)
        score_hrt = scorer.distmult(head, r_dm, tail)
        score_trh = scorer.distmult(tail, r_dm, head)
        assert torch.allclose(score_hrt, score_trh, atol=1e-6)


class TestRotatEComponent:
    """Tests for the RotatE sub-scorer component."""

    def test_rotate_score_perfect_rotation(self, scorer: HybridLinkScorer) -> None:
        """RotatE gives score 0 when t = h * r in complex space."""
        complex_dim = EMBEDDING_DIM // 2

        h_real = torch.randn(1, EMBEDDING_DIM)
        rel_idx = torch.tensor([0])

        # Get phase angles from the model
        phases = scorer.rel_embed_rotate(rel_idx)  # [1, complex_dim]

        # Compute expected t in complex space: t = h * r
        h_c = torch.view_as_complex(h_real.view(-1, complex_dim, 2))
        r_c = torch.polar(torch.ones_like(phases), phases)
        t_c = h_c * r_c
        t_real = torch.view_as_real(t_c).view(1, EMBEDDING_DIM)

        score = scorer.rotate(h_real, phases, t_real)
        assert score.item() == pytest.approx(0.0, abs=1e-4)

    def test_rotate_score_negative_for_random(self, scorer: HybridLinkScorer) -> None:
        """RotatE gives negative score for random (non-matching) triples."""
        torch.manual_seed(42)
        head = torch.randn(4, EMBEDDING_DIM)
        tail = torch.randn(4, EMBEDDING_DIM)
        rel_idx = torch.tensor([0, 1, 2, 3])

        phases = scorer.rel_embed_rotate(rel_idx)
        scores = scorer.rotate(head, phases, tail)
        # RotatE score = -||h*r - t||, so random triples give negative scores
        assert (scores < 0).all()

    def test_rotate_relation_embedding_dim(self, scorer: HybridLinkScorer) -> None:
        """RotatE relation embedding has half the entity dimension."""
        rel_idx = torch.tensor([0])
        phases = scorer.rel_embed_rotate(rel_idx)
        assert phases.shape == (1, EMBEDDING_DIM // 2)


class TestNeuralScorer:
    """Tests for the neural MLP sub-scorer."""

    def test_neural_scorer_output_shape(self, scorer: HybridLinkScorer) -> None:
        """Neural scorer MLP maps [B, 3*d] -> [B, 1]."""
        concat_input = torch.randn(BATCH_SIZE, EMBEDDING_DIM * 3)
        output = scorer.neural_scorer(concat_input)
        assert output.shape == (BATCH_SIZE, 1)

    def test_neural_scorer_squeeze(self, scorer: HybridLinkScorer) -> None:
        """After squeeze, neural score is [B]."""
        concat_input = torch.randn(BATCH_SIZE, EMBEDDING_DIM * 3)
        output = scorer.neural_scorer(concat_input).squeeze(-1)
        assert output.shape == (BATCH_SIZE,)

    def test_neural_scorer_nonlinear(self, scorer: HybridLinkScorer) -> None:
        """Neural scorer is not a linear function (GELU nonlinearity)."""
        torch.manual_seed(0)
        x1 = torch.randn(1, EMBEDDING_DIM * 3)
        x2 = torch.randn(1, EMBEDDING_DIM * 3)

        out1 = scorer.neural_scorer(x1).item()
        out2 = scorer.neural_scorer(x2).item()
        out_sum = scorer.neural_scorer(x1 + x2).item()

        # A linear function would satisfy f(x1 + x2) = f(x1) + f(x2)
        # A nonlinear function will not (with high probability)
        assert out_sum != pytest.approx(out1 + out2, abs=1e-3)


class TestGradientFlow:
    """Tests for gradient flow through all components."""

    def test_gradients_flow_to_all_parameters(
        self, scorer: HybridLinkScorer, batch_inputs
    ) -> None:
        """Backward pass produces gradients for all parameters."""
        head, relation_idx, tail = batch_inputs
        head.requires_grad_(True)
        tail.requires_grad_(True)

        scores = scorer(head, relation_idx, tail)
        loss = scores.sum()
        loss.backward()

        # Verify gradients for input tensors
        assert head.grad is not None
        assert tail.grad is not None

        # Verify gradients for all model parameters
        for name, param in scorer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_to_weight_net(self, scorer: HybridLinkScorer) -> None:
        """Gradient flows through the weight_net."""
        head = torch.randn(2, EMBEDDING_DIM)
        tail = torch.randn(2, EMBEDDING_DIM)
        rel_idx = torch.tensor([0, 1])

        scores = scorer(head, rel_idx, tail)
        scores.sum().backward()

        assert scorer.weight_net.weight.grad is not None
        assert scorer.weight_net.bias.grad is not None

    def test_gradient_to_relation_embeddings(self, scorer: HybridLinkScorer) -> None:
        """Gradient flows to both relation embedding tables."""
        head = torch.randn(2, EMBEDDING_DIM)
        tail = torch.randn(2, EMBEDDING_DIM)
        rel_idx = torch.tensor([0, 1])

        scores = scorer(head, rel_idx, tail)
        scores.sum().backward()

        assert scorer.rel_embed_distmult.weight.grad is not None
        assert scorer.rel_embed_rotate.weight.grad is not None


class TestRelationTypes:
    """Tests for single and multiple relation types."""

    def test_single_relation_type(self) -> None:
        """Scorer works with a single relation type (num_relations=1)."""
        scorer = HybridLinkScorer(
            embedding_dim=EMBEDDING_DIM,
            num_relations=1,
            neural_hidden_dim=32,
        )
        head = torch.randn(4, EMBEDDING_DIM)
        tail = torch.randn(4, EMBEDDING_DIM)
        rel_idx = torch.zeros(4, dtype=torch.long)

        scores = scorer(head, rel_idx, tail)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_all_relation_types(self, scorer: HybridLinkScorer) -> None:
        """Scorer produces valid scores for every relation type."""
        for rel_id in range(NUM_RELATIONS):
            head = torch.randn(2, EMBEDDING_DIM)
            tail = torch.randn(2, EMBEDDING_DIM)
            rel_idx = torch.full((2,), rel_id, dtype=torch.long)

            scores = scorer(head, rel_idx, tail)
            assert scores.shape == (2,)
            assert torch.isfinite(scores).all(), (
                f"Non-finite score for relation {rel_id}"
            )

    def test_mixed_relations_in_batch(self, scorer: HybridLinkScorer) -> None:
        """Different relation types in the same batch produce valid results."""
        head = torch.randn(NUM_RELATIONS, EMBEDDING_DIM)
        tail = torch.randn(NUM_RELATIONS, EMBEDDING_DIM)
        rel_idx = torch.arange(NUM_RELATIONS)

        scores = scorer(head, rel_idx, tail)
        assert scores.shape == (NUM_RELATIONS,)
        assert torch.isfinite(scores).all()

    def test_different_relations_give_different_scores(
        self, scorer: HybridLinkScorer
    ) -> None:
        """Same (head, tail) pair scored with different relations gives
        different scores (due to distinct relation embeddings/weights)."""
        torch.manual_seed(42)
        head = torch.randn(1, EMBEDDING_DIM).expand(NUM_RELATIONS, -1)
        tail = torch.randn(1, EMBEDDING_DIM).expand(NUM_RELATIONS, -1)
        rel_idx = torch.arange(NUM_RELATIONS)

        scores = scorer(head, rel_idx, tail)
        # With different relation embeddings, scores should not all be equal
        assert not torch.allclose(
            scores, scores[0].expand_as(scores), atol=1e-6
        )


class TestFourComponentScoring:
    """Tests for 4-component scoring: DistMult + RotatE + Neural + Poincare."""

    def test_all_four_components_contribute(self, scorer: HybridLinkScorer) -> None:
        """All four sub-scorers produce non-zero contributions."""
        torch.manual_seed(42)
        head = torch.randn(4, EMBEDDING_DIM)
        tail = torch.randn(4, EMBEDDING_DIM)
        rel_idx = torch.tensor([0, 1, 2, 3])

        r_dm = scorer.rel_embed_distmult(rel_idx)
        r_rot = scorer.rel_embed_rotate(rel_idx)
        r_poincare = scorer.rel_embed_poincare(rel_idx)

        score_dm = scorer.distmult(head, r_dm, tail)
        score_rot = scorer.rotate(head, r_rot, tail)
        score_nn = scorer.neural_scorer(
            torch.cat([head, r_dm, tail], dim=-1)
        ).squeeze(-1)
        score_hyp = scorer.poincare(head, r_poincare, tail)

        # All sub-scores should be non-zero for random inputs
        assert score_dm.abs().sum() > 1e-6
        assert score_rot.abs().sum() > 1e-6
        assert score_nn.abs().sum() > 1e-6
        assert score_hyp.abs().sum() > 1e-6

    def test_weights_sum_to_one_all_components(self, scorer: HybridLinkScorer) -> None:
        """Per-relation weights are softmax over all sub-scorers (sum=1)."""
        for rel_id in range(NUM_RELATIONS):
            rel_idx = torch.tensor([rel_id])
            r_dm = scorer.rel_embed_distmult(rel_idx)
            weights = torch.softmax(scorer.weight_net(r_dm), dim=-1)
            assert weights.shape == (1, HybridLinkScorer.NUM_SUB_SCORERS)
            assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)
            assert (weights > 0).all()

    def test_backward_through_all_four_paths(self, scorer: HybridLinkScorer) -> None:
        """Backward pass produces gradients for all 4 scoring paths."""
        head = torch.randn(4, EMBEDDING_DIM, requires_grad=True)
        tail = torch.randn(4, EMBEDDING_DIM, requires_grad=True)
        rel_idx = torch.tensor([0, 1, 2, 3])

        scores = scorer(head, rel_idx, tail)
        scores.sum().backward()

        # All relation embedding tables get gradients
        assert scorer.rel_embed_distmult.weight.grad is not None
        assert scorer.rel_embed_rotate.weight.grad is not None
        assert scorer.rel_embed_poincare.weight.grad is not None
        # Weight net gets gradients
        assert scorer.weight_net.weight.grad is not None
        # Neural scorer layers get gradients
        for param in scorer.neural_scorer.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_poincare_component_shape(self, scorer: HybridLinkScorer) -> None:
        """Poincare sub-scorer produces [batch] tensor."""
        torch.manual_seed(42)
        head = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        tail = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        rel_idx = torch.randint(0, NUM_RELATIONS, (BATCH_SIZE,))

        r_poincare = scorer.rel_embed_poincare(rel_idx)
        score_hyp = scorer.poincare(head, r_poincare, tail)
        assert score_hyp.shape == (BATCH_SIZE,)

    def test_poincare_relation_embedding_dim(self, scorer: HybridLinkScorer) -> None:
        """Poincare relation embedding has full entity dimension."""
        rel_idx = torch.tensor([0])
        r_poincare = scorer.rel_embed_poincare(rel_idx)
        assert r_poincare.shape == (1, EMBEDDING_DIM)

    def test_all_subscorers_produce_batch_tensors(
        self, scorer: HybridLinkScorer
    ) -> None:
        """All 4 sub-scorers produce [batch] shaped output."""
        torch.manual_seed(42)
        B = 5
        head = torch.randn(B, EMBEDDING_DIM)
        tail = torch.randn(B, EMBEDDING_DIM)
        rel_idx = torch.randint(0, NUM_RELATIONS, (B,))

        r_dm = scorer.rel_embed_distmult(rel_idx)
        r_rot = scorer.rel_embed_rotate(rel_idx)
        r_poincare = scorer.rel_embed_poincare(rel_idx)

        assert scorer.distmult(head, r_dm, tail).shape == (B,)
        assert scorer.rotate(head, r_rot, tail).shape == (B,)
        assert scorer.neural_scorer(
            torch.cat([head, r_dm, tail], dim=-1)
        ).squeeze(-1).shape == (B,)
        assert scorer.poincare(head, r_poincare, tail).shape == (B,)


class TestInitialization:
    """Tests for weight initialization."""

    def test_weight_net_bias_zeros(self, scorer: HybridLinkScorer) -> None:
        """weight_net bias is initialized to zeros (for roughly uniform weights)."""
        assert torch.allclose(
            scorer.weight_net.bias,
            torch.zeros_like(scorer.weight_net.bias),
        )

    def test_initial_weights_roughly_uniform(self, scorer: HybridLinkScorer) -> None:
        """With zero bias initialization, initial softmax weights are near 1/N."""
        n = HybridLinkScorer.NUM_SUB_SCORERS
        rel_idx = torch.tensor([0])
        r_dm = scorer.rel_embed_distmult(rel_idx)
        weights = torch.softmax(scorer.weight_net(r_dm), dim=-1)
        # Each weight should be roughly 1/N (uniform) with zero bias
        for i in range(n):
            assert weights[0, i].item() == pytest.approx(1.0 / n, abs=0.3)

    def test_embedding_dim_stored(self, scorer: HybridLinkScorer) -> None:
        """Scorer stores embedding_dim attribute."""
        assert scorer.embedding_dim == EMBEDDING_DIM
        assert scorer.complex_dim == EMBEDDING_DIM // 2
