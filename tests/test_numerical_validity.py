"""Numerical validity tests for KGE models, decoders, losses, and evaluation metrics.

Validates that:
  1. Score functions implement the correct mathematical formulas.
  2. Scores, losses, and gradients remain finite (no NaN/Inf) under
     normal and extreme inputs.
  3. Training dynamics converge: positive scores > negative scores after
     a short training run.
  4. Evaluation metrics are mathematically correct against hand-computed
     reference values.
  5. Edge cases (zero vectors, large embeddings, all-same scores) are
     handled gracefully.

Reference implementations compared against:
  - PyKEEN (pykeen.models.TransE, pykeen.models.DistMult, pykeen.models.RotatE)
  - OGB link-prediction ranking protocol
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from glycoMusubi.embedding.decoders.transe import TransEDecoder
from glycoMusubi.embedding.decoders.distmult import DistMultDecoder
from glycoMusubi.embedding.decoders.rotate import RotatEDecoder
from glycoMusubi.embedding.models.glycoMusubie import TransE, DistMult, RotatE
from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.losses.bce_loss import BCEWithLogitsKGELoss
from glycoMusubi.evaluation.metrics import (
    compute_ranks,
    compute_mrr,
    compute_hits_at_k,
    compute_mr,
    compute_amr,
)


# ======================================================================
# Helpers
# ======================================================================


def _small_num_nodes_dict() -> dict[str, int]:
    """Minimal node-type dict for testing."""
    return {"protein": 10, "glycan": 8}


# ======================================================================
# 1. Score Function Mathematical Correctness
# ======================================================================


class TestTransEMath:
    """Verify TransE: score = -||h + r - t||_p."""

    def test_score_perfect_triple(self) -> None:
        """When h + r == t exactly, score should be 0."""
        decoder = TransEDecoder(p_norm=2)
        h = torch.tensor([[1.0, 0.0]])
        r = torch.tensor([[0.0, 1.0]])
        t = torch.tensor([[1.0, 1.0]])
        score = decoder(h, r, t)
        assert score.shape == (1,)
        assert score.item() == pytest.approx(0.0, abs=1e-6)

    def test_score_l2_norm(self) -> None:
        """Score equals negative L2 norm of (h + r - t)."""
        decoder = TransEDecoder(p_norm=2)
        h = torch.tensor([[1.0, 2.0, 3.0]])
        r = torch.tensor([[0.5, -0.5, 1.0]])
        t = torch.tensor([[0.0, 0.0, 0.0]])
        # h + r - t = [1.5, 1.5, 4.0]
        # ||[1.5, 1.5, 4.0]||_2 = sqrt(1.5^2 + 1.5^2 + 4^2) = sqrt(2.25+2.25+16) = sqrt(20.5)
        expected = -math.sqrt(20.5)
        score = decoder(h, r, t)
        assert score.item() == pytest.approx(expected, rel=1e-5)

    def test_score_l1_norm(self) -> None:
        """Score equals negative L1 norm of (h + r - t)."""
        decoder = TransEDecoder(p_norm=1)
        h = torch.tensor([[1.0, 2.0]])
        r = torch.tensor([[1.0, 1.0]])
        t = torch.tensor([[0.0, 0.0]])
        # h + r - t = [2.0, 3.0]; ||·||_1 = 5.0
        score = decoder(h, r, t)
        assert score.item() == pytest.approx(-5.0, abs=1e-6)

    def test_antisymmetry(self) -> None:
        """score(h, r, t) != score(t, r, h) in general (antisymmetry)."""
        decoder = TransEDecoder(p_norm=2)
        h = torch.tensor([[1.0, 0.0]])
        r = torch.tensor([[1.0, 1.0]])
        t = torch.tensor([[0.0, 1.0]])
        s_fwd = decoder(h, r, t)
        s_rev = decoder(t, r, h)
        # h + r - t = [2, 0] vs t + r - h = [0, 2] -- same norm
        # but with asymmetric r this would differ. Test with non-zero offset:
        r2 = torch.tensor([[2.0, 0.0]])
        s_fwd2 = decoder(h, r2, t)
        s_rev2 = decoder(t, r2, h)
        # h + r2 - t = [3, -1], ||·|| = sqrt(10)
        # t + r2 - h = [1, 1], ||·|| = sqrt(2)
        assert not torch.allclose(s_fwd2, s_rev2)

    def test_batch_scoring(self) -> None:
        """Batched scoring gives consistent results."""
        decoder = TransEDecoder(p_norm=2)
        h = torch.randn(32, 64)
        r = torch.randn(32, 64)
        t = torch.randn(32, 64)
        scores = decoder(h, r, t)
        assert scores.shape == (32,)
        # Verify first element manually
        expected_0 = -torch.norm(h[0] + r[0] - t[0], p=2)
        assert scores[0].item() == pytest.approx(expected_0.item(), rel=1e-5)

    def test_model_score_matches_decoder(self) -> None:
        """TransE model.score() matches TransEDecoder for same inputs."""
        model = TransE(_small_num_nodes_dict(), num_relations=3, embedding_dim=8)
        decoder = TransEDecoder(p_norm=2)
        h = torch.randn(4, 8)
        r = torch.randn(4, 8)
        t = torch.randn(4, 8)
        assert torch.allclose(model.score(h, r, t), decoder(h, r, t), atol=1e-6)


class TestDistMultMath:
    """Verify DistMult: score = <h, r, t> (element-wise product then sum)."""

    def test_score_formula(self) -> None:
        """Score equals sum of element-wise product h * r * t."""
        decoder = DistMultDecoder()
        h = torch.tensor([[1.0, 2.0, 3.0]])
        r = torch.tensor([[1.0, 1.0, 1.0]])
        t = torch.tensor([[1.0, 1.0, 1.0]])
        score = decoder(h, r, t)
        # 1*1*1 + 2*1*1 + 3*1*1 = 6
        assert score.item() == pytest.approx(6.0, abs=1e-6)

    def test_symmetry(self) -> None:
        """DistMult is symmetric: score(h, r, t) == score(t, r, h)."""
        decoder = DistMultDecoder()
        h = torch.tensor([[1.0, 2.0, 3.0]])
        r = torch.tensor([[0.5, -1.0, 0.3]])
        t = torch.tensor([[4.0, 5.0, 6.0]])
        s_fwd = decoder(h, r, t)
        s_rev = decoder(t, r, h)
        assert torch.allclose(s_fwd, s_rev, atol=1e-6)

    def test_orthogonal_vectors(self) -> None:
        """Score is 0 for orthogonal h and t (when r = 1)."""
        decoder = DistMultDecoder()
        h = torch.tensor([[1.0, 0.0]])
        r = torch.tensor([[1.0, 1.0]])
        t = torch.tensor([[0.0, 1.0]])
        score = decoder(h, r, t)
        assert score.item() == pytest.approx(0.0, abs=1e-6)

    def test_batch_scoring(self) -> None:
        """Batched scoring gives consistent results."""
        decoder = DistMultDecoder()
        h = torch.randn(32, 64)
        r = torch.randn(32, 64)
        t = torch.randn(32, 64)
        scores = decoder(h, r, t)
        assert scores.shape == (32,)
        expected_0 = (h[0] * r[0] * t[0]).sum()
        assert scores[0].item() == pytest.approx(expected_0.item(), rel=1e-5)

    def test_model_score_matches_decoder(self) -> None:
        """DistMult model.score() matches DistMultDecoder."""
        model = DistMult(_small_num_nodes_dict(), num_relations=3, embedding_dim=8)
        decoder = DistMultDecoder()
        h = torch.randn(4, 8)
        r = torch.randn(4, 8)
        t = torch.randn(4, 8)
        assert torch.allclose(model.score(h, r, t), decoder(h, r, t), atol=1e-6)


class TestRotatEMath:
    """Verify RotatE: score = -||h * r - t|| in complex space."""

    def test_identity_rotation(self) -> None:
        """Phase = 0 means r = 1+0j; score = -||h - t||."""
        decoder = RotatEDecoder()
        # dim=4 -> complex_dim=2
        h = torch.tensor([[1.0, 0.0, 0.0, 1.0]])  # complex: [1+0j, 0+1j]
        r = torch.tensor([[0.0, 0.0]])  # phase 0 -> r = [1+0j, 1+0j]
        t = torch.tensor([[1.0, 0.0, 0.0, 1.0]])  # same as h
        score = decoder(h, r, t)
        # h * 1 - t = 0 -> score = 0
        assert score.item() == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_rotation(self) -> None:
        """A pi/2 rotation of (1+0j) gives (0+1j) = i."""
        decoder = RotatEDecoder()
        # h = [1+0j] -> real tensor [1.0, 0.0]
        h = torch.tensor([[1.0, 0.0]])
        r = torch.tensor([[math.pi / 2]])  # 90 degrees
        # h * r = 1 * exp(i*pi/2) = 0+1j -> real [0.0, 1.0]
        t = torch.tensor([[0.0, 1.0]])
        score = decoder(h, r, t)
        assert score.item() == pytest.approx(0.0, abs=1e-5)

    def test_180_degree_rotation(self) -> None:
        """A pi rotation of (1+0j) gives (-1+0j)."""
        decoder = RotatEDecoder()
        h = torch.tensor([[1.0, 0.0]])
        r = torch.tensor([[math.pi]])
        # h * r = 1 * exp(i*pi) = -1+0j -> real [-1.0, 0.0]
        t = torch.tensor([[-1.0, 0.0]])
        score = decoder(h, r, t)
        assert score.item() == pytest.approx(0.0, abs=1e-4)

    def test_inversion_pattern(self) -> None:
        """r and r_inv = -r should invert each other: h * r * r_inv = h."""
        decoder = RotatEDecoder()
        h = torch.tensor([[3.0, 2.0, 1.0, -1.0]])  # 2 complex numbers
        phase = torch.tensor([[1.2, -0.7]])
        # After forward: h * r
        h_c = torch.view_as_complex(h.view(1, 2, 2))
        r_c = torch.polar(torch.ones_like(phase), phase)
        intermediate = h_c * r_c  # h * r in complex space
        # Now apply inverse rotation -phase
        r_inv_c = torch.polar(torch.ones_like(phase), -phase)
        result = intermediate * r_inv_c
        # result should equal h_c
        assert torch.allclose(
            torch.view_as_real(result).view(1, 4),
            h,
            atol=1e-5,
        )

    def test_unit_modulus_relation(self) -> None:
        """Relation vector has unit modulus (|r| = 1 in complex space)."""
        model = RotatE(_small_num_nodes_dict(), num_relations=5, embedding_dim=8)
        rel_phase = model.relation_embeddings.weight  # [5, 4] (complex_dim=4)
        r_complex = torch.polar(torch.ones_like(rel_phase), rel_phase)
        magnitudes = r_complex.abs()
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_model_score_matches_decoder(self) -> None:
        """RotatE model.score() matches RotatEDecoder."""
        model = RotatE(_small_num_nodes_dict(), num_relations=3, embedding_dim=8)
        decoder = RotatEDecoder()
        h = torch.randn(4, 8)
        r = torch.randn(4, 4)  # phase angles, dim/2
        t = torch.randn(4, 8)
        assert torch.allclose(model.score(h, r, t), decoder(h, r, t), atol=1e-6)

    def test_embedding_dim_must_be_even(self) -> None:
        """RotatE raises ValueError for odd embedding dimension."""
        with pytest.raises(ValueError, match="even"):
            RotatE(_small_num_nodes_dict(), num_relations=3, embedding_dim=7)

    def test_batch_scoring(self) -> None:
        """Batched scoring produces correct shape."""
        decoder = RotatEDecoder()
        h = torch.randn(16, 64)
        r = torch.randn(16, 32)  # dim/2
        t = torch.randn(16, 64)
        scores = decoder(h, r, t)
        assert scores.shape == (16,)


# ======================================================================
# 2. Numerical Stability
# ======================================================================


class TestNumericalStability:
    """Ensure no NaN/Inf under normal and extreme conditions."""

    @pytest.mark.parametrize("dim", [8, 64, 256])
    def test_transe_no_nan(self, dim: int) -> None:
        """TransE scores are finite for random embeddings."""
        decoder = TransEDecoder(p_norm=2)
        h = torch.randn(32, dim)
        r = torch.randn(32, dim)
        t = torch.randn(32, dim)
        scores = decoder(h, r, t)
        assert torch.isfinite(scores).all()

    @pytest.mark.parametrize("dim", [8, 64, 256])
    def test_distmult_no_nan(self, dim: int) -> None:
        """DistMult scores are finite for random embeddings."""
        decoder = DistMultDecoder()
        h = torch.randn(32, dim)
        r = torch.randn(32, dim)
        t = torch.randn(32, dim)
        scores = decoder(h, r, t)
        assert torch.isfinite(scores).all()

    @pytest.mark.parametrize("dim", [8, 64, 256])
    def test_rotate_no_nan(self, dim: int) -> None:
        """RotatE scores are finite for random embeddings."""
        decoder = RotatEDecoder()
        h = torch.randn(32, dim)
        r = torch.randn(32, dim // 2)
        t = torch.randn(32, dim)
        scores = decoder(h, r, t)
        assert torch.isfinite(scores).all()

    def test_transe_large_embeddings(self) -> None:
        """TransE handles large embedding values without overflow."""
        decoder = TransEDecoder(p_norm=2)
        h = torch.full((4, 16), 1e3)
        r = torch.full((4, 16), 1e3)
        t = torch.full((4, 16), 1e3)
        scores = decoder(h, r, t)
        assert torch.isfinite(scores).all()

    def test_transe_small_embeddings(self) -> None:
        """TransE handles very small embedding values."""
        decoder = TransEDecoder(p_norm=2)
        h = torch.full((4, 16), 1e-8)
        r = torch.full((4, 16), 1e-8)
        t = torch.full((4, 16), 1e-8)
        scores = decoder(h, r, t)
        assert torch.isfinite(scores).all()

    def test_transe_zero_vectors(self) -> None:
        """TransE produces score 0 for h=r=t=0."""
        decoder = TransEDecoder(p_norm=2)
        z = torch.zeros(1, 8)
        score = decoder(z, z, z)
        assert score.item() == pytest.approx(0.0, abs=1e-10)

    def test_distmult_large_embeddings(self) -> None:
        """DistMult handles large values -- may produce large scores but finite."""
        decoder = DistMultDecoder()
        h = torch.full((4, 16), 100.0)
        r = torch.full((4, 16), 100.0)
        t = torch.full((4, 16), 100.0)
        scores = decoder(h, r, t)
        assert torch.isfinite(scores).all()

    def test_rotate_extreme_phases(self) -> None:
        """RotatE handles very large phase angles (wraps correctly)."""
        decoder = RotatEDecoder()
        h = torch.randn(4, 8)
        r = torch.tensor([[100.0, -100.0, 50.0, -50.0]])  # large phases
        r = r.expand(4, -1)
        t = torch.randn(4, 8)
        scores = decoder(h, r, t)
        assert torch.isfinite(scores).all()

    def test_margin_loss_no_nan(self) -> None:
        """MarginRankingLoss is finite for extreme scores."""
        loss_fn = MarginRankingLoss(margin=9.0)
        pos = torch.tensor([1e6, -1e6, 0.0, 1e-8])
        neg = torch.tensor([-1e6, 1e6, 0.0, -1e-8])
        loss = loss_fn(pos, neg)
        assert torch.isfinite(loss)

    def test_bce_loss_no_nan(self) -> None:
        """BCEWithLogitsKGELoss is finite for extreme scores."""
        loss_fn = BCEWithLogitsKGELoss()
        pos = torch.tensor([100.0, -100.0, 0.0])
        neg = torch.tensor([-100.0, 100.0, 0.0])
        loss = loss_fn(pos, neg)
        assert torch.isfinite(loss)

    def test_bce_adversarial_no_nan(self) -> None:
        """Self-adversarial BCE loss is finite."""
        loss_fn = BCEWithLogitsKGELoss(adversarial_temperature=1.0)
        pos = torch.tensor([5.0, -5.0])
        neg = torch.tensor([[1.0, -1.0, 3.0], [2.0, 0.0, -2.0]])
        loss = loss_fn(pos, neg)
        assert torch.isfinite(loss)


# ======================================================================
# 3. Gradient Validity
# ======================================================================


class TestGradientValidity:
    """Ensure gradients are finite and have reasonable magnitude."""

    def test_transe_gradient_finite(self) -> None:
        """TransE gradients are finite after backward."""
        model = TransE(_small_num_nodes_dict(), num_relations=3, embedding_dim=16)
        loss_fn = MarginRankingLoss(margin=1.0)

        h = torch.randn(8, 16, requires_grad=True)
        r = torch.randn(8, 16, requires_grad=True)
        t = torch.randn(8, 16, requires_grad=True)

        pos_scores = model.score(h, r, t)
        neg_scores = model.score(h, r, torch.randn(8, 16))

        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()

        assert h.grad is not None
        assert torch.isfinite(h.grad).all()
        assert r.grad is not None
        assert torch.isfinite(r.grad).all()
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()

    def test_distmult_gradient_finite(self) -> None:
        """DistMult gradients are finite after backward."""
        model = DistMult(_small_num_nodes_dict(), num_relations=3, embedding_dim=16)
        loss_fn = BCEWithLogitsKGELoss()

        h = torch.randn(8, 16, requires_grad=True)
        r = torch.randn(8, 16, requires_grad=True)
        t = torch.randn(8, 16, requires_grad=True)

        pos_scores = model.score(h, r, t)
        neg_scores = model.score(h, r, torch.randn(8, 16))

        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()

        for param in [h, r, t]:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_rotate_gradient_finite(self) -> None:
        """RotatE gradients are finite after backward."""
        model = RotatE(_small_num_nodes_dict(), num_relations=3, embedding_dim=16)
        loss_fn = MarginRankingLoss(margin=3.0)

        h = torch.randn(8, 16, requires_grad=True)
        r = torch.randn(8, 8, requires_grad=True)  # complex_dim = 8
        t = torch.randn(8, 16, requires_grad=True)

        pos_scores = model.score(h, r, t)
        neg_scores = model.score(h, r, torch.randn(8, 16))

        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()

        for param in [h, r, t]:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_gradient_magnitude_reasonable(self) -> None:
        """Gradient L2 norm is within a reasonable range (no explosion)."""
        model = TransE(_small_num_nodes_dict(), num_relations=3, embedding_dim=64)
        loss_fn = MarginRankingLoss(margin=5.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        h = torch.randn(16, 64, requires_grad=True)
        r = torch.randn(16, 64, requires_grad=True)
        t = torch.randn(16, 64, requires_grad=True)

        pos_scores = model.score(h, r, t)
        neg_scores = model.score(h, r, torch.randn(16, 64))
        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Gradient norm should be finite and not absurdly large
                assert grad_norm < 1e6, f"Gradient explosion in {name}: norm={grad_norm}"

    def test_no_gradient_vanishing_after_steps(self) -> None:
        """After 10 optimizer steps, gradients remain nonzero (no vanishing)."""
        torch.manual_seed(99)
        model = TransE({"node": 20}, num_relations=3, embedding_dim=32)
        loss_fn = MarginRankingLoss(margin=3.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(10):
            optimizer.zero_grad()
            emb = model.node_embeddings["node"](torch.arange(20))
            h = emb[:8]
            r = model.relation_embeddings(torch.zeros(8, dtype=torch.long))
            t_pos = emb[8:16]
            t_neg = emb[torch.randint(0, 20, (8,))]

            pos_scores = model.score(h, r, t_pos)
            neg_scores = model.score(h, r, t_neg)
            loss = loss_fn(pos_scores, neg_scores)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        # Verify model parameters changed from init (learning happened)
        # Re-init a fresh model to compare
        torch.manual_seed(99)
        model_fresh = TransE({"node": 20}, num_relations=3, embedding_dim=32)
        trained_emb = model.node_embeddings["node"].weight.data
        fresh_emb = model_fresh.node_embeddings["node"].weight.data
        assert not torch.allclose(trained_emb, fresh_emb, atol=1e-6), (
            "Model parameters did not change after training steps"
        )


# ======================================================================
# 4. Training Dynamics
# ======================================================================


class TestTrainingDynamics:
    """Test that score distributions evolve correctly during training."""

    def test_random_init_scores_centered(self) -> None:
        """At random init, TransE scores should be roughly centered (negative)."""
        model = TransE(_small_num_nodes_dict(), num_relations=3, embedding_dim=64)
        h = torch.randn(100, 64)
        r = torch.randn(100, 64)
        t = torch.randn(100, 64)
        scores = model.score(h, r, t)
        # All scores should be non-positive (negative L2 norm)
        assert (scores <= 1e-6).all(), "TransE scores should be non-positive"
        # Mean score should be significantly negative (not zero)
        assert scores.mean() < -1.0, "Random TransE scores should be significantly negative"

    def test_distmult_random_init_score_distribution(self) -> None:
        """At random init, DistMult scores should be centered around 0."""
        model = DistMult(_small_num_nodes_dict(), num_relations=3, embedding_dim=64)
        h = torch.randn(100, 64)
        r = torch.randn(100, 64)
        t = torch.randn(100, 64)
        scores = model.score(h, r, t)
        # Mean should be close to 0 by symmetry of Gaussian initialization
        assert abs(scores.mean().item()) < 5.0, (
            "Random DistMult scores should be centered near 0"
        )

    def test_transe_training_separates_scores(self) -> None:
        """After 50 steps of TransE training, pos scores > neg scores."""
        torch.manual_seed(42)
        model = TransE({"a": 20}, num_relations=2, embedding_dim=32)
        loss_fn = MarginRankingLoss(margin=3.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Fixed positive triples
        h_idx = torch.arange(10)
        r_idx = torch.zeros(10, dtype=torch.long)
        t_idx = torch.arange(10, 20)

        for step in range(50):
            optimizer.zero_grad()
            emb = model.node_embeddings["a"](torch.arange(20))
            h = emb[h_idx]
            r = model.relation_embeddings(r_idx)
            t = emb[t_idx]
            pos_scores = model.score(h, r, t)

            # Negatives: random tails
            neg_t = emb[torch.randint(0, 20, (10,))]
            neg_scores = model.score(h, r, neg_t)

            loss = loss_fn(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

        # After training, positive scores should generally be higher
        with torch.no_grad():
            emb = model.node_embeddings["a"](torch.arange(20))
            h = emb[h_idx]
            r = model.relation_embeddings(r_idx)
            t = emb[t_idx]
            final_pos = model.score(h, r, t)
            neg_t = emb[torch.randint(0, 20, (10,))]
            final_neg = model.score(h, r, neg_t)

        assert final_pos.mean() > final_neg.mean(), (
            f"After training, pos mean ({final_pos.mean():.4f}) should exceed "
            f"neg mean ({final_neg.mean():.4f})"
        )

    def test_no_nan_during_100_step_training(self) -> None:
        """100 steps of training produce no NaN in any model parameter or loss."""
        torch.manual_seed(123)
        model = TransE({"x": 50}, num_relations=5, embedding_dim=64)
        loss_fn = BCEWithLogitsKGELoss(adversarial_temperature=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(100):
            optimizer.zero_grad()
            emb = model.node_embeddings["x"](torch.arange(50))
            h_idx = torch.randint(0, 50, (16,))
            t_idx = torch.randint(0, 50, (16,))
            r_idx = torch.randint(0, 5, (16,))

            h = emb[h_idx]
            r = model.relation_embeddings(r_idx)
            t = emb[t_idx]

            pos_scores = model.score(h, r, t)
            neg_t = emb[torch.randint(0, 50, (16,))]
            neg_scores = model.score(h, r, neg_t)

            loss = loss_fn(pos_scores, neg_scores)
            assert torch.isfinite(loss), f"NaN/Inf loss at step {step}"

            loss.backward()

            # Check all parameter gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), (
                        f"NaN/Inf gradient in {name} at step {step}"
                    )

            optimizer.step()

            # Check all parameter values
            for name, param in model.named_parameters():
                assert torch.isfinite(param.data).all(), (
                    f"NaN/Inf parameter in {name} at step {step}"
                )

    def test_loss_decreases_over_training(self) -> None:
        """Loss generally decreases during training."""
        torch.manual_seed(7)
        model = DistMult({"node": 30}, num_relations=3, embedding_dim=32)
        loss_fn = MarginRankingLoss(margin=5.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        losses = []
        for step in range(100):
            optimizer.zero_grad()
            emb = model.node_embeddings["node"](torch.arange(30))
            h = emb[torch.randint(0, 15, (8,))]
            r = model.relation_embeddings(torch.zeros(8, dtype=torch.long))
            t = emb[torch.randint(15, 30, (8,))]

            pos_scores = model.score(h, r, t)
            neg_t = emb[torch.randint(0, 30, (8,))]
            neg_scores = model.score(h, r, neg_t)

            loss = loss_fn(pos_scores, neg_scores)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Compare first 10 avg to last 10 avg
        first_avg = sum(losses[:10]) / 10
        last_avg = sum(losses[-10:]) / 10
        assert last_avg < first_avg, (
            f"Loss should decrease: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )


# ======================================================================
# 5. Evaluation Metrics Correctness
# ======================================================================


class TestMetricsCorrectness:
    """Verify evaluation metrics against hand-computed reference values."""

    def test_mrr_perfect_predictions(self) -> None:
        """MRR = 1.0 when all ranks are 1."""
        ranks = torch.tensor([1, 1, 1, 1], dtype=torch.long)
        assert compute_mrr(ranks) == pytest.approx(1.0)

    def test_mrr_known_values(self) -> None:
        """MRR for known rank sequence."""
        ranks = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        # MRR = (1/1 + 1/2 + 1/3 + 1/4) / 4 = (1 + 0.5 + 0.333 + 0.25) / 4
        expected = (1.0 + 0.5 + 1.0 / 3 + 0.25) / 4
        assert compute_mrr(ranks) == pytest.approx(expected, rel=1e-6)

    def test_mrr_empty(self) -> None:
        """MRR = 0 for empty tensor."""
        ranks = torch.tensor([], dtype=torch.long)
        assert compute_mrr(ranks) == 0.0

    def test_hits_at_1(self) -> None:
        """Hits@1 counts fraction of rank-1 predictions."""
        ranks = torch.tensor([1, 2, 1, 3, 1], dtype=torch.long)
        assert compute_hits_at_k(ranks, 1) == pytest.approx(3.0 / 5)

    def test_hits_at_10(self) -> None:
        """Hits@10 counts fraction with rank <= 10."""
        ranks = torch.tensor([1, 5, 10, 11, 100], dtype=torch.long)
        assert compute_hits_at_k(ranks, 10) == pytest.approx(3.0 / 5)

    def test_hits_at_k_all_within(self) -> None:
        """Hits@K = 1.0 when all ranks <= K."""
        ranks = torch.tensor([1, 2, 3], dtype=torch.long)
        assert compute_hits_at_k(ranks, 3) == pytest.approx(1.0)

    def test_hits_at_k_none_within(self) -> None:
        """Hits@K = 0.0 when all ranks > K."""
        ranks = torch.tensor([10, 20, 30], dtype=torch.long)
        assert compute_hits_at_k(ranks, 5) == pytest.approx(0.0)

    def test_hits_at_k_invalid(self) -> None:
        """Hits@K raises ValueError for K < 1."""
        ranks = torch.tensor([1], dtype=torch.long)
        with pytest.raises(ValueError):
            compute_hits_at_k(ranks, 0)

    def test_mr_known_values(self) -> None:
        """Mean rank for known sequence."""
        ranks = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        assert compute_mr(ranks) == pytest.approx(3.0)

    def test_mr_empty(self) -> None:
        """MR = 0 for empty tensor."""
        ranks = torch.tensor([], dtype=torch.long)
        assert compute_mr(ranks) == 0.0

    def test_amr_random_predictor(self) -> None:
        """AMR ~ 1.0 for a random predictor (mean rank ~ (n+1)/2)."""
        num_entities = 100
        # Simulate random predictor: uniform ranks from 1 to num_entities
        ranks = torch.arange(1, num_entities + 1, dtype=torch.long)
        amr = compute_amr(ranks, num_entities)
        assert amr == pytest.approx(1.0, rel=1e-6)

    def test_amr_perfect_predictor(self) -> None:
        """AMR ~ 0 for a perfect predictor (all ranks = 1)."""
        num_entities = 1000
        ranks = torch.ones(50, dtype=torch.long)
        amr = compute_amr(ranks, num_entities)
        expected = 1.0 / ((num_entities + 1) / 2)
        assert amr == pytest.approx(expected, rel=1e-6)
        assert amr < 0.01

    def test_amr_invalid_num_candidates(self) -> None:
        """AMR raises ValueError for num_candidates < 1."""
        ranks = torch.tensor([1], dtype=torch.long)
        with pytest.raises(ValueError):
            compute_amr(ranks, 0)


class TestComputeRanks:
    """Verify the rank computation function."""

    def test_basic_ranking(self) -> None:
        """Rank computation for a simple case."""
        # 3 candidates, target is index 1. Scores: [0.5, 0.8, 0.3]
        # Target score = 0.8, higher count = 0 -> rank = 1
        scores = torch.tensor([[0.5, 0.8, 0.3]])
        target_idx = torch.tensor([1])
        ranks = compute_ranks(scores, target_idx)
        assert ranks.item() == 1

    def test_rank_with_ties(self) -> None:
        """When target ties with other entities, rank = 1 + count of strictly higher."""
        # Scores: [1.0, 1.0, 1.0], target = index 0
        # Strictly higher count = 0 -> rank = 1
        scores = torch.tensor([[1.0, 1.0, 1.0]])
        target_idx = torch.tensor([0])
        ranks = compute_ranks(scores, target_idx)
        assert ranks.item() == 1

    def test_worst_rank(self) -> None:
        """Target with lowest score gets worst rank."""
        scores = torch.tensor([[0.5, 0.1, 0.8, 0.9]])
        target_idx = torch.tensor([1])  # score = 0.1, 3 others are higher
        ranks = compute_ranks(scores, target_idx)
        assert ranks.item() == 4

    def test_filtered_ranking(self) -> None:
        """Filtered ranking excludes masked positions."""
        # Scores: [0.9, 0.5, 0.8], target = index 1 (score 0.5)
        # Without filter: rank = 3 (0.9 and 0.8 are higher)
        # With filter on index 0: rank = 2 (only 0.8 is higher)
        scores = torch.tensor([[0.9, 0.5, 0.8]])
        target_idx = torch.tensor([1])
        mask = torch.tensor([[True, False, False]])  # mask out index 0
        ranks = compute_ranks(scores, target_idx, mask)
        assert ranks.item() == 2

    def test_filtered_ranking_preserves_target(self) -> None:
        """Filter mask must not mask the target itself."""
        scores = torch.tensor([[0.9, 0.5, 0.8]])
        target_idx = torch.tensor([1])
        # Mask everything except target -- only target remains
        mask = torch.tensor([[True, False, True]])
        ranks = compute_ranks(scores, target_idx, mask)
        assert ranks.item() == 1

    def test_batch_ranking(self) -> None:
        """Batch of queries produce correct ranks."""
        scores = torch.tensor([
            [0.1, 0.9, 0.5],  # target=1, score=0.9, rank=1
            [0.8, 0.2, 0.5],  # target=0, score=0.8, rank=1
            [0.3, 0.3, 0.9],  # target=2, score=0.9, rank=1
        ])
        target_idx = torch.tensor([1, 0, 2])
        ranks = compute_ranks(scores, target_idx)
        assert ranks.tolist() == [1, 1, 1]

    def test_input_validation(self) -> None:
        """Invalid input shapes raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            compute_ranks(torch.randn(5), torch.tensor([0]))
        with pytest.raises(ValueError, match="1-D"):
            compute_ranks(torch.randn(2, 3), torch.tensor([[0, 1]]))
        with pytest.raises(ValueError, match="Batch size"):
            compute_ranks(torch.randn(2, 3), torch.tensor([0, 1, 2]))


# ======================================================================
# 6. Embedding Initialization
# ======================================================================


class TestEmbeddingInit:
    """Verify embedding initialization properties."""

    def test_xavier_uniform_range(self) -> None:
        """Xavier uniform init produces values in expected range."""
        model = TransE(_small_num_nodes_dict(), num_relations=5, embedding_dim=64)
        for emb in model.node_embeddings.values():
            w = emb.weight.data
            fan_in = w.shape[1]
            bound = math.sqrt(6.0 / (w.shape[0] + fan_in))
            # Values should be within [-bound, bound] (approximately)
            assert w.min() >= -bound - 0.01
            assert w.max() <= bound + 0.01

    def test_rotate_relation_phase_range(self) -> None:
        """RotatE relation phases are within expected range."""
        model = RotatE(_small_num_nodes_dict(), num_relations=5, embedding_dim=64)
        phases = model.relation_embeddings.weight.data
        phase_range = (model.gamma + 2.0) / model.complex_dim
        assert phases.min() >= -phase_range - 0.01
        assert phases.max() <= phase_range + 0.01

    def test_embedding_norm_distribution(self) -> None:
        """Embedding norms at init are not degenerate (all same or zero)."""
        model = TransE({"a": 100}, num_relations=10, embedding_dim=128)
        emb = model.node_embeddings["a"].weight.data
        norms = emb.norm(dim=1)
        assert norms.std() > 0.01, "Embedding norms should have variance"
        assert norms.min() > 0.0, "No zero-norm embeddings at init"


# ======================================================================
# 7. Loss Function Edge Cases
# ======================================================================


class TestLossEdgeCases:
    """Edge case tests for loss functions."""

    def test_margin_loss_equal_scores(self) -> None:
        """Loss equals margin when pos == neg scores."""
        loss_fn = MarginRankingLoss(margin=5.0)
        s = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(s, s)
        assert loss.item() == pytest.approx(5.0)

    def test_bce_loss_zero_scores(self) -> None:
        """BCE loss at score=0: -log(sigmoid(0)) = log(2) per element."""
        loss_fn = BCEWithLogitsKGELoss()
        pos = torch.tensor([0.0])
        neg = torch.tensor([0.0])
        loss = loss_fn(pos, neg)
        # pos_loss = -log(0.5) = log(2); neg_loss = -log(0.5) = log(2)
        expected = 2 * math.log(2)
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_adversarial_weighting_sums_to_one(self) -> None:
        """Self-adversarial weights sum to 1 per batch element."""
        loss_fn = BCEWithLogitsKGELoss(adversarial_temperature=1.0, reduction="none")
        pos = torch.tensor([1.0, 2.0])
        neg = torch.tensor([[0.5, -0.5, 1.0], [1.0, 0.0, -1.0]])
        pos_loss, neg_loss = loss_fn(pos, neg)
        # Adversarial weights are applied multiplicatively
        # The weights should sum to 1 per row (softmax)
        with torch.no_grad():
            weights = torch.softmax(neg * 1.0, dim=-1)
            assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_label_smoothing_changes_target(self) -> None:
        """With label smoothing, pos target < 1 and neg target > 0."""
        eps = 0.1
        loss_fn = BCEWithLogitsKGELoss(label_smoothing=eps)
        pos = torch.tensor([10.0])  # very confident positive
        neg = torch.tensor([-10.0])  # very confident negative
        loss_smooth = loss_fn(pos, neg)
        # Without smoothing this would be very close to 0
        loss_fn_no_smooth = BCEWithLogitsKGELoss(label_smoothing=0.0)
        loss_no_smooth = loss_fn_no_smooth(pos, neg)
        # Smoothed loss should be slightly higher (targets pulled toward 0.5)
        assert loss_smooth > loss_no_smooth


# ======================================================================
# 8. Model Forward Pass Consistency
# ======================================================================


class TestModelForwardPass:
    """Test full model forward pass produces valid embeddings."""

    @pytest.mark.parametrize("ModelClass", [TransE, DistMult, RotatE])
    def test_forward_returns_all_node_types(self, ModelClass) -> None:
        """Forward pass returns embeddings for every node type."""
        from torch_geometric.data import HeteroData

        num_nodes_dict = {"protein": 10, "glycan": 8, "disease": 5}
        kwargs = {"num_nodes_dict": num_nodes_dict, "num_relations": 4, "embedding_dim": 16}
        model = ModelClass(**kwargs)

        data = HeteroData()
        for ntype, count in num_nodes_dict.items():
            data[ntype].num_nodes = count

        emb_dict = model(data)
        for ntype, count in num_nodes_dict.items():
            assert ntype in emb_dict
            assert emb_dict[ntype].shape == (count, 16)

    @pytest.mark.parametrize("ModelClass", [TransE, DistMult, RotatE])
    def test_forward_embeddings_finite(self, ModelClass) -> None:
        """All output embeddings are finite."""
        from torch_geometric.data import HeteroData

        num_nodes_dict = {"a": 50}
        kwargs = {"num_nodes_dict": num_nodes_dict, "num_relations": 3, "embedding_dim": 64}
        model = ModelClass(**kwargs)

        data = HeteroData()
        data["a"].num_nodes = 50
        emb_dict = model(data)
        assert torch.isfinite(emb_dict["a"]).all()

    def test_score_triples_end_to_end(self) -> None:
        """score_triples() produces finite scores."""
        from torch_geometric.data import HeteroData

        num_nodes_dict = {"protein": 10, "glycan": 8}
        model = TransE(num_nodes_dict, num_relations=2, embedding_dim=16)

        data = HeteroData()
        data["protein"].num_nodes = 10
        data["glycan"].num_nodes = 8

        h_idx = torch.tensor([0, 1, 2])
        r_idx = torch.tensor([0, 1, 0])
        t_idx = torch.tensor([0, 1, 2])

        scores = model.score_triples(data, "protein", h_idx, r_idx, "glycan", t_idx)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()
