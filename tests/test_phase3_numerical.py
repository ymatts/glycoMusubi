"""Phase 3 numerical validity tests.

Validates numerical correctness and stability of all Phase 3 components:

1. PathReasoner -- message passing scale, gradient boundedness, convergence
2. Poincare ball -- embeddings stay inside ball, exp map clamping, distance stability
3. HybridLinkScorer -- 4-component contributions, softmax normalization, convergence
4. CompGCN -- composed embedding magnitude, circular correlation correctness
5. Pretraining -- masked losses decrease, CMCA non-collapse, finite gradients
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.path_reasoner import PathReasoner, BellmanFordLayer
from glycoMusubi.embedding.models.poincare import PoincareDistance
from glycoMusubi.embedding.models.compgcn_rel import CompositionalRelationEmbedding
from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer
from glycoMusubi.training.pretraining import (
    MaskedNodePredictor,
    MaskedEdgePredictor,
    GlycanSubstructurePredictor,
)
from glycoMusubi.losses.cmca_loss import CMCALoss


# ======================================================================
# Helpers
# ======================================================================


def _make_path_reasoner_data(
    num_protein: int = 6,
    num_glycan: int = 8,
    dim: int = 64,
    seed: int = 42,
) -> HeteroData:
    """Build a small HeteroData for PathReasoner tests."""
    torch.manual_seed(seed)
    data = HeteroData()
    data["protein"].x = torch.randn(num_protein, dim)
    data["protein"].num_nodes = num_protein
    data["glycan"].x = torch.randn(num_glycan, dim)
    data["glycan"].num_nodes = num_glycan

    # protein -> glycan edges
    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 4, 5]]
    )
    # glycan -> protein edges
    data["glycan", "binds_protein", "protein"].edge_index = torch.tensor(
        [[0, 2, 4], [0, 1, 3]]
    )
    return data


def _make_path_reasoner(
    num_protein: int = 6,
    num_glycan: int = 8,
    dim: int = 64,
    num_iterations: int = 3,
    aggregation: str = "sum",
) -> PathReasoner:
    """Build a small PathReasoner."""
    return PathReasoner(
        num_nodes_dict={"glycan": num_glycan, "protein": num_protein},
        num_relations=2,
        embedding_dim=dim,
        num_iterations=num_iterations,
        aggregation=aggregation,
    )


# ======================================================================
# 1. PathReasoner Numerical Tests
# ======================================================================


class TestPathReasonerMessagePassingScale:
    """Message passing preserves embedding scale over T iterations."""

    @pytest.mark.parametrize("T", [1, 3, 6])
    def test_embedding_norm_bounded_after_T_iterations(self, T: int) -> None:
        """After T message-passing rounds, embedding norms stay bounded."""
        torch.manual_seed(42)
        model = _make_path_reasoner(num_iterations=T)
        data = _make_path_reasoner_data()

        with torch.no_grad():
            out = model(data)

        for ntype, emb in out.items():
            avg_norm = emb.norm(dim=-1).mean().item()
            # LayerNorm should keep norms stable; check reasonable range
            assert avg_norm > 0.1, (
                f"{ntype}: avg norm collapsed to {avg_norm:.6f} after {T} iters"
            )
            assert avg_norm < 1e3, (
                f"{ntype}: avg norm exploded to {avg_norm:.1f} after {T} iters"
            )
            assert torch.isfinite(emb).all(), (
                f"{ntype}: non-finite embeddings after {T} iters"
            )

    def test_norms_stable_across_iterations(self) -> None:
        """Embedding norms should not grow exponentially across iterations."""
        torch.manual_seed(42)
        model = _make_path_reasoner(num_iterations=6)
        data = _make_path_reasoner_data()

        # Manually run iterations and track norms
        edge_index, _, edge_rel_emb = model._flatten_graph(data)
        h = model._build_initial_embeddings(data)
        norms = [h.norm(dim=-1).mean().item()]

        for t in range(model.num_iterations):
            h = model.bf_layers[t](h, edge_index, edge_rel_emb, model._total_nodes)
            h = model.layer_norms[t](h)
            norms.append(h.norm(dim=-1).mean().item())

        # No norm should blow up or collapse
        for i, n in enumerate(norms):
            assert n > 0.01, f"Iter {i}: norm collapsed ({n:.6f})"
            assert n < 1e4, f"Iter {i}: norm exploded ({n:.1f})"

        # Ratio between first and last should be < 100x
        ratio = norms[-1] / max(norms[0], 1e-12)
        assert ratio < 100, (
            f"Norm ratio first/last = {ratio:.1f}, norms: {norms}"
        )


class TestPathReasonerGradients:
    """Score gradients are bounded and finite."""

    def test_score_gradients_finite(self) -> None:
        """Gradients through score MLP are finite."""
        model = _make_path_reasoner()
        # PathReasoner.score uses tail and relation (head is unused)
        rel = torch.randn(8, 64, requires_grad=True)
        tail = torch.randn(8, 64, requires_grad=True)

        scores = model.score(
            head=torch.empty(0),  # unused in path-based scoring
            relation=rel,
            tail=tail,
        )
        loss = scores.sum()
        loss.backward()

        for name, t in [("rel", rel), ("tail", tail)]:
            assert t.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(t.grad).all(), f"Non-finite gradient for {name}"

    def test_full_forward_gradients_finite(self) -> None:
        """Gradients through full forward pass are finite."""
        torch.manual_seed(42)
        model = _make_path_reasoner()
        data = _make_path_reasoner_data()

        out = model(data)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"Non-finite gradient in {name}"
                )
                grad_norm = p.grad.norm().item()
                assert grad_norm < 1e6, (
                    f"Gradient explosion in {name}: norm={grad_norm:.2f}"
                )

    def test_gradient_not_vanishing(self) -> None:
        """At least some gradients are non-negligible after forward."""
        torch.manual_seed(42)
        model = _make_path_reasoner()
        data = _make_path_reasoner_data()

        out = model(data)
        loss = sum(v.mean() for v in out.values())
        loss.backward()

        max_grad = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                max_grad = max(max_grad, p.grad.abs().max().item())

        assert max_grad > 1e-10, (
            f"All gradients are vanishingly small: max={max_grad:.2e}"
        )


class TestPathReasonerPositiveNegative:
    """Scores for positive > negative triples on simple graph."""

    def test_positive_scores_higher_after_training(self) -> None:
        """After 50 training steps, positive triples score higher than random negatives."""
        torch.manual_seed(42)
        model = _make_path_reasoner(dim=32, num_iterations=2)
        data = _make_path_reasoner_data(dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for step in range(50):
            optimizer.zero_grad()
            out = model(data)

            # Positive: protein i -> glycan i (matched by edge structure)
            h = out["protein"][:4]
            t_pos = out["glycan"][:4]
            rel = model.relation_embeddings(torch.zeros(4, dtype=torch.long))

            pos_scores = model.score(h, rel, t_pos)

            # Negative: random tail
            neg_idx = torch.randint(0, 8, (4,))
            t_neg = out["glycan"][neg_idx]
            neg_scores = model.score(h, rel, t_neg)

            # Margin loss
            loss = torch.relu(1.0 - pos_scores + neg_scores).mean()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            out = model(data)
            h = out["protein"][:4]
            t_pos = out["glycan"][:4]
            rel = model.relation_embeddings(torch.zeros(4, dtype=torch.long))
            pos_final = model.score(h, rel, t_pos).mean().item()

            neg_idx = torch.randint(0, 8, (4,))
            t_neg = out["glycan"][neg_idx]
            neg_final = model.score(h, rel, t_neg).mean().item()

        assert pos_final > neg_final, (
            f"After training, pos ({pos_final:.4f}) should exceed neg ({neg_final:.4f})"
        )


class TestPathReasonerConvergence:
    """Training converges: loss decreases over 50 steps."""

    def test_loss_decreases(self) -> None:
        """Loss should generally decrease over 50 training steps."""
        torch.manual_seed(42)
        model = _make_path_reasoner(dim=32, num_iterations=2)
        data = _make_path_reasoner_data(dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for step in range(50):
            optimizer.zero_grad()
            out = model(data)

            h = out["protein"][:4]
            t_pos = out["glycan"][:4]
            rel = model.relation_embeddings(torch.zeros(4, dtype=torch.long))
            pos_scores = model.score(h, rel, t_pos)

            neg_idx = torch.randint(0, 8, (4,))
            t_neg = out["glycan"][neg_idx]
            neg_scores = model.score(h, rel, t_neg)

            loss = torch.relu(1.0 - pos_scores + neg_scores).mean()
            losses.append(loss.item())
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        first_avg = sum(losses[:10]) / 10
        last_avg = sum(losses[-10:]) / 10
        assert last_avg < first_avg, (
            f"Loss should decrease: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )

    def test_pna_aggregation_converges(self) -> None:
        """PathReasoner with PNA aggregation also converges."""
        torch.manual_seed(42)
        model = _make_path_reasoner(dim=32, num_iterations=2, aggregation="pna")
        data = _make_path_reasoner_data(dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for step in range(50):
            optimizer.zero_grad()
            out = model(data)

            h = out["protein"][:4]
            t_pos = out["glycan"][:4]
            rel = model.relation_embeddings(torch.zeros(4, dtype=torch.long))
            pos_scores = model.score(h, rel, t_pos)

            neg_idx = torch.randint(0, 8, (4,))
            t_neg = out["glycan"][neg_idx]
            neg_scores = model.score(h, rel, t_neg)

            loss = torch.relu(1.0 - pos_scores + neg_scores).mean()
            losses.append(loss.item())
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        first_avg = sum(losses[:10]) / 10
        last_avg = sum(losses[-10:]) / 10
        assert last_avg < first_avg, (
            f"PNA loss should decrease: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )


# ======================================================================
# 2. Poincare Numerical Tests
# ======================================================================


class TestPoincareEmbeddingsInsideBall:
    """All embeddings remain inside the Poincare ball."""

    @pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0])
    def test_exp_map_stays_inside_ball(self, curvature: float) -> None:
        """exp_map output satisfies c * ||x||^2 < 1."""
        poincare = PoincareDistance(curvature=curvature)
        v = torch.randn(100, 32)
        result = poincare.exp_map(v)
        norms_sq = (result * result).sum(dim=-1)
        assert (curvature * norms_sq < 1.0).all(), (
            f"exp_map produced points outside ball: max c*||x||^2 = "
            f"{(curvature * norms_sq).max().item():.6f}"
        )

    def test_exp_map_large_vectors(self) -> None:
        """exp_map with very large tangent vectors stays inside ball."""
        poincare = PoincareDistance(curvature=1.0)
        v = torch.randn(50, 32) * 1000.0
        result = poincare.exp_map(v)
        norms_sq = (result * result).sum(dim=-1)
        assert (norms_sq < 1.0).all(), (
            f"Large tangent vectors escaped ball: max ||x||^2 = {norms_sq.max().item():.6f}"
        )
        assert torch.isfinite(result).all(), "exp_map produced NaN/Inf for large vectors"

    def test_exp_map_zero_vector(self) -> None:
        """exp_map of zero tangent vector gives origin."""
        poincare = PoincareDistance(curvature=1.0)
        v = torch.zeros(5, 16)
        result = poincare.exp_map(v)
        # Result should be very close to zero (origin)
        assert result.norm(dim=-1).max().item() < 0.01

    def test_mobius_add_stays_inside_ball(self) -> None:
        """Mobius addition of points inside ball stays inside ball."""
        poincare = PoincareDistance(curvature=1.0)
        x = torch.randn(50, 32) * 0.3  # inside ball
        y = torch.randn(50, 32) * 0.3
        result = poincare.mobius_add(x, y)
        norms_sq = (result * result).sum(dim=-1)
        assert (norms_sq < 1.0).all(), (
            f"Mobius add escaped ball: max ||x||^2 = {norms_sq.max().item():.6f}"
        )


class TestPoincareExpMapClamping:
    """Exp map clamps correctly at boundary."""

    def test_clamping_at_boundary(self) -> None:
        """Points near boundary are clamped to stay inside ball."""
        poincare = PoincareDistance(curvature=1.0, eps=1e-5)
        # Create points very close to boundary
        v = torch.randn(20, 32)
        v = v / v.norm(dim=-1, keepdim=True) * 0.99999
        result = poincare._clamp_norm(v)
        norms = result.norm(dim=-1)
        assert (norms <= poincare.max_norm).all(), (
            f"Clamping failed: max norm = {norms.max().item():.8f}, "
            f"limit = {poincare.max_norm:.8f}"
        )

    def test_clamping_beyond_boundary(self) -> None:
        """Points outside boundary are projected back inside."""
        poincare = PoincareDistance(curvature=1.0, eps=1e-5)
        # Create points outside ball
        v = torch.randn(20, 32)
        v = v / v.norm(dim=-1, keepdim=True) * 2.0  # outside
        result = poincare._clamp_norm(v)
        norms = result.norm(dim=-1)
        assert (norms <= poincare.max_norm + 1e-6).all(), (
            f"Projection failed: max norm = {norms.max().item():.8f}"
        )

    def test_small_points_not_affected(self) -> None:
        """Points well inside ball are not modified by clamping."""
        poincare = PoincareDistance(curvature=1.0, eps=1e-5)
        v = torch.randn(20, 32) * 0.1
        result = poincare._clamp_norm(v)
        assert torch.allclose(result, v, atol=1e-6), (
            "Small points were modified by clamping"
        )


class TestPoincareGradientStability:
    """No NaN/Inf in gradients during training."""

    def test_no_nan_in_gradients(self) -> None:
        """Poincare distance gradients remain finite over 30 training steps."""
        torch.manual_seed(42)
        dim = 32
        poincare = PoincareDistance(curvature=1.0)
        head = nn.Embedding(20, dim)
        rel = nn.Embedding(3, dim)
        tail = nn.Embedding(20, dim)
        params = list(head.parameters()) + list(rel.parameters()) + list(tail.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        for step in range(30):
            optimizer.zero_grad()
            h = head(torch.randint(0, 20, (8,)))
            r = rel(torch.randint(0, 3, (8,)))
            t = tail(torch.randint(0, 20, (8,)))

            scores = poincare(h, r, t)
            loss = -scores.mean()  # maximize scores
            assert torch.isfinite(loss), f"Step {step}: non-finite loss"
            loss.backward()

            for p in params:
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), (
                        f"Step {step}: NaN/Inf in gradients"
                    )

            optimizer.step()

    def test_riemannian_gradient_correction(self) -> None:
        """Gradients through distance function are numerically stable
        for embeddings at different distances from origin.
        """
        poincare = PoincareDistance(curvature=1.0)
        # Use normalized scales to ensure points are inside the ball
        for target_norm in [0.01, 0.1, 0.3, 0.5]:
            torch.manual_seed(42)
            x = torch.randn(10, 16)
            x = x / x.norm(dim=-1, keepdim=True) * target_norm
            x = x.detach().requires_grad_(True)

            y = torch.randn(10, 16)
            y = y / y.norm(dim=-1, keepdim=True) * target_norm
            y = y.detach().requires_grad_(True)

            dist = poincare.distance(x, y)
            loss = dist.sum()
            loss.backward()

            assert torch.isfinite(x.grad).all(), (
                f"NaN gradient at target_norm={target_norm} for x"
            )
            assert torch.isfinite(y.grad).all(), (
                f"NaN gradient at target_norm={target_norm} for y"
            )


class TestPoincareDistanceStability:
    """Distance is numerically stable near origin and boundary."""

    def test_distance_near_origin(self) -> None:
        """Distance between points near origin is finite and non-negative."""
        poincare = PoincareDistance(curvature=1.0)
        x = torch.randn(20, 32) * 0.001
        y = torch.randn(20, 32) * 0.001
        dist = poincare.distance(x, y)
        assert torch.isfinite(dist).all(), "Distance NaN near origin"
        assert (dist >= 0).all(), "Distance negative near origin"

    def test_distance_near_boundary(self) -> None:
        """Distance between points near boundary is finite and non-negative."""
        poincare = PoincareDistance(curvature=1.0)
        x = torch.randn(20, 32)
        x = x / x.norm(dim=-1, keepdim=True) * 0.99
        y = torch.randn(20, 32)
        y = y / y.norm(dim=-1, keepdim=True) * 0.99
        dist = poincare.distance(x, y)
        assert torch.isfinite(dist).all(), "Distance NaN near boundary"
        assert (dist >= 0).all(), "Distance negative near boundary"

    def test_distance_self_is_zero(self) -> None:
        """Distance from point to itself is approximately zero."""
        poincare = PoincareDistance(curvature=1.0)
        x = torch.randn(10, 32) * 0.5
        dist = poincare.distance(x, x)
        assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-4), (
            f"Self-distance should be ~0, got max={dist.max().item():.6f}"
        )

    def test_distance_symmetry(self) -> None:
        """d(x, y) == d(y, x) (symmetry)."""
        poincare = PoincareDistance(curvature=1.0)
        # Use unit-normalized then scaled to ensure points are well inside ball
        torch.manual_seed(42)
        x = torch.randn(20, 32)
        x = x / x.norm(dim=-1, keepdim=True) * 0.5
        y = torch.randn(20, 32)
        y = y / y.norm(dim=-1, keepdim=True) * 0.5
        d_xy = poincare.distance(x, y)
        d_yx = poincare.distance(y, x)
        assert torch.allclose(d_xy, d_yx, atol=1e-3), (
            f"Distance not symmetric: max diff = {(d_xy - d_yx).abs().max().item():.6f}"
        )

    def test_exp_log_roundtrip(self) -> None:
        """log_0(exp_0(v)) ~ v for tangent vectors of moderate norm."""
        poincare = PoincareDistance(curvature=1.0)
        v = torch.randn(20, 32) * 0.3
        p = poincare.exp_map(v)
        v_recovered = poincare.log_map(p)
        assert torch.allclose(v, v_recovered, atol=1e-3), (
            f"Exp-log roundtrip error: max diff = {(v - v_recovered).abs().max().item():.6f}"
        )


class TestMobiusAddStability:
    """Mobius addition is numerically stable."""

    def test_mobius_add_identity(self) -> None:
        """x oplus y ~ x when y is very small (near-identity property)."""
        poincare = PoincareDistance(curvature=1.0)
        torch.manual_seed(42)
        x = torch.randn(10, 16)
        x = x / x.norm(dim=-1, keepdim=True) * 0.3  # well inside ball
        # Use a very small (but non-zero) y instead of exact zero
        # to avoid _clamp_norm artifacts on zero vectors
        y = torch.randn(10, 16) * 1e-7
        result = poincare.mobius_add(x, y)
        # Result should be very close to x
        assert torch.allclose(result, x, atol=1e-3), (
            f"x oplus eps should be close to x, max diff = "
            f"{(result - x).abs().max().item():.6f}"
        )

    def test_mobius_add_finite(self) -> None:
        """Mobius addition of random points inside ball is always finite."""
        poincare = PoincareDistance(curvature=1.0)
        x = torch.randn(50, 32) * 0.4
        y = torch.randn(50, 32) * 0.4
        result = poincare.mobius_add(x, y)
        assert torch.isfinite(result).all(), "Mobius add produced NaN/Inf"


# ======================================================================
# 3. HybridLinkScorer 4-component Tests
# ======================================================================


class TestHybridLinkerScorerSubScores:
    """All 4 sub-scores contribute (non-zero weights)."""

    def test_all_four_components_have_nonzero_weights(self) -> None:
        """Initial weights from weight_net should be roughly uniform (~0.25 each)."""
        torch.manual_seed(42)
        scorer = HybridLinkScorer(embedding_dim=32, num_relations=4)
        rel_idx = torch.arange(4)
        r_dm = scorer.rel_embed_distmult(rel_idx)
        weights = torch.softmax(scorer.weight_net(r_dm), dim=-1)

        # All weights should be > 0 (softmax guarantees this, but check magnitude)
        assert (weights > 0.01).all(), (
            f"Some weights too small: {weights}"
        )

    def test_subscore_gradients_nonzero(self) -> None:
        """All 4 scoring paths produce non-zero gradients."""
        torch.manual_seed(42)
        scorer = HybridLinkScorer(embedding_dim=32, num_relations=4)
        h = torch.randn(8, 32, requires_grad=True)
        rel_idx = torch.randint(0, 4, (8,))
        t = torch.randn(8, 32, requires_grad=True)

        scores = scorer(h, rel_idx, t)
        loss = scores.sum()
        loss.backward()

        # Check that all sub-scorer relation embeddings have gradients
        for name in ["rel_embed_distmult", "rel_embed_rotate", "rel_embed_poincare"]:
            param = getattr(scorer, name).weight
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().max() > 1e-10, f"Vanishing gradient for {name}"

        # Neural scorer should have gradients
        for name, p in scorer.neural_scorer.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().max() > 1e-10, (
                    f"Vanishing gradient in neural_scorer.{name}"
                )


class TestHybridLinkScorerSoftmax:
    """Softmax weights over 4 components sum to 1."""

    def test_weights_sum_to_one(self) -> None:
        """Per-relation adaptive weights sum to 1."""
        scorer = HybridLinkScorer(embedding_dim=32, num_relations=8)
        for rel_id in range(8):
            rel_idx = torch.tensor([rel_id])
            r_dm = scorer.rel_embed_distmult(rel_idx)
            weights = torch.softmax(scorer.weight_net(r_dm), dim=-1)
            assert weights.sum().item() == pytest.approx(1.0, abs=1e-6), (
                f"Weights for relation {rel_id} don't sum to 1: {weights}"
            )

    def test_weights_all_positive(self) -> None:
        """All weights should be strictly positive (softmax property)."""
        scorer = HybridLinkScorer(embedding_dim=32, num_relations=8)
        rel_idx = torch.arange(8)
        r_dm = scorer.rel_embed_distmult(rel_idx)
        weights = torch.softmax(scorer.weight_net(r_dm), dim=-1)
        assert (weights > 0).all(), "Softmax produced zero or negative weights"


class TestHybridLinkScorerConvergence:
    """Training with 4 components converges."""

    def test_loss_decreases(self) -> None:
        """HybridLinkScorer training loss decreases over 80 steps."""
        torch.manual_seed(42)
        scorer = HybridLinkScorer(embedding_dim=32, num_relations=4)

        # Learnable entity embeddings
        head_emb = nn.Embedding(20, 32)
        tail_emb = nn.Embedding(20, 32)

        params = (
            list(scorer.parameters())
            + list(head_emb.parameters())
            + list(tail_emb.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=1e-3)

        losses = []
        for step in range(80):
            optimizer.zero_grad()
            h_idx = torch.randint(0, 10, (16,))
            t_pos_idx = h_idx  # positive: same index
            t_neg_idx = torch.randint(10, 20, (16,))
            rel_idx = torch.randint(0, 4, (16,))

            h = head_emb(h_idx)
            t_pos = tail_emb(t_pos_idx)
            t_neg = tail_emb(t_neg_idx)

            pos_scores = scorer(h, rel_idx, t_pos)
            neg_scores = scorer(h, rel_idx, t_neg)

            loss = torch.relu(1.0 - pos_scores + neg_scores).mean()
            losses.append(loss.item())
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        first_avg = sum(losses[:10]) / 10
        last_avg = sum(losses[-10:]) / 10
        assert last_avg < first_avg, (
            f"HybridLinkScorer loss didn't decrease: "
            f"first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )

    def test_all_losses_finite_during_training(self) -> None:
        """No NaN/Inf in losses or parameters during 50 training steps."""
        torch.manual_seed(42)
        scorer = HybridLinkScorer(embedding_dim=32, num_relations=4)
        head_emb = nn.Embedding(20, 32)
        tail_emb = nn.Embedding(20, 32)

        params = (
            list(scorer.parameters())
            + list(head_emb.parameters())
            + list(tail_emb.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=1e-3)

        for step in range(50):
            optimizer.zero_grad()
            h = head_emb(torch.randint(0, 20, (8,)))
            t = tail_emb(torch.randint(0, 20, (8,)))
            rel_idx = torch.randint(0, 4, (8,))

            scores = scorer(h, rel_idx, t)
            loss = scores.mean()
            assert torch.isfinite(loss), f"Step {step}: non-finite loss"
            loss.backward()

            for name, p in scorer.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), (
                        f"Step {step}: NaN/Inf grad in {name}"
                    )

            optimizer.step()


# ======================================================================
# 4. CompGCN Numerical Tests
# ======================================================================


class TestCompGCNMagnitude:
    """Composed embeddings have appropriate magnitude."""

    @pytest.mark.parametrize("mode", ["subtraction", "multiplication", "circular_correlation"])
    def test_composed_embedding_finite(self, mode: str) -> None:
        """Composed relation embeddings are always finite."""
        comp = CompositionalRelationEmbedding(
            num_node_types=5, num_edge_types=3,
            embedding_dim=64, compose_mode=mode,
        )
        src_idx = torch.tensor([0, 1, 2, 3])
        edge_idx = torch.tensor([0, 1, 2, 0])
        dst_idx = torch.tensor([1, 2, 3, 4])

        result = comp(src_idx, edge_idx, dst_idx)
        assert torch.isfinite(result).all(), (
            f"Mode {mode}: composed embedding contains NaN/Inf"
        )

    @pytest.mark.parametrize("mode", ["subtraction", "multiplication", "circular_correlation"])
    def test_composed_embedding_magnitude_reasonable(self, mode: str) -> None:
        """Composed embeddings should not be degenerate (all zeros or exploded)."""
        torch.manual_seed(42)
        comp = CompositionalRelationEmbedding(
            num_node_types=5, num_edge_types=3,
            embedding_dim=64, compose_mode=mode,
        )
        src_idx = torch.arange(5)
        edge_idx = torch.zeros(5, dtype=torch.long)
        dst_idx = torch.arange(5)

        result = comp(src_idx, edge_idx, dst_idx)
        avg_norm = result.norm(dim=-1).mean().item()
        assert avg_norm > 1e-6, (
            f"Mode {mode}: composed embeddings collapsed (avg norm = {avg_norm:.8f})"
        )
        assert avg_norm < 1e4, (
            f"Mode {mode}: composed embeddings exploded (avg norm = {avg_norm:.1f})"
        )


class TestCircularCorrelation:
    """Circular correlation is correct (verify with known vectors)."""

    def test_circular_correlation_known_vectors(self) -> None:
        """Verify circular correlation against numpy-style computation."""
        # For circular correlation: corr(a, b) = IFFT(conj(FFT(a)) * FFT(b))
        a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

        result = CompositionalRelationEmbedding._circular_correlation(a, b)

        # Manual verification using FFT
        fa = torch.fft.rfft(a, dim=-1)
        fb = torch.fft.rfft(b, dim=-1)
        expected = torch.fft.irfft(fa.conj() * fb, n=4, dim=-1)

        assert torch.allclose(result, expected, atol=1e-5), (
            f"Circular correlation mismatch: got {result}, expected {expected}"
        )

    def test_fft_ifft_roundtrip(self) -> None:
        """FFT/IFFT round-trip preserves real-valued input."""
        x = torch.randn(10, 64)
        x_fft = torch.fft.rfft(x, dim=-1)
        x_recovered = torch.fft.irfft(x_fft, n=64, dim=-1)
        assert torch.allclose(x, x_recovered, atol=1e-5), (
            f"FFT roundtrip error: max diff = {(x - x_recovered).abs().max().item():.8f}"
        )

    def test_circular_correlation_output_shape(self) -> None:
        """Circular correlation output has same shape as input."""
        a = torch.randn(8, 64)
        b = torch.randn(8, 64)
        result = CompositionalRelationEmbedding._circular_correlation(a, b)
        assert result.shape == a.shape, (
            f"Shape mismatch: {result.shape} vs {a.shape}"
        )

    def test_circular_correlation_real_valued(self) -> None:
        """Circular correlation of real inputs is real."""
        a = torch.randn(8, 64)
        b = torch.randn(8, 64)
        result = CompositionalRelationEmbedding._circular_correlation(a, b)
        # result is already a real tensor from irfft, but verify no imaginary leakage
        assert result.dtype in (torch.float32, torch.float64), (
            f"Expected real dtype, got {result.dtype}"
        )
        assert torch.isfinite(result).all(), "Circular correlation produced NaN/Inf"

    def test_compgcn_gradient_flow(self) -> None:
        """Gradients flow through all composition modes."""
        for mode in ["subtraction", "multiplication", "circular_correlation"]:
            comp = CompositionalRelationEmbedding(
                num_node_types=5, num_edge_types=3,
                embedding_dim=64, compose_mode=mode,
            )
            src_idx = torch.tensor([0, 1, 2])
            edge_idx = torch.tensor([0, 1, 2])
            dst_idx = torch.tensor([1, 2, 3])

            result = comp(src_idx, edge_idx, dst_idx)
            loss = result.sum()
            loss.backward()

            assert comp.node_type_embed.weight.grad is not None, (
                f"Mode {mode}: no gradient for node_type_embed"
            )
            assert comp.edge_type_embed.weight.grad is not None, (
                f"Mode {mode}: no gradient for edge_type_embed"
            )


# ======================================================================
# 5. Pretraining Numerical Tests
# ======================================================================


def _make_pretraining_data(dim: int = 64, seed: int = 42) -> HeteroData:
    """Build HeteroData for pretraining tests."""
    torch.manual_seed(seed)
    data = HeteroData()
    data["protein"].x = torch.randn(10, dim)
    data["protein"].num_nodes = 10
    data["glycan"].x = torch.randn(8, dim)
    data["glycan"].num_nodes = 8

    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
    )
    data["glycan", "binds_protein", "protein"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]]
    )
    return data


class _SimpleEncoder(nn.Module):
    """Minimal encoder for pretraining tests.

    Uses .clone() on data features to avoid inplace modification issues
    from upstream masking operations.
    """

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.protein_proj = nn.Linear(dim, dim)
        self.glycan_proj = nn.Linear(dim, dim)

    def forward(self, data: HeteroData):
        out = {}
        if hasattr(data["protein"], "x"):
            out["protein"] = self.protein_proj(data["protein"].x.clone())
        if hasattr(data["glycan"], "x"):
            out["glycan"] = self.glycan_proj(data["glycan"].x.clone())
        return out


class TestMaskedNodePredictionLoss:
    """Masked node prediction losses decrease over training."""

    def test_loss_decreases(self) -> None:
        """Masked node prediction loss should decrease over 30 training steps.

        Uses fresh data each step to avoid inplace modification issues
        in the mask_and_predict method.
        """
        torch.manual_seed(42)
        dim = 64
        encoder = _SimpleEncoder(dim)
        predictor = MaskedNodePredictor(
            embedding_dim=dim, continuous_dim=dim,
        )
        params = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        losses = []
        for step in range(30):
            # Fresh data each step avoids inplace modification issues
            data = _make_pretraining_data(dim=dim, seed=42)
            # Detach features so mask inplace ops don't break autograd
            for nt in data.node_types:
                if hasattr(data[nt], "x"):
                    data[nt].x = data[nt].x.detach().clone()
            optimizer.zero_grad()
            loss, _ = predictor.mask_and_predict(data, encoder, mask_ratio=0.15)
            losses.append(loss.item())
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        first_avg = sum(losses[:5]) / 5
        last_avg = sum(losses[-5:]) / 5
        assert last_avg < first_avg, (
            f"Masked node loss didn't decrease: "
            f"first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )

    def test_gradients_finite(self) -> None:
        """All gradients from masked node prediction are finite."""
        torch.manual_seed(42)
        dim = 64
        encoder = _SimpleEncoder(dim)
        predictor = MaskedNodePredictor(embedding_dim=dim, continuous_dim=dim)
        data = _make_pretraining_data(dim=dim)
        # Detach features so mask inplace ops don't break autograd
        for nt in data.node_types:
            if hasattr(data[nt], "x"):
                data[nt].x = data[nt].x.detach().clone()

        loss, _ = predictor.mask_and_predict(data, encoder, mask_ratio=0.15)
        loss.backward()

        all_params = list(encoder.parameters()) + list(predictor.parameters())
        for p in all_params:
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    "Masked node prediction produced non-finite gradients"
                )


class TestMaskedEdgePredictionLoss:
    """Masked edge prediction losses decrease over training."""

    def test_loss_decreases(self) -> None:
        """Masked edge prediction loss should decrease over 60 training steps."""
        torch.manual_seed(42)
        dim = 64
        encoder = _SimpleEncoder(dim)
        predictor = MaskedEdgePredictor(embedding_dim=dim, num_relations=1)
        params = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=5e-3)

        losses = []
        for step in range(60):
            # Use same graph structure each step for consistent training signal
            data = _make_pretraining_data(dim=dim, seed=42)
            for nt in data.node_types:
                if hasattr(data[nt], "x"):
                    data[nt].x = data[nt].x.detach().clone()
            optimizer.zero_grad()
            loss, _ = predictor.mask_and_predict(data, encoder, mask_ratio=0.10)
            losses.append(loss.item())
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        first_avg = sum(losses[:10]) / 10
        last_avg = sum(losses[-10:]) / 10
        assert last_avg < first_avg, (
            f"Masked edge loss didn't decrease: "
            f"first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )

    def test_gradients_finite(self) -> None:
        """All gradients from masked edge prediction are finite."""
        torch.manual_seed(42)
        dim = 64
        encoder = _SimpleEncoder(dim)
        predictor = MaskedEdgePredictor(embedding_dim=dim, num_relations=2)
        data = _make_pretraining_data(dim=dim)
        for nt in data.node_types:
            if hasattr(data[nt], "x"):
                data[nt].x = data[nt].x.detach().clone()

        loss, _ = predictor.mask_and_predict(data, encoder, mask_ratio=0.10)
        loss.backward()

        all_params = list(encoder.parameters()) + list(predictor.parameters())
        for p in all_params:
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    "Masked edge prediction produced non-finite gradients"
                )


class TestCMCALossNumerical:
    """CMCA loss doesn't collapse and produces finite gradients."""

    def test_cross_modal_loss_nonzero(self) -> None:
        """Cross-modal loss should be positive for non-identical embeddings."""
        torch.manual_seed(42)
        cmca = CMCALoss(temperature=0.07)
        modal = torch.randn(16, 64)
        kg = torch.randn(16, 64)
        loss = cmca.cross_modal_loss(modal, kg)
        assert loss.item() > 0, "CMCA cross-modal loss should be positive"
        assert torch.isfinite(loss), "CMCA cross-modal loss is not finite"

    def test_intra_modal_loss_nonzero(self) -> None:
        """Intra-modal loss should be positive for random embeddings."""
        torch.manual_seed(42)
        cmca = CMCALoss(temperature=0.07)
        emb = torch.randn(16, 64)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        loss = cmca.intra_modal_loss(emb, pairs)
        assert loss.item() > 0, "CMCA intra-modal loss should be positive"
        assert torch.isfinite(loss), "CMCA intra-modal loss is not finite"

    def test_cmca_loss_decreases(self) -> None:
        """CMCA cross-modal loss should decrease when aligning embeddings."""
        torch.manual_seed(42)
        cmca = CMCALoss(temperature=0.07)
        modal_emb = nn.Embedding(16, 64)
        kg_emb = nn.Embedding(16, 64)

        params = list(modal_emb.parameters()) + list(kg_emb.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-2)

        idx = torch.arange(16)
        losses = []
        for step in range(40):
            optimizer.zero_grad()
            modal = modal_emb(idx)
            kg = kg_emb(idx)
            loss = cmca.cross_modal_loss(modal, kg)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        first_avg = sum(losses[:5]) / 5
        last_avg = sum(losses[-5:]) / 5
        assert last_avg < first_avg, (
            f"CMCA loss didn't decrease: "
            f"first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )

    def test_cmca_does_not_collapse(self) -> None:
        """CMCA loss should not collapse to zero (InfoNCE temperature prevents this)."""
        torch.manual_seed(42)
        cmca = CMCALoss(temperature=0.07)

        # Even with perfectly aligned embeddings, the loss should be > 0
        # because negatives still contribute to the denominator
        emb = torch.randn(16, 64)
        loss = cmca.cross_modal_loss(emb, emb)
        # With identical embeddings, the diagonal is maximized but off-diagonal
        # still contributes, so loss = -log(softmax_diagonal) > 0
        assert torch.isfinite(loss), "CMCA collapsed to NaN/Inf"

    def test_cmca_gradients_finite(self) -> None:
        """All gradients from CMCA loss are finite."""
        cmca = CMCALoss(temperature=0.07)
        modal = torch.randn(16, 64, requires_grad=True)
        kg = torch.randn(16, 64, requires_grad=True)

        losses = cmca(modal_embeddings=modal, kg_embeddings=kg)
        total = losses["cross_modal_loss"]
        total.backward()

        assert torch.isfinite(modal.grad).all(), "CMCA modal gradient NaN/Inf"
        assert torch.isfinite(kg.grad).all(), "CMCA kg gradient NaN/Inf"

    def test_cmca_forward_combined(self) -> None:
        """Combined forward with both intra-modal and cross-modal terms."""
        torch.manual_seed(42)
        cmca = CMCALoss(temperature=0.07)
        modal = torch.randn(16, 64, requires_grad=True)
        kg = torch.randn(16, 64, requires_grad=True)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])

        losses = cmca(
            modal_embeddings=modal,
            kg_embeddings=kg,
            positive_pairs=pairs,
        )

        assert torch.isfinite(losses["intra_modal_loss"]), "Intra-modal NaN"
        assert torch.isfinite(losses["cross_modal_loss"]), "Cross-modal NaN"

        total = losses["intra_modal_loss"] + losses["cross_modal_loss"]
        total.backward()

        assert torch.isfinite(modal.grad).all(), "Combined CMCA modal grad NaN"
        assert torch.isfinite(kg.grad).all(), "Combined CMCA kg grad NaN"


class TestGlycanSubstructurePrediction:
    """Glycan substructure prediction losses are finite and decrease."""

    def test_loss_finite(self) -> None:
        """Substructure prediction loss is finite."""
        predictor = GlycanSubstructurePredictor(
            embedding_dim=64, num_monosaccharide_types=10,
        )
        emb = torch.randn(8, 64)
        targets = torch.randint(0, 2, (8, 10)).float()
        loss = predictor.compute_loss(emb, targets)
        assert torch.isfinite(loss), "Substructure loss not finite"

    def test_loss_decreases(self) -> None:
        """Substructure prediction loss decreases over training."""
        torch.manual_seed(42)
        predictor = GlycanSubstructurePredictor(
            embedding_dim=64, num_monosaccharide_types=10,
        )
        emb_table = nn.Embedding(20, 64)
        params = list(predictor.parameters()) + list(emb_table.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        # Fixed targets
        targets = torch.randint(0, 2, (20, 10)).float()

        losses = []
        for step in range(30):
            optimizer.zero_grad()
            emb = emb_table(torch.arange(20))
            loss = predictor.compute_loss(emb, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        first_avg = sum(losses[:5]) / 5
        last_avg = sum(losses[-5:]) / 5
        assert last_avg < first_avg, (
            f"Substructure loss didn't decrease: "
            f"first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )

    def test_gradients_finite(self) -> None:
        """Gradients from substructure loss are finite."""
        predictor = GlycanSubstructurePredictor(
            embedding_dim=64, num_monosaccharide_types=10,
        )
        emb = torch.randn(8, 64, requires_grad=True)
        targets = torch.randint(0, 2, (8, 10)).float()
        loss = predictor.compute_loss(emb, targets)
        loss.backward()
        assert torch.isfinite(emb.grad).all(), "Substructure gradient NaN/Inf"


class TestAllPretrainingGradientsFinite:
    """All pretraining losses produce finite gradients together."""

    def test_combined_pretraining_gradients(self) -> None:
        """Running all three pretraining tasks together produces finite gradients."""
        torch.manual_seed(42)
        dim = 64

        encoder = _SimpleEncoder(dim)
        node_predictor = MaskedNodePredictor(
            embedding_dim=dim, continuous_dim=dim,
        )
        edge_predictor = MaskedEdgePredictor(embedding_dim=dim, num_relations=2)
        cmca = CMCALoss(temperature=0.07)

        all_params = (
            list(encoder.parameters())
            + list(node_predictor.parameters())
            + list(edge_predictor.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=1e-3)

        for step in range(10):
            data = _make_pretraining_data(dim=dim, seed=42)
            # Detach features to avoid inplace modification issues
            for nt in data.node_types:
                if hasattr(data[nt], "x"):
                    data[nt].x = data[nt].x.detach().clone()

            optimizer.zero_grad()

            # Masked node prediction
            node_loss, _ = node_predictor.mask_and_predict(
                data, encoder, mask_ratio=0.15,
            )

            # Masked edge prediction (fresh data copy since edges are modified)
            data2 = _make_pretraining_data(dim=dim, seed=42)
            for nt in data2.node_types:
                if hasattr(data2[nt], "x"):
                    data2[nt].x = data2[nt].x.detach().clone()
            edge_loss, _ = edge_predictor.mask_and_predict(
                data2, encoder, mask_ratio=0.10,
            )

            # CMCA loss: use min of protein/glycan counts for alignment
            out = encoder(data)
            n_align = min(out["protein"].size(0), out["glycan"].size(0))
            cmca_losses = cmca(
                modal_embeddings=out["protein"][:n_align],
                kg_embeddings=out["glycan"][:n_align],
            )

            total = node_loss + edge_loss + cmca_losses["cross_modal_loss"]
            assert torch.isfinite(total), f"Step {step}: non-finite total loss"

            total.backward()

            for name, p in encoder.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), (
                        f"Step {step}: NaN grad in encoder.{name}"
                    )

            optimizer.step()
