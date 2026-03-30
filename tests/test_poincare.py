"""Unit tests for PoincareDistance (Poincare ball model).

Tests cover:
  - Distance metric properties: non-negative, zero iff equal, symmetric, triangle inequality
  - Exponential map: result inside ball
  - Log-exp round-trip: log_0(exp_0(v)) approx v
  - Mobius addition: x + 0 = x
  - Numerical stability: near-boundary points, no NaN/Inf
  - Gradient flow: finite gradients
  - Curvature parameter effect
  - Forward scoring function
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.embedding.models.poincare import PoincareDistance


# ======================================================================
# Constants
# ======================================================================

DIM = 16
BATCH = 8


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def poincare() -> PoincareDistance:
    """Standard PoincareDistance with default curvature=1."""
    return PoincareDistance(curvature=1.0)


@pytest.fixture()
def poincare_c2() -> PoincareDistance:
    """PoincareDistance with curvature=2."""
    return PoincareDistance(curvature=2.0)


@pytest.fixture()
def small_points() -> tuple[torch.Tensor, torch.Tensor]:
    """Two batches of small-norm points safely inside the unit ball."""
    torch.manual_seed(42)
    x = torch.randn(BATCH, DIM) * 0.3
    y = torch.randn(BATCH, DIM) * 0.3
    return x, y


# ======================================================================
# Distance metric properties
# ======================================================================


class TestDistanceProperties:
    """Tests that Poincare distance satisfies metric axioms."""

    def test_non_negative(self, poincare: PoincareDistance, small_points) -> None:
        """d(x, y) >= 0 for all x, y."""
        x, y = small_points
        d = poincare.distance(x, y)
        assert (d >= -1e-6).all(), f"Negative distances found: {d}"

    def test_zero_iff_equal(self, poincare: PoincareDistance) -> None:
        """d(x, x) == 0."""
        torch.manual_seed(7)
        x = torch.randn(BATCH, DIM) * 0.3
        d = poincare.distance(x, x)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-4)

    def test_positive_for_distinct(self, poincare: PoincareDistance) -> None:
        """d(x, y) > 0 when x != y."""
        torch.manual_seed(42)
        x = torch.randn(BATCH, DIM) * 0.3
        y = x + 0.1  # shift so distinct
        d = poincare.distance(x, y)
        assert (d > 1e-6).all()

    def test_symmetric(self, poincare: PoincareDistance, small_points) -> None:
        """d(x, y) == d(y, x)."""
        x, y = small_points
        d_xy = poincare.distance(x, y)
        d_yx = poincare.distance(y, x)
        assert torch.allclose(d_xy, d_yx, atol=1e-5)

    def test_triangle_inequality(self, poincare: PoincareDistance) -> None:
        """d(x, z) <= d(x, y) + d(y, z)."""
        torch.manual_seed(123)
        x = torch.randn(BATCH, DIM) * 0.2
        y = torch.randn(BATCH, DIM) * 0.2
        z = torch.randn(BATCH, DIM) * 0.2

        d_xz = poincare.distance(x, z)
        d_xy = poincare.distance(x, y)
        d_yz = poincare.distance(y, z)

        assert (d_xz <= d_xy + d_yz + 1e-4).all(), (
            "Triangle inequality violated"
        )


# ======================================================================
# Exponential map
# ======================================================================


class TestExponentialMap:
    """Tests for exp_map (exponential map from tangent space to ball)."""

    def test_exp_map_inside_ball(self, poincare: PoincareDistance) -> None:
        """exp_0(v) lands inside the Poincare ball: c * ||result||^2 < 1."""
        torch.manual_seed(42)
        v = torch.randn(BATCH, DIM) * 2.0  # large tangent vectors
        result = poincare.exp_map(v)

        norms_sq = (result * result).sum(dim=-1)
        # c * ||x||^2 must be < 1
        assert (poincare.c * norms_sq < 1.0).all()

    def test_exp_map_zero_gives_origin(self, poincare: PoincareDistance) -> None:
        """exp_0(0) = 0 (the origin)."""
        v = torch.zeros(BATCH, DIM)
        result = poincare.exp_map(v)
        # tanh(0) = 0, so result should be near zero
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_exp_map_from_nonorigin(self, poincare: PoincareDistance) -> None:
        """exp_x(v) stays inside ball when x is a non-origin base point."""
        torch.manual_seed(42)
        x = torch.randn(BATCH, DIM) * 0.3  # inside ball
        v = torch.randn(BATCH, DIM) * 0.5
        result = poincare.exp_map(v, x=x)

        norms_sq = (result * result).sum(dim=-1)
        assert (poincare.c * norms_sq < 1.0).all()

    def test_exp_map_large_tangent(self, poincare: PoincareDistance) -> None:
        """Even for very large tangent vectors, exp_0(v) stays in the ball."""
        v = torch.randn(BATCH, DIM) * 100.0
        result = poincare.exp_map(v)

        norms_sq = (result * result).sum(dim=-1)
        assert (poincare.c * norms_sq < 1.0).all()
        assert torch.isfinite(result).all()


# ======================================================================
# Log-Exp round-trip
# ======================================================================


class TestLogExpRoundTrip:
    """Tests for log_0(exp_0(v)) approx v."""

    def test_round_trip_small_vectors(self, poincare: PoincareDistance) -> None:
        """log_0(exp_0(v)) approx v for small tangent vectors."""
        torch.manual_seed(42)
        v = torch.randn(BATCH, DIM) * 0.3
        recovered = poincare.log_map(poincare.exp_map(v))
        assert torch.allclose(recovered, v, atol=1e-4), (
            f"Max error: {(recovered - v).abs().max().item()}"
        )

    def test_round_trip_moderate_vectors(self, poincare: PoincareDistance) -> None:
        """log_0(exp_0(v)) approx v for moderate tangent vectors."""
        torch.manual_seed(99)
        v = torch.randn(BATCH, DIM) * 1.0
        recovered = poincare.log_map(poincare.exp_map(v))
        assert torch.allclose(recovered, v, atol=1e-3), (
            f"Max error: {(recovered - v).abs().max().item()}"
        )

    def test_exp_log_round_trip(self, poincare: PoincareDistance) -> None:
        """exp_0(log_0(y)) approx y for points in the ball."""
        torch.manual_seed(42)
        y = torch.randn(BATCH, DIM) * 0.4  # inside ball
        y = poincare._clamp_norm(y)  # ensure safely inside
        recovered = poincare.exp_map(poincare.log_map(y))
        assert torch.allclose(recovered, y, atol=1e-3)


# ======================================================================
# Mobius addition
# ======================================================================


class TestMobiusAddition:
    """Tests for Mobius addition in the Poincare ball."""

    def test_identity_right(self, poincare: PoincareDistance) -> None:
        """x + 0 = x (zero is the identity element for Mobius addition)."""
        torch.manual_seed(42)
        # Use small scale to keep ||x|| < 1/sqrt(c) (ball boundary)
        x = torch.randn(BATCH, DIM) * 0.1
        zero = torch.zeros_like(x)
        result = poincare.mobius_add(x, zero)
        assert torch.allclose(result, x, atol=1e-5)

    def test_identity_left(self, poincare: PoincareDistance) -> None:
        """0 + x = x."""
        torch.manual_seed(42)
        x = torch.randn(BATCH, DIM) * 0.1
        zero = torch.zeros_like(x)
        result = poincare.mobius_add(zero, x)
        assert torch.allclose(result, x, atol=1e-5)

    def test_result_inside_ball(self, poincare: PoincareDistance) -> None:
        """Mobius addition result stays inside the ball."""
        torch.manual_seed(42)
        x = torch.randn(BATCH, DIM) * 0.5
        y = torch.randn(BATCH, DIM) * 0.5
        result = poincare.mobius_add(x, y)

        norms_sq = (result * result).sum(dim=-1)
        assert (poincare.c * norms_sq < 1.0).all()


# ======================================================================
# Numerical stability
# ======================================================================


class TestNumericalStability:
    """Tests for numerical stability near ball boundary."""

    def test_near_boundary_distance(self, poincare: PoincareDistance) -> None:
        """Distance between near-boundary points is finite and non-negative."""
        # Create points very close to boundary (||x|| close to 1/sqrt(c) = 1.0)
        torch.manual_seed(42)
        direction = torch.randn(BATCH, DIM)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        x = direction * 0.999  # very close to boundary
        y = -direction * 0.999

        d = poincare.distance(x, y)
        assert torch.isfinite(d).all(), "NaN or Inf in near-boundary distance"
        assert (d >= 0).all()

    def test_near_boundary_exp_map(self, poincare: PoincareDistance) -> None:
        """exp_map at near-boundary base point produces finite results."""
        torch.manual_seed(42)
        direction = torch.randn(BATCH, DIM)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        x = direction * 0.99  # near boundary
        v = torch.randn(BATCH, DIM) * 0.1

        result = poincare.exp_map(v, x=x)
        assert torch.isfinite(result).all()

    def test_near_boundary_log_map(self, poincare: PoincareDistance) -> None:
        """log_map at near-boundary point produces finite tangent vectors."""
        torch.manual_seed(42)
        direction = torch.randn(BATCH, DIM)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        y = direction * 0.99

        result = poincare.log_map(y)
        assert torch.isfinite(result).all()

    def test_near_boundary_mobius_add(self, poincare: PoincareDistance) -> None:
        """Mobius addition of near-boundary points is finite."""
        torch.manual_seed(42)
        direction = torch.randn(BATCH, DIM)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        x = direction * 0.99
        y = -direction * 0.5

        result = poincare.mobius_add(x, y)
        assert torch.isfinite(result).all()

    def test_zero_vector_distance(self, poincare: PoincareDistance) -> None:
        """Distance from origin to a point is finite."""
        torch.manual_seed(42)
        origin = torch.zeros(BATCH, DIM)
        y = torch.randn(BATCH, DIM) * 0.5

        d = poincare.distance(origin, y)
        assert torch.isfinite(d).all()
        assert (d >= 0).all()

    def test_forward_no_nan_inf(self, poincare: PoincareDistance) -> None:
        """Forward scoring produces no NaN or Inf."""
        torch.manual_seed(42)
        head = torch.randn(BATCH, DIM) * 0.5
        relation = torch.randn(BATCH, DIM) * 0.3
        tail = torch.randn(BATCH, DIM) * 0.5

        scores = poincare(head, relation, tail)
        assert torch.isfinite(scores).all()


# ======================================================================
# Gradient flow
# ======================================================================


class TestGradientFlow:
    """Tests for gradient flow through Poincare operations."""

    def test_distance_gradient(self, poincare: PoincareDistance) -> None:
        """Gradients through distance are finite."""
        torch.manual_seed(42)
        x = (torch.randn(BATCH, DIM) * 0.1).requires_grad_(True)
        y = (torch.randn(BATCH, DIM) * 0.1).requires_grad_(True)

        d = poincare.distance(x, y)
        d.sum().backward()

        assert x.grad is not None
        assert y.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(y.grad).all()

    def test_exp_map_gradient(self, poincare: PoincareDistance) -> None:
        """Gradients through exp_map are finite."""
        torch.manual_seed(42)
        v = torch.randn(BATCH, DIM, requires_grad=True)
        result = poincare.exp_map(v)
        result.sum().backward()

        assert v.grad is not None
        assert torch.isfinite(v.grad).all()

    def test_forward_gradient(self, poincare: PoincareDistance) -> None:
        """Gradients through forward scoring are finite."""
        torch.manual_seed(42)
        head = torch.randn(BATCH, DIM, requires_grad=True)
        relation = torch.randn(BATCH, DIM, requires_grad=True)
        tail = torch.randn(BATCH, DIM, requires_grad=True)

        scores = poincare(head, relation, tail)
        scores.sum().backward()

        assert head.grad is not None
        assert relation.grad is not None
        assert tail.grad is not None
        assert torch.isfinite(head.grad).all()
        assert torch.isfinite(relation.grad).all()
        assert torch.isfinite(tail.grad).all()

    def test_near_boundary_gradient(self, poincare: PoincareDistance) -> None:
        """Gradients are finite even for near-boundary points."""
        torch.manual_seed(42)
        direction = torch.randn(BATCH, DIM)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        x = (direction * 0.95).requires_grad_(True)
        y = (torch.randn(BATCH, DIM) * 0.1).requires_grad_(True)

        d = poincare.distance(x, y)
        d.sum().backward()

        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(y.grad).all()


# ======================================================================
# Curvature parameter effect
# ======================================================================


class TestCurvature:
    """Tests for curvature parameter behavior."""

    def test_higher_curvature_larger_distance(
        self, poincare: PoincareDistance, poincare_c2: PoincareDistance
    ) -> None:
        """Larger curvature produces larger distances for same points."""
        torch.manual_seed(42)
        x = torch.randn(BATCH, DIM) * 0.2
        y = torch.randn(BATCH, DIM) * 0.2

        d_c1 = poincare.distance(x, y)
        d_c2 = poincare_c2.distance(x, y)

        # With curvature c, d_c = (2/sqrt(c)) * arctanh(...).
        # For small norms, higher c stretches more due to the ball being smaller.
        # Generally larger c means larger distances for same Euclidean positions.
        assert (d_c2 > d_c1 - 1e-4).all()

    def test_invalid_curvature(self) -> None:
        """Non-positive curvature raises ValueError."""
        with pytest.raises(ValueError, match="Curvature must be positive"):
            PoincareDistance(curvature=0.0)
        with pytest.raises(ValueError, match="Curvature must be positive"):
            PoincareDistance(curvature=-1.0)

    def test_max_norm_depends_on_curvature(self) -> None:
        """max_norm = (1 - eps) / sqrt(c)."""
        pd = PoincareDistance(curvature=4.0)
        expected = (1.0 - pd.eps) / (4.0 ** 0.5)
        assert pd.max_norm == pytest.approx(expected)


# ======================================================================
# Forward scoring
# ======================================================================


class TestForwardScoring:
    """Tests for the forward scoring function S_hyp(h, r, t) = -d_c(...)."""

    def test_output_shape(self, poincare: PoincareDistance) -> None:
        """Forward returns [batch] tensor."""
        torch.manual_seed(42)
        head = torch.randn(BATCH, DIM)
        rel = torch.randn(BATCH, DIM)
        tail = torch.randn(BATCH, DIM)

        scores = poincare(head, rel, tail)
        assert scores.shape == (BATCH,)

    def test_scores_are_non_positive(self, poincare: PoincareDistance) -> None:
        """Scores = -distance, so scores <= 0."""
        torch.manual_seed(42)
        head = torch.randn(BATCH, DIM) * 0.3
        rel = torch.randn(BATCH, DIM) * 0.3
        tail = torch.randn(BATCH, DIM) * 0.3

        scores = poincare(head, rel, tail)
        assert (scores <= 1e-5).all(), "Scores should be non-positive"

    def test_perfect_triple_high_score(self, poincare: PoincareDistance) -> None:
        """When head + relation = tail (in tangent space), score is near 0 (maximum)."""
        torch.manual_seed(42)
        head = torch.randn(1, DIM) * 0.3
        relation = torch.randn(1, DIM) * 0.1
        # If tail = head + relation, then exp_0(h + r) = exp_0(t), so distance is 0
        tail = head + relation

        score = poincare(head, relation, tail)
        # Score should be close to 0 (maximum possible score)
        assert score.item() == pytest.approx(0.0, abs=1e-3)

    def test_relation_affects_score(self, poincare: PoincareDistance) -> None:
        """Different relations yield different scores for same (h, t)."""
        torch.manual_seed(42)
        head = torch.randn(1, DIM) * 0.3
        tail = torch.randn(1, DIM) * 0.3
        r1 = torch.randn(1, DIM) * 0.3
        r2 = torch.randn(1, DIM) * 0.3

        s1 = poincare(head, r1, tail)
        s2 = poincare(head, r2, tail)

        assert not torch.allclose(s1, s2, atol=1e-6)
