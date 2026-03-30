"""Unit tests for CompositionalRelationEmbedding (CompGCN-style relation composition).

Tests cover:
  - Subtraction compose mode: e_src - e_edge + e_dst
  - Multiplication compose mode: e_src * e_edge * e_dst
  - Circular correlation compose mode: IFFT(conj(FFT(a)) * FFT(b))
  - Output dimension matches embedding_dim
  - Different modes produce different embeddings
  - Gradient flow through compose operations
  - Invalid compose mode rejection
  - Batch and scalar input handling
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.embedding.models.compgcn_rel import CompositionalRelationEmbedding


# ======================================================================
# Constants
# ======================================================================

NUM_NODE_TYPES = 5
NUM_EDGE_TYPES = 8
EMBEDDING_DIM = 32
BATCH = 4


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def sub_model() -> CompositionalRelationEmbedding:
    """CompGCN with subtraction compose mode."""
    torch.manual_seed(42)
    return CompositionalRelationEmbedding(
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        embedding_dim=EMBEDDING_DIM,
        compose_mode="subtraction",
    )


@pytest.fixture()
def mul_model() -> CompositionalRelationEmbedding:
    """CompGCN with multiplication compose mode."""
    torch.manual_seed(42)
    return CompositionalRelationEmbedding(
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        embedding_dim=EMBEDDING_DIM,
        compose_mode="multiplication",
    )


@pytest.fixture()
def corr_model() -> CompositionalRelationEmbedding:
    """CompGCN with circular correlation compose mode."""
    torch.manual_seed(42)
    return CompositionalRelationEmbedding(
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        embedding_dim=EMBEDDING_DIM,
        compose_mode="circular_correlation",
    )


@pytest.fixture()
def batch_indices():
    """Batch of type indices for testing."""
    src = torch.tensor([0, 1, 2, 3])
    edge = torch.tensor([0, 2, 4, 6])
    dst = torch.tensor([1, 3, 4, 0])
    return src, edge, dst


# ======================================================================
# Subtraction mode
# ======================================================================


class TestSubtractionMode:
    """Tests for subtraction compose: e_src - e_edge + e_dst."""

    def test_subtraction_correctness(
        self, sub_model: CompositionalRelationEmbedding, batch_indices
    ) -> None:
        """Output equals e_src - e_edge + e_dst."""
        src, edge, dst = batch_indices
        result = sub_model(src, edge, dst)

        # Manual computation
        e_src = sub_model.node_type_embed(src)
        e_edge = sub_model.edge_type_embed(edge)
        e_dst = sub_model.node_type_embed(dst)
        expected = e_src - e_edge + e_dst

        assert torch.allclose(result, expected, atol=1e-6)

    def test_subtraction_zero_edge(
        self, sub_model: CompositionalRelationEmbedding
    ) -> None:
        """When edge embedding is zero, result = e_src + e_dst."""
        # Manually zero out edge embedding
        with torch.no_grad():
            sub_model.edge_type_embed.weight[0].zero_()

        src = torch.tensor([1])
        edge = torch.tensor([0])  # zeroed out
        dst = torch.tensor([2])

        result = sub_model(src, edge, dst)
        e_src = sub_model.node_type_embed(src)
        e_dst = sub_model.node_type_embed(dst)
        expected = e_src + e_dst

        assert torch.allclose(result, expected, atol=1e-6)


# ======================================================================
# Multiplication mode
# ======================================================================


class TestMultiplicationMode:
    """Tests for multiplication compose: e_src * e_edge * e_dst."""

    def test_multiplication_correctness(
        self, mul_model: CompositionalRelationEmbedding, batch_indices
    ) -> None:
        """Output equals e_src * e_edge * e_dst."""
        src, edge, dst = batch_indices
        result = mul_model(src, edge, dst)

        e_src = mul_model.node_type_embed(src)
        e_edge = mul_model.edge_type_embed(edge)
        e_dst = mul_model.node_type_embed(dst)
        expected = e_src * e_edge * e_dst

        assert torch.allclose(result, expected, atol=1e-6)

    def test_multiplication_commutative_src_dst(
        self, mul_model: CompositionalRelationEmbedding
    ) -> None:
        """Multiplication is commutative w.r.t. swapping src and dst indices."""
        src = torch.tensor([0, 1, 2])
        edge = torch.tensor([0, 1, 2])
        dst = torch.tensor([3, 4, 0])

        result_fwd = mul_model(src, edge, dst)
        result_rev = mul_model(dst, edge, src)

        assert torch.allclose(result_fwd, result_rev, atol=1e-6)


# ======================================================================
# Circular correlation mode
# ======================================================================


class TestCircularCorrelationMode:
    """Tests for circular correlation compose: IFFT(conj(FFT(e_src * e_edge)) * FFT(e_dst))."""

    def test_circular_correlation_correctness(
        self, corr_model: CompositionalRelationEmbedding, batch_indices
    ) -> None:
        """Output matches manual circular correlation computation."""
        src, edge, dst = batch_indices
        result = corr_model(src, edge, dst)

        e_src = corr_model.node_type_embed(src)
        e_edge = corr_model.edge_type_embed(edge)
        e_dst = corr_model.node_type_embed(dst)

        # Manual circular correlation
        a = e_src * e_edge
        fa = torch.fft.rfft(a, dim=-1)
        fb = torch.fft.rfft(e_dst, dim=-1)
        expected = torch.fft.irfft(fa.conj() * fb, n=a.size(-1), dim=-1)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_circular_correlation_real_output(
        self, corr_model: CompositionalRelationEmbedding, batch_indices
    ) -> None:
        """Circular correlation output is real-valued (no complex residue)."""
        src, edge, dst = batch_indices
        result = corr_model(src, edge, dst)
        assert result.dtype in (torch.float32, torch.float64)
        assert torch.isfinite(result).all()


# ======================================================================
# Output dimension
# ======================================================================


class TestOutputDimension:
    """Tests that output dimension matches embedding_dim."""

    def test_subtraction_output_dim(
        self, sub_model: CompositionalRelationEmbedding, batch_indices
    ) -> None:
        """Subtraction mode output is [batch, embedding_dim]."""
        src, edge, dst = batch_indices
        result = sub_model(src, edge, dst)
        assert result.shape == (BATCH, EMBEDDING_DIM)

    def test_multiplication_output_dim(
        self, mul_model: CompositionalRelationEmbedding, batch_indices
    ) -> None:
        """Multiplication mode output is [batch, embedding_dim]."""
        src, edge, dst = batch_indices
        result = mul_model(src, edge, dst)
        assert result.shape == (BATCH, EMBEDDING_DIM)

    def test_circular_correlation_output_dim(
        self, corr_model: CompositionalRelationEmbedding, batch_indices
    ) -> None:
        """Circular correlation mode output is [batch, embedding_dim]."""
        src, edge, dst = batch_indices
        result = corr_model(src, edge, dst)
        assert result.shape == (BATCH, EMBEDDING_DIM)

    def test_scalar_input_output_dim(
        self, sub_model: CompositionalRelationEmbedding
    ) -> None:
        """Scalar (0-dim) indices produce [embedding_dim] output."""
        src = torch.tensor(0)
        edge = torch.tensor(1)
        dst = torch.tensor(2)
        result = sub_model(src, edge, dst)
        assert result.shape == (EMBEDDING_DIM,)


# ======================================================================
# Different modes produce different embeddings
# ======================================================================


class TestModesDiffer:
    """Tests that different compose modes produce different results."""

    def test_all_modes_differ(self) -> None:
        """Same inputs with different modes produce different outputs."""
        torch.manual_seed(42)
        models = {}
        for mode in ("subtraction", "multiplication", "circular_correlation"):
            m = CompositionalRelationEmbedding(
                num_node_types=NUM_NODE_TYPES,
                num_edge_types=NUM_EDGE_TYPES,
                embedding_dim=EMBEDDING_DIM,
                compose_mode=mode,
            )
            models[mode] = m

        # Use same weights across all models for fair comparison
        shared_node_weight = models["subtraction"].node_type_embed.weight.data.clone()
        shared_edge_weight = models["subtraction"].edge_type_embed.weight.data.clone()
        for m in models.values():
            m.node_type_embed.weight.data.copy_(shared_node_weight)
            m.edge_type_embed.weight.data.copy_(shared_edge_weight)

        src = torch.tensor([0, 1, 2])
        edge = torch.tensor([0, 1, 2])
        dst = torch.tensor([3, 4, 0])

        results = {mode: m(src, edge, dst) for mode, m in models.items()}

        assert not torch.allclose(
            results["subtraction"], results["multiplication"], atol=1e-6
        )
        assert not torch.allclose(
            results["subtraction"], results["circular_correlation"], atol=1e-6
        )
        assert not torch.allclose(
            results["multiplication"], results["circular_correlation"], atol=1e-6
        )


# ======================================================================
# Gradient flow
# ======================================================================


class TestGradientFlow:
    """Tests for gradient flow through compose operations."""

    @pytest.mark.parametrize("mode", ["subtraction", "multiplication", "circular_correlation"])
    def test_gradient_to_embeddings(self, mode: str) -> None:
        """Gradients flow to both node_type_embed and edge_type_embed."""
        torch.manual_seed(42)
        model = CompositionalRelationEmbedding(
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            embedding_dim=EMBEDDING_DIM,
            compose_mode=mode,
        )

        src = torch.tensor([0, 1, 2])
        edge = torch.tensor([0, 1, 2])
        dst = torch.tensor([3, 4, 0])

        result = model(src, edge, dst)
        loss = result.sum()
        loss.backward()

        assert model.node_type_embed.weight.grad is not None, (
            f"No gradient for node_type_embed in {mode} mode"
        )
        assert model.edge_type_embed.weight.grad is not None, (
            f"No gradient for edge_type_embed in {mode} mode"
        )

    @pytest.mark.parametrize("mode", ["subtraction", "multiplication", "circular_correlation"])
    def test_gradients_are_finite(self, mode: str) -> None:
        """All gradients are finite (no NaN/Inf)."""
        torch.manual_seed(42)
        model = CompositionalRelationEmbedding(
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            embedding_dim=EMBEDDING_DIM,
            compose_mode=mode,
        )

        src = torch.tensor([0, 1])
        edge = torch.tensor([0, 1])
        dst = torch.tensor([1, 0])

        result = model(src, edge, dst)
        result.sum().backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in {name} for {mode} mode"
                )


# ======================================================================
# Invalid mode and edge cases
# ======================================================================


class TestInvalidMode:
    """Tests for invalid compose mode rejection."""

    def test_invalid_mode_raises(self) -> None:
        """Invalid compose_mode raises ValueError."""
        with pytest.raises(ValueError, match="compose_mode must be one of"):
            CompositionalRelationEmbedding(
                num_node_types=5,
                num_edge_types=3,
                embedding_dim=32,
                compose_mode="invalid_mode",
            )

    def test_valid_modes_accepted(self) -> None:
        """All three valid modes are accepted without error."""
        for mode in ("subtraction", "multiplication", "circular_correlation"):
            model = CompositionalRelationEmbedding(
                num_node_types=5,
                num_edge_types=3,
                embedding_dim=32,
                compose_mode=mode,
            )
            assert model.compose_mode == mode


class TestXavierInit:
    """Tests for weight initialization."""

    def test_embeddings_not_zero(self) -> None:
        """Xavier init produces non-zero embeddings."""
        model = CompositionalRelationEmbedding(
            num_node_types=5,
            num_edge_types=3,
            embedding_dim=32,
        )
        assert model.node_type_embed.weight.abs().sum() > 0
        assert model.edge_type_embed.weight.abs().sum() > 0

    def test_embedding_dim_stored(self) -> None:
        """Model stores embedding_dim as attribute."""
        model = CompositionalRelationEmbedding(
            num_node_types=5,
            num_edge_types=3,
            embedding_dim=64,
        )
        assert model.embedding_dim == 64
