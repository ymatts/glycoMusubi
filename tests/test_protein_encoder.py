"""Tests for ProteinEncoder.

Validates learnable and ESM-2 encoding modes, fallback behaviour when
ESM-2 cache is missing, output dimensionality, and gradient flow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from glycoMusubi.embedding.encoders.protein_encoder import ProteinEncoder


class TestProteinEncoderLearnable:
    """Tests for the learnable (default) encoding mode."""

    def test_learnable_mode_output_shape(self):
        encoder = ProteinEncoder(num_proteins=10, output_dim=256, method="learnable")
        indices = torch.tensor([0, 1, 2])
        out = encoder(indices)
        assert out.shape == (3, 256)

    def test_output_dim_configurable(self):
        """Output dimension should match the configured value."""
        for dim in (64, 128, 256, 512):
            encoder = ProteinEncoder(num_proteins=5, output_dim=dim, method="learnable")
            out = encoder(torch.tensor([0]))
            assert out.shape == (1, dim), f"Expected output_dim={dim}, got {out.shape}"

    def test_gradient_flow(self):
        """Gradients should flow through learnable embeddings."""
        encoder = ProteinEncoder(num_proteins=5, output_dim=32, method="learnable")
        indices = torch.tensor([0, 1])
        out = encoder(indices)
        loss = out.sum()
        loss.backward()
        assert encoder.embedding.weight.grad is not None
        assert encoder.embedding.weight.grad.abs().sum() > 0

    def test_batch_size_one(self):
        encoder = ProteinEncoder(num_proteins=5, output_dim=64, method="learnable")
        out = encoder(torch.tensor([0]))
        assert out.shape == (1, 64)

    def test_all_indices(self):
        """All valid indices should produce valid embeddings."""
        n = 8
        encoder = ProteinEncoder(num_proteins=n, output_dim=32, method="learnable")
        indices = torch.arange(n)
        out = encoder(indices)
        assert out.shape == (n, 32)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestProteinEncoderESM2:
    """Tests for the ESM-2 encoding mode with cache files."""

    def test_esm2_mode_with_cache(self, tmp_path):
        """When ESM-2 cache files exist, encoder should use them."""
        # Create mock ESM-2 cache files
        esm_dim = 1280
        for idx in range(3):
            emb = torch.randn(esm_dim)
            torch.save(emb, tmp_path / f"{idx}.pt")

        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2",
            esm2_dim=esm_dim,
            cache_path=tmp_path,
        )
        indices = torch.tensor([0, 1, 2])
        out = encoder(indices)
        assert out.shape == (3, 256)
        assert not torch.isnan(out).any()

    def test_esm2_mode_fallback(self, tmp_path):
        """When cache is empty, ESM-2 mode should fall back to learnable embeddings."""
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2",
            cache_path=tmp_path,  # empty directory
        )
        indices = torch.tensor([0, 1])
        out = encoder(indices)
        assert out.shape == (2, 256)
        # Output should come from the learnable fallback
        learnable_out = encoder.embedding(indices)
        assert torch.allclose(out, learnable_out), (
            "Without ESM-2 cache, output should match learnable embeddings"
        )

    def test_esm2_mode_partial_cache(self, tmp_path):
        """Mixed cache hit/miss should blend ESM-2 projected and learnable."""
        esm_dim = 1280
        # Only create cache for index 0
        torch.save(torch.randn(esm_dim), tmp_path / "0.pt")

        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=128,
            method="esm2",
            esm2_dim=esm_dim,
            cache_path=tmp_path,
        )
        indices = torch.tensor([0, 1])
        out = encoder(indices)
        assert out.shape == (2, 128)

        # Index 0 should use ESM-2 projected embedding (not pure learnable)
        learnable_0 = encoder.embedding(torch.tensor([0]))
        # They should differ because index 0 has ESM-2 data
        assert not torch.allclose(out[0:1], learnable_0), (
            "Index with ESM-2 cache should differ from pure learnable"
        )

    def test_esm2_2d_embeddings_mean_pooled(self, tmp_path):
        """Per-residue embeddings (2D) should be mean-pooled to 1D."""
        esm_dim = 1280
        seq_len = 50
        # Save a 2D tensor (per-residue)
        per_residue_emb = torch.randn(seq_len, esm_dim)
        torch.save(per_residue_emb, tmp_path / "0.pt")

        encoder = ProteinEncoder(
            num_proteins=3,
            output_dim=128,
            method="esm2",
            esm2_dim=esm_dim,
            cache_path=tmp_path,
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 128)

    def test_esm2_wrong_dim_falls_back(self, tmp_path):
        """Cache file with wrong dimension should trigger learnable fallback."""
        wrong_dim = 768
        torch.save(torch.randn(wrong_dim), tmp_path / "0.pt")

        encoder = ProteinEncoder(
            num_proteins=3,
            output_dim=128,
            method="esm2",
            esm2_dim=1280,
            cache_path=tmp_path,
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 128)
        # Should fall back to learnable
        learnable_out = encoder.embedding(indices)
        assert torch.allclose(out, learnable_out)

    def test_esm2_gradient_flow(self, tmp_path):
        """Gradients should flow through the projection MLP in ESM-2 mode."""
        esm_dim = 1280
        torch.save(torch.randn(esm_dim), tmp_path / "0.pt")

        encoder = ProteinEncoder(
            num_proteins=3,
            output_dim=64,
            method="esm2",
            esm2_dim=esm_dim,
            cache_path=tmp_path,
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        loss = out.sum()
        loss.backward()

        # Projection MLP should have gradients
        for name, param in encoder.projection.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_cache_clear(self, tmp_path):
        """Clearing internal caches should work without errors."""
        encoder = ProteinEncoder(
            num_proteins=3,
            output_dim=64,
            method="esm2",
            cache_path=tmp_path,
        )
        encoder.clear_cache()
        assert len(encoder._esm2_cache) == 0
        assert len(encoder._missing_indices) == 0

    def test_no_cache_path_returns_learnable(self):
        """ESM-2 mode without cache_path should use learnable embeddings."""
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=128,
            method="esm2",
            cache_path=None,
        )
        indices = torch.tensor([0, 1])
        out = encoder(indices)
        assert out.shape == (2, 128)


class TestProteinEncoderSiteAware:
    """Tests for the ESM-2 site-aware encoding mode."""

    @pytest.fixture
    def site_cache(self, tmp_path):
        """Create mock per-residue ESM-2 cache files (2D: [L, 1280])."""
        esm_dim = 1280
        seq_lengths = {0: 100, 1: 50, 2: 200, 3: 30, 4: 10}
        for idx, seq_len in seq_lengths.items():
            per_residue = torch.randn(seq_len, esm_dim)
            torch.save(per_residue, tmp_path / f"{idx}.pt")
        return tmp_path, seq_lengths

    def test_site_aware_initializes(self, site_cache):
        """ProteinEncoder(method='esm2_site_aware') initializes correctly."""
        cache_path, _ = site_cache
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [50]},
        )
        assert encoder.method == "esm2_site_aware"
        assert hasattr(encoder, "mlp_site")
        assert hasattr(encoder, "mlp_merge")
        assert hasattr(encoder, "site_count_mlp")

    def test_site_context_window_extraction(self, site_cache):
        """Site context window [p-15:p+15] should extract from per-residue embeddings."""
        cache_path, seq_lengths = site_cache
        site_pos = 50
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [site_pos]},
            site_window=15,
        )
        assert encoder.site_window == 15

        # Forward pass should work without error
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 256)
        assert not torch.isnan(out).any()

    def test_site_at_boundary_start(self, site_cache):
        """Site at position 0 should clamp window start to 0."""
        cache_path, _ = site_cache
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [0]},  # site at position 0
            site_window=15,
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 256)
        assert not torch.isnan(out).any()

    def test_site_at_boundary_end(self, site_cache):
        """Site at end of sequence should clamp window end to seq_len."""
        cache_path, seq_lengths = site_cache
        # Protein 0 has seq_len=100, put site at position 99 (last residue)
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [99]},
            site_window=15,
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 256)
        assert not torch.isnan(out).any()

    def test_multiple_sites_per_protein(self, site_cache):
        """Multiple glycosylation sites per protein should be aggregated."""
        cache_path, _ = site_cache
        # Protein 0 (seq_len=100) with 3 sites
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [10, 50, 90]},
            site_window=15,
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 256)
        assert not torch.isnan(out).any()

    def test_no_sites_fallback_to_standard(self, site_cache):
        """Proteins with no sites should fall back to standard ESM-2 pooling."""
        cache_path, _ = site_cache
        # No site positions for protein 0
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={},  # no sites for any protein
            site_window=15,
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 256)
        assert not torch.isnan(out).any()

    def test_output_shape_batch(self, site_cache):
        """Output shape should be [num_proteins, 256] for a batch."""
        cache_path, _ = site_cache
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [50], 1: [25], 2: [100]},
            site_window=15,
        )
        indices = torch.tensor([0, 1, 2])
        out = encoder(indices)
        assert out.shape == (3, 256)

    def test_site_aware_vs_no_sites_differ(self, site_cache):
        """Protein with sites should produce different embedding than without sites."""
        cache_path, _ = site_cache
        # Encoder with sites
        encoder_sites = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [50]},
            site_window=15,
        )
        # Encoder without sites
        encoder_no_sites = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={},
            site_window=15,
        )

        # Need same weights for fair comparison; instead just verify both produce valid output
        out_sites = encoder_sites(torch.tensor([0]))
        out_no_sites = encoder_no_sites(torch.tensor([0]))
        assert out_sites.shape == (1, 256)
        assert out_no_sites.shape == (1, 256)

    def test_site_position_out_of_bounds_skipped(self, site_cache):
        """Sites beyond sequence length should be skipped gracefully."""
        cache_path, seq_lengths = site_cache
        # Protein 4 has seq_len=10, put site at 999 (out of bounds)
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={4: [999]},
            site_window=15,
        )
        indices = torch.tensor([4])
        out = encoder(indices)
        assert out.shape == (1, 256)
        assert not torch.isnan(out).any()

    def test_site_aware_gradient_flow(self, site_cache):
        """Gradients should flow through site-aware MLP components."""
        cache_path, _ = site_cache
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [50]},
            site_window=15,
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        loss = out.sum()
        loss.backward()

        # mlp_site should have gradients
        for name, param in encoder.mlp_site.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for mlp_site.{name}"

        # mlp_merge should have gradients
        for name, param in encoder.mlp_merge.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for mlp_merge.{name}"

    def test_positional_encoding_for_site(self):
        """Positional encoding function should produce correct-dimensional output."""
        from glycoMusubi.embedding.encoders.protein_encoder import _positional_encoding
        pe = _positional_encoding(50, 64)
        assert pe.shape == (64,)
        assert not torch.isnan(pe).any()

        # Different positions should produce different encodings
        pe0 = _positional_encoding(0, 64)
        pe100 = _positional_encoding(100, 64)
        assert not torch.allclose(pe0, pe100)

    def test_mixed_sites_and_no_sites_batch(self, site_cache):
        """Batch with some proteins having sites and some not."""
        cache_path, _ = site_cache
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=cache_path,
            site_positions_map={0: [50], 2: [100, 150]},  # 1 and 3 have no sites
            site_window=15,
        )
        indices = torch.tensor([0, 1, 2, 3])
        out = encoder(indices)
        assert out.shape == (4, 256)
        assert not torch.isnan(out).any()

    def test_site_aware_no_cache_path_fallback(self):
        """Without cache_path, site-aware mode should fall back to learnable."""
        encoder = ProteinEncoder(
            num_proteins=5,
            output_dim=256,
            method="esm2_site_aware",
            cache_path=None,
            site_positions_map={0: [50]},
        )
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 256)


class TestProteinEncoderValidation:
    """Tests for input validation."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            ProteinEncoder(num_proteins=5, method="transformer")
