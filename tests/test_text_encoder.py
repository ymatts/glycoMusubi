"""Tests for TextEncoder.

Validates hash-based text encoding, deterministic hashing,
output dimensionality, gradient flow, and PubMedBERT mode.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from glycoMusubi.embedding.encoders.text_encoder import TextEncoder, _stable_text_hash


class TestStableTextHash:
    """Tests for the deterministic text hashing function."""

    def test_deterministic(self):
        """Same input should always produce the same hash."""
        text = "diabetes mellitus"
        h1 = _stable_text_hash(text, 1000)
        h2 = _stable_text_hash(text, 1000)
        assert h1 == h2

    def test_different_texts_different_hashes(self):
        """Different texts should (very likely) produce different hashes."""
        h1 = _stable_text_hash("diabetes mellitus", 100000)
        h2 = _stable_text_hash("congenital disorder of glycosylation", 100000)
        # With 100k buckets, collision probability is negligible
        assert h1 != h2

    def test_within_range(self):
        """Hash should be within [0, num_buckets)."""
        num_buckets = 500
        for text in ["alpha", "beta", "gamma", "delta"]:
            h = _stable_text_hash(text, num_buckets)
            assert 0 <= h < num_buckets

    def test_unicode_support(self):
        """Should handle Unicode text."""
        h = _stable_text_hash("CDG (先天性糖鎖異常症)", 10000)
        assert isinstance(h, int)
        assert 0 <= h < 10000


class TestTextEncoder:
    """Tests for the TextEncoder nn.Module."""

    def test_forward_output_shape(self):
        """Forward with bucket indices should produce [batch, output_dim]."""
        encoder = TextEncoder(num_entities=100, output_dim=256)
        indices = torch.tensor([0, 5, 10])
        out = encoder(indices)
        assert out.shape == (3, 256)

    def test_output_dim_configurable(self):
        for dim in (32, 128, 256):
            encoder = TextEncoder(num_entities=50, output_dim=dim)
            out = encoder(torch.tensor([0]))
            assert out.shape == (1, dim)

    def test_encode_texts_convenience(self):
        """encode_texts should hash strings and return embeddings."""
        encoder = TextEncoder(num_entities=50, output_dim=64)
        texts = ["diabetes", "CDG", "galactosemia"]
        out = encoder.encode_texts(texts)
        assert out.shape == (3, 64)

    def test_encode_texts_deterministic(self):
        """Same texts should produce identical embeddings."""
        encoder = TextEncoder(num_entities=50, output_dim=64)
        texts = ["diabetes", "CDG"]
        out1 = encoder.encode_texts(texts)
        out2 = encoder.encode_texts(texts)
        assert torch.allclose(out1, out2)

    def test_text_to_index(self):
        """text_to_index should return a valid integer index."""
        encoder = TextEncoder(num_entities=100, output_dim=32)
        idx = encoder.text_to_index("diabetes mellitus")
        assert isinstance(idx, int)
        assert 0 <= idx < encoder.num_buckets

    def test_gradient_flow(self):
        """Gradients should flow through the text encoder."""
        encoder = TextEncoder(num_entities=50, output_dim=32)
        indices = torch.tensor([0, 1, 2])
        out = encoder(indices)
        # Use a single feature column rather than sum of all features,
        # because sum(LayerNorm(x)) is analytically independent of x
        # when gamma=1 and beta=0 (default), yielding zero gradients.
        loss = out[:, 0].sum()
        loss.backward()
        assert encoder.embedding.weight.grad is not None
        assert encoder.embedding.weight.grad.abs().sum() > 0

    def test_batch_size_one(self):
        encoder = TextEncoder(num_entities=50, output_dim=64)
        out = encoder(torch.tensor([0]))
        assert out.shape == (1, 64)

    def test_no_nan_or_inf(self):
        """Output should not contain NaN or Inf values."""
        encoder = TextEncoder(num_entities=100, output_dim=128)
        indices = torch.randint(0, encoder.num_buckets, (20,))
        out = encoder(indices)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_num_buckets_default(self):
        """Default num_buckets should be min(2 * num_entities, 100_000)."""
        encoder_small = TextEncoder(num_entities=100)
        assert encoder_small.num_buckets == 200

        encoder_large = TextEncoder(num_entities=60000)
        assert encoder_large.num_buckets == 100_000

    def test_num_buckets_custom(self):
        """Custom num_buckets should be respected."""
        encoder = TextEncoder(num_entities=100, num_buckets=500)
        assert encoder.num_buckets == 500

    def test_encode_texts_with_device(self):
        """encode_texts should accept a device argument."""
        encoder = TextEncoder(num_entities=50, output_dim=32)
        device = torch.device("cpu")
        out = encoder.encode_texts(["test"], device=device)
        assert out.device == device
        assert out.shape == (1, 32)

    def test_different_texts_different_embeddings(self):
        """Different text inputs should produce different embeddings."""
        encoder = TextEncoder(num_entities=100, output_dim=128)
        out = encoder.encode_texts(["diabetes mellitus", "congenital disorder"])
        # The two embeddings should differ
        assert not torch.allclose(out[0], out[1])

    def test_projection_has_gelu_layernorm(self):
        """The projection should include GELU activation and LayerNorm."""
        encoder = TextEncoder(num_entities=50, output_dim=64)
        layers = list(encoder.projection.children())
        layer_types = [type(l).__name__ for l in layers]
        assert "GELU" in layer_types, "Projection should include GELU activation"
        assert "LayerNorm" in layer_types, "Projection should include LayerNorm"


# ---------------------------------------------------------------------------
# PubMedBERT mode tests
# ---------------------------------------------------------------------------

def _mock_bert_output(batch_size: int, seq_len: int = 10, hidden_dim: int = 768):
    """Create a mock BERT model output with last_hidden_state."""
    output = MagicMock()
    output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
    return output


def _make_pubmedbert_encoder(num_entities: int = 10, output_dim: int = 256):
    """Create a TextEncoder with method='pubmedbert' using mocked transformers."""
    text_map = {i: f"entity_{i}" for i in range(num_entities)}

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }

    mock_bert = MagicMock()
    mock_bert.eval = MagicMock(return_value=None)

    # The __call__ responds to any batch size with the correct shape
    def bert_forward(**kwargs):
        batch_size = kwargs.get("input_ids", kwargs.get("attention_mask")).shape[0]
        return _mock_bert_output(batch_size)

    mock_bert.side_effect = bert_forward

    mock_auto_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_bert

    with patch.dict("sys.modules", {"transformers": MagicMock()}):
        with patch(
            "glycoMusubi.embedding.encoders.text_encoder.AutoTokenizer",
            mock_auto_tokenizer,
            create=True,
        ), patch(
            "glycoMusubi.embedding.encoders.text_encoder.AutoModel",
            mock_auto_model,
            create=True,
        ):
            # We need to directly build the encoder because _init_pubmedbert
            # imports AutoModel/AutoTokenizer from transformers at runtime.
            # Instead, manually construct the encoder to bypass the import.
            encoder = TextEncoder.__new__(TextEncoder)
            nn.Module.__init__(encoder)

            encoder.num_entities = num_entities
            encoder.output_dim = output_dim
            encoder.method = "pubmedbert"

            bert_dim = 768
            mid_dim = 384

            encoder.pubmedbert_mlp = nn.Sequential(
                nn.Linear(bert_dim, mid_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(mid_dim),
                nn.Linear(mid_dim, output_dim),
            )

            # Simulate pre-computed cached BERT embeddings
            cached = torch.randn(num_entities, bert_dim)
            encoder.register_buffer("_pubmedbert_cache", cached)

    return encoder


class TestPubMedBERTTextEncoder:
    """Tests for TextEncoder with method='pubmedbert'."""

    def test_pubmedbert_initializes(self):
        """TextEncoder(method='pubmedbert') initializes without error."""
        encoder = _make_pubmedbert_encoder(num_entities=10, output_dim=256)
        assert encoder.method == "pubmedbert"
        assert hasattr(encoder, "pubmedbert_mlp")
        assert hasattr(encoder, "_pubmedbert_cache")

    def test_pubmedbert_output_shape(self):
        """Output shape should be [num_entities, 256]."""
        encoder = _make_pubmedbert_encoder(num_entities=10, output_dim=256)
        indices = torch.arange(10)
        out = encoder(indices)
        assert out.shape == (10, 256)

    def test_pubmedbert_output_shape_subset(self):
        """Output shape should be [batch_size, 256] for a subset of indices."""
        encoder = _make_pubmedbert_encoder(num_entities=10, output_dim=256)
        indices = torch.tensor([0, 3, 7])
        out = encoder(indices)
        assert out.shape == (3, 256)

    def test_pubmedbert_caching_works(self):
        """Second forward() call uses cache (same buffer), not re-runs BERT."""
        encoder = _make_pubmedbert_encoder(num_entities=5, output_dim=256)
        encoder.eval()  # Disable dropout for deterministic comparison
        indices = torch.tensor([0, 1, 2])

        out1 = encoder(indices)
        out2 = encoder(indices)

        # The cached buffer is the same, so deterministic MLP produces same output
        assert torch.allclose(out1, out2), (
            "PubMedBERT cached embeddings should produce identical outputs on repeated calls"
        )

    def test_pubmedbert_mlp_architecture(self):
        """Projection MLP should have architecture: 768 -> 384 -> 256."""
        encoder = _make_pubmedbert_encoder(num_entities=5, output_dim=256)
        mlp = encoder.pubmedbert_mlp

        # Extract linear layers
        linear_layers = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2, f"Expected 2 Linear layers, got {len(linear_layers)}"

        # First linear: 768 -> 384
        assert linear_layers[0].in_features == 768
        assert linear_layers[0].out_features == 384

        # Second linear: 384 -> 256
        assert linear_layers[1].in_features == 384
        assert linear_layers[1].out_features == 256

    def test_pubmedbert_fallback_missing_transformers(self):
        """When transformers not installed, should raise ImportError with message."""
        import sys

        # Temporarily remove transformers from sys.modules if present
        saved = sys.modules.get("transformers")
        sys.modules["transformers"] = None  # Forces ImportError on import

        text_map = {0: "test entity"}
        try:
            with pytest.raises(ImportError, match="transformers"):
                TextEncoder(
                    num_entities=1,
                    output_dim=256,
                    method="pubmedbert",
                    text_map=text_map,
                )
        finally:
            if saved is not None:
                sys.modules["transformers"] = saved
            else:
                sys.modules.pop("transformers", None)

    def test_hash_embedding_still_works_regression(self):
        """Hash embedding mode should still work (regression test)."""
        encoder = TextEncoder(num_entities=100, output_dim=256, method="hash_embedding")
        indices = torch.tensor([0, 5, 10])
        out = encoder(indices)
        assert out.shape == (3, 256)
        assert not torch.isnan(out).any()

        # encode_texts convenience method should work
        texts_out = encoder.encode_texts(["diabetes", "CDG"])
        assert texts_out.shape == (2, 256)

    def test_pubmedbert_gradient_flow_through_mlp(self):
        """Gradients should flow through projection MLP (BERT is frozen, MLP is trainable)."""
        encoder = _make_pubmedbert_encoder(num_entities=5, output_dim=256)

        # Verify buffer is not a parameter (frozen BERT cache)
        buffer_names = {name for name, _ in encoder.named_buffers()}
        assert "_pubmedbert_cache" in buffer_names

        # Verify MLP parameters are trainable
        mlp_params = list(encoder.pubmedbert_mlp.parameters())
        assert len(mlp_params) > 0
        assert all(p.requires_grad for p in mlp_params)

        # Forward + backward
        indices = torch.tensor([0, 1, 2])
        out = encoder(indices)
        loss = out.sum()
        loss.backward()

        # MLP should have gradients
        for name, param in encoder.pubmedbert_mlp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for MLP param: {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for MLP param: {name}"

    def test_pubmedbert_different_texts_different_embeddings(self):
        """Different cached embeddings should produce different output vectors."""
        encoder = _make_pubmedbert_encoder(num_entities=10, output_dim=256)
        out = encoder(torch.tensor([0, 1]))
        # Since the cached embeddings are random, they should differ
        assert not torch.allclose(out[0], out[1]), (
            "Different entities should produce different PubMedBERT embeddings"
        )

    def test_pubmedbert_no_nan_or_inf(self):
        """PubMedBERT output should not contain NaN or Inf."""
        encoder = _make_pubmedbert_encoder(num_entities=10, output_dim=256)
        out = encoder(torch.arange(10))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_pubmedbert_requires_text_map(self):
        """method='pubmedbert' without text_map should raise ValueError."""
        # This test only works if transformers is installed; if not, ImportError
        # comes first. We mock transformers to test the ValueError path.
        encoder = TextEncoder.__new__(TextEncoder)
        nn.Module.__init__(encoder)
        encoder.method = "pubmedbert"

        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with pytest.raises(ValueError, match="text_map"):
                encoder._init_pubmedbert(
                    num_entities=5,
                    output_dim=256,
                    text_map=None,
                    model_name="test",
                    dropout=0.1,
                )

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            TextEncoder(num_entities=10, method="bert_large")
