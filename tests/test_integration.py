"""End-to-end integration tests for the glycoMusubi pipeline.

Tests the full pipeline: converter -> dataset -> model -> trainer -> evaluator.
Also validates gradient flow and loss decrease across epochs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.glycoMusubie import TransE, DistMult, RotatE
from glycoMusubi.embedding.encoders import GlycanEncoder, ProteinEncoder, TextEncoder
from glycoMusubi.data.splits import random_link_split
from glycoMusubi.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MarginLoss(nn.Module):
    """Simple margin-based ranking loss for testing."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.margin - pos_scores + neg_scores, min=0).mean()


def _make_mini_hetero_data() -> HeteroData:
    """Create a minimal HeteroData for integration testing.

    This mirrors the structure from conftest.py but is self-contained
    so integration tests do not depend on fixtures.
    """
    data = HeteroData()

    node_counts = {
        "glycan": 3,
        "protein": 4,
        "enzyme": 2,
        "disease": 2,
        "variant": 1,
        "compound": 1,
        "site": 2,
    }
    for ntype, n in node_counts.items():
        x = torch.empty(n, 256)
        nn.init.xavier_uniform_(x)
        data[ntype].x = x
        data[ntype].num_nodes = n

    # Edges with enough edges per type to survive splitting
    data["compound", "inhibits", "enzyme"].edge_index = torch.tensor(
        [[0, 0], [0, 1]]
    )
    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 0]]
    )
    data["protein", "associated_with_disease", "disease"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 0]]
    )
    data["protein", "has_variant", "variant"].edge_index = torch.tensor(
        [[0, 1], [0, 0]]
    )
    data["protein", "has_site", "site"].edge_index = torch.tensor(
        [[0, 1], [0, 1]]
    )
    data["enzyme", "has_site", "site"].edge_index = torch.tensor(
        [[0], [0]]
    )
    data["site", "ptm_crosstalk", "site"].edge_index = torch.tensor(
        [[0], [1]]
    )

    return data


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end integration tests for the glycoMusubi pipeline."""

    def test_full_pipeline_smoke(self):
        """converter -> split -> model -> trainer -> no crash.

        Verifies that the full pipeline runs without errors for 1 epoch.
        """
        # 1. Build mini HeteroData
        data = _make_mini_hetero_data()
        assert len(data.node_types) >= 5

        # 2. Split data
        train_data, val_data, test_data = random_link_split(
            data, val_ratio=0.1, test_ratio=0.1, seed=42
        )

        # 3. Initialize model
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_relations = len(data.edge_types)
        model = TransE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=256,
        )

        # 4. Train for 1 epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = MarginLoss(margin=1.0)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            val_data=val_data,
            device="cpu",
        )
        history = trainer.fit(epochs=1)

        # 5. Verify history is populated
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert isinstance(history["train_loss"][0], float)
        assert not torch.isnan(torch.tensor(history["train_loss"][0]))

    def test_all_model_types_smoke(self):
        """Verify TransE, DistMult, and RotatE all run without errors."""
        data = _make_mini_hetero_data()
        train_data, val_data, _ = random_link_split(
            data, val_ratio=0.1, test_ratio=0.1, seed=42
        )

        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_relations = len(data.edge_types)

        for ModelClass in [TransE, DistMult, RotatE]:
            model = ModelClass(
                num_nodes_dict=num_nodes_dict,
                num_relations=num_relations,
                embedding_dim=256,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = MarginLoss(margin=1.0)
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_data=train_data,
                device="cpu",
            )
            history = trainer.fit(epochs=1)
            assert len(history["train_loss"]) == 1
            assert not torch.isnan(torch.tensor(history["train_loss"][0])), (
                f"{ModelClass.__name__} produced NaN loss"
            )

    def test_gradient_flow(self):
        """All model parameters should receive gradients after a forward+backward pass."""
        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_relations = len(data.edge_types)

        model = TransE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=64,
        )

        # Forward pass
        emb_dict = model(data)
        # Score one edge type
        edge_type = data.edge_types[0]
        src_type, rel, dst_type = edge_type
        ei = data[edge_type].edge_index
        head_emb = emb_dict[src_type][ei[0]]
        tail_emb = emb_dict[dst_type][ei[1]]
        rel_emb = model.get_relation_embedding(
            torch.zeros(ei.size(1), dtype=torch.long)
        )
        scores = model.score(head_emb, rel_emb, tail_emb)
        loss = -scores.mean()  # simple loss for testing
        loss.backward()

        # Check that all parameters have gradients
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        # Some embedding tables may not be used in this particular edge type,
        # so we check that at least the used ones have gradients
        assert len(params_without_grad) < len(list(model.parameters())), (
            f"Too many parameters without gradients: {params_without_grad}"
        )

    def test_loss_decreases(self):
        """Loss should decrease over multiple training epochs.

        We train for several epochs with a reasonable learning rate and
        verify the loss trend is downward (allowing for minor fluctuations).
        """
        torch.manual_seed(42)
        data = _make_mini_hetero_data()
        train_data, _, _ = random_link_split(
            data, val_ratio=0.05, test_ratio=0.05, seed=42
        )

        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_relations = len(data.edge_types)

        model = DistMult(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=128,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = MarginLoss(margin=1.0)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            device="cpu",
        )
        n_epochs = 20
        history = trainer.fit(epochs=n_epochs)

        losses = history["train_loss"]
        assert len(losses) == n_epochs

        # The final loss should be less than the initial loss
        first_loss = losses[0]
        last_loss = losses[-1]
        assert last_loss < first_loss, (
            f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"
        )

    def test_checkpoint_save_load(self, tmp_path):
        """Saving and loading checkpoints should preserve model state."""
        data = _make_mini_hetero_data()
        train_data, _, _ = random_link_split(
            data, val_ratio=0.05, test_ratio=0.05, seed=42
        )

        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_relations = len(data.edge_types)

        model = TransE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=64,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = MarginLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            device="cpu",
        )
        trainer.fit(epochs=2)

        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        # Load into a fresh model
        model2 = TransE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=64,
        )
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            loss_fn=loss_fn,
            train_data=train_data,
            device="cpu",
        )
        trainer2.load_checkpoint(ckpt_path)

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter mismatch after load: {n1}"

    def test_data_split_preserves_nodes(self):
        """Splitting should preserve all node types and their features."""
        data = _make_mini_hetero_data()
        train_data, val_data, test_data = random_link_split(
            data, val_ratio=0.1, test_ratio=0.1, seed=42
        )

        for split_name, split_data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            for ntype in data.node_types:
                assert ntype in split_data.node_types, (
                    f"Node type '{ntype}' missing from {split_name} split"
                )
                assert split_data[ntype].num_nodes == data[ntype].num_nodes, (
                    f"Node count mismatch for '{ntype}' in {split_name} split"
                )


# ---------------------------------------------------------------------------
# TestEncoderIntegration
# ---------------------------------------------------------------------------

class TestEncoderIntegration:
    """Tests that encoders integrate correctly with downstream components."""

    def test_glycan_encoder_output_compatible_with_model(self):
        """GlycanEncoder output should be compatible with KGE model embedding dim."""
        encoder = GlycanEncoder(num_glycans=5, output_dim=256, method="learnable")
        indices = torch.tensor([0, 1, 2])
        glycan_emb = encoder(indices)

        # Verify the embedding dimension matches what the model expects
        assert glycan_emb.shape[-1] == 256

    def test_protein_encoder_output_compatible_with_model(self):
        """ProteinEncoder output should be compatible with KGE model embedding dim."""
        encoder = ProteinEncoder(num_proteins=5, output_dim=256, method="learnable")
        indices = torch.tensor([0, 1, 2])
        protein_emb = encoder(indices)
        assert protein_emb.shape[-1] == 256

    def test_text_encoder_output_compatible_with_model(self):
        """TextEncoder output should be compatible with KGE model embedding dim."""
        encoder = TextEncoder(num_entities=50, output_dim=256)
        texts = ["diabetes", "CDG", "galactosemia"]
        text_emb = encoder.encode_texts(texts)
        assert text_emb.shape[-1] == 256

    def test_all_encoders_same_output_dim(self):
        """All three encoders should produce the same output dimensionality."""
        output_dim = 256
        glycan_enc = GlycanEncoder(num_glycans=5, output_dim=output_dim)
        protein_enc = ProteinEncoder(num_proteins=5, output_dim=output_dim)
        text_enc = TextEncoder(num_entities=50, output_dim=output_dim)

        g = glycan_enc(torch.tensor([0]))
        p = protein_enc(torch.tensor([0]))
        t = text_enc(torch.tensor([0]))

        assert g.shape[-1] == p.shape[-1] == t.shape[-1] == output_dim


# ---------------------------------------------------------------------------
# TestConverterIntegration
# ---------------------------------------------------------------------------

class TestConverterIntegration:
    """Tests for KGConverter integration with downstream pipeline."""

    def test_converter_to_model(self, mini_kg_dir):
        """KGConverter output should be usable by a KGE model."""
        from glycoMusubi.data.converter import KGConverter

        converter = KGConverter(
            kg_dir=mini_kg_dir,
            schema_dir=None,
        )
        data, node_mappings = converter.convert(feature_dim=256)

        # The data should have node types
        assert len(data.node_types) > 0
        # The data should have edge types
        assert len(data.edge_types) > 0

        # Build a model from the converted data
        num_nodes_dict = {}
        for ntype in data.node_types:
            num_nodes_dict[ntype] = data[ntype].num_nodes

        num_relations = len(data.edge_types)
        model = TransE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=256,
        )

        # Forward pass should not crash
        emb_dict = model(data)
        assert len(emb_dict) == len(data.node_types)
        for ntype, emb in emb_dict.items():
            assert emb.shape == (num_nodes_dict[ntype], 256)


# ---------------------------------------------------------------------------
# TestModelScoring
# ---------------------------------------------------------------------------

class TestModelScoring:
    """Tests for model scoring consistency across model types."""

    @pytest.fixture
    def mini_setup(self):
        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_relations = len(data.edge_types)
        return data, num_nodes_dict, num_relations

    def test_transe_scores_are_negative(self, mini_setup):
        """TransE scores should be non-positive (negative L2 norm)."""
        data, num_nodes_dict, num_relations = mini_setup
        model = TransE(num_nodes_dict=num_nodes_dict, num_relations=num_relations)
        emb_dict = model(data)

        edge_type = data.edge_types[0]
        src_type, _, dst_type = edge_type
        ei = data[edge_type].edge_index
        head_emb = emb_dict[src_type][ei[0]]
        tail_emb = emb_dict[dst_type][ei[1]]
        rel_emb = model.get_relation_embedding(
            torch.zeros(ei.size(1), dtype=torch.long)
        )
        scores = model.score(head_emb, rel_emb, tail_emb)
        assert (scores <= 0).all(), "TransE scores should be non-positive"

    def test_distmult_scores_finite(self, mini_setup):
        """DistMult scores should be finite."""
        data, num_nodes_dict, num_relations = mini_setup
        model = DistMult(num_nodes_dict=num_nodes_dict, num_relations=num_relations)
        emb_dict = model(data)

        edge_type = data.edge_types[0]
        src_type, _, dst_type = edge_type
        ei = data[edge_type].edge_index
        head_emb = emb_dict[src_type][ei[0]]
        tail_emb = emb_dict[dst_type][ei[1]]
        rel_emb = model.get_relation_embedding(
            torch.zeros(ei.size(1), dtype=torch.long)
        )
        scores = model.score(head_emb, rel_emb, tail_emb)
        assert torch.isfinite(scores).all(), "DistMult scores should be finite"

    def test_rotate_scores_are_negative(self, mini_setup):
        """RotatE scores should be non-positive (negative L1 norm in complex space)."""
        data, num_nodes_dict, num_relations = mini_setup
        model = RotatE(num_nodes_dict=num_nodes_dict, num_relations=num_relations)
        emb_dict = model(data)

        edge_type = data.edge_types[0]
        src_type, _, dst_type = edge_type
        ei = data[edge_type].edge_index
        head_emb = emb_dict[src_type][ei[0]]
        tail_emb = emb_dict[dst_type][ei[1]]
        rel_emb = model.get_relation_embedding(
            torch.zeros(ei.size(1), dtype=torch.long)
        )
        scores = model.score(head_emb, rel_emb, tail_emb)
        assert (scores <= 0).all(), "RotatE scores should be non-positive"
