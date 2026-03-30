"""Unit tests for NodeClassifier.

Tests cover:
  - Instantiation with single and multiple task heads
  - Forward pass produces correct output shapes (num_nodes, num_classes)
  - Gradient flow through classification heads
  - Multiple task heads work independently
  - Dropout active during training, inactive during eval
  - Parameter count check (~50K per head with default dims)
  - Unknown task raises KeyError
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.embedding.decoders.node_classifier import NodeClassifier


# ======================================================================
# Constants
# ======================================================================

EMBEDDING_DIM = 256
HIDDEN_DIM = 128
NUM_NODES = 20


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def single_task_classifier() -> NodeClassifier:
    """NodeClassifier with a single task head."""
    return NodeClassifier(
        embed_dim=EMBEDDING_DIM,
        task_configs={"glycan_taxonomy": 10},
    )


@pytest.fixture()
def multi_task_classifier() -> NodeClassifier:
    """NodeClassifier with three task heads."""
    return NodeClassifier(
        embed_dim=EMBEDDING_DIM,
        task_configs={
            "glycan_taxonomy": 10,
            "protein_function": 25,
            "disease_category": 8,
        },
    )


@pytest.fixture()
def node_embeddings() -> torch.Tensor:
    """Random node embeddings [NUM_NODES, EMBEDDING_DIM]."""
    torch.manual_seed(42)
    return torch.randn(NUM_NODES, EMBEDDING_DIM)


# ======================================================================
# TestInstantiation
# ======================================================================


class TestInstantiation:
    """Tests for NodeClassifier construction."""

    def test_single_head_creation(self, single_task_classifier: NodeClassifier) -> None:
        """Single-task classifier creates one MLP head."""
        assert len(single_task_classifier.heads) == 1
        assert "glycan_taxonomy" in single_task_classifier.heads

    def test_multi_head_creation(self, multi_task_classifier: NodeClassifier) -> None:
        """Multi-task classifier creates one head per task."""
        assert len(multi_task_classifier.heads) == 3
        for task in ("glycan_taxonomy", "protein_function", "disease_category"):
            assert task in multi_task_classifier.heads

    def test_task_configs_stored(self, multi_task_classifier: NodeClassifier) -> None:
        """task_configs dict is stored correctly."""
        assert multi_task_classifier.task_configs["glycan_taxonomy"] == 10
        assert multi_task_classifier.task_configs["protein_function"] == 25
        assert multi_task_classifier.task_configs["disease_category"] == 8


# ======================================================================
# TestOutputShape
# ======================================================================


class TestOutputShape:
    """Tests for forward output shapes."""

    def test_single_task_output_shape(
        self,
        single_task_classifier: NodeClassifier,
        node_embeddings: torch.Tensor,
    ) -> None:
        """Output shape is (num_nodes, num_classes) for single task."""
        logits = single_task_classifier(node_embeddings, "glycan_taxonomy")
        assert logits.shape == (NUM_NODES, 10)

    def test_multi_task_output_shapes(
        self,
        multi_task_classifier: NodeClassifier,
        node_embeddings: torch.Tensor,
    ) -> None:
        """Each task head produces correct shape."""
        expected = {
            "glycan_taxonomy": (NUM_NODES, 10),
            "protein_function": (NUM_NODES, 25),
            "disease_category": (NUM_NODES, 8),
        }
        for task, shape in expected.items():
            logits = multi_task_classifier(node_embeddings, task)
            assert logits.shape == shape, (
                f"Shape mismatch for task '{task}': expected {shape}, got {logits.shape}"
            )

    def test_single_node_input(self, single_task_classifier: NodeClassifier) -> None:
        """Forward works with a single node."""
        single_emb = torch.randn(1, EMBEDDING_DIM)
        logits = single_task_classifier(single_emb, "glycan_taxonomy")
        assert logits.shape == (1, 10)

    def test_output_dtype(
        self,
        single_task_classifier: NodeClassifier,
        node_embeddings: torch.Tensor,
    ) -> None:
        """Output is float32."""
        logits = single_task_classifier(node_embeddings, "glycan_taxonomy")
        assert logits.dtype == torch.float32

    def test_output_finite(
        self,
        multi_task_classifier: NodeClassifier,
        node_embeddings: torch.Tensor,
    ) -> None:
        """All outputs are finite."""
        for task in ("glycan_taxonomy", "protein_function", "disease_category"):
            logits = multi_task_classifier(node_embeddings, task)
            assert torch.isfinite(logits).all(), f"Non-finite logits for task '{task}'"


# ======================================================================
# TestGradientFlow
# ======================================================================


class TestGradientFlow:
    """Tests for gradient flow through classification heads."""

    def test_gradients_flow_to_input(
        self, single_task_classifier: NodeClassifier,
    ) -> None:
        """Gradients flow back to input embeddings."""
        emb = torch.randn(NUM_NODES, EMBEDDING_DIM, requires_grad=True)
        logits = single_task_classifier(emb, "glycan_taxonomy")
        loss = logits.sum()
        loss.backward()
        assert emb.grad is not None

    def test_gradients_flow_to_all_head_params(
        self, single_task_classifier: NodeClassifier,
    ) -> None:
        """All parameters in the MLP head receive gradients."""
        emb = torch.randn(NUM_NODES, EMBEDDING_DIM)
        logits = single_task_classifier(emb, "glycan_taxonomy")
        loss = logits.sum()
        loss.backward()

        for name, param in single_task_classifier.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_independent_head_gradients(
        self, multi_task_classifier: NodeClassifier,
    ) -> None:
        """Only the used head receives gradients, not other heads."""
        emb = torch.randn(NUM_NODES, EMBEDDING_DIM)
        logits = multi_task_classifier(emb, "glycan_taxonomy")
        loss = logits.sum()
        loss.backward()

        # glycan_taxonomy head should have gradients
        for param in multi_task_classifier.heads["glycan_taxonomy"].parameters():
            assert param.grad is not None

        # Other heads should NOT have gradients
        for param in multi_task_classifier.heads["protein_function"].parameters():
            assert param.grad is None
        for param in multi_task_classifier.heads["disease_category"].parameters():
            assert param.grad is None


# ======================================================================
# TestDropoutBehavior
# ======================================================================


class TestDropoutBehavior:
    """Tests for dropout during training vs eval."""

    def test_train_mode_stochastic(self) -> None:
        """In training mode, outputs vary due to dropout."""
        classifier = NodeClassifier(
            embed_dim=EMBEDDING_DIM,
            task_configs={"task": 5},
            dropout=0.5,  # High dropout for observable variance
        )
        classifier.train()
        torch.manual_seed(42)
        emb = torch.randn(50, EMBEDDING_DIM)

        outputs = []
        for _ in range(5):
            outputs.append(classifier(emb, "task").detach().clone())

        # With high dropout, at least some outputs should differ
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Training mode should produce stochastic outputs with dropout"

    def test_eval_mode_deterministic(self) -> None:
        """In eval mode, outputs are deterministic."""
        classifier = NodeClassifier(
            embed_dim=EMBEDDING_DIM,
            task_configs={"task": 5},
            dropout=0.5,
        )
        classifier.eval()
        emb = torch.randn(10, EMBEDDING_DIM)

        out1 = classifier(emb, "task")
        out2 = classifier(emb, "task")
        assert torch.allclose(out1, out2)


# ======================================================================
# TestParameterCount
# ======================================================================


class TestParameterCount:
    """Tests for approximate parameter counts."""

    def test_single_head_param_count(self) -> None:
        """Single head with 10 classes should have ~50K params.

        Linear(256, 128): 256*128 + 128 = 32,896
        Linear(128, 10):  128*10  + 10  = 1,290
        Total:                             34,186
        """
        classifier = NodeClassifier(
            embed_dim=EMBEDDING_DIM,
            task_configs={"task": 10},
        )
        n_params = sum(p.numel() for p in classifier.parameters())
        assert 30_000 < n_params < 70_000, f"Expected ~50K params, got {n_params:,}"

    def test_more_classes_more_params(self) -> None:
        """More output classes leads to more parameters."""
        small = NodeClassifier(embed_dim=EMBEDDING_DIM, task_configs={"t": 5})
        large = NodeClassifier(embed_dim=EMBEDDING_DIM, task_configs={"t": 100})
        n_small = sum(p.numel() for p in small.parameters())
        n_large = sum(p.numel() for p in large.parameters())
        assert n_large > n_small

    def test_multi_task_param_count_additive(self) -> None:
        """Multi-task total params equals sum of individual heads."""
        tasks = {"a": 10, "b": 20, "c": 5}
        multi = NodeClassifier(embed_dim=EMBEDDING_DIM, task_configs=tasks)
        multi_total = sum(p.numel() for p in multi.parameters())

        individual_total = 0
        for name, n_cls in tasks.items():
            single = NodeClassifier(embed_dim=EMBEDDING_DIM, task_configs={name: n_cls})
            individual_total += sum(p.numel() for p in single.parameters())

        assert multi_total == individual_total


# ======================================================================
# TestErrorHandling
# ======================================================================


class TestErrorHandling:
    """Tests for error cases."""

    def test_unknown_task_raises_keyerror(
        self, single_task_classifier: NodeClassifier,
    ) -> None:
        """Requesting an unknown task raises KeyError."""
        emb = torch.randn(5, EMBEDDING_DIM)
        with pytest.raises(KeyError, match="Unknown classification task"):
            single_task_classifier(emb, "nonexistent_task")

    def test_error_message_lists_available_tasks(
        self, multi_task_classifier: NodeClassifier,
    ) -> None:
        """Error message includes the list of available tasks."""
        emb = torch.randn(5, EMBEDDING_DIM)
        with pytest.raises(KeyError, match="Available tasks"):
            multi_task_classifier(emb, "bad_task")
