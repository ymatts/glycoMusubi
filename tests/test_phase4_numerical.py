"""Numerical validity tests for Phase 4 components.

Verifies finite outputs, correct value ranges, gradient flow, loss convergence,
and edge-case handling for decoders, metrics, and evaluation utilities.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from glycoMusubi.embedding.decoders.node_classifier import NodeClassifier
from glycoMusubi.embedding.decoders.graph_level_decoder import GraphLevelDecoder
from glycoMusubi.evaluation.statistical_tests import (
    auto_test,
    bootstrap_ci,
    cohens_d,
    holm_bonferroni,
)
from glycoMusubi.evaluation.kg_quality import compute_kg_quality
from glycoMusubi.evaluation.glyco_metrics import (
    cross_modal_alignment_score,
    glycan_structure_recovery,
    taxonomy_hierarchical_consistency,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture()
def node_classifier():
    """NodeClassifier with two task heads."""
    torch.manual_seed(0)
    return NodeClassifier(
        embed_dim=64,
        task_configs={"taxonomy": 5, "function": 3},
        hidden_dim=32,
        dropout=0.0,
    )


@pytest.fixture()
def graph_decoder():
    """GraphLevelDecoder with deterministic init."""
    torch.manual_seed(0)
    return GraphLevelDecoder(
        embed_dim=64,
        num_classes=4,
        hidden_dim=32,
        dropout=0.0,
    )


# =====================================================================
# 1. Node Classification Decoder (6 tests)
# =====================================================================


class TestNodeClassifierNumerical:
    """Numerical validity for NodeClassifier."""

    def test_logits_finite_for_random_input(self, node_classifier):
        """Logits must be finite (no NaN/Inf) for random input."""
        torch.manual_seed(1)
        x = torch.randn(16, 64)
        logits = node_classifier(x, "taxonomy")
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
        assert logits.shape == (16, 5)

    def test_softmax_sums_to_one(self, node_classifier):
        """Softmax of logits must sum to 1.0 within tolerance."""
        torch.manual_seed(2)
        x = torch.randn(32, 64)
        logits = node_classifier(x, "taxonomy")
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
            f"Softmax sums deviate from 1.0: max deviation = {(sums - 1.0).abs().max().item()}"
        )

    def test_gradients_finite_and_nonvanishing(self, node_classifier):
        """Gradients must be finite and non-zero after backward."""
        torch.manual_seed(3)
        x = torch.randn(8, 64)
        logits = node_classifier(x, "taxonomy")
        labels = torch.randint(0, 5, (8,))
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        for name, param in node_classifier.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in {name}"
                )
                assert param.grad.abs().sum() > 0, (
                    f"Vanishing gradient in {name}"
                )

    def test_loss_decreases_over_training(self, node_classifier):
        """Cross-entropy loss must decrease over 20 training steps."""
        torch.manual_seed(4)
        x = torch.randn(32, 64)
        labels = torch.randint(0, 5, (32,))
        optimizer = torch.optim.Adam(node_classifier.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            logits = node_classifier(x, "taxonomy")
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_independent_task_head_gradients(self, node_classifier):
        """Different task heads must produce independent gradients."""
        torch.manual_seed(5)
        x = torch.randn(8, 64)

        # Forward through "taxonomy" head only
        node_classifier.zero_grad()
        logits_tax = node_classifier(x, "taxonomy")
        loss_tax = logits_tax.sum()
        loss_tax.backward()

        # The "function" head should have zero gradients
        func_head = node_classifier.heads["function"]
        for name, param in func_head.named_parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() == 0, (
                    f"Function head param '{name}' received gradient from taxonomy head"
                )

    def test_large_input_norms_produce_finite_logits(self, node_classifier):
        """Large input embeddings (norm=100) must still produce finite logits."""
        torch.manual_seed(6)
        x = torch.randn(8, 64)
        x = x / x.norm(dim=-1, keepdim=True) * 100.0
        logits = node_classifier(x, "taxonomy")
        assert torch.isfinite(logits).all(), (
            "Large-norm inputs caused non-finite logits"
        )


# =====================================================================
# 2. Graph-Level Decoder (6 tests)
# =====================================================================


class TestGraphLevelDecoderNumerical:
    """Numerical validity for GraphLevelDecoder."""

    def test_gate_values_in_zero_one(self, graph_decoder):
        """AttentiveReadout gate values must be in [0, 1] (sigmoid)."""
        torch.manual_seed(10)
        x = torch.randn(20, 64)
        gate = torch.sigmoid(graph_decoder.gate_linear(x))
        assert (gate >= 0.0).all() and (gate <= 1.0).all(), (
            f"Gate values out of [0,1]: min={gate.min():.6f}, max={gate.max():.6f}"
        )

    def test_graph_embedding_norms_bounded(self, graph_decoder):
        """Graph embedding norms must be bounded (< 1e3)."""
        torch.manual_seed(11)
        x = torch.randn(50, 64)
        out = graph_decoder(x, batch=None)
        norm = out.norm().item()
        assert norm < 1e3, f"Graph embedding norm too large: {norm}"

    def test_gradients_flow_through_attention_gates(self, graph_decoder):
        """Gradients must flow through the attention gate parameters."""
        torch.manual_seed(12)
        x = torch.randn(10, 64)
        out = graph_decoder(x, batch=None)
        loss = out.sum()
        loss.backward()

        assert graph_decoder.gate_linear.weight.grad is not None, (
            "No gradient for gate_linear.weight"
        )
        assert graph_decoder.gate_linear.weight.grad.abs().sum() > 0, (
            "Vanishing gradient in gate_linear.weight"
        )

    def test_loss_decreases_over_training(self, graph_decoder):
        """Loss must decrease over 20 training steps."""
        torch.manual_seed(13)
        x = torch.randn(30, 64)
        batch = torch.tensor([0]*10 + [1]*10 + [2]*10)
        labels = torch.randint(0, 4, (3,))
        optimizer = torch.optim.Adam(graph_decoder.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            logits = graph_decoder(x, batch=batch)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_single_node_graph_valid_output(self, graph_decoder):
        """Single-node graph must produce valid finite output."""
        torch.manual_seed(14)
        x = torch.randn(1, 64)
        out = graph_decoder(x, batch=None)
        assert out.shape == (1, 4)
        assert torch.isfinite(out).all(), "Single-node graph produced non-finite output"

    def test_batch_produces_correct_number_of_outputs(self, graph_decoder):
        """Batch of multiple graphs must produce one output per graph."""
        torch.manual_seed(15)
        # 3 graphs with 5, 8, and 7 nodes
        x = torch.randn(20, 64)
        batch = torch.cat([
            torch.full((5,), 0, dtype=torch.long),
            torch.full((8,), 1, dtype=torch.long),
            torch.full((7,), 2, dtype=torch.long),
        ])
        out = graph_decoder(x, batch=batch)
        assert out.shape == (3, 4), (
            f"Expected (3, 4) output for 3 graphs, got {out.shape}"
        )
        assert torch.isfinite(out).all()


# =====================================================================
# 3. Downstream Task Metrics (8 tests)
# =====================================================================


class TestDownstreamTaskMetrics:
    """Numerical range tests for downstream evaluation metrics."""

    def test_auc_roc_in_zero_one(self):
        """AUC-ROC must be in [0, 1] for valid inputs."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(20)
        y_true = rng.randint(0, 2, 100)
        y_scores = rng.rand(100)
        auc = roc_auc_score(y_true, y_scores)
        assert 0.0 <= auc <= 1.0, f"AUC-ROC out of range: {auc}"

    def test_auc_roc_approx_half_for_random(self):
        """AUC-ROC should be approximately 0.5 for random predictions."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(21)
        y_true = rng.randint(0, 2, 10000)
        y_scores = rng.rand(10000)
        auc = roc_auc_score(y_true, y_scores)
        assert abs(auc - 0.5) < 0.05, (
            f"AUC-ROC for random predictions far from 0.5: {auc:.4f}"
        )

    def test_auc_pr_in_zero_one(self):
        """AUC-PR must be in [0, 1]."""
        from sklearn.metrics import average_precision_score

        rng = np.random.RandomState(22)
        y_true = rng.randint(0, 2, 100)
        y_scores = rng.rand(100)
        auc_pr = average_precision_score(y_true, y_scores)
        assert 0.0 <= auc_pr <= 1.0, f"AUC-PR out of range: {auc_pr}"

    def test_f1_in_zero_one(self):
        """F1 score must be in [0, 1]."""
        from sklearn.metrics import f1_score

        rng = np.random.RandomState(23)
        y_true = rng.randint(0, 2, 100)
        y_pred = rng.randint(0, 2, 100)
        f1 = f1_score(y_true, y_pred)
        assert 0.0 <= f1 <= 1.0, f"F1 out of range: {f1}"

    def test_mcc_in_minus_one_to_one(self):
        """MCC must be in [-1, 1]."""
        from sklearn.metrics import matthews_corrcoef

        rng = np.random.RandomState(24)
        y_true = rng.randint(0, 2, 100)
        y_pred = rng.randint(0, 2, 100)
        mcc = matthews_corrcoef(y_true, y_pred)
        assert -1.0 <= mcc <= 1.0, f"MCC out of range: {mcc}"

    def test_ndcg_at_k_in_zero_one(self):
        """NDCG@K must be in [0, 1]."""
        from glycoMusubi.evaluation.tasks.disease_association import DiseaseAssociationTask

        rng = np.random.RandomState(25)
        scores = rng.rand(50)
        true_indices = np.array([2, 7, 15, 30])
        for k in [5, 10, 20, 50]:
            ndcg = DiseaseAssociationTask._compute_ndcg(scores, true_indices, k)
            assert 0.0 <= ndcg <= 1.0, f"NDCG@{k} out of range: {ndcg}"

    def test_recall_at_k_in_zero_one(self):
        """Recall@K must be in [0, 1]."""
        from glycoMusubi.evaluation.tasks.disease_association import DiseaseAssociationTask

        ranked_indices = np.arange(100)
        true_indices = np.array([0, 5, 50, 99])
        for k in [1, 5, 10, 50, 100]:
            recall = DiseaseAssociationTask._compute_recall_at_k(
                ranked_indices, true_indices, k
            )
            assert 0.0 <= recall <= 1.0, f"Recall@{k} out of range: {recall}"

    def test_enrichment_factor_nonnegative(self):
        """Enrichment factor must be non-negative."""
        from glycoMusubi.evaluation.tasks.drug_target import DrugTargetTask

        rng = np.random.RandomState(26)
        y_true = rng.randint(0, 2, 200).astype(float)
        y_scores = rng.rand(200)
        ef = DrugTargetTask._enrichment_factor(y_true, y_scores, fraction=0.01)
        assert ef >= 0.0, f"Enrichment factor negative: {ef}"

        # Also test with no positives
        y_all_neg = np.zeros(100)
        ef_zero = DrugTargetTask._enrichment_factor(y_all_neg, rng.rand(100))
        assert ef_zero == 0.0


# =====================================================================
# 4. Statistical Tests (6 tests)
# =====================================================================


class TestStatisticalTestsNumerical:
    """Numerical validity for statistical testing utilities."""

    def test_p_values_in_zero_one(self):
        """p-values from auto_test must be in [0, 1]."""
        rng = np.random.RandomState(30)
        scores_a = rng.rand(20)
        scores_b = rng.rand(20)
        result = auto_test(scores_a, scores_b)
        assert 0.0 <= result["p_value"] <= 1.0, (
            f"p-value out of range: {result['p_value']}"
        )
        assert 0.0 <= result["normality_p"] <= 1.0, (
            f"normality p out of range: {result['normality_p']}"
        )

    def test_holm_bonferroni_adjusted_geq_unadjusted(self):
        """Holm-Bonferroni adjusted p-values must be >= unadjusted."""
        raw_p = [0.001, 0.01, 0.05, 0.1, 0.5]
        adjusted = holm_bonferroni(raw_p)
        for raw, adj in zip(raw_p, adjusted):
            assert adj >= raw - 1e-12, (
                f"Adjusted p-value {adj} < raw {raw}"
            )
        # Adjusted p-values also capped at 1.0
        for adj in adjusted:
            assert adj <= 1.0 + 1e-12, f"Adjusted p > 1.0: {adj}"

    def test_cohens_d_zero_for_identical(self):
        """Cohen's d must be 0 for identical distributions."""
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(group, group.copy())
        assert abs(d) < 1e-10, f"Cohen's d not zero for identical groups: {d}"

    def test_cohens_d_magnitude_for_known_shift(self):
        """Cohen's d ~= 1.0 for a 1-std shift between groups."""
        rng = np.random.RandomState(31)
        n = 10000
        group1 = rng.normal(0.0, 1.0, n)
        group2 = rng.normal(1.0, 1.0, n)
        d = cohens_d(group1, group2)
        # Should be approximately -1.0 (group1 mean < group2 mean)
        assert abs(abs(d) - 1.0) < 0.1, (
            f"Cohen's d for 1-std shift should be ~1.0, got {d:.4f}"
        )

    def test_bootstrap_ci_lower_lt_upper(self):
        """Bootstrap CI lower bound must be < upper bound."""
        rng = np.random.RandomState(32)
        data = rng.normal(5.0, 2.0, 100)
        lower, upper = bootstrap_ci(np.mean, data, n_bootstrap=5000, ci=0.95)
        assert lower < upper, f"CI lower ({lower}) >= upper ({upper})"

    def test_bootstrap_ci_contains_true_mean(self):
        """Bootstrap CI should contain true mean ~95% of time."""
        true_mean = 10.0
        n_trials = 200
        contains = 0
        for seed in range(n_trials):
            rng = np.random.RandomState(seed + 1000)
            data = rng.normal(true_mean, 1.0, 50)
            lower, upper = bootstrap_ci(
                np.mean, data, n_bootstrap=2000, ci=0.95, rng_seed=seed
            )
            if lower <= true_mean <= upper:
                contains += 1

        coverage = contains / n_trials
        # Allow some slack: 95% CI should cover ~90-100% of the time
        assert coverage >= 0.85, (
            f"Bootstrap 95% CI coverage too low: {coverage:.2%} "
            f"({contains}/{n_trials} trials)"
        )


# =====================================================================
# 5. KG Quality Metrics (4 tests)
# =====================================================================


class TestKGQualityNumerical:
    """Numerical validity for KG quality metrics."""

    @pytest.fixture()
    def simple_hetero_data(self):
        """Create a minimal HeteroData for testing."""
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["glycan"].num_nodes = 10
        data["protein"].num_nodes = 15
        data["glycan", "binds", "protein"].edge_index = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
        )
        data["protein", "interacts", "protein"].edge_index = torch.tensor(
            [[0, 1, 5], [1, 2, 6]], dtype=torch.long
        )
        return data

    def test_all_metrics_finite_nonneg(self, simple_hetero_data):
        """All scalar quality metrics must be finite and non-negative."""
        metrics = compute_kg_quality(simple_hetero_data)
        for key, val in metrics.items():
            if key == "per_type_coverage":
                for cov_key, cov_val in val.items():
                    assert math.isfinite(cov_val), (
                        f"per_type_coverage[{cov_key}] not finite: {cov_val}"
                    )
                    assert cov_val >= 0.0, (
                        f"per_type_coverage[{cov_key}] negative: {cov_val}"
                    )
            else:
                assert math.isfinite(val), f"Metric '{key}' not finite: {val}"
                assert val >= 0.0, f"Metric '{key}' negative: {val}"

    def test_density_in_zero_one(self, simple_hetero_data):
        """Graph density must be in [0, 1]."""
        metrics = compute_kg_quality(simple_hetero_data)
        assert 0.0 <= metrics["graph_density"] <= 1.0, (
            f"Density out of range: {metrics['graph_density']}"
        )

    def test_entropy_nonnegative(self, simple_hetero_data):
        """Relation entropy must be non-negative (Shannon entropy >= 0)."""
        metrics = compute_kg_quality(simple_hetero_data)
        assert metrics["relation_entropy"] >= 0.0, (
            f"Negative entropy: {metrics['relation_entropy']}"
        )

    def test_coverage_values_in_zero_one(self, simple_hetero_data):
        """Per-type coverage must be in [0, 1] and sum to 1."""
        metrics = compute_kg_quality(simple_hetero_data)
        coverage = metrics["per_type_coverage"]
        total = 0.0
        for node_type, cov in coverage.items():
            assert 0.0 <= cov <= 1.0, (
                f"Coverage for '{node_type}' out of range: {cov}"
            )
            total += cov
        assert abs(total - 1.0) < 1e-10, (
            f"Coverage values do not sum to 1.0: {total}"
        )


# =====================================================================
# 6. Glyco Metrics (4 tests)
# =====================================================================


class TestGlycoMetricsNumerical:
    """Numerical validity for glyco-specific evaluation metrics."""

    def test_gsr_in_minus_one_to_one(self):
        """GSR (Spearman correlation) must be in [-1, 1]."""
        torch.manual_seed(40)
        sims = torch.rand(100)
        dists = torch.rand(100)
        gsr = glycan_structure_recovery(sims, dists)
        assert -1.0 <= gsr <= 1.0, f"GSR out of range: {gsr}"

        # Perfect correlation: identical rankings
        ordered = torch.arange(50, dtype=torch.float)
        gsr_perfect = glycan_structure_recovery(ordered, ordered)
        assert abs(gsr_perfect - 1.0) < 1e-6, (
            f"GSR for identical rankings should be 1.0, got {gsr_perfect}"
        )

    def test_cas_finite_positive(self):
        """CAS (cross-modal alignment score) must be finite and positive."""
        torch.manual_seed(41)
        glycan_emb = torch.randn(10, 32)
        protein_emb = torch.randn(20, 32)
        # Known pairs: glycan 0 -> protein 0, glycan 1 -> protein 1, ...
        known_pairs = torch.tensor([[i, i] for i in range(5)])
        cas = cross_modal_alignment_score(glycan_emb, protein_emb, known_pairs)
        assert math.isfinite(cas), f"CAS not finite: {cas}"
        assert cas > 0.0, f"CAS should be positive (MRR > 0), got {cas}"
        # MRR is at most 1.0
        assert cas <= 1.0, f"CAS > 1.0: {cas}"

    def test_taxonomy_consistency_in_zero_one(self):
        """Taxonomy hierarchical consistency must be in [0, 1]."""
        # All predictions correct -> consistency = 1.0
        preds_correct = {
            "kingdom": torch.tensor([0, 1, 2, 0, 1]),
            "phylum": torch.tensor([0, 1, 2, 0, 1]),
            "class": torch.tensor([0, 1, 2, 0, 1]),
        }
        labels = {
            "kingdom": torch.tensor([0, 1, 2, 0, 1]),
            "phylum": torch.tensor([0, 1, 2, 0, 1]),
            "class": torch.tensor([0, 1, 2, 0, 1]),
        }
        thc = taxonomy_hierarchical_consistency(preds_correct, labels)
        assert thc == 1.0, f"Perfect predictions should give THC=1.0, got {thc}"

        # Mixed predictions
        preds_mixed = {
            "kingdom": torch.tensor([0, 1, 2, 0, 1]),
            "phylum": torch.tensor([0, 0, 2, 0, 1]),  # one wrong at phylum
        }
        labels_mixed = {
            "kingdom": torch.tensor([0, 1, 2, 0, 1]),
            "phylum": torch.tensor([0, 1, 2, 0, 1]),
        }
        thc_mixed = taxonomy_hierarchical_consistency(preds_mixed, labels_mixed)
        assert 0.0 <= thc_mixed <= 1.0, f"THC out of range: {thc_mixed}"

    def test_glyco_metrics_handle_edge_cases(self):
        """All glyco metrics must handle empty/minimal input gracefully."""
        # GSR with fewer than 2 elements
        gsr_empty = glycan_structure_recovery(
            torch.tensor([1.0]), torch.tensor([2.0])
        )
        assert gsr_empty == 0.0, f"GSR for single element should be 0.0, got {gsr_empty}"

        # CAS with no known pairs
        cas_empty = cross_modal_alignment_score(
            torch.randn(5, 16),
            torch.randn(5, 16),
            torch.zeros(0, 2, dtype=torch.long),
        )
        assert cas_empty == 0.0, f"CAS with no pairs should be 0.0, got {cas_empty}"

        # THC with single level -> should return 1.0
        thc_single = taxonomy_hierarchical_consistency(
            {"kingdom": torch.tensor([0, 1])},
            {"kingdom": torch.tensor([0, 1])},
        )
        assert thc_single == 1.0, (
            f"THC with single level should be 1.0, got {thc_single}"
        )
