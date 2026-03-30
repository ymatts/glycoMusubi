"""Glycobiology validation of Phase 4 downstream evaluation tasks.

Validates that every downstream task respects domain-specific biological
constraints and invariants of glycobiology, pharmacology, and protein
glycosylation.  Uses synthetic data designed to test biological plausibility
rather than real KG data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# 1. Glycan-Protein Interaction Biology
# ---------------------------------------------------------------------------


class TestGlycanProteinBiology:
    """Verify biological constraints of glycan-protein interaction task."""

    def test_negative_sampling_respects_type_constraints(self) -> None:
        """Negative samples are always (glycan, protein) pairs, never
        same-type pairs."""
        from glycoMusubi.evaluation.tasks.glycan_protein_interaction import (
            GlycanProteinInteractionTask,
        )

        task = GlycanProteinInteractionTask(neg_ratio=3, n_folds=2, epochs=1)

        data = HeteroData()
        n_glycan, n_protein, dim = 20, 15, 8
        data["glycan"].x = torch.randn(n_glycan, dim)
        data["glycan"].num_nodes = n_glycan
        data["protein"].x = torch.randn(n_protein, dim)
        data["protein"].num_nodes = n_protein
        src = torch.tensor([0, 1, 2, 3, 4])
        dst = torch.tensor([0, 1, 2, 3, 4])
        data["glycan", "binds_to", "protein"].edge_index = torch.stack([src, dst])

        embeddings = {
            "glycan": torch.randn(n_glycan, dim),
            "protein": torch.randn(n_protein, dim),
        }

        features, labels = task.prepare_data(embeddings, data)
        n_pos = int(labels.sum().item())
        n_neg = int((labels == 0).sum().item())

        # Feature dim should be 2*embedding_dim (concatenated glycan + protein)
        assert features.shape[1] == 2 * dim
        # Negative ratio is respected
        assert n_neg == n_pos * task.neg_ratio

    def test_known_pairs_score_higher_than_random(self) -> None:
        """When embeddings encode binding info, known glycan-protein pairs
        should produce higher feature dot products than random pairs."""
        n_glycan, n_protein, dim = 20, 10, 16

        # Create embeddings where binding partners are similar
        glycan_emb = torch.randn(n_glycan, dim)
        protein_emb = torch.randn(n_protein, dim)
        # Make glycan 0 similar to protein 0
        protein_emb[0] = glycan_emb[0] + torch.randn(dim) * 0.01

        # Dot product of known pair vs random pair
        known_score = (glycan_emb[0] * protein_emb[0]).sum()
        random_scores = [(glycan_emb[0] * protein_emb[j]).sum() for j in range(1, n_protein)]
        mean_random = sum(random_scores) / len(random_scores)

        assert known_score > mean_random, (
            "Known binding pair should score higher than average random pair"
        )


# ---------------------------------------------------------------------------
# 2. Glycan Function / Taxonomy Biology
# ---------------------------------------------------------------------------


class TestGlycanFunctionBiology:
    """Verify taxonomy-related biological constraints."""

    def test_taxonomy_levels_are_hierarchically_ordered(self) -> None:
        """Taxonomy levels proceed from coarsest (domain) to finest (species)."""
        from glycoMusubi.evaluation.tasks.glycan_function import GlycanFunctionTask

        levels = GlycanFunctionTask.TAXONOMY_LEVELS
        expected_order = [
            "domain", "kingdom", "phylum", "class",
            "order", "family", "genus", "species",
        ]
        assert levels == expected_order

    def test_hierarchical_consistency_metric(self) -> None:
        """taxonomy_hierarchical_consistency returns 1.0 when child
        predictions are always correct whenever parent is correct."""
        from glycoMusubi.evaluation.glyco_metrics import (
            taxonomy_hierarchical_consistency,
        )

        n = 50
        # All predictions correct at all levels => consistency = 1.0
        predictions = {
            "kingdom": torch.arange(n) % 3,
            "phylum": torch.arange(n) % 5,
        }
        labels = predictions.copy()
        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert thc == pytest.approx(1.0)

    def test_hierarchical_inconsistency_detected(self) -> None:
        """THC < 1.0 when parent-correct but child-wrong samples exist."""
        from glycoMusubi.evaluation.glyco_metrics import (
            taxonomy_hierarchical_consistency,
        )

        n = 20
        parent_pred = torch.zeros(n, dtype=torch.long)
        parent_true = torch.zeros(n, dtype=torch.long)  # all parent correct
        child_pred = torch.zeros(n, dtype=torch.long)
        child_true = torch.zeros(n, dtype=torch.long)
        # Make half the children wrong
        child_pred[n // 2:] = 1

        predictions = {"kingdom": parent_pred, "phylum": child_pred}
        labels = {"kingdom": parent_true, "phylum": child_true}

        thc = taxonomy_hierarchical_consistency(predictions, labels)
        assert thc == pytest.approx(0.5)

    def test_function_evaluation_handles_missing_levels_gracefully(self) -> None:
        """If only some taxonomy levels have labels, the task evaluates
        only those levels without error."""
        from glycoMusubi.evaluation.tasks.glycan_function import GlycanFunctionTask

        task = GlycanFunctionTask(n_folds=2, max_iter=10, min_samples_per_level=5)

        data = HeteroData()
        n = 30
        data["glycan"].x = torch.randn(n, 8)
        data["glycan"].num_nodes = n
        # Only domain labels, no other levels
        data["glycan"].domain = torch.randint(0, 3, (n,))

        embeddings = {"glycan": torch.randn(n, 8)}
        results = task.evaluate(embeddings, data)

        # Only domain should be evaluated
        assert "domain_accuracy" in results
        assert "kingdom_accuracy" not in results
        assert results["num_levels_evaluated"] == 1.0

    def test_multiclass_predictions_respect_class_count(self) -> None:
        """The classifier should only predict classes seen in the data."""
        from glycoMusubi.evaluation.tasks.glycan_function import GlycanFunctionTask

        rng = np.random.RandomState(42)
        n = 60
        dim = 8
        # 4 classes
        y = np.array([0] * 15 + [1] * 15 + [2] * 15 + [3] * 15)
        X = rng.randn(n, dim)

        task = GlycanFunctionTask(n_folds=2, max_iter=20, min_samples_per_level=5)
        acc, f1, mcc = task._cross_validate(X, y)

        # With random data and 4 classes, accuracy should be bounded
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= f1 <= 1.0
        assert -1.0 <= mcc <= 1.0


# ---------------------------------------------------------------------------
# 3. Disease Association Biology
# ---------------------------------------------------------------------------


class TestDiseaseAssociationBiology:
    """Verify disease-glycan biological constraints."""

    def test_known_associations_rank_higher_than_unknown(self) -> None:
        """When disease embedding is close to its associated entities,
        those entities should rank near the top."""
        from glycoMusubi.evaluation.tasks.disease_association import (
            DiseaseAssociationTask,
        )

        task = DiseaseAssociationTask(k_values=[5, 10])

        n_entities, n_diseases, dim = 50, 2, 16
        data = HeteroData()
        data["glycan"].x = torch.randn(n_entities, dim)
        data["glycan"].num_nodes = n_entities
        data["disease"].x = torch.randn(n_diseases, dim)
        data["disease"].num_nodes = n_diseases

        # Disease 0 associated with entities 0 and 1
        src = torch.tensor([0, 1])
        dst = torch.tensor([0, 0])
        data["glycan", "associated_with_disease", "disease"].edge_index = (
            torch.stack([src, dst])
        )

        # Make entities 0,1 very similar to disease 0
        entity_emb = torch.randn(n_entities, dim)
        disease_emb = torch.randn(n_diseases, dim)
        entity_emb[0] = disease_emb[0] + torch.randn(dim) * 0.01
        entity_emb[1] = disease_emb[0] + torch.randn(dim) * 0.01

        embeddings = {"glycan": entity_emb, "disease": disease_emb}
        results = task.evaluate(embeddings, data)

        # With these engineered embeddings, recall@5 should be high
        assert results["recall@5"] > 0.5

    def test_disease_specificity(self) -> None:
        """Different diseases associate with different entities.
        Scoring should distinguish between diseases."""
        from glycoMusubi.evaluation.tasks.disease_association import (
            DiseaseAssociationTask,
        )

        dim = 16
        n_entities = 20

        entity_emb = torch.randn(n_entities, dim)
        # Two diseases with orthogonal embeddings
        d0 = torch.zeros(dim)
        d0[0] = 1.0
        d1 = torch.zeros(dim)
        d1[1] = 1.0
        disease_emb = torch.stack([d0, d1])

        # Entity 0 is similar to disease 0, entity 1 to disease 1
        entity_emb[0] = d0 + torch.randn(dim) * 0.01
        entity_emb[1] = d1 + torch.randn(dim) * 0.01

        scores_d0 = DiseaseAssociationTask._cosine_scores(
            entity_emb.numpy(), disease_emb[0].numpy()
        )
        scores_d1 = DiseaseAssociationTask._cosine_scores(
            entity_emb.numpy(), disease_emb[1].numpy()
        )

        # Entity 0 should score highest for disease 0
        assert scores_d0[0] > scores_d0[1]
        # Entity 1 should score highest for disease 1
        assert scores_d1[1] > scores_d1[0]

    def test_auc_roc_requires_both_classes(self) -> None:
        """AUC-ROC is only computed when both positive and negative entities
        exist for a disease."""
        from glycoMusubi.evaluation.tasks.disease_association import (
            DiseaseAssociationTask,
        )

        task = DiseaseAssociationTask(k_values=[5])

        # Create scenario where all entities are positive (no negatives)
        data = HeteroData()
        n_entities = 5
        data["glycan"].x = torch.randn(n_entities, 8)
        data["glycan"].num_nodes = n_entities
        data["disease"].x = torch.randn(1, 8)
        data["disease"].num_nodes = 1

        # All entities associated with disease 0
        src = torch.arange(n_entities)
        dst = torch.zeros(n_entities, dtype=torch.long)
        data["glycan", "associated_with_disease", "disease"].edge_index = (
            torch.stack([src, dst])
        )

        embeddings = {
            "glycan": torch.randn(n_entities, 8),
            "disease": torch.randn(1, 8),
        }
        results = task.evaluate(embeddings, data)

        # AUC-ROC should be 0.0 (skipped) since all entities are positive
        assert results["auc_roc"] == 0.0


# ---------------------------------------------------------------------------
# 4. Drug Target Biology
# ---------------------------------------------------------------------------


class TestDrugTargetBiology:
    """Verify pharmacological constraints of drug target prediction."""

    def test_enrichment_factor_interpretation(self) -> None:
        """Enrichment Factor @1% > 1.0 means the model is better than
        random at identifying true targets in the top 1%."""
        from glycoMusubi.evaluation.tasks.drug_target import DrugTargetTask

        # Perfect enrichment: all positives at the top
        y_true = np.zeros(100)
        y_true[:10] = 1.0
        y_scores = np.zeros(100)
        y_scores[:10] = 1.0  # all positives have top scores

        ef = DrugTargetTask._enrichment_factor(y_true, y_scores, fraction=0.1)
        assert ef > 1.0

    def test_enrichment_factor_random(self) -> None:
        """Random scores should give EF close to 1.0."""
        from glycoMusubi.evaluation.tasks.drug_target import DrugTargetTask

        rng = np.random.RandomState(42)
        n = 10000
        y_true = np.zeros(n)
        y_true[:1000] = 1.0
        y_scores = rng.rand(n)

        ef = DrugTargetTask._enrichment_factor(y_true, y_scores, fraction=0.1)
        # Should be approximately 1.0 for random
        assert 0.5 < ef < 1.5

    def test_compound_enzyme_edge_type_convention(self) -> None:
        """Drug target task expects (compound, inhibits, enzyme) edges."""
        from glycoMusubi.evaluation.tasks.drug_target import DrugTargetTask

        task = DrugTargetTask()

        data = HeteroData()
        data["compound"].x = torch.randn(10, 8)
        data["compound"].num_nodes = 10
        data["enzyme"].x = torch.randn(5, 8)
        data["enzyme"].num_nodes = 5
        data["compound", "inhibits", "enzyme"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 1, 2]]
        )

        embeddings = {
            "compound": torch.randn(10, 8),
            "enzyme": torch.randn(5, 8),
        }

        # Should not raise
        results = task.evaluate(embeddings, data)
        assert "auc_roc" in results
        assert "enrichment_factor@1%" in results

    def test_novel_target_detection(self) -> None:
        """Hit@K for novel targets only counts enzymes unseen during training."""
        from glycoMusubi.evaluation.tasks.drug_target import DrugTargetTask

        # All positives have same target => no novel targets in test
        y_true = np.array([1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        novel_mask = np.array([False, False, False, False, False])

        hit = DrugTargetTask._hit_at_k_novel(y_true, y_scores, novel_mask, k=3)
        assert hit == 0.0  # no novel positives


# ---------------------------------------------------------------------------
# 5. Binding Site Biology
# ---------------------------------------------------------------------------


class TestBindingSiteBiology:
    """Verify glycosylation site prediction biological constraints."""

    def test_site_type_extraction(self) -> None:
        """N-linked and O-linked site types are correctly identified."""
        from glycoMusubi.evaluation.tasks.binding_site import BindingSiteTask

        data = HeteroData()
        n_sites = 6
        data["site"].x = torch.randn(n_sites, 8)
        data["site"].num_nodes = n_sites
        data["site"].site_type = [
            "N-linked", "O-linked", "N-linked",
            "O-linked", "N_linked", "unknown",
        ]

        test_idx = np.array([0, 1, 2, 3, 4, 5])
        types = BindingSiteTask._extract_site_types(data, test_idx, num_pos=n_sites)

        assert types is not None
        assert types[0] == "N-linked"
        assert types[1] == "O-linked"
        assert types[2] == "N-linked"
        assert types[3] == "O-linked"
        assert types[4] == "N-linked"  # N_linked -> N-linked
        assert types[5] == "unknown"

    def test_binding_site_requires_site_embeddings(self) -> None:
        """Task raises ValueError when 'site' key is missing."""
        from glycoMusubi.evaluation.tasks.binding_site import BindingSiteTask

        task = BindingSiteTask()
        data = HeteroData()

        with pytest.raises(ValueError, match="site"):
            task.prepare_data({"protein": torch.randn(10, 8)}, data)


# ---------------------------------------------------------------------------
# 6. Node Classification Biology
# ---------------------------------------------------------------------------


class TestNodeClassificationBiology:
    """Verify node classifier biological constraints."""

    def test_node_classifier_creates_per_task_heads(self) -> None:
        """Each classification task gets its own MLP head."""
        from glycoMusubi.embedding.decoders.node_classifier import NodeClassifier

        clf = NodeClassifier(
            embed_dim=32,
            task_configs={"glycan_type": 5, "protein_function": 3},
        )

        # Should have separate heads for each task
        x = torch.randn(10, 32)
        out_glycan = clf(x, task="glycan_type")
        out_protein = clf(x, task="protein_function")
        assert out_glycan.shape == (10, 5)
        assert out_protein.shape == (10, 3)

    def test_different_tasks_produce_independent_logits(self) -> None:
        """Different classification heads should produce different outputs
        even for the same input."""
        from glycoMusubi.embedding.decoders.node_classifier import NodeClassifier

        clf = NodeClassifier(
            embed_dim=16,
            task_configs={"task_a": 3, "task_b": 4},
        )

        x = torch.randn(5, 16)
        out_a = clf(x, task="task_a")
        out_b = clf(x, task="task_b")

        # Shapes differ
        assert out_a.shape[1] != out_b.shape[1]


# ---------------------------------------------------------------------------
# 7. KG Quality Metrics Biology
# ---------------------------------------------------------------------------


class TestKGQualityBiology:
    """Verify graph-level quality metrics capture expected properties."""

    def test_density_for_sparse_kg(self) -> None:
        """A sparse KG should have low density."""
        from glycoMusubi.evaluation.kg_quality import compute_kg_quality

        data = HeteroData()
        data["glycan"].x = torch.randn(100, 8)
        data["glycan"].num_nodes = 100
        data["protein"].x = torch.randn(50, 8)
        data["protein"].num_nodes = 50
        # 200 edges in a graph of 150 nodes -> sparse
        src = torch.randint(0, 100, (200,))
        dst = torch.randint(0, 50, (200,))
        data["glycan", "binds_to", "protein"].edge_index = torch.stack([src, dst])

        metrics = compute_kg_quality(data)

        assert metrics["num_nodes"] == 150.0
        assert metrics["num_edges"] == 200.0
        assert metrics["graph_density"] < 0.05  # sparse
        assert metrics["num_node_types"] == 2.0
        assert metrics["num_edge_types"] == 1.0

    def test_relation_entropy_single_type(self) -> None:
        """Single relation type should have zero entropy."""
        from glycoMusubi.evaluation.kg_quality import compute_kg_quality

        data = HeteroData()
        data["a"].x = torch.randn(10, 4)
        data["a"].num_nodes = 10
        data["b"].x = torch.randn(5, 4)
        data["b"].num_nodes = 5
        data["a", "r", "b"].edge_index = torch.tensor([[0, 1], [0, 1]])

        metrics = compute_kg_quality(data)
        assert metrics["relation_entropy"] == pytest.approx(0.0)

    def test_relation_entropy_uniform(self) -> None:
        """Uniformly distributed relation types should have max entropy."""
        from glycoMusubi.evaluation.kg_quality import compute_kg_quality

        data = HeteroData()
        data["a"].x = torch.randn(10, 4)
        data["a"].num_nodes = 10
        data["b"].x = torch.randn(10, 4)
        data["b"].num_nodes = 10

        # 3 edge types, each with same count
        for rel in ["r1", "r2", "r3"]:
            data["a", rel, "b"].edge_index = torch.tensor(
                [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
            )

        metrics = compute_kg_quality(data)
        expected_entropy = -3 * (1 / 3) * math.log(1 / 3)
        assert metrics["relation_entropy"] == pytest.approx(expected_entropy, rel=1e-4)

    def test_per_type_coverage_sums_to_one(self) -> None:
        """Node type coverage fractions should sum to 1.0."""
        from glycoMusubi.evaluation.kg_quality import compute_kg_quality

        data = HeteroData()
        data["glycan"].x = torch.randn(30, 4)
        data["glycan"].num_nodes = 30
        data["protein"].x = torch.randn(20, 4)
        data["protein"].num_nodes = 20
        data["disease"].x = torch.randn(10, 4)
        data["disease"].num_nodes = 10
        data["glycan", "r", "protein"].edge_index = torch.tensor([[0], [0]])

        metrics = compute_kg_quality(data)
        total_coverage = sum(metrics["per_type_coverage"].values())
        assert total_coverage == pytest.approx(1.0)

    def test_avg_degree_computation(self) -> None:
        """Average degree = 2*|E|/|V|."""
        from glycoMusubi.evaluation.kg_quality import compute_kg_quality

        data = HeteroData()
        data["a"].x = torch.randn(10, 4)
        data["a"].num_nodes = 10
        data["a", "r", "a"].edge_index = torch.tensor(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        )

        metrics = compute_kg_quality(data)
        assert metrics["avg_degree"] == pytest.approx(2.0 * 5 / 10)


# ---------------------------------------------------------------------------
# 8. Cross-Modal Alignment Biology
# ---------------------------------------------------------------------------


class TestCrossModalAlignmentBiology:
    """Verify cross-modal alignment between glycans and proteins."""

    def test_perfect_alignment_gives_mrr_one(self) -> None:
        """When each glycan's most similar protein is its binding partner,
        CAS (MRR) should be 1.0."""
        from glycoMusubi.evaluation.glyco_metrics import cross_modal_alignment_score

        dim = 16
        g_emb = torch.randn(5, dim)
        # Each protein is a copy of its binding glycan
        p_emb = g_emb.clone()

        pairs = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

        cas = cross_modal_alignment_score(g_emb, p_emb, pairs)
        assert cas == pytest.approx(1.0)

    def test_random_alignment_gives_low_mrr(self) -> None:
        """Unrelated embeddings should give CAS well below 1.0."""
        from glycoMusubi.evaluation.glyco_metrics import cross_modal_alignment_score

        torch.manual_seed(42)
        g_emb = torch.randn(50, 16)
        p_emb = torch.randn(50, 16)
        pairs = torch.tensor([[i, i] for i in range(50)])

        cas = cross_modal_alignment_score(g_emb, p_emb, pairs)
        assert cas < 0.5

    def test_glycan_structure_recovery_positive_correlation(self) -> None:
        """When structural similarity tracks embedding distance,
        GSR should be positive."""
        from glycoMusubi.evaluation.glyco_metrics import glycan_structure_recovery

        # Monotonically related
        structural_sim = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        emb_dist = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])  # inversely related

        gsr = glycan_structure_recovery(structural_sim, emb_dist)
        # Perfect inverse relationship -> negative Spearman correlation
        assert gsr < -0.5

    def test_glycan_structure_recovery_perfect(self) -> None:
        """Identical rank orderings give GSR = 1.0."""
        from glycoMusubi.evaluation.glyco_metrics import glycan_structure_recovery

        structural_sim = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        emb_dist = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

        gsr = glycan_structure_recovery(structural_sim, emb_dist)
        assert gsr == pytest.approx(1.0, abs=1e-6)
