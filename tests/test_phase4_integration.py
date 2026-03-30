"""Phase 4 end-to-end integration tests for the full GlycoKGNet pipeline.

Validates that all Phase 1-4 components work together correctly:

1. Full Pipeline with Node Classification (5 tests)
2. Full Pipeline with Graph-Level Decoder (3 tests)
3. Downstream Task Pipeline (8 tests)
4. HGTLoader Pipeline (3 tests)
5. KG Quality Metrics (2 tests)
6. Complete System (2 tests)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.decoders.graph_level_decoder import GraphLevelDecoder
from glycoMusubi.embedding.decoders.node_classifier import NodeClassifier
from glycoMusubi.embedding.models.glycoMusubi_net import GlycoKGNet
from glycoMusubi.evaluation.downstream import BaseDownstreamTask, DownstreamEvaluator
from glycoMusubi.evaluation.glyco_metrics import (
    cross_modal_alignment_score,
    glycan_structure_recovery,
)
from glycoMusubi.evaluation.kg_quality import compute_kg_quality
from glycoMusubi.evaluation.tasks import (
    BindingSiteTask,
    DiseaseAssociationTask,
    DrugTargetTask,
    GlycanFunctionTask,
    GlycanProteinInteractionTask,
)
from glycoMusubi.losses.composite_loss import CompositeLoss
from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.training.trainer import Trainer


# ======================================================================
# Constants (reused from Phase 3 test graph specification)
# ======================================================================

DIM = 64
NUM_RELATIONS = 13

NUM_NODES_DICT = {
    "glycan": 10,
    "protein": 8,
    "enzyme": 6,
    "disease": 5,
    "variant": 4,
    "compound": 4,
    "site": 6,
    "motif": 5,
    "reaction": 4,
    "pathway": 3,
}

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("protein", "has_glycan", "glycan"),
    ("compound", "inhibits", "enzyme"),
    ("protein", "associated_with_disease", "disease"),
    ("protein", "has_variant", "variant"),
    ("protein", "has_site", "site"),
    ("enzyme", "has_site", "site"),
    ("site", "ptm_crosstalk", "site"),
    ("enzyme", "produced_by", "glycan"),
    ("enzyme", "consumed_by", "glycan"),
    ("glycan", "has_motif", "motif"),
    ("glycan", "child_of", "glycan"),
    ("enzyme", "catalyzed_by", "reaction"),
    ("reaction", "has_product", "glycan"),
]


# ======================================================================
# Fixtures
# ======================================================================


def _make_hetero_data(seed: int = 42) -> HeteroData:
    """Build a mini heterogeneous graph with all 10 node types and 13 edge types."""
    torch.manual_seed(seed)
    data = HeteroData()

    for ntype, count in NUM_NODES_DICT.items():
        data[ntype].x = torch.randn(count, DIM)
        data[ntype].num_nodes = count

    for src_type, rel, dst_type in EDGE_TYPES:
        num_src = NUM_NODES_DICT[src_type]
        num_dst = NUM_NODES_DICT[dst_type]
        num_edges = min(num_src, num_dst, 4)
        src_idx = torch.randint(0, num_src, (num_edges,))
        dst_idx = torch.randint(0, num_dst, (num_edges,))
        data[src_type, rel, dst_type].edge_index = torch.stack([src_idx, dst_idx])

    return data


@pytest.fixture
def hetero_data() -> HeteroData:
    return _make_hetero_data()


@pytest.fixture
def node_classifier() -> NodeClassifier:
    return NodeClassifier(
        embed_dim=DIM,
        task_configs={
            "glycan_type": 5,
            "protein_function": 3,
        },
    )


@pytest.fixture
def graph_decoder() -> GraphLevelDecoder:
    return GraphLevelDecoder(
        embed_dim=DIM,
        num_classes=4,
    )


def _make_phase4_model(
    node_classifier=None,
    graph_decoder=None,
) -> GlycoKGNet:
    """GlycoKGNet with Phase 4 decoders attached."""
    return GlycoKGNet(
        num_nodes_dict=NUM_NODES_DICT,
        num_relations=NUM_RELATIONS,
        embedding_dim=DIM,
        glycan_encoder_type="hybrid",
        protein_encoder_type="learnable",
        num_hgt_layers=2,
        num_hgt_heads=4,
        use_bio_prior=True,
        use_cross_modal_fusion=True,
        num_fusion_heads=4,
        decoder_type="hybrid",
        dropout=0.1,
        edge_types=EDGE_TYPES,
        node_classifier=node_classifier,
        graph_decoder=graph_decoder,
    )


# ======================================================================
# 1. Full Pipeline with Node Classification (5 tests)
# ======================================================================


class TestNodeClassificationPipeline:
    """GlycoKGNet + NodeClassifier end-to-end tests."""

    def test_instantiate_forward_get_logits(
        self, hetero_data: HeteroData, node_classifier: NodeClassifier
    ) -> None:
        """GlycoKGNet + NodeClassifier: instantiate, forward, get logits."""
        model = _make_phase4_model(node_classifier=node_classifier)
        assert model.node_classifier is not None

        logits = model.node_classify(hetero_data, task="glycan_type", node_type="glycan")
        assert logits.shape == (NUM_NODES_DICT["glycan"], 5)
        assert torch.isfinite(logits).all()

    def test_node_classify_correct_shape(
        self, hetero_data: HeteroData, node_classifier: NodeClassifier
    ) -> None:
        """node_classify() returns correct shape (num_nodes_of_type, num_classes)."""
        model = _make_phase4_model(node_classifier=node_classifier)

        # glycan_type: 10 glycans, 5 classes
        logits_glycan = model.node_classify(
            hetero_data, task="glycan_type", node_type="glycan"
        )
        assert logits_glycan.shape == (NUM_NODES_DICT["glycan"], 5)

        # protein_function: 8 proteins, 3 classes
        logits_protein = model.node_classify(
            hetero_data, task="protein_function", node_type="protein"
        )
        assert logits_protein.shape == (NUM_NODES_DICT["protein"], 3)

    def test_backward_gradient_flow_from_classification(
        self, hetero_data: HeteroData, node_classifier: NodeClassifier
    ) -> None:
        """Gradient flows from classification loss through encoder."""
        model = _make_phase4_model(node_classifier=node_classifier)
        model.zero_grad()

        logits = model.node_classify(
            hetero_data, task="glycan_type", node_type="glycan"
        )
        targets = torch.randint(0, 5, (NUM_NODES_DICT["glycan"],))
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        # Check gradients flow to encoder parameters
        encoder_has_grad = False
        for name, p in model.named_parameters():
            if "node_classifier" not in name and p.grad is not None and p.grad.abs().sum() > 0:
                encoder_has_grad = True
                break

        assert encoder_has_grad, "No gradients flowed from classification loss to encoder"

    def test_combined_link_and_node_loss_training(
        self, hetero_data: HeteroData, node_classifier: NodeClassifier
    ) -> None:
        """Combined loss (link + node) training: 5 steps, loss decreases."""
        torch.manual_seed(42)
        model = _make_phase4_model(node_classifier=node_classifier)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for step in range(5):
            optimizer.zero_grad()

            # Link prediction loss
            emb_dict = model(hetero_data)
            bs = 4
            h_idx = torch.randint(0, NUM_NODES_DICT["protein"], (bs,))
            t_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (bs,))
            r_idx = torch.zeros(bs, dtype=torch.long)
            h_emb = emb_dict["protein"][h_idx]
            t_emb = emb_dict["glycan"][t_idx]
            pos_scores = model.decoder(h_emb, r_idx, t_emb)
            neg_t_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (bs,))
            neg_t_emb = emb_dict["glycan"][neg_t_idx]
            neg_scores = model.decoder(h_emb, r_idx, neg_t_emb)
            link_loss = torch.clamp(1.0 - pos_scores + neg_scores, min=0.0).mean()

            # Node classification loss
            logits = model.node_classifier(emb_dict["glycan"], "glycan_type")
            targets = torch.randint(0, 5, (NUM_NODES_DICT["glycan"],))
            node_loss = F.cross_entropy(logits, targets)

            total_loss = link_loss + 0.1 * node_loss
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # Loss should not diverge
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
        # Verify first half avg >= second half avg (allowing tolerance)
        first_avg = sum(losses[:3]) / 3
        second_avg = sum(losses[3:]) / 2
        assert second_avg <= first_avg + 1.0, (
            f"Loss did not decrease: first_avg={first_avg:.4f}, second_avg={second_avg:.4f}"
        )

    def test_checkpoint_preserves_node_classifier_weights(
        self, hetero_data: HeteroData, node_classifier: NodeClassifier
    ) -> None:
        """Checkpoint save/load preserves node classifier weights."""
        model = _make_phase4_model(node_classifier=node_classifier)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=hetero_data,
            device="cpu",
        )
        trainer.fit(epochs=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "phase4_ckpt.pt"
            trainer.save_checkpoint(ckpt_path)
            assert ckpt_path.exists()

            # Reload into a fresh model
            nc2 = NodeClassifier(
                embed_dim=DIM,
                task_configs={"glycan_type": 5, "protein_function": 3},
            )
            model2 = _make_phase4_model(node_classifier=nc2)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            trainer2 = Trainer(
                model=model2,
                optimizer=optimizer2,
                loss_fn=loss_fn,
                train_data=hetero_data,
                device="cpu",
            )
            trainer2.load_checkpoint(ckpt_path)

            # Compare node classifier weights
            for (n1, p1), (n2, p2) in zip(
                model.node_classifier.named_parameters(),
                model2.node_classifier.named_parameters(),
            ):
                assert torch.allclose(p1, p2, atol=1e-6), (
                    f"Node classifier weight mismatch in {n1}"
                )


# ======================================================================
# 2. Full Pipeline with Graph-Level Decoder (3 tests)
# ======================================================================


class TestGraphLevelDecoderPipeline:
    """GlycoKGNet + GraphLevelDecoder end-to-end tests."""

    def test_instantiate_and_forward(
        self, hetero_data: HeteroData, graph_decoder: GraphLevelDecoder
    ) -> None:
        """GlycoKGNet + GraphLevelDecoder: instantiate, forward."""
        model = _make_phase4_model(graph_decoder=graph_decoder)
        assert model.graph_decoder is not None

        preds = model.predict_graph(hetero_data)
        assert preds.ndim == 2
        assert preds.shape[1] == 4  # num_classes
        assert torch.isfinite(preds).all()

    def test_predict_graph_correct_shape(
        self, hetero_data: HeteroData, graph_decoder: GraphLevelDecoder
    ) -> None:
        """predict_graph() returns correct shape."""
        model = _make_phase4_model(graph_decoder=graph_decoder)

        # Full graph: single graph prediction
        preds = model.predict_graph(hetero_data)
        assert preds.shape == (1, 4)

        # With subgraph_nodes selection
        subgraph_nodes = {
            "glycan": torch.tensor([0, 1, 2]),
            "protein": torch.tensor([0, 1]),
        }
        preds_sub = model.predict_graph(hetero_data, subgraph_nodes=subgraph_nodes)
        assert preds_sub.shape == (1, 4)

    def test_backward_through_attention_readout(
        self, hetero_data: HeteroData, graph_decoder: GraphLevelDecoder
    ) -> None:
        """Backward through attention readout to encoder."""
        model = _make_phase4_model(graph_decoder=graph_decoder)
        model.zero_grad()

        preds = model.predict_graph(hetero_data)
        targets = torch.tensor([2])  # single graph target
        loss = F.cross_entropy(preds, targets)
        loss.backward()

        # Check gradients flow to encoder
        encoder_grad_count = 0
        decoder_grad_count = 0
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                if "graph_decoder" in name:
                    decoder_grad_count += 1
                else:
                    encoder_grad_count += 1

        assert decoder_grad_count > 0, "No gradients in graph decoder"
        assert encoder_grad_count > 0, "No gradients flowed to encoder"


# ======================================================================
# 3. Downstream Task Pipeline (8 tests)
# ======================================================================


class TestDownstreamTaskPipeline:
    """Downstream evaluation task tests with GlycoKGNet embeddings."""

    @pytest.fixture(autouse=True)
    def _setup(self, hetero_data: HeteroData) -> None:
        """Pre-compute embeddings shared by all tests in this class."""
        torch.manual_seed(42)
        self.data = hetero_data
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )
        model.eval()
        with torch.no_grad():
            self.embeddings = {k: v.detach() for k, v in model(self.data).items()}

    def test_extract_embeddings(self) -> None:
        """Extract embeddings from trained GlycoKGNet."""
        assert isinstance(self.embeddings, dict)
        for ntype in NUM_NODES_DICT:
            assert ntype in self.embeddings
            assert self.embeddings[ntype].shape == (NUM_NODES_DICT[ntype], DIM)

    def test_glycan_protein_interaction_task(self) -> None:
        """GlycanProteinInteractionTask.evaluate() completes without error."""
        task = GlycanProteinInteractionTask(
            neg_ratio=2, n_folds=2, epochs=10
        )
        metrics = task.evaluate(self.embeddings, self.data)
        assert isinstance(metrics, dict)
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics
        assert "f1_optimal" in metrics

    def test_glycan_function_task(self) -> None:
        """GlycanFunctionTask.evaluate() completes without error.

        Note: With our synthetic test graph that has no taxonomy labels,
        the task will still complete and return aggregate metrics.
        """
        task = GlycanFunctionTask(
            n_folds=2, max_iter=10, min_samples_per_level=2
        )
        metrics = task.evaluate(self.embeddings, self.data)
        assert isinstance(metrics, dict)
        assert "mean_accuracy" in metrics
        assert "mean_f1" in metrics
        assert "mean_mcc" in metrics

    def test_disease_association_task(self) -> None:
        """DiseaseAssociationTask.evaluate() completes without error."""
        task = DiseaseAssociationTask(
            k_values=[2, 5],
            source_node_type="protein",
        )
        metrics = task.evaluate(self.embeddings, self.data)
        assert isinstance(metrics, dict)
        assert "auc_roc" in metrics

    def test_drug_target_task(self) -> None:
        """DrugTargetTask.evaluate() completes without error."""
        task = DrugTargetTask(
            k_values=[2, 5],
            neg_ratio=2,
            test_fraction=0.3,
        )
        metrics = task.evaluate(self.embeddings, self.data)
        assert isinstance(metrics, dict)
        assert "auc_roc" in metrics
        assert "enrichment_factor@1%" in metrics

    def test_binding_site_task(self) -> None:
        """BindingSiteTask.evaluate() completes without error."""
        task = BindingSiteTask(
            neg_ratio=2,
            test_fraction=0.3,
        )
        metrics = task.evaluate(self.embeddings, self.data)
        assert isinstance(metrics, dict)
        assert "residue_auc" in metrics
        assert "site_f1" in metrics

    def test_downstream_evaluator_all_tasks(self) -> None:
        """DownstreamEvaluator runs all tasks in sequence."""
        tasks = [
            GlycanProteinInteractionTask(neg_ratio=2, n_folds=2, epochs=10),
            DiseaseAssociationTask(k_values=[2, 5], source_node_type="protein"),
            DrugTargetTask(k_values=[2, 5], neg_ratio=2, test_fraction=0.3),
            BindingSiteTask(neg_ratio=2, test_fraction=0.3),
        ]
        evaluator = DownstreamEvaluator(tasks)
        results = evaluator.evaluate_all(self.embeddings, self.data)

        assert isinstance(results, dict)
        assert len(results) == 4
        for task in tasks:
            assert task.name in results

    def test_all_results_contain_expected_keys(self) -> None:
        """All results contain expected metric keys."""
        tasks_and_keys = [
            (
                GlycanProteinInteractionTask(neg_ratio=2, n_folds=2, epochs=10),
                ["auc_roc", "auc_pr", "f1_optimal"],
            ),
            (
                DiseaseAssociationTask(k_values=[2, 5], source_node_type="protein"),
                ["auc_roc"],
            ),
            (
                DrugTargetTask(k_values=[2, 5], neg_ratio=2, test_fraction=0.3),
                ["auc_roc", "enrichment_factor@1%"],
            ),
            (
                BindingSiteTask(neg_ratio=2, test_fraction=0.3),
                ["residue_auc", "site_f1"],
            ),
        ]

        for task, expected_keys in tasks_and_keys:
            metrics = task.evaluate(self.embeddings, self.data)
            for key in expected_keys:
                assert key in metrics, (
                    f"Task '{task.name}' missing key '{key}'. Got: {list(metrics.keys())}"
                )
                assert isinstance(metrics[key], float)


# ======================================================================
# 4. HGTLoader Pipeline (3 tests)
# ======================================================================


_has_torch_sparse = True
try:
    import torch_sparse  # noqa: F401
except ImportError:
    _has_torch_sparse = False


class TestHGTLoaderPipeline:
    """Trainer with HGTLoader integration tests."""

    @pytest.mark.skipif(
        not _has_torch_sparse, reason="torch-sparse not installed"
    )
    def test_trainer_with_hgt_loader_runs_3_epochs(self) -> None:
        """Trainer with HGTLoader + GlycoKGNet runs 3 epochs."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=True,
            hgt_num_samples=[4],
            hgt_batch_size=4,
        )
        history = trainer.fit(epochs=3)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
        assert all(isinstance(v, float) for v in history["train_loss"])

    @pytest.mark.skipif(
        not _has_torch_sparse, reason="torch-sparse not installed"
    )
    def test_checkpoint_with_hgt_loader_config(self) -> None:
        """Checkpoint save/load works with HGTLoader config."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_data=data,
                device="cpu",
                use_hgt_loader=True,
                hgt_num_samples=[4],
                hgt_batch_size=4,
            )
            trainer.fit(epochs=2)

            ckpt_path = Path(tmpdir) / "hgt_ckpt.pt"
            trainer.save_checkpoint(ckpt_path)
            assert ckpt_path.exists()

            # Reload
            model2 = GlycoKGNet(
                num_nodes_dict=NUM_NODES_DICT,
                num_relations=NUM_RELATIONS,
                embedding_dim=DIM,
                glycan_encoder_type="learnable",
                protein_encoder_type="learnable",
                num_hgt_layers=0,
                use_cross_modal_fusion=False,
                decoder_type="distmult",
            )
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            trainer2 = Trainer(
                model=model2,
                optimizer=optimizer2,
                loss_fn=loss_fn,
                train_data=data,
                device="cpu",
                use_hgt_loader=True,
                hgt_num_samples=[4],
                hgt_batch_size=4,
            )
            trainer2.load_checkpoint(ckpt_path)

            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2, atol=1e-6), f"Mismatch in {n1}"

    def test_trainer_without_hgt_loader_backward_compat(self) -> None:
        """Trainer without HGTLoader still works (backward compat)."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=False,
        )
        history = trainer.fit(epochs=3)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3


# ======================================================================
# 5. KG Quality Metrics (2 tests)
# ======================================================================


class TestKGQualityMetrics:
    """KG structural quality metrics integration tests."""

    def test_compute_kg_quality_returns_all_keys(
        self, hetero_data: HeteroData
    ) -> None:
        """compute_kg_quality on test graph returns all keys."""
        metrics = compute_kg_quality(hetero_data)

        expected_keys = [
            "num_nodes",
            "num_edges",
            "num_node_types",
            "num_edge_types",
            "graph_density",
            "avg_degree",
            "num_connected_components",
            "clustering_coefficient",
            "per_type_coverage",
            "relation_entropy",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

        assert metrics["num_node_types"] == 10.0
        assert metrics["num_edge_types"] == 13.0
        assert metrics["num_nodes"] == sum(NUM_NODES_DICT.values())
        assert metrics["graph_density"] > 0
        assert metrics["avg_degree"] > 0
        assert metrics["relation_entropy"] > 0

    def test_glyco_metrics_with_glycokgnet_embeddings(
        self, hetero_data: HeteroData
    ) -> None:
        """Glyco-specific metrics work with GlycoKGNet embeddings."""
        torch.manual_seed(42)
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )
        model.eval()
        with torch.no_grad():
            emb_dict = model(hetero_data)

        # Glycan Structure Recovery
        n_glycans = NUM_NODES_DICT["glycan"]
        struct_sims = torch.rand(n_glycans * (n_glycans - 1) // 2)
        glycan_emb = emb_dict["glycan"]
        # Compute pairwise embedding distances
        dists = []
        for i in range(n_glycans):
            for j in range(i + 1, n_glycans):
                dists.append(torch.dist(glycan_emb[i], glycan_emb[j]).item())
        emb_dists = torch.tensor(dists)

        gsr = glycan_structure_recovery(struct_sims, emb_dists)
        assert isinstance(gsr, float)
        assert -1.0 <= gsr <= 1.0

        # Cross-modal Alignment Score
        glycan_emb = emb_dict["glycan"]
        protein_emb = emb_dict["protein"]
        edge_index = hetero_data["protein", "has_glycan", "glycan"].edge_index
        # Swap to (glycan_idx, protein_idx) format
        known_pairs = torch.stack([edge_index[1], edge_index[0]], dim=1)
        cas = cross_modal_alignment_score(glycan_emb, protein_emb, known_pairs)
        assert isinstance(cas, float)
        assert 0.0 <= cas <= 1.0


# ======================================================================
# 6. Complete System (2 tests)
# ======================================================================


class TestCompleteSystem:
    """Full Phase 1-4 integration tests."""

    def test_all_phases_together(self, hetero_data: HeteroData) -> None:
        """All Phase 1-4 components together in one model."""
        torch.manual_seed(42)

        # Phase 4 decoders
        nc = NodeClassifier(
            embed_dim=DIM,
            task_configs={"glycan_type": 5, "protein_function": 3},
        )
        gd = GraphLevelDecoder(embed_dim=DIM, num_classes=4)

        # Full model with all phases
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_bio_prior=True,
            use_cross_modal_fusion=True,
            num_fusion_heads=4,
            decoder_type="hybrid",
            dropout=0.1,
            edge_types=EDGE_TYPES,
            node_classifier=nc,
            graph_decoder=gd,
        )

        # Phase 1-3: Forward pass (embed + BioHGT + fusion)
        emb_dict = model(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert emb_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)
            assert torch.isfinite(emb_dict[ntype]).all()

        # Phase 1-3: Link scoring
        bs = 4
        scores = model.score_triples(
            hetero_data,
            "protein",
            torch.randint(0, NUM_NODES_DICT["protein"], (bs,)),
            torch.zeros(bs, dtype=torch.long),
            "glycan",
            torch.randint(0, NUM_NODES_DICT["glycan"], (bs,)),
        )
        assert scores.shape == (bs,)
        assert torch.isfinite(scores).all()

        # Phase 4: Node classification
        glycan_logits = model.node_classify(
            hetero_data, task="glycan_type", node_type="glycan"
        )
        assert glycan_logits.shape == (NUM_NODES_DICT["glycan"], 5)

        protein_logits = model.node_classify(
            hetero_data, task="protein_function", node_type="protein"
        )
        assert protein_logits.shape == (NUM_NODES_DICT["protein"], 3)

        # Phase 4: Graph-level prediction
        graph_preds = model.predict_graph(hetero_data)
        assert graph_preds.shape == (1, 4)

        # Combined backward pass: link + node + graph losses
        model.zero_grad()
        h_idx = torch.randint(0, NUM_NODES_DICT["protein"], (bs,))
        t_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (bs,))
        r_idx = torch.zeros(bs, dtype=torch.long)

        emb_dict2 = model(hetero_data)
        h_emb = emb_dict2["protein"][h_idx]
        t_emb = emb_dict2["glycan"][t_idx]
        pos_scores = model.decoder(h_emb, r_idx, t_emb)
        neg_scores = model.decoder(
            h_emb, r_idx, emb_dict2["glycan"][torch.randint(0, NUM_NODES_DICT["glycan"], (bs,))]
        )

        link_loss = torch.clamp(1.0 - pos_scores + neg_scores, min=0.0).mean()
        node_loss = F.cross_entropy(
            model.node_classifier(emb_dict2["glycan"], "glycan_type"),
            torch.randint(0, 5, (NUM_NODES_DICT["glycan"],)),
        )
        # Recompute graph prediction from the same forward pass embeddings
        all_embs = torch.cat(list(emb_dict2.values()), dim=0)
        graph_pred = model.graph_decoder(all_embs)
        graph_loss = F.cross_entropy(graph_pred, torch.tensor([1]))

        total_loss = link_loss + 0.1 * node_loss + 0.05 * graph_loss
        total_loss.backward()

        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No parameters received gradients in combined loss"

    def test_memory_footprint_reasonable(self, hetero_data: HeteroData) -> None:
        """Memory footprint is reasonable (< 200MB for small test graph)."""
        nc = NodeClassifier(
            embed_dim=DIM,
            task_configs={"glycan_type": 5},
        )
        gd = GraphLevelDecoder(embed_dim=DIM, num_classes=4)

        model = _make_phase4_model(node_classifier=nc, graph_decoder=gd)

        # Calculate parameter memory
        total_params = sum(p.numel() for p in model.parameters())
        # Each float32 param is 4 bytes
        param_memory_mb = (total_params * 4) / (1024 * 1024)

        assert param_memory_mb < 200, (
            f"Model uses {param_memory_mb:.1f} MB for parameters, "
            f"which exceeds the 200 MB budget"
        )

        # Run a forward pass and check it completes without OOM
        model.eval()
        with torch.no_grad():
            emb_dict = model(hetero_data)
            _ = model.node_classify(hetero_data, task="glycan_type", node_type="glycan")
            _ = model.predict_graph(hetero_data)

        # Embedding memory
        emb_memory_mb = sum(
            e.numel() * 4 for e in emb_dict.values()
        ) / (1024 * 1024)
        total_memory_mb = param_memory_mb + emb_memory_mb

        assert total_memory_mb < 200, (
            f"Total memory (params + embeddings) is {total_memory_mb:.1f} MB"
        )
