"""Evaluation module for glycoMusubi knowledge graph embeddings.

Provides rank-based metrics, filtered link prediction evaluation,
embedding visualization utilities, multi-seed evaluation, KG quality
metrics, and glycan-specific evaluation metrics.
"""

from glycoMusubi.evaluation.metrics import (
    compute_ranks,
    compute_mrr,
    compute_hits_at_k,
    compute_mr,
    compute_amr,
)
from glycoMusubi.evaluation.link_prediction import LinkPredictionEvaluator
from glycoMusubi.evaluation.visualize import EmbeddingVisualizer
from glycoMusubi.evaluation.multi_seed import multi_seed_evaluation
from glycoMusubi.evaluation.downstream import BaseDownstreamTask, DownstreamEvaluator
from glycoMusubi.evaluation.statistical_tests import (
    auto_test,
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d,
    delong_test,
    holm_bonferroni,
)
from glycoMusubi.evaluation.kg_quality import compute_kg_quality
from glycoMusubi.evaluation.glyco_metrics import (
    glycan_structure_recovery,
    cross_modal_alignment_score,
    taxonomy_hierarchical_consistency,
)

__all__ = [
    "compute_ranks",
    "compute_mrr",
    "compute_hits_at_k",
    "compute_mr",
    "compute_amr",
    "LinkPredictionEvaluator",
    "EmbeddingVisualizer",
    "multi_seed_evaluation",
    "BaseDownstreamTask",
    "DownstreamEvaluator",
    "auto_test",
    "benjamini_hochberg",
    "bootstrap_ci",
    "cohens_d",
    "delong_test",
    "holm_bonferroni",
    "compute_kg_quality",
    "glycan_structure_recovery",
    "cross_modal_alignment_score",
    "taxonomy_hierarchical_consistency",
]
