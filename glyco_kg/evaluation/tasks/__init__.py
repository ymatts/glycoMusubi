"""Downstream evaluation tasks for glycoMusubi embeddings.

Each task takes pre-computed node embeddings and the KG data,
trains a lightweight classifier, and reports task-specific metrics.
"""

from glycoMusubi.evaluation.tasks.binding_site import BindingSiteTask
from glycoMusubi.evaluation.tasks.disease_association import DiseaseAssociationTask
from glycoMusubi.evaluation.tasks.drug_target import DrugTargetTask
from glycoMusubi.evaluation.tasks.glycan_function import GlycanFunctionTask
from glycoMusubi.evaluation.tasks.glycan_protein_interaction import (
    GlycanProteinInteractionTask,
)
from glycoMusubi.evaluation.tasks.immunogenicity import ImmunogenicityTask

__all__ = [
    "BindingSiteTask",
    "DiseaseAssociationTask",
    "DrugTargetTask",
    "GlycanFunctionTask",
    "GlycanProteinInteractionTask",
    "ImmunogenicityTask",
]
