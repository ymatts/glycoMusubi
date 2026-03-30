"""Configuration management for glycoMusubi embedding pipeline.

Dataclass-based configuration with OmegaConf YAML loading support.
Each config section mirrors the YAML structure under ``configs/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Settings for KG data loading and preprocessing."""

    kg_dir: str = "kg"
    node_file: str = "nodes.tsv"
    edge_file: str = "edges.tsv"
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    num_neg_samples: int = 64
    split_seed: Optional[int] = None  # defaults to global seed when None
    max_eval_triples: int = 0  # 0 = use all; >0 = subsample for faster eval
    # Inductive evaluation settings
    holdout_ratio: float = 0.15  # fraction of entities to hold out for inductive eval
    inductive_min_degree: int = 2  # minimum degree for hold-out entities


@dataclass
class ModelConfig:
    """KGE model hyper-parameters."""

    name: str = "TransE"
    embedding_dim: int = 256
    margin: float = 1.0
    p_norm: int = 2
    # DistMult / RotatE extras
    regularization: float = 0.0
    reg_type: str = "l2"  # l2 | l3
    # GlycoKGNet-specific (ignored by TransE/DistMult/RotatE)
    glycan_encoder_type: str = "learnable"
    protein_encoder_type: str = "learnable"
    num_hgt_layers: int = 4
    num_hgt_heads: int = 8
    use_bio_prior: bool = True
    use_cross_modal_fusion: bool = True
    num_fusion_heads: int = 4
    decoder_type: str = "hybrid"
    dropout: float = 0.1
    esm2_cache_path: Optional[str] = None
    text_node_types: Optional[List[str]] = None
    # Loss function selection
    loss_fn: str = "margin"  # margin | bce | composite
    adversarial_temperature: Optional[float] = None
    label_smoothing: float = 0.0
    # CompositeLoss params
    lambda_struct: float = 0.1
    lambda_hyp: float = 0.01
    lambda_reg: float = 0.01
    # Glycan function features
    use_glycan_function_features: bool = False
    # Inductive mode settings
    inductive: bool = False
    site_map_path: Optional[str] = None
    wurcs_cache_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training loop settings."""

    epochs: int = 200
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # adam | sgd | adagrad
    scheduler: str = "none"  # none | cosine | step
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5
    grad_clip: float = 0.0  # 0 = disabled
    early_stopping_patience: int = 20
    eval_every: int = 10
    save_every: int = 50
    num_workers: int = 4
    relation_balance_alpha: float = 0.0
    relation_balance_max_weight: float = 5.0
    # Mini-batch / HGT loader settings
    use_hgt_loader: bool = False
    hgt_batch_size: int = 1024
    hgt_num_samples: Optional[List[int]] = None  # e.g. [512, 256]
    gradient_accumulation_steps: int = 1
    max_edges_per_type: int = 0  # 0 = no limit; >0 = subsample per edge type per step
    num_negatives: int = 1  # negatives per positive (K>1 for self-adversarial BCE)
    function_aware_negatives: bool = False  # restrict has_glycan neg pool to function-bearing glycans


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default"
    seed: int = 42
    embedding_dim: int = 256
    device: str = "auto"  # auto | cuda | cpu
    output_dir: str = "experiments"
    log_level: str = "INFO"
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_enabled: bool = False
    deterministic: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        """Return ``<output_dir>/<name>``."""
        return Path(self.output_dir) / self.name

    def resolve_device(self) -> str:
        """Resolve ``'auto'`` to a concrete device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _merge_yaml_files(*paths: str) -> DictConfig:
    """Load and merge multiple YAML files (later files override earlier)."""
    merged = OmegaConf.create({})
    for p in paths:
        if os.path.isfile(p):
            cfg = OmegaConf.load(p)
            merged = OmegaConf.merge(merged, cfg)
    return merged  # type: ignore[return-value]


def _dictconfig_to_experiment(cfg: DictConfig) -> ExperimentConfig:
    """Convert an OmegaConf DictConfig to an ExperimentConfig dataclass."""
    plain: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    data_dict = plain.pop("data", {})
    model_dict = plain.pop("model", {})
    training_dict = plain.pop("training", {})

    data_cfg = DataConfig(**{k: v for k, v in data_dict.items() if k in DataConfig.__dataclass_fields__})
    model_cfg = ModelConfig(**{k: v for k, v in model_dict.items() if k in ModelConfig.__dataclass_fields__})
    training_cfg = TrainingConfig(**{k: v for k, v in training_dict.items() if k in TrainingConfig.__dataclass_fields__})

    top_fields = {k for k in ExperimentConfig.__dataclass_fields__ if k not in ("data", "model", "training")}
    top_dict = {k: v for k, v in plain.items() if k in top_fields}

    return ExperimentConfig(data=data_cfg, model=model_cfg, training=training_cfg, **top_dict)


def load_experiment_config(
    config_dir: str = "configs",
    experiment: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` by merging YAML layers.

    Merge order (later wins):
        1. ``configs/base.yaml``
        2. ``configs/model/<model_name>.yaml``  (from base or experiment)
        3. ``configs/training/default.yaml``
        4. ``configs/experiment/<experiment>.yaml``  (if supplied)
        5. CLI overrides  (``key=value`` strings)

    Parameters
    ----------
    config_dir:
        Root directory containing YAML configs.
    experiment:
        Name of an experiment file (without ``.yaml``).
    overrides:
        OmegaConf dot-list overrides, e.g. ``["training.lr=0.01"]``.

    Returns
    -------
    ExperimentConfig
    """
    base_path = os.path.join(config_dir, "base.yaml")
    training_path = os.path.join(config_dir, "training", "default.yaml")

    merged = _merge_yaml_files(base_path, training_path)

    # Detect model name to load model-specific config
    model_name: Optional[str] = None
    if experiment is not None:
        exp_path = os.path.join(config_dir, "experiment", f"{experiment}.yaml")
        if os.path.isfile(exp_path):
            exp_cfg = OmegaConf.load(exp_path)
            model_name = OmegaConf.select(exp_cfg, "model.name", default=None)

    if model_name is None:
        model_name = OmegaConf.select(merged, "model.name", default=None)

    if model_name is not None:
        model_path = os.path.join(config_dir, "model", f"{model_name.lower()}.yaml")
        merged = OmegaConf.merge(merged, _merge_yaml_files(model_path))

    # Merge experiment on top
    if experiment is not None:
        exp_path = os.path.join(config_dir, "experiment", f"{experiment}.yaml")
        merged = OmegaConf.merge(merged, _merge_yaml_files(exp_path))

    # CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli_cfg)

    return _dictconfig_to_experiment(merged)
