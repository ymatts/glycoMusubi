# glycoMusubi v0.2 Technical Design Document

**Version**: 0.2.0
**Date**: 2026-02-13
**Status**: Implementation Complete (48/48 design components, 100% coverage)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Knowledge Graph Schema](#2-knowledge-graph-schema)
3. [Four-Stage Architecture](#3-four-stage-architecture)
   - 3.1 [Stage 1: Modality-Specific Encoders](#31-stage-1-modality-specific-encoders)
   - 3.2 [Stage 2: BioHGT Message Passing](#32-stage-2-biohgt-message-passing)
   - 3.3 [Stage 3: Cross-Modal Attention Fusion](#33-stage-3-cross-modal-attention-fusion)
   - 3.4 [Stage 4: Task-Specific Decoders](#34-stage-4-task-specific-decoders)
4. [Training Strategy](#4-training-strategy)
   - 4.1 [Multi-Task Learning Objective](#41-multi-task-learning-objective)
   - 4.2 [Self-Supervised Pre-training](#42-self-supervised-pre-training)
   - 4.3 [Scalable Training Pipeline](#43-scalable-training-pipeline)
   - 4.4 [Callbacks and Checkpointing](#44-callbacks-and-checkpointing)
5. [Evaluation Framework](#5-evaluation-framework)
   - 5.1 [Intrinsic Evaluation](#51-intrinsic-evaluation-link-prediction)
   - 5.2 [Downstream Tasks](#52-downstream-evaluation-tasks)
   - 5.3 [KG Quality Metrics](#53-kg-quality-metrics)
   - 5.4 [Glyco-Specific Metrics](#54-glyco-specific-metrics)
   - 5.5 [Statistical Testing](#55-statistical-testing)
6. [Configuration System](#6-configuration-system)
7. [Module Reference](#7-module-reference)
8. [Parameter Budget](#8-parameter-budget)
9. [Data Integration](#9-data-integration)
10. [CLI and Pipeline](#10-cli-and-pipeline)
11. [Comparison with Existing Approaches](#11-comparison-with-existing-approaches)

---

## 1. System Overview

glycoMusubi is a deep learning framework for knowledge graph embedding in glycobiology. It jointly encodes heterogeneous biological entities (glycans, proteins, enzymes, diseases, compounds, PTM sites, motifs, reactions, and pathways) into a unified embedding space using a four-stage neural architecture.

### 1.1 Design Principles

- **Multi-modal**: Glycan tree structures (WURCS/IUPAC), protein sequences (ESM-2), textual metadata (PubMedBERT), and KG topology are integrated through modality-specific encoders and cross-modal fusion.
- **Biology-aware**: Biosynthetic pathway priors and PTM crosstalk knowledge are injected as inductive biases in the graph transformer attention mechanism.
- **Compact**: ~13M trainable parameters (excluding frozen foundation models) enable single-GPU training and fast iteration on biologically-sized datasets.
- **Evaluation-first**: Comprehensive evaluation spanning intrinsic KG quality, 6 downstream biological tasks, glyco-specific metrics, and rigorous statistical testing.

### 1.2 Package Structure

```
glycoMusubi/
  __init__.py                     # v0.2.0
  embedding/
    encoders/
      glycan_encoder.py           # Phase 1 learnable/WURCS/hybrid glycan encoder
      glycan_tree_encoder.py      # Phase 2 Tree-MPNN encoder with branching-aware pooling
      protein_encoder.py          # ESM-2 + site-aware protein encoder
      text_encoder.py             # PubMedBERT-based text encoder
      wurcs_tree_parser.py        # WURCS string -> monosaccharide tree parser
    models/
      base.py                     # BaseKGEModel abstract base class
      glycoMusubie.py                # TransE, DistMult, RotatE shallow models
      glycoMusubi_net.py             # GlycoKGNet unified 4-stage model
      biohgt.py                   # Biology-Aware Heterogeneous Graph Transformer
      cross_modal_fusion.py       # Gated cross-attention fusion
      path_reasoner.py            # NBFNet-style Bellman-Ford path reasoner
      poincare.py                 # Poincare ball model for hyperbolic scoring
      compgcn_rel.py              # CompGCN compositional relation embeddings
    decoders/
      distmult.py                 # DistMult bilinear decoder
      transe.py                   # TransE translational decoder
      rotate.py                   # RotatE complex-rotation decoder
      hybrid_scorer.py            # Hybrid link scorer (DM+RotatE+Neural+Poincare)
      node_classifier.py          # Per-task MLP node classification heads
      graph_level_decoder.py      # AttentiveReadout + MLP graph-level decoder
  training/
    trainer.py                    # Training loop with AMP, HGTLoader, gradient accumulation
    pretraining.py                # Masked node/edge prediction, glycan substructure
    callbacks.py                  # EarlyStopping, ModelCheckpoint, MetricsLogger
  evaluation/
    link_prediction.py            # Filtered link prediction evaluator
    metrics.py                    # MRR, Hits@K, MR, AMR
    downstream.py                 # BaseDownstreamTask ABC + DownstreamEvaluator
    tasks/                        # 6 downstream task implementations
    statistical_tests.py          # Auto-test, Holm-Bonferroni, BH-FDR, DeLong
    kg_quality.py                 # Graph density, entropy, connected components
    glyco_metrics.py              # GSR, CAS, THC
    multi_seed.py                 # Multi-seed robustness evaluation
    visualize.py                  # Embedding visualization utilities
  data/                           # KGConverter, GlycoKGDataset
  etl/                            # Extract-Transform-Load pipeline
  utils/                          # Config, logging, reproducibility, scatter ops
```

---

## 2. Knowledge Graph Schema

### 2.1 Node Types (10)

| Node Type  | Description                           | Primary Data Source     |
|------------|---------------------------------------|------------------------|
| `glycan`   | Glycan structures (WURCS/IUPAC)       | GlyTouCan, GlyGen      |
| `protein`  | Protein sequences                     | UniProt                |
| `enzyme`   | Glycosyltransferases / glycosidases   | UniProt, CAZy          |
| `disease`  | Disease entities                      | OMIM, DisGeNET         |
| `variant`  | Protein/gene variants                 | UniProt, ClinVar       |
| `compound` | Small molecule inhibitors             | ChEMBL                 |
| `site`     | Glycosylation / PTM sites             | UniProt, PhosphoSite   |
| `motif`    | Glycan structural motifs              | GlyTouCan              |
| `reaction` | Biochemical reactions                 | Reactome               |
| `pathway`  | Metabolic / signaling pathways        | Reactome, KEGG         |

### 2.2 Edge Types (13)

| Edge Type                    | Source Type | Target Type | Pattern           |
|------------------------------|-------------|-------------|-------------------|
| `has_glycan`                 | protein     | glycan      | Antisymmetric 1:N |
| `inhibits`                   | compound    | enzyme      | Antisymmetric N:1 |
| `associated_with_disease`    | protein     | disease     | N:N               |
| `has_variant`                | protein     | variant     | Antisymmetric     |
| `has_site`                   | protein     | site        | Antisymmetric     |
| `has_site` (enzyme)          | enzyme      | site        | Antisymmetric     |
| `ptm_crosstalk`              | site        | site        | Symmetric         |
| `produced_by`                | enzyme      | glycan      | Antisymmetric     |
| `consumed_by`                | enzyme      | glycan      | Antisymmetric     |
| `has_motif`                  | glycan      | motif       | Hierarchical      |
| `child_of`                   | glycan      | glycan      | Hierarchical      |
| `catalyzed_by`               | enzyme      | reaction    | Antisymmetric     |
| `has_product`                | reaction    | glycan      | Antisymmetric     |

All edge types are defined in `biohgt.py:DEFAULT_EDGE_TYPES` and used consistently throughout the pipeline.

---

## 3. Four-Stage Architecture

The unified `GlycoKGNet` model (`glycoMusubi_net.py`) implements a four-stage pipeline:

```
Input HeteroData
      |
      v
  Stage 1: Modality-Specific Encoders
  (GlycanTreeEncoder, ProteinEncoder, TextEncoder, learnable fallbacks)
      |
      v  Dict[str, Tensor[N_type, 256]]
  Stage 2: BioHGT Message Passing
  (4 layers x 8 heads, bio-prior attention biases)
      |
      v  Dict[str, Tensor[N_type, 256]]
  Stage 3: Cross-Modal Attention Fusion
  (Gated cross-attention for glycan/protein/text modalities)
      |
      v  Dict[str, Tensor[N_type, 256]]
  Stage 4: Task-Specific Decoders
  (HybridLinkScorer | NodeClassifier | GraphLevelDecoder)
      |
      v  Tensor[B] scores / Tensor[N, C] logits
```

### 3.1 Stage 1: Modality-Specific Encoders

#### 3.1.1 GlycanTreeEncoder

**File**: `encoders/glycan_tree_encoder.py`
**Parameters**: ~1.2M

Encodes glycan branching tree structures from WURCS strings into 256-dimensional embeddings using a hierarchical message-passing architecture.

**Node feature construction** (d=56):
- Monosaccharide type embedding: `Embedding(60, 32)` (~60 vocabulary)
- Anomeric configuration: `Embedding(3, 4)` (alpha, beta, unknown)
- Ring form: `Embedding(4, 4)` (pyranose, furanose, open, unknown)
- Modification encoding: `Linear(NUM_MODIFICATIONS, 16)` (multi-hot projected)

**Architecture**:

1. **Input projection**: `Linear(56, 256) -> GELU -> Dropout(0.1)`

2. **3 Bottom-up Tree-MPNN layers** (`TreeMPNNLayer`):
   For each node *v* in bottom-up topological order:
   ```
   Child aggregation:
     MSG(h_c, e_vc) = MLP_child([h_c || e_vc])     # d_model + d_edge -> d_model
     alpha_c = softmax_children(MLP_attn([h_c || h_v]))
     m_v = sum_c(alpha_c * MSG_c)

   Sibling aggregation:
     s_v = MLP_sibling(mean({h_s : s in siblings(v)}))

   GRU update:
     h_v^(l+1) = GRU(h_v^(l), [m_v || s_v])
     h_v^(l+1) = LayerNorm(h_v^(l+1))
   ```

3. **1 Top-down refinement layer** (`TopDownRefinement`):
   ```
   h_v^refined = MLP([h_v^bu || h_parent^refined])
   ```
   Root node uses zero vector for parent context.

4. **Branching-Aware Attention Pooling** (`BranchingAwarePooling`, 4 heads):
   ```
   Multi-head attention pooling:
     For each head k:
       score_i = MLP_k(h_i)
       alpha_i = softmax_per_graph(score_i)
       pool_k = sum_i(alpha_i * h_i)
     h_global = Linear(concat(pool_1, ..., pool_4))

   Branch feature:
     h_branch = mean(h_i for branching nodes i)

   Depth encoding:
     d_enc = Embedding(max_depth_per_graph)    # max_depth=32, d=8

   Fusion:
     output = MLP([h_global || h_branch || d_enc])    # 256*2+8 -> 256
   ```

**Edge features** (`LinkageEncoder`, d_edge=24):
- Parent carbon position: one-hot(7)
- Child carbon position: one-hot(7)
- Bond type: one-hot(3) (alpha, beta, unknown)
- `Linear(17, 24)` projection

#### 3.1.2 ProteinEncoder

**File**: `encoders/protein_encoder.py`
**Parameters**: ~0.8M (projection only; ESM-2 frozen)

Three encoding strategies:

| Mode               | Input          | Architecture                              |
|--------------------|----------------|-------------------------------------------|
| `learnable`        | Node index     | `Embedding(num_proteins, 256)`            |
| `esm2`             | Cached `.pt`   | Mean-pool ESM-2 -> `MLP(1280->512->256)`  |
| `esm2_site_aware`  | Per-residue    | Site-context extraction + merge MLP       |

**Site-aware mode** (`esm2_site_aware`):

For each glycosylation site at position *p*:
```
window_mean = mean(per_residue[p-15 : p+15])     # [esm2_dim]
global_mean = mean(per_residue)                    # [esm2_dim]
pe = sinusoidal_PE(p, dim=64)                      # [64]

site_context = MLP_site([window_mean || global_mean || pe])  # 1280+1280+64 -> 256
```

Site contexts are aggregated and merged:
```
seq_emb = MLP_proj(global_mean)                                 # 1280 -> 256
agg_sites = mean(site_contexts)                                 # [256]
count_enc = MLP_count(num_sites)                                # [32]
output = MLP_merge([seq_emb || agg_sites || count_enc])         # 256+256+32 -> 256
```

**Batched processing** (Phase 4+ optimization):
Proteins are classified into 3 groups for efficient computation:
1. **Site-aware**: per-residue ESM-2 + known sites (sequential due to variable site counts)
2. **ESM-2 only**: batched projection MLP forward
3. **Learnable**: batched embedding lookup

#### 3.1.3 TextEncoder

**File**: `encoders/text_encoder.py`
**Parameters**: ~0.3M

Encodes disease names and pathway descriptions using PubMedBERT (frozen) with a projection MLP.

```
input: node index
output: Embedding(num_entities, 256)    # learnable fallback
```

When PubMedBERT is available:
```
cls_token = PubMedBERT(text)[CLS]      # [768]
output = MLP(768 -> 384 -> 256)
```

### 3.2 Stage 2: BioHGT Message Passing

**File**: `models/biohgt.py`
**Parameters**: ~8.5M (4 layers)

Biology-Aware Heterogeneous Graph Transformer with type-specific transforms and domain-informed attention priors.

#### 3.2.1 BioHGTLayer

For each target node *i* of type T_i, aggregating from neighbors *j* of type T_j connected by relation r:

**Type-specific Q/K/V transforms**:
```
Q_i = W_Q^{T_i} * h_i + b_Q^{T_i}     # [E, out_dim] per node type
K_j = W_K^{T_j} * h_j + b_K^{T_j}
V_j = W_V^{T_j} * h_j + b_V^{T_j}
```

**Multi-head reshape** (H=8, d_k=32):
```
Q = Q.view(-1, 8, 32)    # [E, H, d_k]
K = K.view(-1, 8, 32)
V = V.view(-1, 8, 32)
```

**Relation-conditioned attention**:
```
W_attn^r ∈ R^{H x d_k x d_k}     # per-relation attention weight
K_rel = einsum("ehd,hdf->ehf", K, W_attn^r)
attn_logits = (Q * K_rel).sum(dim=-1) / sqrt(d_k)    # [E, H]
```

**Biology-aware prior** (`BioPrior`):
```
if relation in {produced_by, consumed_by}:
    bias = pathway_MLP([pathway_emb_src || pathway_emb_dst])
elif relation == ptm_crosstalk:
    dist_feat = Linear(|pos_src - pos_dst|)
    bias = crosstalk_MLP(dist_feat)
else:
    bias = learnable_scalar[relation]

attn_logits += bias.unsqueeze(-1)    # broadcast across heads
```

**Message aggregation**:
```
alpha = scatter_softmax(attn_logits, dst_idx)    # [E, H]
W_msg^r ∈ R^{H x d_k x d_k}
V_rel = einsum("ehd,hdf->ehf", V, W_msg^r)
msg = (alpha.unsqueeze(-1) * V_rel).view(-1, out_dim)
agg = scatter_sum(msg, dst_idx)
```

**Residual + FFN**:
```
h' = LayerNorm(h + Dropout(agg))
h'' = LayerNorm(h' + Dropout(FFN(h')))

FFN = Linear(256, 1024) -> GELU -> Dropout -> Linear(1024, 256)
```

#### 3.2.2 CompGCN Compositional Relations

**File**: `models/compgcn_rel.py`

```
RelEmb(r) = Compose(NodeTypeEmb(T_src), EdgeTypeEmb(edge_type), NodeTypeEmb(T_dst))
```

Three composition modes:
- **Subtraction**: `e_src - e_edge + e_dst`
- **Multiplication**: `e_src * e_edge * e_dst`
- **Circular correlation**: `IFFT(conj(FFT(e_src * e_edge)) * FFT(e_dst))`

#### 3.2.3 BioHGT Default Configuration

| Parameter       | Value |
|-----------------|-------|
| Layers          | 4     |
| Heads           | 8     |
| d_k             | 32    |
| FFN inner dim   | 1024  |
| Dropout         | 0.1   |
| Bio-prior       | On    |
| Hidden dim      | 256   |

### 3.3 Stage 3: Cross-Modal Attention Fusion

**File**: `models/cross_modal_fusion.py`
**Parameters**: ~0.5M

Integrates modality-specific features with KG-derived features using gated cross-attention. Applied only to node types with external modality encoders (glycan, protein, text-encoded disease/pathway).

```
Cross-attention:
  Q = h_KG                    # from BioHGT output
  K = V = h_modality           # from Stage 1 encoder
  h_cross = MultiheadAttention(Q, K, V)    # 4 heads

Gated fusion:
  gate = sigmoid(MLP([h_KG || h_cross]))    # 256*2 -> 256 -> 1
  h_fused = gate * h_KG + (1 - gate) * h_cross

Output:
  LayerNorm(h_fused)
```

For node types without modality features: `h_fused = h_KG` (passthrough).

Supports a boolean mask for partial fusion (only nodes with available modality features are processed).

### 3.4 Stage 4: Task-Specific Decoders

#### 3.4.1 HybridLinkScorer

**File**: `decoders/hybrid_scorer.py`
**Parameters**: ~1.2M

Combines four scoring paradigms with per-relation adaptive weights:

```
score(h, r, t) = w1(r) * DistMult(h, r, t)
               + w2(r) * RotatE(h, r, t)
               + w3(r) * Neural(h, r, t)
               + w4(r) * Poincare(h, r, t)
```

**Sub-scorers**:

| Scorer     | Formula                                            | Relation Dim |
|------------|----------------------------------------------------|--------------|
| DistMult   | `<h, r, t>` (element-wise product + sum)           | 256          |
| RotatE     | `-||h * exp(i*theta_r) - t||` (complex rotation)   | 128 (d/2)    |
| Neural     | `MLP([h || r || t])` (3*256 -> 512 -> 1)           | 256          |
| Poincare   | `-d_H(exp_0(h + r), exp_0(t))` (hyperbolic)        | 256          |

**Per-relation adaptive weights**:
```
[w1, w2, w3, w4] = softmax(Linear(rel_embed_distmult(r)))
```
Each relation type learns which scoring strategy is most effective.

#### 3.4.2 DistMult Fallback

When `HybridLinkScorer` is unavailable, `GlycoKGNet` falls back to simple DistMult scoring:
```
score(h, r, t) = (h * r * t).sum(dim=-1)
```

#### 3.4.3 NodeClassifier

**File**: `decoders/node_classifier.py`

Per-task two-layer MLP heads stored in `nn.ModuleDict`:
```
logits = Linear(128, num_classes)(
    Dropout(0.1)(
        GELU(
            Linear(256, 128)(h_node)
        )
    )
)
```

#### 3.4.4 GraphLevelDecoder

**File**: `decoders/graph_level_decoder.py`

AttentiveReadout pooling + MLP prediction:
```
gate_i = sigmoid(Linear(h_i))             # [N, 1]
h_transformed = Linear(h_i)               # [N, embed_dim]
gated = gate_i * h_transformed             # [N, embed_dim]
h_graph = scatter_sum(gated, batch)        # [B, embed_dim]
output = MLP(h_graph)                      # [B, num_classes]
```

### 3.5 PathReasoner (Auxiliary Model)

**File**: `models/path_reasoner.py`
**Parameters**: ~2-4M

NBFNet-style Bellman-Ford GNN for path-based link prediction.

**Architecture**:
- Query-conditioned initialization: only source node *h* receives non-zero representation
- T iterations of Bellman-Ford propagation (default T=6)
- Relation-conditioned message: `MSG(h_u, r) = MLP(h_u + e_r)`
- Aggregation modes: `sum`, `mean`, or `pna` (sum+mean+max with learned combination)
- Inverse edges appended automatically
- Scoring: `MLP([h_tail || e_relation]) -> scalar`

**Batched query scoring** (Phase 4+ optimization):
Queries are processed via super-graph replication with `max_parallel=64` chunking:
```
For B queries, replicate graph structure B times:
  batched_edge_index = edge_index + per-query node offsets
  h_init[b*N + head[b]] = entity_emb[head[b]]

BF iterations on [B*N] node super-graph
  -> reshape to [B, N, dim]
  -> score_mlp([h, r_expanded]) -> [B, N]
```

### 3.6 Poincare Ball Model

**File**: `models/poincare.py`

Implements the Poincare ball model B^d_c for hyperbolic embeddings:

- Mobius addition: `x ⊕_c y`
- Exponential map: `exp_x^c(v)` (tangent space -> ball)
- Logarithmic map: `log_x^c(y)` (ball -> tangent space)
- Poincare distance: `d_c(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕_c y||)`

**Scoring function**:
```
S_hyp(h, r, t) = -d_c(exp_0(e_h + r), exp_0(e_t))
```

Used by `HybridLinkScorer` for scoring hierarchical relations (`child_of`, `has_motif`).

---

## 4. Training Strategy

### 4.1 Multi-Task Learning Objective

```
L_total = lambda_1 * L_link         # link prediction (margin/InfoNCE)
        + lambda_2 * L_node         # node classification (cross-entropy)
        + lambda_3 * L_contrastive  # cross-modal alignment (InfoNCE)
        + lambda_4 * L_masked       # masked prediction (MSE + CE)

Default weights: lambda_1=1.0, lambda_2=0.5, lambda_3=0.3, lambda_4=0.2
```

### 4.2 Self-Supervised Pre-training

**File**: `training/pretraining.py`

Three pre-training objectives:

#### 4.2.1 Masked Node Feature Prediction (`MaskedNodePredictor`)

- Mask 15% of node features with learnable `[MASK]` token
- Predict original features from context
- Loss: MSE for continuous features, CE for categorical
- Applied per node type

#### 4.2.2 Masked Edge Prediction (`MaskedEdgePredictor`)

- Remove 10% of edges per edge type
- Predict edge existence (BCE) and relation type (CE)
- Negative sampling via random tail permutation
- Edges restored after prediction

#### 4.2.3 Glycan Substructure Prediction (`GlycanSubstructurePredictor`)

- Given glycan KG embedding, predict constituent monosaccharides
- Multi-label binary classification
- MLP: `Linear(256, 256) -> ReLU -> Dropout(0.1) -> Linear(256, num_mono_types)`
- Loss: binary cross-entropy with logits

### 4.3 Scalable Training Pipeline

**File**: `training/trainer.py`

The `Trainer` class supports both full-batch and mini-batch training modes:

#### 4.3.1 Full-Batch Training

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_data=train_data,
    mixed_precision=True,
)
history = trainer.fit(epochs=200, validate_every=5)
```

#### 4.3.2 HGTLoader Mini-Batch Training

Type-aware neighbor sampling for large graphs:

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_data=train_data,
    use_hgt_loader=True,
    hgt_num_samples=[15],          # neighbors per edge type per layer
    hgt_batch_size=1024,           # target seed nodes
    gradient_accumulation_steps=4,  # effective batch = 4096
    mixed_precision=True,
    amp_dtype=torch.float16,
)
```

#### 4.3.3 Learning Rate Scheduling

Default: `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`

The scheduler is resolved from the `scheduler` parameter:
- `None` -> default CAWR scheduler
- `"none"` -> no scheduling
- `"cosine_warm_restarts"` -> explicit CAWR
- `_LRScheduler` instance -> used as-is

#### 4.3.4 Mixed Precision Training

- AMP context with configurable dtype (`float16` or `bfloat16`)
- `GradScaler` for float16 on CUDA
- Gradient clipping with `grad_clip_norm`

### 4.4 Callbacks and Checkpointing

**File**: `training/callbacks.py`

| Callback          | Description                                                 |
|-------------------|-------------------------------------------------------------|
| `EarlyStopping`   | Stop when monitored metric (default `mrr`) stalls for N epochs |
| `ModelCheckpoint` | Save `best.pt` on metric improvement, optionally `last.pt`  |
| `MetricsLogger`   | Log to JSON-lines, W&B (wandb), and TensorBoard            |

**Checkpoint format**:
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,     # if scheduler is active
    "scaler_state_dict": dict,        # if mixed precision with GradScaler
}
```

---

## 5. Evaluation Framework

### 5.1 Intrinsic Evaluation: Link Prediction

**File**: `evaluation/link_prediction.py`

**Protocol**: Filtered ranking -- for each test triple `(h, r, t)`, all candidates are scored and known-true triples (excluding the test triple itself) are removed before computing metrics.

**Interface**: Models must implement the `ScorableModel` protocol:
```python
class ScorableModel(Protocol):
    def score_t(self, head, relation, num_entities) -> Tensor[B, E]:  # tail prediction
    def score_h(self, tail, relation, num_entities) -> Tensor[B, E]:  # head prediction
```

**Metrics**:

| Metric   | Formula                         | Target |
|----------|---------------------------------|--------|
| MRR      | Mean reciprocal rank            | > 0.35 |
| Hits@1   | Fraction of rank = 1            | > 0.25 |
| Hits@3   | Fraction of rank <= 3           | > 0.38 |
| Hits@10  | Fraction of rank <= 10          | > 0.50 |
| MR       | Mean rank                       | -      |
| AMR      | Adjusted mean rank              | -      |

**Result container** (`LinkPredictionResult`):
- `metrics`: Overall averaged metrics
- `head_metrics` / `tail_metrics`: Direction-specific metrics
- `per_relation`: Per-relation breakdown
- `num_triples`: Number of test triples evaluated

**Data split protocol**:
- 80/10/10 train/val/test (stratified random)
- Inverse relation leak prevention: `(h,r,t) in train -> (t,r_inv,h) not in test`
- 5 random seeds for reproducibility

### 5.2 Downstream Evaluation Tasks

**File**: `evaluation/downstream.py`

Abstract base class `BaseDownstreamTask` with methods:
- `prepare_data(embeddings, data) -> tuple`: Prepare train/test data from KG embeddings
- `evaluate(embeddings, data) -> Dict[str, float]`: Run evaluation
- `name -> str`: Human-readable task name

`DownstreamEvaluator` orchestrates running multiple tasks with optional multi-seed robustness evaluation.

#### 5.2.1 Six Downstream Tasks

**File**: `evaluation/tasks/`

| # | Task                                 | File                             | Metrics                                |
|---|--------------------------------------|----------------------------------|----------------------------------------|
| 1 | Glycan-Protein Interaction           | `glycan_protein_interaction.py`  | AUC-ROC > 0.85, AUC-PR > 0.60, F1     |
| 2 | Glycan Function (GlycanML)           | `glycan_function.py`             | Acc > 0.75, Macro F1 > 0.65, MCC       |
| 3 | Glycan Immunogenicity                | `immunogenicity.py`              | AUC-ROC > 0.80, Sens/Spec > 0.75 (*)  |
| 4 | Glycan-Disease Association           | `disease_association.py`         | AUC-ROC > 0.80, Recall@K, NDCG@K      |
| 5 | Glycosylation Binding Site           | `binding_site.py`                | Residue AUC > 0.85, Site F1 > 0.70    |
| 6 | Drug Target Identification           | `drug_target.py`                 | AUC-ROC > 0.80, Hit@K, EF > 10x       |

(*) Immunogenicity requires a label alignment step that maps external dataset IDs
(e.g., GlycanML internal numeric IDs / IUPAC strings) to GlyTouCan accessions used
by KG glycan nodes.

**Multi-seed evaluation**:
```python
evaluator = DownstreamEvaluator(tasks=[...])
results = evaluator.evaluate_multi_seed(
    model_factory=lambda: GlycoKGNet(...),
    data=hetero_data,
    seeds=[42, 123, 456, 789, 1024],
    train_fn=training_function,
)
# results: {task_name: {metric: {"mean": float, "std": float}}}
```

### 5.3 KG Quality Metrics

**File**: `evaluation/kg_quality.py`

```python
metrics = compute_kg_quality(data: HeteroData) -> Dict[str, float]
```

| Metric                       | Formula                                        |
|------------------------------|------------------------------------------------|
| `graph_density`              | `2|E| / (|V|(|V|-1))`                          |
| `avg_degree`                 | `2|E| / |V|`                                   |
| `num_connected_components`   | Weakly connected components (via networkx)     |
| `clustering_coefficient`     | Average local clustering coefficient           |
| `per_type_coverage`          | `{node_type: count / total_nodes}`             |
| `relation_entropy`           | Shannon entropy: `-sum(p * log(p))` over types |

### 5.4 Glyco-Specific Metrics

**File**: `evaluation/glyco_metrics.py`

#### 5.4.1 Glycan Structure Recovery (GSR)

```python
gsr = glycan_structure_recovery(
    structural_similarities: Tensor[N],    # e.g., Tanimoto on glycan fingerprints
    embedding_distances: Tensor[N],        # pairwise embedding distances
) -> float                                  # Spearman rank correlation [-1, 1]
```

Measures how well the embedding space preserves structural relationships between glycans.

#### 5.4.2 Cross-Modal Alignment Score (CAS)

```python
cas = cross_modal_alignment_score(
    glycan_emb: Tensor[G, d],
    protein_emb: Tensor[P, d],
    known_pairs: Tensor[K, 2],    # (glycan_idx, protein_idx) pairs
) -> float                         # Mean reciprocal rank
```

For each known glycan-protein binding pair, computes the rank of the true protein partner among all proteins by cosine similarity, then returns the average reciprocal rank.

#### 5.4.3 Taxonomy Hierarchical Consistency (THC)

```python
thc = taxonomy_hierarchical_consistency(
    predictions: Dict[str, Tensor],   # {level_name: predicted_class_indices}
    labels: Dict[str, Tensor],        # {level_name: ground_truth_indices}
) -> float                             # Fraction [0, 1]
```

Measures the fraction of instances where a correct parent-level prediction also has a correct child-level prediction. Ordered from coarsest to finest taxonomy level.

### 5.5 Statistical Testing

**File**: `evaluation/statistical_tests.py`

#### 5.5.1 Automatic Test Selection (`auto_test`)

```python
result = auto_test(scores_a, scores_b, alpha=0.05)
# -> {"statistic", "p_value", "test" (0=t-test, 1=Wilcoxon), "normality_p"}
```

1. Shapiro-Wilk normality test on paired differences
2. If normal (p >= alpha): paired t-test
3. If non-normal: Wilcoxon signed-rank test

#### 5.5.2 Multiple Comparison Corrections

**Holm-Bonferroni** (FWER control):
```python
adjusted = holm_bonferroni(p_values: List[float], alpha=0.05) -> List[float]
```

**Benjamini-Hochberg** (FDR control):
```python
rejected, adjusted = benjamini_hochberg(p_values: List[float], alpha=0.05)
# -> (List[bool], List[float])
```

BH is less conservative than Holm-Bonferroni and yields more power in large-scale testing.

#### 5.5.3 Effect Size

```python
d = cohens_d(group1: ndarray, group2: ndarray) -> float
```

Uses pooled standard deviation. Positive when mean(group1) > mean(group2).

#### 5.5.4 Bootstrap Confidence Intervals

```python
lower, upper = bootstrap_ci(
    statistic_fn=np.mean,
    data=scores,
    n_bootstrap=10000,
    ci=0.95,
    rng_seed=42,
)
```

#### 5.5.5 DeLong's Test

```python
p_value = delong_test(y_true, scores_a, scores_b) -> float
```

Compares two AUC values from correlated ROC curves using DeLong's nonparametric approach.

---

## 6. Configuration System

### 6.1 Base Configuration

**File**: `configs/base.yaml`

```yaml
seed: 42
embedding_dim: 256
device: auto          # auto | cuda | cpu
output_dir: experiments
log_level: INFO
deterministic: true

data:
  kg_dir: kg
  node_file: nodes.tsv
  edge_file: edges.tsv
  validation_ratio: 0.1
  test_ratio: 0.1
  num_neg_samples: 64
```

### 6.2 Model Configuration

**File**: `configs/model/glycoMusubi_net.yaml`

```yaml
model:
  name: GlycoKGNet
  embedding_dim: 256

  # Stage 1: Modality-specific encoders
  glycan_encoder_type: learnable     # learnable | wurcs_features | hybrid | tree_mpnn
  protein_encoder_type: learnable    # learnable | esm2 | esm2_site_aware

  # Stage 2: BioHGT
  num_hgt_layers: 4
  num_hgt_heads: 8
  use_bio_prior: true

  # Stage 3: Cross-Modal Fusion
  use_cross_modal_fusion: true
  num_fusion_heads: 4

  # Stage 4: Decoder
  decoder_type: hybrid               # hybrid | distmult

  # General
  dropout: 0.1

  # Data paths (override per environment)
  esm2_cache_path: null
  wurcs_map: null
  text_node_types:
    - disease
    - pathway
```

### 6.3 Configuration Loading

OmegaConf-based hierarchical configuration with CLI overrides:

```bash
# Full pipeline with defaults
python scripts/embedding_pipeline.py

# Override hyperparameters from CLI
python scripts/embedding_pipeline.py --experiment baseline_transe \
    training.lr=0.01 training.epochs=100

# Dry-run: show resolved config
python scripts/embedding_pipeline.py --dry-run
```

---

## 7. Module Reference

### 7.1 Models

| Class                              | File                       | Base Class      | Key Methods                                    |
|------------------------------------|----------------------------|-----------------|------------------------------------------------|
| `BaseKGEModel`                     | `models/base.py`           | `ABC, nn.Module`| `forward`, `score`, `score_triples`, `get_embeddings` |
| `GlycoKGNet`                       | `models/glycoMusubi_net.py`   | `BaseKGEModel`  | `encode`, `node_classify`, `predict_graph`     |
| `BioHGTLayer`                      | `models/biohgt.py`         | `nn.Module`     | `forward(x_dict, edge_index_dict)`             |
| `BioHGT`                           | `models/biohgt.py`         | `BaseKGEModel`  | `forward(data)`, `score`                       |
| `BioPrior`                         | `models/biohgt.py`         | `nn.Module`     | `forward(relation, src_idx, dst_idx)`          |
| `CrossModalFusion`                 | `models/cross_modal_fusion.py` | `nn.Module` | `forward(h_kg, h_modality, mask)`              |
| `PathReasoner`                     | `models/path_reasoner.py`  | `BaseKGEModel`  | `forward`, `score_query`                       |
| `BellmanFordLayer`                 | `models/path_reasoner.py`  | `nn.Module`     | `forward(h, edge_index, edge_rel_emb)`         |
| `PoincareDistance`                 | `models/poincare.py`       | `nn.Module`     | `forward`, `distance`, `mobius_add`, `exp_map`  |
| `CompositionalRelationEmbedding`   | `models/compgcn_rel.py`    | `nn.Module`     | `forward(src_type, edge_type, dst_type)`       |

### 7.2 Encoders

| Class                | File                          | Key Methods                                 |
|----------------------|-------------------------------|---------------------------------------------|
| `GlycanEncoder`      | `encoders/glycan_encoder.py`  | `forward(indices)`                          |
| `GlycanTreeEncoder`  | `encoders/glycan_tree_encoder.py` | `forward(trees)`, `encode_wurcs(wurcs_list)` |
| `ProteinEncoder`     | `encoders/protein_encoder.py` | `forward(indices)`, `clear_cache()`         |
| `TextEncoder`        | `encoders/text_encoder.py`    | `forward(indices)`                          |
| `LinkageEncoder`     | `encoders/glycan_tree_encoder.py` | `forward(parent_carbon, child_carbon, bond_type)` |
| `TreeMPNNLayer`      | `encoders/glycan_tree_encoder.py` | `forward(h, edge_index, edge_attr, ...)`   |
| `TopDownRefinement`  | `encoders/glycan_tree_encoder.py` | `forward(h, topo_order_td, parent_map)`    |
| `BranchingAwarePooling` | `encoders/glycan_tree_encoder.py` | `forward(h, batch, is_branch, depth)`   |

### 7.3 Decoders

| Class               | File                          | Scoring Formula                               |
|----------------------|-------------------------------|-----------------------------------------------|
| `DistMultDecoder`    | `decoders/distmult.py`        | `<h, r, t>`                                   |
| `TransEDecoder`      | `decoders/transe.py`          | `-||h + r - t||`                               |
| `RotatEDecoder`      | `decoders/rotate.py`          | `-||h * exp(i*theta_r) - t||`                  |
| `HybridLinkScorer`   | `decoders/hybrid_scorer.py`   | `w1*DM + w2*RotatE + w3*Neural + w4*Poincare` |
| `NodeClassifier`     | `decoders/node_classifier.py` | `MLP(h) -> logits[num_classes]`                |
| `GraphLevelDecoder`  | `decoders/graph_level_decoder.py` | `AttentiveReadout(h) -> MLP -> logits`     |

### 7.4 Training

| Class                         | File                     | Description                               |
|-------------------------------|--------------------------|-------------------------------------------|
| `Trainer`                     | `training/trainer.py`    | Main training loop with AMP, HGTLoader    |
| `MaskedNodePredictor`         | `training/pretraining.py`| Masked node feature prediction            |
| `MaskedEdgePredictor`         | `training/pretraining.py`| Masked edge prediction                    |
| `GlycanSubstructurePredictor` | `training/pretraining.py`| Monosaccharide composition prediction     |
| `Callback`                    | `training/callbacks.py`  | Base callback class                       |
| `EarlyStopping`               | `training/callbacks.py`  | Early stopping on metric plateau          |
| `ModelCheckpoint`             | `training/callbacks.py`  | Save best/last checkpoints                |
| `MetricsLogger`               | `training/callbacks.py`  | JSON-lines / W&B / TensorBoard logging    |

### 7.5 Evaluation

| Class / Function                | File                              | Description                          |
|---------------------------------|-----------------------------------|--------------------------------------|
| `LinkPredictionEvaluator`       | `evaluation/link_prediction.py`   | Filtered link prediction evaluation  |
| `BaseDownstreamTask`            | `evaluation/downstream.py`        | ABC for downstream tasks             |
| `DownstreamEvaluator`           | `evaluation/downstream.py`        | Multi-task/multi-seed evaluator      |
| `auto_test`                     | `evaluation/statistical_tests.py` | Normality-aware paired test          |
| `holm_bonferroni`               | `evaluation/statistical_tests.py` | FWER multiple comparison correction  |
| `benjamini_hochberg`            | `evaluation/statistical_tests.py` | FDR multiple comparison correction   |
| `cohens_d`                      | `evaluation/statistical_tests.py` | Effect size                          |
| `bootstrap_ci`                  | `evaluation/statistical_tests.py` | Bootstrap confidence intervals       |
| `delong_test`                   | `evaluation/statistical_tests.py` | AUC comparison test                  |
| `compute_kg_quality`            | `evaluation/kg_quality.py`        | Graph structural quality metrics     |
| `glycan_structure_recovery`     | `evaluation/glyco_metrics.py`     | GSR (Spearman correlation)           |
| `cross_modal_alignment_score`   | `evaluation/glyco_metrics.py`     | CAS (mean reciprocal rank)           |
| `taxonomy_hierarchical_consistency` | `evaluation/glyco_metrics.py` | THC (hierarchical consistency)       |

---

## 8. Parameter Budget

| Component                      | Trainable Parameters |
|--------------------------------|----------------------|
| GlycanTreeEncoder              | ~1.2M                |
| ProteinEncoder (projection)    | ~0.8M                |
| TextEncoder (projection)       | ~0.3M                |
| BioHGT (4 layers, 8 heads)    | ~8.5M                |
| CrossModalFusion               | ~0.5M                |
| HybridLinkScorer               | ~1.2M                |
| NodeClassifier heads (3x)      | ~0.15M               |
| GraphLevelDecoder              | ~0.2M                |
| Embeddings (node/relation)     | ~0.1M                |
| **Total Trainable**            | **~13M**             |

| Frozen Foundation Model        | Parameters | Purpose                    |
|--------------------------------|------------|----------------------------|
| ESM-2 (esm2_t33_650M_UR50D)   | 650M       | Protein per-residue embeds |
| PubMedBERT                     | 110M       | Disease/pathway text       |

**Design rationale**: The compact 13M trainable parameter budget enables:
- Single GPU training (A100 40GB or 2x V100 32GB)
- Reduced overfitting on biologically-sized datasets (typically 10K-100K samples)
- Fast iteration and hyperparameter tuning
- Foundation model features are pre-computed and cached (one-time cost)

---

## 9. Data Integration

### 9.1 External Data Sources

| Source              | Data Provided                                  | Node Types Populated            |
|---------------------|------------------------------------------------|---------------------------------|
| GlyGen              | Glycan structures (WURCS), interactions         | glycan, protein                 |
| GlyTouCan           | Glycan ontology, structure repository           | glycan, motif                   |
| UniProt             | Protein sequences, glycosylation sites          | protein, enzyme, site, variant  |
| ChEMBL              | Enzyme inhibitor assay data                     | compound                        |
| PhosphoSitePlus     | PTM crosstalk, site information                 | site                            |
| PTMCode             | PTM co-evolution data                           | site                            |
| OMIM                | Disease-gene associations                       | disease                         |
| Reactome            | Pathways, reactions                             | pathway, reaction               |
| KEGG                | Metabolic pathway data                          | pathway                         |
| CAZy                | Carbohydrate-active enzyme classification       | enzyme                          |

### 9.2 Feature Pre-computation

| Feature Type        | Method                                    | Storage                |
|---------------------|-------------------------------------------|------------------------|
| ESM-2 embeddings    | `esm2_t33_650M_UR50D` inference           | `{idx}.pt` per protein |
| WURCS tree parsing  | `wurcs_tree_parser.parse_wurcs_to_tree()`  | On-the-fly             |
| PubMedBERT          | CLS token extraction                      | Cached per entity      |
| Glycan fingerprints | Tanimoto-compatible fingerprints           | Pre-computed array     |

### 9.3 ETL Pipeline

The `glycoMusubi.etl` module handles data ingestion and knowledge graph construction:
- TSV-based node and edge files
- `KGConverter` for converting to PyG-compatible triples
- `GlycoKGDataset` for serialization/deserialization

### 9.4 Label Alignment Layer (Required for Some Downstream Tasks)

Some downstream labels are published in identifier spaces different from the KG node IDs.
In particular, immunogenicity labels may be provided with dataset-internal IDs or IUPAC
strings, while KG glycans use GlyTouCan accessions.

Therefore, the pipeline must include an explicit label alignment step:

1. Normalize source labels to `(source_id_or_iupac, label)`.
2. Resolve `source_id_or_iupac -> GlyTouCan AC` using a mapping table or structure-based conversion.
3. Emit `data_clean/glycan_immunogenicity.tsv` with columns `glycan_id, label`.
4. During `build`, merge labels into glycan node metadata (`immunogenicity`).
5. During `featurize`, attach `data["glycan"].y` from metadata.

Without Step 2, the immunogenicity task is expected to be skipped / empty.

---

## 10. CLI and Pipeline

**File**: `scripts/embedding_pipeline.py`

Three-stage pipeline:

| Stage       | Function        | Description                                        |
|-------------|-----------------|-----------------------------------------------------|
| `featurize` | `run_featurize` | Convert KG TSV -> PyTorch-Geometric `GlycoKGDataset` |
| `train`     | `run_train`     | Train KGE model via `Trainer`                        |
| `evaluate`  | `run_evaluate`  | Link prediction evaluation + embedding export        |

**Usage**:

```bash
# Full pipeline with default config
python scripts/embedding_pipeline.py

# Specific experiment with CLI overrides
python scripts/embedding_pipeline.py --experiment baseline_transe \
    training.lr=0.01 training.epochs=100

# Single stage
python scripts/embedding_pipeline.py --stage featurize

# Multiple stages
python scripts/embedding_pipeline.py --stage featurize train

# Dry-run (show resolved config)
python scripts/embedding_pipeline.py --dry-run
```

**Reproducibility**:
- Deterministic mode via `set_deterministic(True)`
- Seed propagation: `set_seed(cfg.seed)` at pipeline start
- All random operations use seeded generators

---

## 11. Comparison with Existing Approaches

| Feature                  | SweetNet | GIFFLAR  | ProtHGT  | BioPathNet | **GlycoKGNet**       |
|--------------------------|----------|----------|----------|------------|----------------------|
| Glycan tree structure    | Flat GNN | Higher-order | N/A  | N/A        | **Tree-MPNN**        |
| Heterogeneous KG         | No       | No       | Yes      | Yes (homog.)| **Yes**             |
| Protein PLM integration  | No       | No       | ESM-2    | No         | **ESM-2 site-aware** |
| Multi-modal fusion       | No       | No       | No       | No         | **Gated cross-attn** |
| Biosynthetic priors      | No       | No       | No       | No         | **BioPrior**         |
| Link prediction          | No       | No       | No       | Path-based | **Hybrid 4-scorer**  |
| Glycan+KG joint          | No       | No       | No       | No         | **Yes**              |
| Contrastive pre-training | No       | No       | Graph CL | No         | **Cross-modal CL**   |
| Hyperbolic scoring       | No       | No       | No       | No         | **Poincare**         |
| Path reasoning           | No       | No       | No       | Yes        | **BF-GNN**           |

### Key Novelties

1. **Glycan Tree-Aware Message Passing within HetKG**: First framework to jointly encode glycan branching tree structure and heterogeneous KG topology.

2. **Biology-Aware Attention Priors**: Learnable biosynthetic pathway order and PTM crosstalk biases in the graph transformer attention mechanism.

3. **Hybrid Link Scoring**: Relation-adaptive weighted combination of DistMult (bilinear), RotatE (rotational), Neural (MLP), and Poincare (hyperbolic) scoring -- each relation type learns which paradigm is most effective.

4. **Cross-Modal Contrastive Alignment**: Pre-training that aligns glycan structure, protein sequence, and KG topology embeddings for downstream transfer.

5. **Comprehensive glycoMusubi System**: End-to-end embedding of 10 node types spanning glycans, proteins, enzymes, diseases, compounds, PTM sites, motifs, reactions, and pathways.

---

## Appendix A: Implementation Status

Design compliance as of v0.2.0: **100% (48/48 components)**

| Phase   | Coverage | Components Added                                               |
|---------|----------|----------------------------------------------------------------|
| Phase 1 | 35%      | BaseKGEModel, shallow models, basic encoders, training loop    |
| Phase 2 | 63%      | BioHGT, TreeEncoder, ESM-2, Poincare, CompGCN, PathReasoner   |
| Phase 3 | 87%      | HybridScorer, pre-training, fusion, evaluation pipeline        |
| Phase 4 | 100%     | Downstream tasks, statistical tests, KG metrics, HGTLoader    |

Test suite: 1,788 tests passing (12 skipped for optional dependencies).

## Appendix B: Hardware Requirements

| Task                    | Minimum Hardware          | Recommended                |
|-------------------------|---------------------------|----------------------------|
| Training (full)         | 1x V100 32GB              | 1x A100 40GB               |
| Training (mini-batch)   | 1x V100 16GB              | 1x A100 40GB               |
| ESM-2 pre-computation   | 1x A100 40GB (one-time)   | 1x A100 80GB               |
| Inference               | 1x V100 16GB              | Any GPU with 8GB+ VRAM     |
| CPU-only (small KG)     | 32GB RAM                  | 64GB RAM                   |

## Appendix C: Reproducibility Seeds

Default evaluation seeds: `[42, 123, 456, 789, 1024]`

All random operations (data splits, negative sampling, model initialization, dropout) respect the configured seed for full reproducibility when `deterministic: true`.
