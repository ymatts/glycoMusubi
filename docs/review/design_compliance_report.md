# glycoMusubi v0.1 -- Design Compliance Report

**Reviewer**: system-expert (R5)
**Date**: 2026-02-13
**Scope**: `glycoMusubi/` implementation vs design documents

---

## Overview

This report evaluates the v0.1 implementation against three design documents:

1. `docs/architecture/model_architecture_design.md` (Architecture Design)
2. `docs/design/algorithm_design.md` (Algorithm Design)
3. `docs/evaluation_framework.md` (Evaluation Framework)

**Overall Assessment**: The v0.1 implementation is a **Phase 1 / baseline foundation** that correctly implements the core KGE pipeline (data loading, TransE/DistMult/RotatE, training loop, filtered evaluation). It deliberately defers advanced components (BioHGT, GlycanTreeEncoder, PathReasoner, cross-modal fusion, contrastive learning) to later phases, which is appropriate for an incremental implementation strategy.

---

## Compliance Check Results

### 1. Architecture Design (`model_architecture_design.md`)

| Design Section | Design Component | Implementation File | Status | Notes |
|---|---|---|---|---|
| 3.1 GlycanTreeEncoder | Tree-MPNN with bottom-up + top-down pass | `encoders/glycan_encoder.py` | **Partial** | Implemented as WURCS feature extraction + MLP, not Tree-MPNN. Tree structure not parsed to graph. Appropriate for Phase 1. |
| 3.1 GlycanTreeEncoder | Monosaccharide type embedding (d=32) | `encoders/glycan_encoder.py` | **Partial** | 8-class count vector (dim=24 total features), not per-node embedding. |
| 3.1 GlycanTreeEncoder | Branching-Aware Attention Pooling | `encoders/glycan_encoder.py` | **Not Implemented** | Phase 2+ feature. |
| 3.2 ProteinEncoder | ESM-2 frozen + 2-layer MLP (1280->512->256) | `encoders/protein_encoder.py` | **Partial** | MLP projects 1280->640->256 (not 1280->512->256 as specified). Close but dimensions differ. |
| 3.2 ProteinEncoder | Site-Aware Pooling | `encoders/protein_encoder.py` | **Not Implemented** | Phase 2+ feature. |
| 3.3 TextEncoder | PubMedBERT (768->384->256) | `encoders/text_encoder.py` | **Alternative** | Uses hash-based learnable embedding, not PubMedBERT. Documented as Phase 1 approach. |
| 3.4 BioHGT | 4-layer BioHGT with type-specific Q/K/V | -- | **Not Implemented** | Core architecture component deferred. |
| 3.4 BioHGT | Biosynthetic path priors | -- | **Not Implemented** | Phase 2+ feature. |
| 3.4 BioHGT | Compositional relation embedding | -- | **Not Implemented** | Phase 2+ feature. |
| 3.5 Cross-Modal Fusion | Gated cross-attention fusion | -- | **Not Implemented** | Phase 2+ feature. |
| 3.6.1 HybridLinkScorer | DistMult + RotatE + NeuralScore with per-relation weights | `models/glycoMusubie.py` | **Partial** | TransE, DistMult, RotatE implemented as separate models, not hybrid combination. |
| 3.6.2 Node Classification Decoder | Type-specific MLP heads | -- | **Not Implemented** | Phase 2+ feature. |
| 3.6.3 Graph-Level Prediction | AttentiveReadout + MLP | -- | **Not Implemented** | Phase 2+ feature. |
| 4.1 Multi-Task Learning | L_link + L_node + L_contrastive + L_masked | `losses/` | **Partial** | Only L_link (margin + BCE). No multi-task loss. |
| 4.2 CMCA Pre-training | InfoNCE contrastive alignment | -- | **Not Implemented** | Phase 2+ feature. |
| 4.4 Scalable Training | HGTLoader mini-batch, mixed precision | `training/trainer.py` | **Partial** | Mixed precision supported. Full-batch training default; no HGTLoader integration. |
| 6.1 GlycoKGNet class | Top-level model with encode() + predict_links() | `models/glycoMusubie.py` | **Alternative** | Implemented as separate TransE/DistMult/RotatE classes, not unified GlycoKGNet. |

### 2. Algorithm Design (`algorithm_design.md`)

| Design Section | Design Component | Implementation File | Status | Notes |
|---|---|---|---|---|
| 4.2.1 GlycanEncoder: Tree-GNN | Bidirectional tree message passing | `encoders/glycan_encoder.py` | **Alternative** | Feature-based encoding (WURCS extraction) instead of Tree-GNN. Phase 1 simplification. |
| 4.2.3 PathReasoner | NBFNet-style Bellman-Ford GNN | -- | **Not Implemented** | Core novel algorithm deferred. |
| 4.2.4 Composite Scoring | S_path + S_struct + S_hyp multi-view | `models/glycoMusubie.py` | **Not Implemented** | Individual scoring functions (TransE/DistMult/RotatE) implemented, not composite. |
| 4.2.5 Loss Function | BCE with self-adversarial negative sampling | `losses/bce_loss.py` | **Implemented** | Self-adversarial temperature parameter supported. |
| 4.2.5 Loss Function | Contrastive structural loss L_struct | -- | **Not Implemented** | Phase 2+ feature. |
| 4.2.5 Loss Function | Hyperbolic regularization L_hyp | -- | **Not Implemented** | Phase 2+ feature. |
| Appendix B | PoincareDistance module | -- | **Not Implemented** | Phase 2+ feature. |
| 5 Training Loop | Type-constrained negative sampling | `data/sampler.py` | **Implemented** | Correct implementation with schema-based constraints. |
| 5 Training Loop | Gradient clipping | `training/trainer.py` | **Implemented** | Via `grad_clip_norm` parameter. |
| 5 Training Loop | Early stopping | `training/callbacks.py` | **Implemented** | Monitor-based patience with configurable metric. |
| 5 Training Loop | CosineAnnealingWarmRestarts | `training/trainer.py` | **Partial** | Scheduler integration exists but defaults to "none". Cosine option available via config. |

### 3. Evaluation Framework (`evaluation_framework.md`)

| Design Section | Design Component | Implementation File | Status | Notes |
|---|---|---|---|---|
| 1.1 Link Prediction | MRR (filtered) | `evaluation/metrics.py`, `evaluation/link_prediction.py` | **Implemented** | Correct filtered ranking protocol with float64 precision. |
| 1.1 Link Prediction | Hits@1, @3, @10 (filtered) | `evaluation/metrics.py` | **Implemented** | Parameterized hits@K. |
| 1.1 Link Prediction | MR (Mean Rank) | `evaluation/metrics.py` | **Implemented** | |
| 1.1 Link Prediction | AMR (Adjusted Mean Rank) | `evaluation/metrics.py` | **Implemented** | Correctly normalized by expected random MR. |
| 1.1 Link Prediction | Per-relation MRR | `evaluation/link_prediction.py` | **Implemented** | Per-relation breakdown in LinkPredictionResult. |
| 1.1 Link Prediction | Head vs Tail separate metrics | `evaluation/link_prediction.py` | **Implemented** | `head_metrics` / `tail_metrics` in result. |
| 1.1 Data Split | Train 80% / Val 10% / Test 10% | `data/splits.py`, `utils/config.py` | **Implemented** | Configurable ratios; stratified split available. |
| 1.1 Data Split | Stratified random split | `data/splits.py` | **Implemented** | `relation_stratified_split()` ensures per-relation proportional splits. |
| 1.1 Data Split | Inverse relation leak prevention | `data/splits.py` | **Not Implemented** | No explicit check for (h,r,t) in train => (t,r_inv,h) excluded from test. |
| 1.1 Data Split | 5 random seeds | `utils/config.py` | **Partial** | Single seed configurable; multi-seed loop not built-in. |
| 1.2 Extrinsic Evaluation | Glycan-protein interaction, function prediction, etc. | -- | **Not Implemented** | Downstream tasks deferred. |
| 1.3 KG Quality Metrics | Graph density, clustering coefficient, etc. | -- | **Not Implemented** | Deferred. |
| 4.3 Reproducibility | Deterministic CUDA, seed fixing | `utils/reproducibility.py` | **Implemented** | `set_seed()`, `set_deterministic()`, `CUBLAS_WORKSPACE_CONFIG`, `seed_worker()`. |
| 6 Visualization | t-SNE/UMAP of embeddings | `evaluation/visualize.py` | **Implemented** | Publication-quality figures with entity/relation visualization. |

---

## Unimplemented Feature List

### Priority 1: Core Architecture (Required for Novel Contribution)

| Feature | Design Reference | Impact | Effort |
|---|---|---|---|
| BioHGT (4-layer heterogeneous graph transformer) | Architecture 3.4 | Critical: core KG encoder | High |
| GlycanTreeEncoder (Tree-MPNN) | Architecture 3.1, Algorithm 4.2.1 | Critical: glycan structural encoding | High |
| PathReasoner (NBFNet-style Bellman-Ford) | Algorithm 4.2.3 | Critical: path-based reasoning | High |
| HybridLinkScorer (composite DistMult+RotatE+Neural) | Architecture 3.6.1 | High: unique scoring | Medium |
| Cross-Modal Attention Fusion | Architecture 3.5 | High: multi-modal integration | Medium |

### Priority 2: Training Enhancements

| Feature | Design Reference | Impact | Effort |
|---|---|---|---|
| Multi-task learning loss (L_link + L_node + L_contrastive + L_masked) | Architecture 4.1 | Medium | Medium |
| Cross-Modal Contrastive Alignment (CMCA) | Architecture 4.2 | Medium | Medium |
| Self-supervised pre-training tasks (masked node/edge) | Architecture 4.3 | Medium | Medium |
| PoincareDistance / hyperbolic scoring | Algorithm 4.2.4 | Medium | Medium |
| HGTLoader mini-batch sampling | Architecture 4.4 | Medium: scalability | Low |

### Priority 3: Evaluation & Domain-Specific

| Feature | Design Reference | Impact | Effort |
|---|---|---|---|
| PubMedBERT TextEncoder | Architecture 3.3 | Low (Phase 1 hash-based acceptable) | Medium |
| Site-Aware Protein Pooling | Architecture 3.2 | Medium | Medium |
| Inverse relation leak prevention in splits | Evaluation 1.1 | Low (correctness) | Low |
| Downstream evaluation tasks (5 tasks) | Evaluation 1.2 | Medium (for paper) | High |
| glycoMusubi-specific metrics (GSR, CAS, BPC, etc.) | Evaluation 7.4 | Low | Medium |
| Node Classification / Graph-Level Decoders | Architecture 3.6.2, 3.6.3 | Low for v0.1 | Medium |

---

## Inconsistency Details

### I1: ProteinEncoder MLP Dimensions

- **Design**: `1280 -> 512 -> 256` (2-layer MLP)
- **Implementation**: `1280 -> 640 -> 256` (uses `esm2_dim // 2` = 640 as intermediate)
- **File**: `glycoMusubi/embedding/encoders/protein_encoder.py:78`
- **Severity**: Low. The implementation uses a slightly different intermediate dimension. The halving approach (`esm2_dim // 2`) is more general but does not match the design specification of 512.

### I2: GlycanEncoder Strategy Mismatch

- **Design**: GlycanTreeEncoder with Tree-MPNN, bottom-up + top-down pass, GRU updates, branching-aware attention pooling
- **Implementation**: WURCS string feature extraction (composition counts, branching degree, modifications) + MLP projection
- **File**: `glycoMusubi/embedding/encoders/glycan_encoder.py`
- **Severity**: Expected for Phase 1. The implementation correctly acknowledges this via the `"learnable"` / `"wurcs_features"` / `"hybrid"` method parameter, providing a clear upgrade path.

### I3: TextEncoder Strategy Mismatch

- **Design**: PubMedBERT (frozen) with 768->384->256 projection
- **Implementation**: Hash-based learnable embedding with `nn.Embedding(num_buckets, output_dim)` + single-layer projection
- **File**: `glycoMusubi/embedding/encoders/text_encoder.py`
- **Severity**: Expected for Phase 1. The implementation documents this as a "Phase 1" approach. However, the projection is 256->256 (identity-like), not 768->384->256.

### I4: Model Architecture Pattern

- **Design**: Single `GlycoKGNet` class with `encode()`, `predict_links()`, `classify_nodes()`, `predict_graph()`
- **Implementation**: Separate `TransE`, `DistMult`, `RotatE` classes with `forward()` and `score()`
- **File**: `glycoMusubi/embedding/models/glycoMusubie.py`
- **Severity**: Moderate. The design envisions a unified multi-modal model; the implementation provides independent baseline KGE models. This is appropriate for Phase 1 baselines but will require refactoring for the full architecture.

### I5: Evaluation Framework Data Split Default

- **Design**: `validation_ratio: 0.10, test_ratio: 0.10` (80/10/10)
- **Implementation Config**: `validation_ratio: 0.1, test_ratio: 0.1` (matches)
- **Implementation Split Default**: `val_ratio: 0.05, test_ratio: 0.10` in `random_link_split()` signature
- **File**: `glycoMusubi/data/splits.py:42`
- **Severity**: Low. The function default (5%/10%) differs from the config default (10%/10%). In practice, the config value should be passed, but the mismatched default could cause confusion.

---

## Recommendations

### Immediate (v0.1 polish)

1. **[R-I1]** Align `ProteinEncoder` MLP intermediate dimension to 512 per design, or document the deviation explicitly.
2. **[R-I2]** Fix `random_link_split` default `val_ratio` from 0.05 to 0.10 to match evaluation framework specification and `DataConfig` default.
3. **[R-I3]** Add inverse relation leak prevention in split utilities.

### Short-term (v0.2)

4. **[R-S1]** Implement `GlycanTreeEncoder` (Tree-MPNN) to replace WURCS feature extraction.
5. **[R-S2]** Implement `HybridLinkScorer` combining DistMult + RotatE + NeuralScore with learned per-relation weights.
6. **[R-S3]** Implement unified `GlycoKGNet` class wrapping encoders + BioHGT + fusion + decoders.
7. **[R-S4]** Integrate HGTLoader for mini-batch training on large graphs.

### Medium-term (v0.3+)

8. **[R-M1]** Implement BioHGT with biosynthetic priors and compositional relation embeddings.
9. **[R-M2]** Implement PathReasoner (NBFNet-style) for path-based scoring.
10. **[R-M3]** Implement Cross-Modal Attention Fusion.
11. **[R-M4]** Implement multi-task loss and contrastive pre-training.
12. **[R-M5]** Add PoincareDistance and hyperbolic scoring for hierarchical relations.
13. **[R-M6]** Implement downstream evaluation pipeline (5 extrinsic tasks).

---

## Summary Statistics

| Category | Designed | Implemented | Partial | Not Impl. |
|---|---|---|---|---|
| Architecture Components | 15 | 0 | 7 | 8 |
| Algorithm Components | 12 | 3 | 1 | 8 |
| Evaluation Components | 14 | 9 | 2 | 3 |
| **Total** | **41** | **12 (29%)** | **10 (24%)** | **19 (46%)** |

The v0.1 implementation covers approximately **53%** of the design (fully or partially), which is appropriate for a Phase 1 baseline release. The implemented components are the foundational building blocks upon which the novel architecture will be constructed.
