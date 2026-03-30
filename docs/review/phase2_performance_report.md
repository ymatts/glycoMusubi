# Phase 2 Performance Benchmark Report: Phase 1 vs Phase 2

**Reviewer**: Computational Science Expert (R4)
**Date**: 2026-02-13
**Status**: PASS (all benchmarks passed)
**Test file**: `tests/test_phase2_benchmark.py` (38 tests, all passing)

---

## 1. Executive Summary

This report compares Phase 1 baseline models (TransE, DistMult, RotatE) against
Phase 2 GlycoKGNet components (BioHGT, CrossModalFusion, HybridLinkScorer) on a
synthetic mini knowledge graph with 76 nodes across 7 types and ~74 edges
across 7 relation types. Embedding dimension d=64 was used for tractable
benchmarking.

**Key findings**:

1. All Phase 2 components produce valid, finite, differentiable outputs.
2. BioHGT achieves 84.3% loss reduction in 50 steps (vs 61-95% for Phase 1).
3. Phase 2 introduces ~828K total parameters vs ~5K for Phase 1 (expected given
   attention layers, FFN, BioPrior, neural scorer MLP).
4. Full Phase 2 pipeline (BioHGT -> CrossModalFusion -> HybridLinkScorer) is
   end-to-end differentiable with gradient flow verified to all components.
5. No numerical instabilities, NaN/Inf values, or training divergence observed.

| Category | Tests | Pass | Fail | Verdict |
|----------|-------|------|------|---------|
| Parameter count comparison | 5 | 5 | 0 | PASS |
| Forward pass correctness | 5 | 5 | 0 | PASS |
| Scoring function comparison | 4 | 4 | 0 | PASS |
| Convergence speed | 3 | 3 | 0 | PASS |
| Memory usage | 4 | 4 | 0 | PASS |
| Timing comparison | 3 | 3 | 0 | PASS |
| Link prediction metrics | 6 | 6 | 0 | PASS |
| Composite loss integration | 1 | 1 | 0 | PASS |
| CrossModalFusion integration | 4 | 4 | 0 | PASS |
| Phase 2 pipeline integration | 3 | 3 | 0 | PASS |
| **Total** | **38** | **38** | **0** | **PASS** |

---

## 2. Parameter Count Comparison

| Model | Parameters | Category |
|-------|-----------|----------|
| TransE | 5,312 | Phase 1 baseline |
| DistMult | 5,312 | Phase 1 baseline |
| RotatE | 5,088 | Phase 1 baseline (half-dim relation) |
| **BioHGT** (2 layers, 4 heads) | **702,862** | Phase 2 encoder |
| HybridLinkScorer | 100,196 | Phase 2 decoder |
| CrossModalFusion | 25,089 | Phase 2 fusion |
| **Phase 2 Total** | **828,147** | Combined |

**Analysis**: Phase 1 models contain only per-type node embedding tables
(N * d) and relation embeddings (R * d), totaling ~5K parameters for the mini
KG. Phase 2 introduces:

- **BioHGT**: Type-specific Q/K/V linear transforms (7 types x 3 x d^2),
  relation-conditioned attention matrices (7 rels x H x d_k^2), FFN layers
  (7 types x 4 x d x d), BioPrior networks, and LayerNorm per type per layer.
- **HybridLinkScorer**: DistMult + RotatE relation embeddings, neural scorer
  MLP (3d -> 512 -> 1), and per-relation weight network (d -> 3).
- **CrossModalFusion**: MultiheadAttention (4 x d^2), gate MLP (2d -> d -> 1),
  and LayerNorm.

The parameter increase is proportional to the architectural complexity and is
expected. At production scale (d=256, full KG), the ratio narrows because node
embedding tables dominate.

**Verdict**: PASS -- parameter counts match architectural expectations.

---

## 3. Memory Usage Comparison

| Model | Memory (bytes) | Memory (KB) |
|-------|---------------|-------------|
| TransE | 21,248 | 20.8 |
| DistMult | 21,248 | 20.8 |
| RotatE | 20,352 | 19.9 |
| **BioHGT** | **2,811,448** | **2,745.6** |
| HybridLinkScorer | 400,784 | 391.4 |
| CrossModalFusion | 100,356 | 98.0 |
| **Phase 2 Total** | **3,312,588** | **3,235.0** |

**Analysis**: Memory footprint is dominated by BioHGT's attention and FFN
parameters. The Phase 2 total of ~3.2 MB is well within acceptable bounds for
a modern ML pipeline. At production scale with d=256, BioHGT memory scales as
O(L * (T * 3d^2 + R * H * d_k^2 + T * 4d^2 + T * 2d)) where L=layers,
T=types, R=relations, H=heads, d_k=d/H.

**Verdict**: PASS -- memory usage is reasonable and bounded.

---

## 4. Forward Pass Timing

| Model | Avg Time (ms) | Relative to TransE |
|-------|--------------|-------------------|
| TransE | 0.041 | 1.0x |
| DistMult | 0.042 | 1.0x |
| RotatE | 0.043 | 1.0x |
| **BioHGT** | **4.788** | **~117x** |

**Analysis**: Phase 1 models perform only embedding table lookups (O(N * d)),
while BioHGT performs multi-layer heterogeneous message passing with attention
computation, scatter operations, and FFN layers. The ~5ms forward pass for
BioHGT on this 76-node mini KG is well within acceptable bounds.

The 117x slowdown relative to TransE is expected: BioHGT performs 2 layers of
attention computation across 7 edge types with Q/K/V transforms, relation-
conditioned attention, scatter softmax, and FFN. This is the computational
cost of learning structure-aware representations rather than static embeddings.

**Verdict**: PASS -- timing is bounded and architecturally expected.

---

## 5. Training Convergence Comparison

All models trained for 50 steps on protein->glycan link prediction with
MarginRankingLoss (margin=5.0).

| Model | LR | Initial Loss | Final Loss | Loss Reduction |
|-------|------|-------------|------------|---------------|
| TransE | 0.01 | 4.984 | 1.946 | 61.0% |
| DistMult | 0.01 | 4.985 | 0.250 | 95.0% |
| RotatE | 0.01 | 5.241 | 0.261 | 95.0% |
| **BioHGT** | **0.001** | **4.609** | **0.725** | **84.3%** |

**Analysis**:

- **DistMult and RotatE** converge fastest (95% reduction), which is expected
  for bilinear/rotational models on small graphs where the scoring functions
  can easily separate positive and negative triples.
- **TransE** converges more slowly (61%) because L2-distance scoring requires
  more precise alignment of head+relation to tail vectors.
- **BioHGT** achieves strong convergence (84.3%) despite using a 10x lower
  learning rate (0.001 vs 0.01). This is necessary because attention-based
  models have more complex loss landscapes. The lower LR prevents instability
  while still achieving substantial loss reduction.

**Important**: BioHGT's advantage is not faster convergence on toy data but
richer representations from message passing. On real KGs with graph structure,
BioHGT is expected to achieve higher final metric quality due to its ability
to propagate information along multi-hop paths and leverage biology-aware
attention priors.

**Verdict**: PASS -- all models converge within budget; BioHGT shows healthy
training dynamics.

---

## 6. Scoring Function Correctness

### 6.1 Phase 1 Scoring Functions

| Model | Formula | Differentiable | Finite Output | Batch Consistent |
|-------|---------|---------------|---------------|-----------------|
| TransE | -\|\|h+r-t\|\|_p | Yes | Yes | Yes |
| DistMult | <h, r, t> | Yes | Yes | Yes |
| RotatE | -\|\|h*r-t\|\| (complex) | Yes | Yes | Yes |

### 6.2 Phase 2 Scoring Functions

| Component | Differentiable | Finite Output | Gradient Flow |
|-----------|---------------|---------------|--------------|
| BioHGT score (DistMult-style) | Yes | Yes | Yes |
| HybridLinkScorer (weighted combination) | Yes | Yes | Yes |
| CompositeLoss (link + contrastive + L2) | Yes | Yes | Yes |

### 6.3 HybridLinkScorer Sub-Score Combination

The HybridLinkScorer combines three sub-scorers with per-relation adaptive
weights:

```
score(h, r, t) = w1(r)*DistMult(h,r,t) + w2(r)*RotatE(h,r,t) + w3(r)*Neural(h,r,t)
```

where `w1, w2, w3 = softmax(W_weight @ r_distmult)`.

Verified properties:
- Output shape matches batch size: `[B]` for input `[B, d]`
- All sub-scores contribute (non-zero weights after initialization)
- Softmax normalization ensures weights sum to 1.0
- Gradients flow to all three sub-scorers and the weight network

**Verdict**: PASS -- all scoring functions are mathematically correct and
numerically stable.

---

## 7. Cross-Modal Fusion Verification

| Test | Result | Details |
|------|--------|---------|
| Shape preservation | PASS | Input `[N, d]` -> Output `[N, d]` |
| Masked fusion | PASS | Unmasked nodes remain unchanged |
| All-False mask passthrough | PASS | Returns h_kg unchanged |
| Differentiability | PASS | Gradients flow to both h_kg and h_modality |

The gated cross-attention mechanism correctly implements:
```
h_fused = gate * h_KG + (1 - gate) * CrossAttn(h_KG, h_modality)
```

**Verdict**: PASS -- CrossModalFusion is correct and compatible with the
pipeline.

---

## 8. End-to-End Phase 2 Pipeline

The full pipeline BioHGT -> CrossModalFusion -> HybridLinkScorer was tested
for:

| Test | Result |
|------|--------|
| Forward pass produces valid scores | PASS |
| Gradients flow through all 3 components | PASS |
| 10-step training reduces loss | PASS |

Gradient flow verification:
- **BioHGT encoder**: gradients reach node embedding tables, input projections,
  attention matrices, FFN layers, and BioPrior parameters.
- **CrossModalFusion**: gradients reach cross-attention Q/K/V/out projections,
  gate MLP, and LayerNorm.
- **HybridLinkScorer**: gradients reach all three relation embedding tables,
  neural scorer MLP, and weight network.

**Verdict**: PASS -- the Phase 2 pipeline is fully end-to-end differentiable.

---

## 9. Link Prediction Metric Infrastructure

| Test | Result |
|------|--------|
| `compute_ranks` returns valid 1-indexed ranks | PASS |
| MRR is in [0, 1] | PASS |
| Hits@K is in [0, 1] for K=1,3,10 | PASS |
| Perfect ranks yield MRR=1, Hits@K=1 | PASS |
| Phase 1 models produce full candidate rankings | PASS |
| BioHGT produces full candidate rankings | PASS |

All models can be evaluated with the standard filtered link prediction
protocol. The evaluation infrastructure (metrics.py, link_prediction.py)
is compatible with both Phase 1 and Phase 2 models.

**Verdict**: PASS -- evaluation infrastructure is ready for both phases.

---

## 10. Summary and Recommendations

### 10.1 Phase 1 vs Phase 2 Comparison Summary

| Dimension | Phase 1 (TransE/DistMult/RotatE) | Phase 2 (BioHGT + Fusion + HybridScorer) |
|-----------|----------------------------------|------------------------------------------|
| Parameters | ~5K | ~828K |
| Memory | ~21 KB | ~3.2 MB |
| Forward time | ~0.04 ms | ~4.8 ms |
| Convergence (50 steps) | 61-95% loss reduction | 84.3% loss reduction |
| Scoring | Single-decoder | Adaptive hybrid (3 sub-scorers) |
| Representation | Static embeddings (lookup) | Message-passing + cross-modal fusion |
| Biology awareness | None | BioPrior (biosynthetic, PTM crosstalk) |
| Multi-modal | No | Yes (gated cross-attention) |

### 10.2 Assessment

The Phase 2 architecture introduces substantial computational overhead (parameter
count, memory, latency) compared to Phase 1 baselines. This is expected and
justified by:

1. **Richer representations**: BioHGT produces structure-aware embeddings via
   heterogeneous graph attention with biology-informed priors, whereas Phase 1
   models use only static embedding lookups.

2. **Multi-modal integration**: CrossModalFusion enables incorporating glycan
   tree structure (WURCS-derived), protein sequence (ESM-2), and text features
   into the KG embedding space.

3. **Adaptive scoring**: HybridLinkScorer combines DistMult, RotatE, and neural
   scoring with per-relation learned weights, providing flexibility to handle
   diverse relation types (symmetric, antisymmetric, compositional).

4. **Domain-specific inductive biases**: BioPrior attention biases for
   enzyme-glycan biosynthetic pathway order and site-site PTM crosstalk
   encode glycobiology domain knowledge directly into the model.

### 10.3 Verdict

**PASS**: All 38 benchmark tests pass. Phase 2 components are correctly
implemented, numerically stable, and end-to-end differentiable. The
performance trade-offs (increased parameters and latency in exchange for
richer representations and biology-aware learning) are well-justified by
the architectural goals outlined in `docs/architecture/model_architecture_design.md`.

---

## Appendix: Test Execution

```
$ python3 -m pytest tests/test_phase2_benchmark.py -v
============================== 38 passed in 2.37s ==============================
```
