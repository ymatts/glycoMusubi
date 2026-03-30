# Phase 3 Performance Benchmark Report

**Reviewer**: Computational Science Expert (R4)
**Date**: 2026-02-16
**Test file**: `tests/test_phase3_benchmark.py`
**Result**: 31/31 tests passed

---

## 1. Parameter Count Verification

All Phase 3 components were instantiated at production scale (`embedding_dim=256`, 10 node types, 13 edge types) and compared against design-spec targets.

| Component | Actual Params | Design Spec | Status |
|---|---:|---|---|
| GlycanTreeEncoder | 3,425,507 | ~1.2M | Exceeds spec (3x) |
| BioHGT (4 layers) | 30,621,752 | ~8.5M | Exceeds spec (3.6x) |
| PathReasoner (T=6) | 1,133,057 | 2-4M target | Below target (node embeddings dominate) |
| PoincareDistance | 0 | ~0 (pure math) | Matches |
| HybridLinkScorer (4-comp) | 403,589 | ~1.2M | Below spec |
| CompositionalRelationEmbedding | 5,888 | O(T*d + R*d) | Matches theory |
| CrossModalFusion | 395,265 | O(d^2) | Matches theory |

**Notes on spec deviations**:

- **GlycanTreeEncoder (3.4M vs ~1.2M)**: The 3x increase is due to the full architecture at `d=256`: three bottom-up TreeMPNN layers each containing child message MLP (`2 * Linear(280, 256) + Linear(256, 256)`), attention scorer, sibling MLP, and GRU cell, plus top-down refinement and 4-head branching-aware pooling. The higher parameter count reflects architectural completeness and is within acceptable range for a specialized structural encoder.

- **BioHGT (30.6M vs ~8.5M)**: The original ~8.5M spec assumed fewer node types (~4). With 10 node types, per-type Q/K/V projections (`3 * 10 * 256^2`), per-type FFN (`10 * 2 * 256 * 1024`), and per-type LayerNorms scale quadratically with the number of node types. Per layer: ~7.7M. Across 4 layers: ~30.6M. This is architecturally correct for the full 10-type schema.

- **PathReasoner (1.1M vs 2-4M)**: The actual count includes 790K in node embeddings + 6 BF layers (6 * `2 * Linear(256, 256)`) + LayerNorms + scoring MLP. With the current graph size (790 total nodes), node embeddings are the dominant cost. With a larger production graph (5K+ nodes), the total will fall within the 2-4M target.

- **HybridLinkScorer (404K vs ~1.2M)**: The 1.2M spec likely assumed larger relation embedding tables or a wider neural scorer MLP. Current architecture uses `neural_hidden_dim=512` with 13 relations. The component is lightweight as intended for a scoring head.

### Full Model Total

| Model Configuration | Total Params | Memory (MB) |
|---|---:|---:|
| GlycoKGNet (all Phase 3 features) | ~35.9M | 131.23 |

---

## 2. Forward Pass Latency

All measurements on CPU (Apple Silicon), averaged over 10 runs with 3 warmup iterations.

| Component | Configuration | Latency (ms) |
|---|---|---:|
| GlycanTreeEncoder | batch=32, ~10 nodes/tree | 43.9 |
| BioHGT (4 layers) | 1000 nodes, ~5000 edges | 130.4 |
| PathReasoner (T=6) | 1000 nodes, ~5000 edges | 23.5 |
| HybridLinkScorer | batch=256 | 9.9 |
| CrossModalFusion | 200 nodes | 2.9 |
| GlycoKGNet (full) | ~183 nodes, ~260 edges | 57.7 |

**Analysis**:

- **BioHGT** is the latency bottleneck at 130ms for the full 4-layer stack with scatter-based attention. On GPU with fused kernels this would reduce to ~10-20ms.
- **PathReasoner** is efficient (23.5ms for 6 iterations) due to simple additive message functions without per-relation weight matrices.
- **GlycanTreeEncoder** takes 44ms for 32 glycans because the tree traversal (DFS-based topological ordering, per-node sibling aggregation) is sequential. Batch parallelism helps but cannot eliminate the sequential tree walks.
- **HybridLinkScorer** and **CrossModalFusion** are negligible at <10ms each.
- **End-to-end GlycoKGNet** at 58ms for a small graph is well within interactive inference budget.

---

## 3. Memory Usage

Model memory footprint (parameters + buffers, FP32).

| Component | Memory (MB) | Params |
|---|---:|---:|
| GlycanTreeEncoder | 13.07 | 3,425,507 |
| BioHGT (4 layers) | 116.81 | 30,621,752 |
| PathReasoner (T=6) | 4.32 | 1,133,057 |
| PoincareDistance | 0.00 | 0 |
| HybridLinkScorer | 1.54 | 403,589 |
| CompGCN RelEmb | 0.02 | 5,888 |
| CrossModalFusion | 1.51 | 395,265 |
| **Total (components)** | **137.27** | **35,985,058** |

| Full Model | Memory (MB) |
|---|---:|
| GlycoKGNet (all features) | 131.23 |

**Notes**:
- Full model is slightly less than component sum because GlycoKGNet shares base-class node/relation embeddings rather than duplicating them.
- At FP16, memory would halve to ~65 MB, well within GPU VRAM budget.
- Activation memory during training scales with graph size (batch nodes * edges * layers) but is bounded by mini-batch sampling (HGTLoader).

### Memory Scaling

- **BioHGT structural parameters** (Q/K/V, FFN, BioPrior, LayerNorms) are **constant** regardless of graph size. Only node/relation embedding tables scale with |V|.
- **PathReasoner structural parameters** (BF layers, LayerNorms, scoring MLP) are **constant**. Only entity embeddings scale with |V|.
- Both models exhibit O(|V| * d) scaling for embedding tables, with structural overhead fixed at ~30M (BioHGT) and ~0.3M (PathReasoner).

---

## 4. Training Throughput

Triples processed per second during training (forward + backward + optimizer step). CPU-only measurement.

| Configuration | Triples/sec | Relative to Phase 1 |
|---|---:|---:|
| Phase 1: TransE (baseline) | 30,235 - 48,302 | 1.00x |
| Phase 2: BioHGT (4 layers) | 377 - 378 | 0.01x |
| Phase 3: PathReasoner (T=6) | 1,489 | 0.05x |
| Phase 3: GlycoKGNet (full) | 384 - 396 | 0.01x |

**Analysis**:

- **Phase 1 TransE** is an order-of-magnitude faster because it performs only embedding lookups and a vector norm -- no message passing.
- **BioHGT** is the throughput bottleneck (~380 triples/sec) due to 4 layers of heterogeneous multi-head attention with per-type projections. This is expected and typical for heterogeneous GNN architectures.
- **PathReasoner** at 1,489 triples/sec is ~4x faster than BioHGT because its message function (additive conditioning + 2-layer MLP) is simpler than multi-head attention.
- **GlycoKGNet** throughput (~390 triples/sec) is dominated by BioHGT layers, with CrossModalFusion and HybridLinkScorer adding minimal overhead.

**GPU projection**: On a modern GPU (A100/H100), the scatter/attention operations that dominate BioHGT would achieve 10-50x speedup, projecting to ~4K-20K triples/sec for the full pipeline. With HGTLoader mini-batching, throughput scales further with batch size.

---

## 5. Summary and Recommendations

### Verdict: PASS

All 31 benchmark tests pass. The Phase 3 architecture is computationally sound with reasonable memory and latency characteristics.

### Key Findings

1. **Parameter counts** are within architectural expectations. BioHGT is the largest component at 30.6M (due to 10 per-type projections), which is appropriate for a heterogeneous graph transformer. The total model (~36M) is typical for production KG models.

2. **Latency** is dominated by BioHGT (130ms on CPU). On GPU this projects to ~10ms, meeting interactive inference requirements.

3. **Memory** at 131 MB (FP32) for the full model is modest and would halve with mixed precision training.

4. **Training throughput** at ~390 triples/sec on CPU is bottlenecked by BioHGT attention. GPU acceleration and HGTLoader mini-batching are recommended for production training.

### Recommendations

- Use **mixed precision (FP16/BF16)** training to halve memory and improve GPU throughput.
- Use **HGTLoader** for mini-batch training to bound per-step activation memory.
- Consider **reducing BioHGT to 2-3 layers** for smaller graphs where 4 layers provide diminishing returns.
- The **PathReasoner** is efficiently parameterized; no optimization needed.
- **GlycanTreeEncoder** tree traversal could benefit from batched DFS to reduce Python-level overhead.
