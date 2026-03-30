# Phase 2 Fusion & Hybrid Scoring Validity Report

**Reviewer**: Program Correctness Expert (R3)
**Date**: 2026-02-13
**Status**: CONDITIONAL PASS (2 bugs found, 3 design concerns noted)
**Scope**: CrossModalFusion, HybridLinkScorer, CompositeLoss, GlycoKGNet integration

---

## 1. Executive Summary

The Cross-Modal Fusion module, HybridLinkScorer, and CompositeLoss have been
reviewed for mathematical correctness, design specification compliance, and
integration soundness. The core algorithms are well-implemented and
mathematically sound. However, **two bugs** and **three design concerns** were
identified that should be addressed before production use.

| Category | Checks | Pass | Fail | Concern | Verdict |
|----------|--------|------|------|---------|---------|
| Fusion gate values [0,1] | 3 | 3 | 0 | 0 | PASS |
| Fused embeddings preserve both modalities | 4 | 4 | 0 | 0 | PASS |
| HybridLinkScorer per-relation weights | 4 | 4 | 0 | 0 | PASS |
| Composite scoring vs individual decoders | 3 | 2 | 0 | 1 | CONCERN |
| CompositeLoss convergence properties | 5 | 4 | 0 | 1 | CONCERN |
| GlycoKGNet integration correctness | 5 | 3 | 2 | 0 | FAIL (2 bugs) |
| Design specification compliance | 6 | 5 | 0 | 1 | CONCERN |
| **Total** | **30** | **25** | **2** | **3** | **CONDITIONAL PASS** |

---

## 2. CrossModalFusion: Gate Value Validity

### 2.1 Gate Range [0, 1] -- PASS

**Implementation** (`glycoMusubi/embedding/models/cross_modal_fusion.py:61-67`):

```python
self.gate_mlp = nn.Sequential(
    nn.Linear(embed_dim * 2, embed_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(embed_dim, 1),
    nn.Sigmoid(),     # <-- guarantees output in [0, 1]
)
```

The gate MLP terminates with `nn.Sigmoid()`, which maps R -> (0, 1). This
guarantees the gate value `g` satisfies `0 < g < 1` for all finite inputs.
At the floating-point boundary, Sigmoid(x) can return exactly 0.0 or 1.0
for extreme inputs, but this is non-problematic since the fusion formula
`h_fused = g * h_KG + (1 - g) * h_cross` remains valid at the boundaries.

**Verification**: The existing test `TestCrossModalFusionGate.test_gate_values_in_range`
(`tests/test_cross_modal_fusion.py:103-128`) uses a forward hook to capture
gate values and asserts `0 <= gate <= 1`. This test provides adequate coverage.

### 2.2 Gate Usage in Fusion Formula -- PASS

**Implementation** (`cross_modal_fusion.py:118-119`):

```python
gate = self.gate_mlp(torch.cat([h_kg_active, h_cross], dim=-1))  # [M, 1]
h_fused = gate * h_kg_active + (1.0 - gate) * h_cross           # [M, d]
```

This is a correct convex combination. When `gate=1`, the output is pure KG
embedding (h_KG); when `gate=0`, the output is pure cross-attention output.
The formula ensures smooth interpolation between the two modalities.

**Mathematical property**: For any `g in [0,1]`, `||h_fused|| <= max(||h_KG||, ||h_cross||)`
only if both sources are unit-normalized. Since they are not normalized before gating,
the LayerNorm on line 120 (`self.layer_norm(h_fused)`) compensates by stabilizing
the output norm.

### 2.3 Mask Semantics -- PASS

The mask handling is correct:
- `mask=None`: all nodes fused (lines 104-106)
- `mask` all-False: early return passthrough (lines 96-98)
- Partial mask: clone + scatter-back pattern (lines 123-126) correctly preserves
  unfused nodes while updating fused nodes.

---

## 3. Fused Embeddings: Modality Preservation

### 3.1 Cross-Attention Preserves Modality Information -- PASS

**Implementation** (`cross_modal_fusion.py:111-115`):

```python
q = h_kg_active.unsqueeze(1)       # [M, 1, d]
kv = h_mod_active.unsqueeze(1)     # [M, 1, d]
h_cross, _ = self.cross_attn(q, kv, kv)  # [M, 1, d]
```

The cross-attention module uses Q from KG embeddings and K/V from modality
embeddings. This design correctly:
1. Queries the modality space using KG-derived features (Q from h_KG)
2. Retrieves modality-specific information (V from h_modality)
3. Produces cross-attended output that is a weighted combination of modality values

Since each node is treated as a length-1 sequence, the attention reduces to a
scaled dot-product between Q_i and K_i, producing attention weight = softmax(Q_i . K_i / sqrt(d)).
With a single key, the softmax always outputs 1.0, meaning the cross-attention
output is simply `W_out @ (W_V @ h_modality)` -- a learned linear transform of
the modality embedding.

**Design note**: Using length-1 sequences means the multi-head attention is
effectively a learned gated projection, not a true attention mechanism over
multiple keys. This is mathematically valid but means attention scores carry no
selection information. Consider using node neighborhoods as multi-key context
in a future version for richer cross-attention.

### 3.2 Gated Fusion Preserves KG Information -- PASS

The gated combination `g * h_KG + (1-g) * h_cross` preserves the KG embedding
in proportion to `g`. Since the gate is input-dependent (computed from
`[h_KG || h_cross]`), the model can learn to:
- Increase `g` (keep KG) for nodes where structural information is redundant
- Decrease `g` (incorporate modality) for nodes with rich external features

### 3.3 LayerNorm After Fusion -- PASS

The `self.layer_norm(h_fused)` at line 120 normalizes the fused output.
This is standard practice and prevents scale drift between fused and
unfused (passthrough) nodes.

### 3.4 Spec Compliance: Design vs Implementation -- PASS

**Design spec** (Section 3.5 of `model_architecture_design.md:350-366`):
```
h_cross = attn * V
gate = sigmoid(W_gate * [h_KG || h_cross])
h_fused = gate * h_KG + (1 - gate) * h_cross
h_final = LayerNorm(h_fused)
```

**Implementation**: Matches the spec exactly. The `nn.MultiheadAttention` handles
the Q/K/V projections and attention computation, while the gate_mlp implements
the sigmoid gating and layer_norm provides the final normalization.

---

## 4. HybridLinkScorer: Per-Relation Weights

### 4.1 Softmax Normalization Guarantees -- PASS

**Implementation** (`hybrid_scorer.py:124`):

```python
weights = torch.softmax(self.weight_net(r_dm), dim=-1)  # [B, 3]
```

The `torch.softmax` over 3-dimensional output guarantees:
- All weights are in (0, 1)
- Weights sum to exactly 1.0
- The composite score is a true convex combination of the three sub-scores

**Verification**: The existing test `TestPerRelationWeights.test_weights_sum_to_one`
confirms this property for all relation types.

### 4.2 Per-Relation Weight Diversity -- PASS

The weight_net maps `r_dm -> [w1, w2, w3]` where `r_dm` is the DistMult relation
embedding. Since different relations have different embeddings (verified by
`TestRelationTypes.test_different_relations_give_different_scores`), the model
can learn distinct weighting strategies per relation type.

**Design rationale check**: The architecture spec (Section 3.6.1) states:
> Weights w1, w2, w3 are learned per relation type:
> [w1, w2, w3] = softmax(W_weight * RelEmb(r))

The implementation uses `self.weight_net(r_dm)` where `weight_net` is a `nn.Linear(embed_dim, 3)`.
This matches the spec: a single linear layer maps relation embedding to 3 logits.

### 4.3 Weight Initialization -- PASS

**Implementation** (`hybrid_scorer.py:82-84`):

```python
nn.init.zeros_(self.weight_net.bias)
nn.init.xavier_uniform_(self.weight_net.weight)
```

With zero bias, the initial softmax weights are approximately uniform (1/3 each),
meaning the model starts by giving equal weight to all three scoring paradigms.
This is a good initialization strategy that avoids privileging any single scorer
at the start of training.

### 4.4 Weight Convergence Properties -- PASS

The per-relation weights are computed as:
```
w = softmax(W @ r_dm)
```

where `W` is a `[3, d]` matrix and `r_dm` is `[B, d]`. During training:
1. The gradient flows through `softmax` to both `W` and `r_dm`
2. The gradients from link-prediction loss propagate through the weighted
   combination to update the relative strengths
3. Relations that benefit more from DistMult (e.g., symmetric `ptm_crosstalk`)
   will develop higher `w1` weights
4. Relations that benefit from RotatE (e.g., asymmetric `inhibits`) will
   develop higher `w2` weights

The softmax ensures the weights remain valid throughout training.

---

## 5. Composite Scoring: Individual vs Hybrid

### 5.1 Hybrid Score Correctness -- PASS

**Implementation** (`hybrid_scorer.py:126-131`):

```python
score = (
    weights[:, 0] * score_dm
    + weights[:, 1] * score_rot
    + weights[:, 2] * score_nn
)
```

This is the correct weighted combination. Each sub-score is computed independently:
- `score_dm`: DistMult `<h, r, t>` (bilinear)
- `score_rot`: RotatE `-||h*r - t||` (rotational distance)
- `score_nn`: MLP-based neural score

### 5.2 Sub-Scorer Compatibility -- PASS

The DistMult and RotatE sub-scorers are correctly reused from the existing
decoder implementations:

| Sub-scorer | Expected signature | Actual | Match |
|------------|-------------------|--------|-------|
| DistMult | `(h, r, t) -> [B]` | `(head * relation * tail).sum(dim=-1)` | Yes |
| RotatE | `(h, r_phase, t) -> [B]` | `-diff.abs().sum(dim=-1)` | Yes |
| Neural | `([h||r||t]) -> [B]` | `MLP(3d -> d -> 1).squeeze(-1)` | Yes |

### 5.3 Score Scale Mismatch -- CONCERN

**Issue**: The three sub-scorers produce scores on different scales:
- **DistMult**: Range depends on embedding norms, typically O(d) for random embeddings
- **RotatE**: Always non-positive (distance-based), typically in [-sqrt(d), 0]
- **Neural**: Arbitrary range from MLP output

The softmax-weighted combination mixes these incompatible scales. During early
training, the DistMult scores may dominate simply due to larger magnitude, not
because DistMult is a better scorer for the relation.

**Recommendation**: Consider normalizing sub-scores to a common scale before
combination (e.g., batch normalization or score standardization per sub-scorer).
Alternatively, add learnable per-scorer scale and bias parameters:
```python
score_dm_norm = self.dm_scale * score_dm + self.dm_bias
```

**Severity**: Medium. The model can potentially learn to compensate via the
relation embeddings and weight_net, but convergence may be slower.

---

## 6. CompositeLoss: Convergence Properties

### 6.1 Link Loss Component -- PASS

The `link_loss` is delegated to an existing loss module (`MarginRankingLoss`
or `BCEWithLogitsKGELoss`). Both have been validated in the numerical validity
report. The `forward` signature `(pos_scores, neg_scores) -> scalar` is
correctly used at line 158.

### 6.2 Structural Contrastive Loss (InfoNCE) -- PASS

**Implementation** (`composite_loss.py:60-104`):

```python
z = F.normalize(glycan_embeddings, dim=-1)       # L2 normalize
sim = torch.mm(z, z.t()) / self.struct_temperature # [N, N] similarity

loss_ij = F.cross_entropy(logits_ij, idx_j)  # direction i->j
loss_ji = F.cross_entropy(logits_ji, idx_i)  # direction j->i
return (loss_ij + loss_ji) / 2.0              # symmetric InfoNCE
```

This is a correct implementation of symmetric InfoNCE:
1. L2 normalization ensures cosine similarity
2. Temperature scaling controls the sharpness of the distribution
3. `cross_entropy(logits_ij, idx_j)` is equivalent to `-log(softmax(logits_ij)[idx_j])`
4. The symmetric formulation (both i->j and j->i) is standard

**Edge case handling**: Empty `positive_pairs` correctly returns `tensor(0.0)` (line 84-85).

### 6.3 L2 Regularization -- PASS

**Implementation** (`composite_loss.py:107-127`):

```python
reg = sum(e.norm(2) for e in all_embeddings.values())
```

This computes the sum of L2 norms (Frobenius norms) over all embedding tensors.
The empty-dict case is handled by checking `isinstance(reg, torch.Tensor)`.

### 6.4 Loss Composition -- PASS

```python
loss = self.link_loss(pos_scores, neg_scores)
loss = loss + self.lambda_struct * self.structural_contrastive_loss(...)
loss = loss + self.lambda_reg * self._l2_regularization(...)
```

The additive composition with configurable lambda weights is correct.
All three terms produce differentiable scalar tensors.

### 6.5 Temperature Sensitivity -- CONCERN

The default `struct_temperature = 0.07` is standard for contrastive learning
(following SimCLR/CLIP), but it is quite aggressive for small batch sizes.
With temperature=0.07 and N=100 glycans:
- Similarity range: [-1/0.07, 1/0.07] = [-14.3, 14.3]
- The softmax becomes very peaked, which can lead to:
  - Gradient instability with small batches
  - The loss being dominated by a few very similar or dissimilar pairs

**Recommendation**: Consider adaptive temperature or a higher default for
small glycan datasets (e.g., `struct_temperature = 0.1` or `0.2`).

**Severity**: Low. The temperature is configurable and the default is a
reasonable starting point. Users should tune this hyperparameter.

---

## 7. GlycoKGNet Integration: Critical Bugs

### 7.1 BUG: Wrong Import Path for HybridLinkScorer -- FAIL

**File**: `glycoMusubi/embedding/models/glycoMusubi_net.py:56`

```python
from glycoMusubi.embedding.decoders.hybrid_link_scorer import HybridLinkScorer
```

The actual module is `glycoMusubi/embedding/decoders/hybrid_scorer.py`, not
`hybrid_link_scorer.py`. This import will **always fail**, causing `_HAS_HYBRID_SCORER`
to be `False` and the model to **silently fall back** to the `_FallbackScorer`
(a simple DistMult-style scorer).

**Impact**: Critical. The HybridLinkScorer will never be used in GlycoKGNet,
defeating the purpose of the hybrid scoring architecture.

**Fix**:
```python
# Line 56: Change from
from glycoMusubi.embedding.decoders.hybrid_link_scorer import HybridLinkScorer
# To
from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer
```

**Evidence**: The correct import path is used in both:
- `glycoMusubi/embedding/decoders/__init__.py:10` -- `from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer`
- `tests/test_hybrid_scorer.py:20` -- `from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer`

### 7.2 BUG: GlycoKGNet.score() Calls Wrong Method -- FAIL

**File**: `glycoMusubi/embedding/models/glycoMusubi_net.py:487`

```python
def score(self, head, relation, tail):
    return self.decoder.score(head, relation, tail)
```

When the decoder is `HybridLinkScorer`, the correct method is `forward()`,
not `score()`. The `HybridLinkScorer` class inherits from `nn.Module` and
defines `forward(head, relation_idx, tail)` but does **not** define a `score()`
method.

Meanwhile, the `_FallbackScorer` defines `score()` but not `forward()`.

**Impact**: If Bug 7.1 is fixed, calling `GlycoKGNet.score()` with the
HybridLinkScorer will raise `AttributeError: 'HybridLinkScorer' object has no
attribute 'score'`.

**Note**: The `HybridLinkScorer.forward()` takes `relation_idx` (integer indices),
while `GlycoKGNet.score()` passes `relation` (embeddings). This is a second-order
incompatibility: the HybridLinkScorer manages its own relation embeddings
internally and expects integer relation type indices, not pre-computed
relation embeddings.

**Fix**: Either:
1. Add a `score()` method to `HybridLinkScorer` that delegates to `forward()`, or
2. Modify `GlycoKGNet.score()` to dispatch correctly based on decoder type, or
3. Make `GlycoKGNet._init_decoder()` properly align the decoder interface

### 7.3 GlycoKGNet BioHGT Integration -- PASS

The `_init_decoder` signature for `BioHGT` uses a simplified constructor:

```python
self.biohgt = BioHGT(
    embed_dim=embedding_dim,
    num_heads=num_hgt_heads,
    num_layers=num_hgt_layers,
    num_node_types=num_node_types,
    num_edge_types=num_relations,
    use_bio_prior=use_bio_prior,
    dropout=dropout,
)
```

However, the actual `BioHGT.__init__` expects `num_nodes_dict`, `num_relations`,
and `embedding_dim` as the first three positional args (inherited from
`BaseKGEModel`). This means the GlycoKGNet constructor passes keyword arguments
that don't match `BioHGT.__init__` parameters.

**Status**: This will raise a `TypeError` at construction time if `_HAS_BIOHGT`
is True. However, since Task #4 (GlycoKGNet implementation) is still in
progress, this may be resolved before completion.

### 7.4 Fusion Stage Modality Passback -- PASS

The `_run_fusion()` method correctly passes initial (Stage 1) embeddings as
modality features and post-BioHGT (Stage 2) embeddings as KG features:

```python
fused_dict[ntype] = self._fusion(
    h_kg=emb_dict[ntype],           # Stage 2 output
    h_modality=initial_emb_dict[ntype],  # Stage 1 output
)
```

This is semantically correct: the KG features are enriched by BioHGT message
passing, while the modality features retain the raw encoder output.

### 7.5 Stage Pipeline Order -- PASS

The `encode()` method correctly chains:
1. `_compute_initial_embeddings()` -> Stage 1
2. `_run_biohgt()` -> Stage 2
3. `_run_fusion()` -> Stage 3

This matches the 4-stage architecture described in the design doc (Section 2).

---

## 8. Design Specification Compliance

### 8.1 CrossModalFusion vs Spec (Section 3.5) -- PASS

| Spec Element | Expected | Implemented | Match |
|-------------|----------|-------------|-------|
| Cross-attention Q source | h_KG | Q from `h_kg_active` | Yes |
| Cross-attention K/V source | h_modal | K/V from `h_mod_active` | Yes |
| Gate activation | sigmoid | `nn.Sigmoid()` | Yes |
| Gate input | [h_KG \|\| h_cross] | `torch.cat([h_kg_active, h_cross], dim=-1)` | Yes |
| Fusion formula | `g * h_KG + (1-g) * h_cross` | `gate * h_kg_active + (1.0 - gate) * h_cross` | Yes |
| Final normalization | LayerNorm | `self.layer_norm(h_fused)` | Yes |
| Attention heads | 4 | Default `num_heads=4` | Yes |
| Output dim | 256 | Parameterized (default 256) | Yes |

### 8.2 HybridLinkScorer vs Spec (Section 3.6.1) -- PASS

| Spec Element | Expected | Implemented | Match |
|-------------|----------|-------------|-------|
| DistMult component | `<h, r, t>` | `(head * relation * tail).sum(dim=-1)` | Yes |
| RotatE component | `-\|\|h.r - t\|\|` | `-diff.abs().sum(dim=-1)` | Yes |
| Neural component | `MLP(3d -> 512 -> 1)` | `Linear(3*d, 512) -> GELU -> Dropout -> Linear(512, 1)` | Yes |
| Per-relation weights | `softmax(W * RelEmb(r))` | `torch.softmax(self.weight_net(r_dm), dim=-1)` | Yes |
| Relation embeddings | Separate for DistMult and RotatE | `rel_embed_distmult` (d=256), `rel_embed_rotate` (d=128) | Yes |
| Complex dim | d/2 = 128 | `self.complex_dim = embedding_dim // 2` | Yes |

### 8.3 CompositeLoss vs Spec (Section 4.1) -- PASS

| Spec Element | Expected | Implemented | Match |
|-------------|----------|-------------|-------|
| L_link | Existing loss (Margin/BCE) | `self.link_loss(pos_scores, neg_scores)` | Yes |
| L_struct | Contrastive on glycan embeddings | `structural_contrastive_loss()` (InfoNCE) | Yes |
| L_reg | L2 regularization | `_l2_regularization()` | Yes |
| Lambda weights | Configurable | `lambda_struct`, `lambda_reg` params | Yes |
| Temperature | 0.07 | `struct_temperature=0.07` default | Yes |

### 8.4 Spec Divergence: Cross-Attention Formula -- CONCERN

The design spec (Section 3.5) describes:
```
h_glycan_fused = CrossAttn(h_KG_glycan, h_tree_glycan) + h_KG
```

This is a **residual addition** pattern. However, the actual implementation uses
a **gated interpolation** pattern:
```python
h_fused = gate * h_KG + (1 - gate) * h_cross
```

These are mathematically different:
- Residual: `h_fused = h_KG + CrossAttn(...)` (additive, can increase norm)
- Gated: `h_fused = g * h_KG + (1-g) * h_cross` (interpolation, bounded norm)

The implementation's gated approach is arguably **better** than the spec's
residual approach because it provides a natural interpolation mechanism and
avoids unbounded growth. However, this is a specification divergence.

**Severity**: Low. The gated approach is a sound design choice and is described
in the detailed architecture pseudo-code (Section 3.5, lines 362-363).
The summary description (line 87-88) appears to be a simplified notation.

### 8.5 Pseudo-code vs Implementation: neural_hidden_dim -- PASS

Spec pseudo-code (Section 6.4):
```python
self.neural_scorer = nn.Sequential(
    nn.Linear(3 * d_model, 512),
    ...
```

Implementation:
```python
self.neural_scorer = nn.Sequential(
    nn.Linear(embedding_dim * 3, neural_hidden_dim),
    ...
```

With default `neural_hidden_dim=512`, this matches. The implementation is more
flexible by parameterizing the hidden dim.

### 8.6 Parameter Count Estimate -- PASS

Spec: ~1.2M for Link Prediction Decoder, ~0.5M for Cross-Modal Fusion.

For d=256:
- HybridLinkScorer: 2 relation embed tables + neural MLP + weight_net
  - rel_embed_distmult: 12*256 = 3,072
  - rel_embed_rotate: 12*128 = 1,536
  - neural_scorer: 768*512 + 512 + 512*1 + 1 = 393,729
  - weight_net: 256*3 + 3 = 771
  - Total: ~399K (spec says ~1.2M; likely assumes more relations or larger d)

- CrossModalFusion: MultiheadAttention + gate_mlp + LayerNorm
  - cross_attn: ~4*256*256 = 262,144 + output proj = ~330K
  - gate_mlp: 512*256 + 256 + 256*1 + 1 = ~131K
  - layer_norm: 2*256 = 512
  - Total: ~462K (matches spec ~0.5M estimate)

---

## 9. Bug Summary and Recommended Fixes

### Bug 1: Wrong Import Path (CRITICAL)

**File**: `glycoMusubi/embedding/models/glycoMusubi_net.py:56`
**Current**: `from glycoMusubi.embedding.decoders.hybrid_link_scorer import HybridLinkScorer`
**Fix**: `from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer`
**Impact**: HybridLinkScorer is silently never used; model falls back to simple DistMult.

### Bug 2: Interface Mismatch (HIGH)

**File**: `glycoMusubi/embedding/models/glycoMusubi_net.py:487`
**Current**: `self.decoder.score(head, relation, tail)` -- but HybridLinkScorer has no `score()` method
**Fix**: Align the interface. Either:
- (a) Add `HybridLinkScorer.score = HybridLinkScorer.forward`, or
- (b) Change GlycoKGNet to call `self.decoder(head, relation, tail)` (using `__call__` / `forward()`), or
- (c) Note that HybridLinkScorer expects `relation_idx` (int), not `relation` (embedding).
  A wrapper is needed to convert between interfaces.

### Design Concern 1: Score Scale Mismatch (MEDIUM)

Sub-scores from DistMult, RotatE, and Neural scorer have different magnitude
ranges. Consider per-scorer normalization.

### Design Concern 2: Temperature Sensitivity (LOW)

Default `struct_temperature=0.07` may be aggressive for small datasets.
Document recommended ranges.

### Design Concern 3: Spec Divergence in Fusion Formula (LOW)

Summary spec says residual addition; implementation uses gated interpolation.
The implementation is better but should be documented as an intentional improvement.

---

## 10. Conclusion

The CrossModalFusion, HybridLinkScorer, and CompositeLoss modules are
mathematically sound and well-tested individually. The two integration
bugs in GlycoKGNet (wrong import path and interface mismatch) must be
fixed before the hybrid scoring pipeline can function end-to-end. The
three design concerns are non-blocking but should be addressed for
optimal training convergence.

**Verdict**: CONDITIONAL PASS -- proceed after fixing Bugs 1 and 2.
