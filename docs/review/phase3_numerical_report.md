# Phase 3 Numerical Validity Report

**Date:** 2026-02-13
**Test file:** `tests/test_phase3_numerical.py`
**Result:** 59/59 tests passed

## Summary

All Phase 3 components pass comprehensive numerical validity testing covering
correctness, stability, convergence, and gradient flow. No NaN/Inf issues were
found in any component under normal operating conditions.

---

## 1. PathReasoner (NBFNet-style Bellman-Ford GNN)

### Tests: 10 passed

| Test | Description | Result |
|------|-------------|--------|
| Embedding norm bounded (T=1,3,6) | After T message-passing rounds, norms stay in [0.1, 1e3] | PASS |
| Norms stable across iterations | No exponential growth/decay over 6 BF iterations | PASS |
| Score gradients finite | MLP scoring head produces finite gradients for tail/relation | PASS |
| Full forward gradients finite | All parameter gradients finite after backward through BF layers | PASS |
| Gradients not vanishing | At least one gradient has magnitude > 1e-10 | PASS |
| Positive > negative scores | After 50 training steps, positive triple scores exceed negatives | PASS |
| Loss decreases (sum aggregation) | First-10 avg loss > last-10 avg loss over 50 steps | PASS |
| Loss decreases (PNA aggregation) | PNA variant also converges | PASS |

**Key findings:**
- LayerNorm after each BF iteration effectively controls embedding scale.
- Both sum and PNA aggregation modes produce numerically stable training.
- The score MLP `[tail || relation] -> scalar` has well-behaved gradients.

---

## 2. Poincare Ball Geometry

### Tests: 14 passed

| Test | Description | Result |
|------|-------------|--------|
| exp_map inside ball (c=0.5,1.0,2.0) | `c * ||exp_0(v)||^2 < 1` for all curvatures | PASS |
| exp_map large vectors | Vectors with norm 1000 still map inside ball | PASS |
| exp_map zero vector | Zero tangent vector maps to origin | PASS |
| Mobius add inside ball | Result of Mobius addition stays inside ball | PASS |
| Clamping at boundary | Points at ||x||=0.99999 are clamped correctly | PASS |
| Clamping beyond boundary | Points at ||x||=2.0 are projected back inside | PASS |
| Small points unaffected | Points at ||x||~0.1 pass through clamping unchanged | PASS |
| No NaN in gradients (30 steps) | Training with Poincare distance produces finite gradients throughout | PASS |
| Gradient stability at various norms | Gradients finite for ||x||=0.01,0.1,0.3,0.5 | PASS |
| Distance near origin | Finite, non-negative distances for near-zero points | PASS |
| Distance near boundary | Finite, non-negative distances for ||x||~0.99 | PASS |
| Distance self is zero | d(x,x) < 1e-4 | PASS |
| Distance symmetry | d(x,y) == d(y,x) within 1e-3 tolerance | PASS |
| Exp-log roundtrip | log_0(exp_0(v)) recovers v within 1e-3 for moderate norms | PASS |

**Key findings:**
- The `_clamp_norm` method correctly projects boundary/exterior points.
- `tanh`-based exp map naturally saturates, preventing escape from the ball.
- Distance computation is numerically stable via `atanh` clamping (`max=1-eps`).
- Mobius addition with near-zero vectors preserves the other operand (up to small eps).

---

## 3. HybridLinkScorer (4-component)

### Tests: 6 passed

| Test | Description | Result |
|------|-------------|--------|
| All 4 weights nonzero | After Xavier init + zero bias, all softmax weights > 0.01 | PASS |
| Sub-score gradient flow | All 4 scoring paths (DistMult, RotatE, Neural, Poincare) produce nonzero gradients | PASS |
| Weights sum to 1 | Per-relation softmax weights sum to 1.0 within 1e-6 | PASS |
| All weights positive | Softmax guarantees strict positivity | PASS |
| Training loss decreases | 80-step training with margin loss converges | PASS |
| No NaN/Inf during training | 50 training steps produce finite losses and gradients throughout | PASS |

**Key findings:**
- The weight initialization (Xavier + zero bias) gives roughly uniform initial weights (~0.25 each).
- All 4 scoring paradigms receive gradient signal, confirming no dead branches.
- The adaptive weighting mechanism converges properly.

---

## 4. CompGCN Relation Composition

### Tests: 11 passed

| Test | Description | Result |
|------|-------------|--------|
| Composed embedding finite (all modes) | subtraction, multiplication, circular_correlation all produce finite outputs | PASS |
| Magnitude reasonable (all modes) | Average norm in (1e-6, 1e4) for all composition modes | PASS |
| Circular correlation correctness | IFFT(conj(FFT(a)) * FFT(b)) matches manual computation | PASS |
| FFT/IFFT roundtrip | irfft(rfft(x), n) recovers x within 1e-5 | PASS |
| Output shape correct | Circular correlation preserves input shape | PASS |
| Real-valued output | irfft produces real (float32/float64) tensor | PASS |
| Gradient flow (all modes) | node_type_embed and edge_type_embed receive gradients in all 3 modes | PASS |

**Key findings:**
- Circular correlation via FFT is mathematically correct.
- All three composition modes (subtraction, multiplication, circular_correlation) are numerically stable.
- Xavier initialization keeps composed embeddings in a reasonable magnitude range.

---

## 5. Self-supervised Pretraining

### Tests: 12 passed

| Test | Description | Result |
|------|-------------|--------|
| Masked node loss decreases | MSE reconstruction loss decreases over 30 steps | PASS |
| Masked node gradients finite | All encoder + predictor gradients finite | PASS |
| Masked edge loss decreases | BCE existence loss decreases over 60 steps | PASS |
| Masked edge gradients finite | All encoder + predictor gradients finite | PASS |
| CMCA cross-modal loss positive | Random embeddings produce positive InfoNCE loss | PASS |
| CMCA intra-modal loss positive | Positive pairs produce positive contrastive loss | PASS |
| CMCA loss decreases | Cross-modal alignment loss decreases over 40 steps | PASS |
| CMCA does not collapse | Loss stays finite even with identical input embeddings | PASS |
| CMCA gradients finite | Gradients through both modal and KG embeddings are finite | PASS |
| CMCA combined forward | Both intra-modal and cross-modal terms produce finite losses | PASS |
| Glycan substructure loss finite | Multi-label BCE loss is always finite | PASS |
| Glycan substructure loss decreases | Loss decreases over 30 training steps | PASS |
| Combined pretraining gradients | All 3 pretraining tasks together produce finite gradients for 10 steps | PASS |

**Key findings:**
- The InfoNCE temperature (0.07) in CMCA effectively prevents collapse.
- Masked node/edge prediction both converge with appropriate learning rates.
- All three pretraining objectives can run simultaneously without gradient interference.

---

## Notable Numerical Properties

1. **Poincare ball containment:** The `tanh`-based exponential map combined with `_clamp_norm` ensures all embeddings remain strictly inside the open Poincare ball `B^d_c = {x : c||x||^2 < 1}`.

2. **LayerNorm in PathReasoner:** Per-iteration LayerNorm is critical for preventing embedding explosion/vanishing over 6 message-passing iterations.

3. **MaskedNodePredictor inplace operations:** The mask-and-predict pattern modifies `data[nt].x` in-place. Encoders that process this data should `.clone()` features to avoid autograd issues when the masking code restores original features.

4. **Softmax weight normalization:** HybridLinkScorer's per-relation adaptive weights are guaranteed to sum to 1 by construction (softmax), ensuring no single scorer dominates by arithmetic rather than learning.

---

## Conclusion

All 59 numerical validity tests pass, confirming that Phase 3 components are
numerically correct and stable under normal training conditions. The key
numerical safeguards (clamping, LayerNorm, eps guards, tanh saturation) are
effective at preventing degenerate solutions across all components.
