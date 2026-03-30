# glycoMusubi v0.1 -- Code Quality Report

**Reviewer**: system-expert (R5)
**Date**: 2026-02-13
**Scope**: All files under `glycoMusubi/`

---

## Overview

This report evaluates the code quality of the glycoMusubi v0.1 implementation across six dimensions: type hints, documentation, error handling, readability/maintainability, DRY compliance, and security.

**Overall Assessment**: The code quality is **high**. The codebase demonstrates professional engineering practices including comprehensive type annotations, detailed docstrings with NumPy-style parameter documentation, clean separation of concerns, and appropriate abstraction levels. The issues identified below are minor and easily addressable.

---

## 1. Type Hints Completeness

**Rating**: Excellent (9/10)

All public and private methods have complete type annotations, including:
- Return types on all functions
- Parameter types with generics (`Dict[str, int]`, `Optional[Union[str, Path]]`)
- `from __future__ import annotations` used consistently for forward references
- Proper use of `Tuple`, `List`, `Sequence`, `Dict` from `typing`

### Minor Issues

| File | Line | Issue | Severity |
|---|---|---|---|
| `data/dataset.py` | 57-58 | `transform`, `pre_transform`, `pre_filter` parameters lack type hints | Low |
| `training/trainer.py` | 14 | `from torch.cuda.amp import GradScaler, autocast` -- deprecated in PyTorch 2.x; should use `torch.amp.GradScaler` and `torch.amp.autocast` | Low |
| `training/callbacks.py` | 27-44 | `Callback` base class uses `trainer: Any` instead of a typed reference | Low |
| `evaluation/visualize.py` | 79 | Return type uses string annotation `"matplotlib.figure.Figure"` -- acceptable but could use `TYPE_CHECKING` import | Low |

### Positive Examples

- `glycoMusubi/evaluation/metrics.py`: All functions have precise tensor shape annotations in docstrings (e.g., `[batch, num_entities]`)
- `glycoMusubi/data/sampler.py`: Return types for `sample()` and `sample_flat()` precisely specified
- `glycoMusubi/utils/config.py`: Dataclass fields all typed with defaults

---

## 2. Docstring Quality

**Rating**: Excellent (9/10)

All public classes and methods have NumPy-style docstrings with:
- One-line summary
- Detailed description where appropriate
- `Parameters` section with types and descriptions
- `Returns` section with types and descriptions
- `Raises` section where exceptions are thrown

### Positive Examples

- `glycoMusubi/evaluation/metrics.py`: Each metric function documents the mathematical definition, edge cases, and references
- `glycoMusubi/evaluation/link_prediction.py`: Protocol class `ScorableModel` clearly documents the expected interface
- `glycoMusubi/embedding/encoders/glycan_encoder.py`: Module-level docstring explains all three encoding strategies

### Minor Issues

| File | Line | Issue | Severity |
|---|---|---|---|
| `training/trainer.py` | 159 | `train_epoch()` docstring could mention it does full-batch by default | Low |
| `losses/bce_loss.py` | 12-15 | Reference to RotatE paper is good; could add bibtex key | Minimal |
| `data/converter.py` | 291-318 | `extract_node_metadata()` docstring does not mention JSON parsing behavior for malformed input | Low |

---

## 3. Error Handling

**Rating**: Good (8/10)

The codebase has appropriate error handling for most boundary conditions:

### Well-Handled Cases

- `evaluation/metrics.py:60-72`: Input shape validation with descriptive `ValueError` messages
- `evaluation/metrics.py:141`: `k < 1` check for `hits_at_k`
- `evaluation/metrics.py:194`: `num_candidates < 1` check for AMR
- `embedding/models/glycoMusubie.py:148`: `embedding_dim % 2 != 0` check for RotatE
- `data/converter.py:160-176`: Graceful fallback from Parquet to TSV with `FileNotFoundError`
- `embedding/encoders/glycan_encoder.py:244-248`: WURCS parsing failures return zero vectors with debug logging
- `embedding/encoders/protein_encoder.py:109-127`: ESM-2 cache loading with graceful fallback on file errors

### Areas for Improvement

| File | Line | Issue | Severity |
|---|---|---|---|
| `data/splits.py` | 67 | Uses `assert` for validation (`assert 0 < val_ratio + test_ratio < 1`). Should use `ValueError` for runtime checks, as asserts can be disabled with `-O`. | Medium |
| `data/splits.py` | 136 | Same `assert` issue in `relation_stratified_split` | Medium |
| `training/trainer.py` | 233 | `torch.load(..., weights_only=False)` -- loading arbitrary objects is a security concern (see Security section) | Medium |
| `data/converter.py` | 255 | Edge loop iterates with `range(len(edges_df))` and uses direct indexing -- no error if column names are wrong (silent data corruption possible) | Low |
| `training/trainer.py` | 261 | `edge_store.edge_index` access without checking if it exists -- could raise `AttributeError` for malformed data | Low |

---

## 4. Readability and Maintainability

**Rating**: Excellent (9/10)

### Strengths

- **Clean module structure**: Each module has a single responsibility (converter, dataset, splits, sampler, etc.)
- **Consistent naming**: snake_case for functions/methods, PascalCase for classes, UPPER_CASE for constants
- **Logical file organization**: `data/`, `embedding/`, `evaluation/`, `losses/`, `training/`, `utils/` mirrors the pipeline stages
- **Well-structured `__init__.py` files**: All packages have clean `__all__` exports
- **Consistent import ordering**: `__future__`, stdlib, third-party, local
- **Internal vs public API**: Private methods prefixed with `_` consistently
- **Section separators**: `# ------` comment blocks clearly delineate code sections within files

### Minor Readability Issues

| File | Line | Issue | Severity |
|---|---|---|---|
| `data/converter.py` | 248-276 | Edge building loop uses raw indexing (`src_ids[i]`, `tgt_ids[i]`). A pandas `itertuples()` or vectorized approach would be clearer and faster. | Low |
| `training/trainer.py` | 258-284 | `_compute_scores` mixes embedding lookup, relation index resolution, and negative sampling in one method. Could benefit from factoring out sub-steps. | Low |
| `evaluation/link_prediction.py` | 302-306 | Per-relation bookkeeping in inner loop creates many small tensor slices. Minor performance concern for large test sets. | Low |
| `embedding/encoders/glycan_encoder.py` | 89-109 | `_parse_wurcs_sections` uses complex string manipulation. A brief inline comment explaining the WURCS format would help readers unfamiliar with the standard. | Low |

---

## 5. DRY Principle Compliance

**Rating**: Good (8/10)

### Duplications Found

| Location 1 | Location 2 | Duplicated Logic | Severity |
|---|---|---|---|
| `models/glycoMusubie.py` TransE.forward() (L56-63) | `models/glycoMusubie.py` DistMult.forward() (L95-102) and RotatE.forward() (L171-178) | Identical embedding lookup logic: `for node_type, emb_module in self.node_embeddings.items(): ... torch.arange(num_nodes) ... emb_module(idx)` | Medium |
| `models/glycoMusubie.py` TransE.score() | `decoders/transe.py` TransEDecoder.forward() | Identical `score(h, r, t) = -||h+r-t||_p` | Medium |
| `models/glycoMusubie.py` DistMult.score() | `decoders/distmult.py` DistMultDecoder.forward() | Identical `score(h, r, t) = <h,r,t>` | Medium |
| `models/glycoMusubie.py` RotatE.score() | `decoders/rotate.py` RotatEDecoder.forward() | Identical complex rotation scoring logic | Medium |
| `data/sampler.py` L29-39 | `data/converter.py` L22-29 | Both load `relation_config.yaml` with nearly identical patterns | Low |

### Recommendations

1. **[DRY-1]** The `forward()` method (embedding lookup) is identical across TransE, DistMult, and RotatE. Extract to `BaseKGEModel.forward()` as a default implementation.
2. **[DRY-2]** The decoder modules (`TransEDecoder`, `DistMultDecoder`, `RotatEDecoder`) duplicate scoring logic from the model classes. Consider having the model classes delegate scoring to the decoder modules, or remove the standalone decoders.
3. **[DRY-3]** Schema loading from YAML is repeated in `converter.py` and `sampler.py`. Extract a shared `_load_schema()` utility.

---

## 6. Security Assessment

**Rating**: Good (8/10)

### Findings

| Finding | File | Line | Severity | Description |
|---|---|---|---|---|
| **S1**: Unsafe `torch.load` | `training/trainer.py` | 233 | Medium | `torch.load(path, weights_only=False)` allows arbitrary code execution via pickle deserialization. Checkpoints from untrusted sources could execute malicious code. Use `weights_only=True` where possible, or validate checkpoint provenance. |
| **S2**: YAML loading without SafeLoader | `data/converter.py` | 28 | Low | `yaml.safe_load()` is correctly used (not `yaml.load()`). No issue. |
| **S3**: YAML loading without SafeLoader | `data/sampler.py` | 32, 45 | Low | `yaml.safe_load()` is correctly used. No issue. |
| **S4**: Path traversal | `embedding/encoders/protein_encoder.py` | 104 | Low | `filepath = self.cache_path / f"{idx}.pt"` -- `idx` is an integer from internal tensor, not user input. No traversal risk. |
| **S5**: Hash collision in TextEncoder | `embedding/encoders/text_encoder.py` | 31 | Low | SHA-256 modulo `num_buckets` is used. Collision resistance is adequate. However, if `num_buckets` is small, different texts may share embeddings. This is by design (feature hashing) and documented. |
| **S6**: Logging user data | `data/converter.py` | 279 | Minimal | Skipped edge count is logged. No sensitive data exposure. |

### Positive Security Practices

- `yaml.safe_load()` consistently used instead of `yaml.load()`
- `torch.load(..., weights_only=True)` used in `protein_encoder.py` L110 for ESM-2 cache loading
- No shell command execution or `eval()`/`exec()` usage
- File paths constructed using `pathlib.Path` (prevents injection)
- `hashlib.sha256` used for deterministic hashing (not Python's randomized `hash()`)

---

## Module-by-Module Quality Summary

| Module | Type Hints | Docstrings | Error Handling | Readability | DRY | Security | Overall |
|---|---|---|---|---|---|---|---|
| `utils/config.py` | 10 | 9 | 8 | 9 | 10 | 10 | **9.3** |
| `utils/logging_setup.py` | 10 | 10 | 9 | 10 | 10 | 10 | **9.8** |
| `utils/reproducibility.py` | 9 | 9 | 9 | 10 | 10 | 10 | **9.5** |
| `data/converter.py` | 9 | 9 | 8 | 8 | 8 | 10 | **8.7** |
| `data/dataset.py` | 8 | 9 | 8 | 9 | 10 | 10 | **9.0** |
| `data/splits.py` | 9 | 9 | 7 | 9 | 9 | 10 | **8.8** |
| `data/sampler.py` | 9 | 9 | 8 | 8 | 8 | 10 | **8.7** |
| `embedding/models/base.py` | 10 | 10 | 8 | 10 | 9 | 10 | **9.5** |
| `embedding/models/glycoMusubie.py` | 9 | 9 | 9 | 9 | 6 | 10 | **8.7** |
| `embedding/decoders/*.py` | 9 | 9 | 8 | 10 | 6 | 10 | **8.7** |
| `embedding/encoders/glycan_encoder.py` | 9 | 9 | 9 | 8 | 10 | 10 | **9.2** |
| `embedding/encoders/protein_encoder.py` | 9 | 9 | 9 | 9 | 10 | 10 | **9.3** |
| `embedding/encoders/text_encoder.py` | 9 | 9 | 8 | 9 | 10 | 10 | **9.2** |
| `losses/margin_loss.py` | 9 | 9 | 8 | 10 | 10 | 10 | **9.3** |
| `losses/bce_loss.py` | 9 | 9 | 8 | 9 | 10 | 10 | **9.2** |
| `training/trainer.py` | 8 | 8 | 7 | 8 | 9 | 8 | **8.0** |
| `training/callbacks.py` | 8 | 9 | 8 | 9 | 9 | 10 | **8.8** |
| `evaluation/metrics.py` | 10 | 10 | 10 | 10 | 10 | 10 | **10.0** |
| `evaluation/link_prediction.py` | 9 | 10 | 9 | 9 | 10 | 10 | **9.5** |
| `evaluation/visualize.py` | 9 | 9 | 8 | 9 | 10 | 10 | **9.2** |
| **Average** | **9.1** | **9.2** | **8.3** | **9.1** | **9.2** | **9.9** | **9.1** |

---

## Prioritized Recommendations

### High Priority

1. **[CQ-H1]** Replace `assert` statements in `data/splits.py` (lines 67, 136) with `if ... raise ValueError(...)` for production safety.
2. **[CQ-H2]** Address `torch.load(weights_only=False)` in `training/trainer.py` L233. Either use `weights_only=True` (requires PyTorch 2.x state_dict format), or add a prominent warning in the docstring.
3. **[CQ-H3]** Extract the duplicated `forward()` embedding lookup logic from TransE/DistMult/RotatE into `BaseKGEModel.forward()`.

### Medium Priority

4. **[CQ-M1]** Resolve the decoder duplication: either have model `.score()` delegate to decoder modules, or remove the standalone decoder classes.
5. **[CQ-M2]** Update `from torch.cuda.amp import GradScaler, autocast` to `from torch.amp import GradScaler, autocast` for PyTorch 2.x compatibility.
6. **[CQ-M3]** Add `TYPE_CHECKING` import pattern for `matplotlib.figure.Figure` return types in visualize.py.
7. **[CQ-M4]** Extract shared YAML schema loading utility from `converter.py` and `sampler.py`.

### Low Priority

8. **[CQ-L1]** Add type annotations to `GlycoKGDataset.__init__()` transform parameters.
9. **[CQ-L2]** Use `Trainer` typed hint instead of `Any` in callback method signatures.
10. **[CQ-L3]** Vectorize edge building in `converter.py` for better performance on large KGs.
11. **[CQ-L4]** Add inline WURCS format explanation in `glycan_encoder.py` `_parse_wurcs_sections()`.

---

## Conclusion

The glycoMusubi v0.1 codebase demonstrates strong engineering practices with an overall quality score of **9.1/10**. The code is well-structured, thoroughly documented, and follows Python best practices. The most significant quality concerns are:

1. The `assert`-based validation in `splits.py` (easy fix)
2. Code duplication between model classes and decoder modules (design decision)
3. The unsafe `torch.load` in the trainer (needs mitigation for untrusted checkpoints)

These are all straightforward to address and do not indicate systemic quality issues. The codebase provides a solid foundation for the planned Phase 2+ feature additions.
