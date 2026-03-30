# Reproducibility Verification Report (R4)

**Reviewer**: Reproducibility & Deterministic Behaviour Specialist
**Date**: 2026-02-13
**Status**: COMPLETE
**Standard**: Nature-level reproducibility (same seed = bit-identical results on CPU)

---

## 1. Executive Summary

This report assesses the reproducibility infrastructure of the glycoMusubi embedding pipeline.
The overall verdict is **PASS with recommendations**: the codebase implements comprehensive
seed management and deterministic controls, ensuring that identical seeds produce identical
results across data splitting, negative sampling, model initialisation, and training on CPU.
Several minor improvements are recommended for multi-GPU and edge-case scenarios.

| Category | Verdict |
|----------|---------|
| Seed fixing coverage (Python/NumPy/PyTorch/CUDA) | **PASS** |
| Data split determinism | **PASS** |
| Negative sampling determinism | **PASS** |
| Model initialisation determinism | **PASS** |
| Training loss determinism (CPU) | **PASS** |
| Checkpoint save/restore fidelity | **PASS** |
| DataLoader worker seed isolation | **PASS** |
| Deterministic mode (`torch.use_deterministic_algorithms`) | **PASS** |
| Non-deterministic operation mitigation | **PASS with note** |
| Multi-GPU reproducibility | **NOT TESTED** (documented) |

---

## 2. Source Code Analysis

### 2.1 `glycoMusubi/utils/reproducibility.py` -- Seed Management

#### `set_seed(seed: int)`

| RNG Source | Fixed? | Mechanism | Line |
|------------|--------|-----------|------|
| Python `random` | Yes | `random.seed(seed)` | L18 |
| NumPy `np.random` | Yes | `np.random.seed(seed)` | L22 |
| PyTorch CPU | Yes | `torch.manual_seed(seed)` | L28 |
| PyTorch CUDA (all GPUs) | Yes | `torch.cuda.manual_seed_all(seed)` | L30 |

**Assessment**: All four critical RNG sources are fixed. The function gracefully handles
missing optional dependencies (NumPy, PyTorch) via try/except ImportError, which is
appropriate for a library that may be imported in testing contexts without full GPU support.

**Note**: `torch.manual_seed(seed)` also seeds all CUDA devices internally (via
`torch.cuda.manual_seed_all` dispatched from the C++ backend), but the explicit
`torch.cuda.manual_seed_all(seed)` call in the code provides additional safety and
clarity. This is correct and robust.

#### `set_deterministic(enabled: bool)`

| Setting | Applied? | Mechanism | Line |
|---------|----------|-----------|------|
| `cudnn.deterministic` | Yes | Direct attribute set | L49 |
| `cudnn.benchmark` | Yes (disabled) | Set to `not enabled` | L50 |
| `torch.use_deterministic_algorithms` | Yes | Conditional on version | L51-52 |
| `CUBLAS_WORKSPACE_CONFIG` | Yes | `os.environ` | L57 |

**Assessment**: Comprehensive. The `CUBLAS_WORKSPACE_CONFIG=:4096:8` setting is
essential for deterministic cuBLAS operations on CUDA 10.2+. The version check
via `hasattr(torch, "use_deterministic_algorithms")` ensures backward compatibility.

**Minor finding**: When `enabled=False`, the `CUBLAS_WORKSPACE_CONFIG` environment
variable is not unset. This is a non-issue in practice (the env var has no effect
when `use_deterministic_algorithms(False)` is called) but could be cleaned up for
completeness.

#### `seed_worker(worker_id: int)`

| Operation | Correct? | Notes |
|-----------|----------|-------|
| Derives seed from `torch.initial_seed()` | Yes | Uses `% 2**32` for 32-bit range |
| Seeds NumPy per worker | Yes | `np.random.seed(worker_seed)` |
| Seeds Python `random` per worker | Yes | `random.seed(worker_seed)` |

**Assessment**: Correct implementation. Each DataLoader worker gets a unique,
deterministic seed derived from the global PyTorch seed. This function should be
passed as `worker_init_fn` to `torch.utils.data.DataLoader`. The `worker_id`
parameter is intentionally unused (annotated with `# noqa: ARG001`) because
PyTorch already incorporates the worker ID into `torch.initial_seed()`.

### 2.2 `glycoMusubi/data/splits.py` -- Data Splitting

Both `random_link_split()` and `relation_stratified_split()` use a **local**
`torch.Generator` seeded from the `seed` parameter:

```python
generator = torch.Generator().manual_seed(seed)
perm = torch.randperm(num_edges, generator=generator)
```

**Key finding**: The use of a local `torch.Generator` rather than the global RNG
state is excellent practice. This means:

1. The split result depends **only** on the `seed` argument, not on any prior
   random operations.
2. The split operation does **not** perturb the global RNG state, preserving
   reproducibility of downstream operations.
3. Multiple calls with the same seed produce identical results regardless of
   execution context.

**Assessment**: **PASS** -- Best practice for reproducible data splitting.

### 2.3 `glycoMusubi/data/sampler.py` -- Negative Sampling

The `TypeConstrainedNegativeSampler.sample()` method accepts an optional
`torch.Generator` parameter:

```python
corrupt_mask = torch.rand(B, K, generator=generator) < self.corrupt_head_prob
idx = torch.randint(len(valid), (n,), generator=generator)
```

**Assessment**: **PASS** -- The sampler correctly threads a generator through all
random operations. When a seeded generator is provided, the output is fully
deterministic.

**Observation**: The `sample()` method iterates over the batch dimension with a
Python loop (`for i in range(B)`). While this is not a performance concern for
testing, it means the generator state advances in a sequential, deterministic
order -- which is correct for reproducibility.

### 2.4 `glycoMusubi/embedding/models/` -- Model Initialisation

All three KGE models (TransE, DistMult, RotatE) inherit from `BaseKGEModel`,
which calls `_init_embeddings()`:

```python
def _init_embeddings(self) -> None:
    for emb in self.node_embeddings.values():
        nn.init.xavier_uniform_(emb.weight)
    nn.init.xavier_uniform_(self.relation_embeddings.weight)
```

**Assessment**: `nn.init.xavier_uniform_` uses the global PyTorch RNG, so
calling `set_seed(seed)` before model construction ensures deterministic
initialisation.

RotatE additionally overrides relation embeddings with `_init_rotate_relations()`:

```python
nn.init.uniform_(self.relation_embeddings.weight, -phase_range, phase_range)
```

This also uses the global RNG and is covered by `set_seed()`.

**Assessment**: **PASS** -- All initialisation is deterministic given `set_seed()`.

### 2.5 `glycoMusubi/training/trainer.py` -- Training Loop

The `Trainer` class implements a standard training loop. Key reproducibility
observations:

| Operation | Deterministic? | Notes |
|-----------|----------------|-------|
| `model.forward(data)` | Yes (lookup only for TransE/DistMult/RotatE) | No random ops |
| `loss_fn(pos_scores, neg_scores)` | Yes | BCE and Margin losses are deterministic |
| `optimizer.zero_grad()` | Yes | No randomness |
| `loss.backward()` | Yes on CPU | Non-deterministic on GPU without `use_deterministic_algorithms` |
| `optimizer.step()` | Yes | Adam/SGD are deterministic given identical gradients |
| `_simple_negative_scores` (fallback) | **Uses global RNG** | `torch.randint` without generator |

**Finding (Minor)**: The `_simple_negative_scores()` fallback in the Trainer uses
`torch.randint(0, num_dst, (edge_index.size(1),), device=self.device)` **without**
an explicit generator. This means it depends on the global RNG state. Since
`set_seed()` fixes the global state and the training loop is sequential, this is
deterministic in practice, but it would be more robust to thread a generator.

**Assessment**: **PASS** -- Training is deterministic on CPU when `set_seed()` and
`set_deterministic(True)` are used before training.

### 2.6 `glycoMusubi/training/callbacks.py` -- Checkpoint System

The `ModelCheckpoint` callback and `Trainer.save_checkpoint()` / `load_checkpoint()`
persist and restore:

| State | Saved? | Restored? |
|-------|--------|-----------|
| `model_state_dict` | Yes | Yes |
| `optimizer_state_dict` | Yes | Yes |
| `scheduler_state_dict` | Yes (if present) | Yes (if present) |
| `scaler_state_dict` (AMP) | Yes (if present) | Yes (if present) |
| `current_epoch` | Yes | Yes |

**Missing from checkpoint**: The global RNG states (Python, NumPy, PyTorch) are
**not** saved in the checkpoint. This means that resuming training from a
checkpoint will produce different results from uninterrupted training unless
the user manually manages RNG state.

**Recommendation**: Save and restore RNG states in checkpoints for perfect
mid-training reproducibility:

```python
state["rng_state"] = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch_cpu": torch.random.get_rng_state(),
    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
}
```

**Assessment**: **PASS** for checkpoint fidelity (parameters restored correctly).
**RECOMMENDATION** for RNG state preservation.

---

## 3. Non-Deterministic Operations Inventory

### 3.1 Operations Checked

| Operation | Present in Code? | Deterministic Alternative? | Status |
|-----------|-----------------|---------------------------|--------|
| `torch.scatter_add` / `scatter_reduce` | Not directly used | `torch.use_deterministic_algorithms` covers this | N/A |
| `index_add_` | Not used | Would need deterministic mode | N/A |
| `torch.nn.functional.embedding_bag` | Not used | N/A | N/A |
| `torch.nn.functional.interpolate` | Not used | N/A | N/A |
| `torch.Tensor.scatter_` | Not used | N/A | N/A |
| Atomic operations (CUDA) | Covered by `set_deterministic(True)` | `CUBLAS_WORKSPACE_CONFIG` set | OK |
| cuDNN autotuning | Disabled by `cudnn.benchmark=False` | N/A | OK |
| `torch.randperm` (splits) | Uses local generator | Deterministic | OK |
| `torch.randint` (sampler) | Uses passed generator | Deterministic | OK |
| `torch.randint` (trainer fallback) | Uses global RNG | Deterministic via `set_seed()` | OK (see note) |
| `torch.rand` (sampler corrupt mask) | Uses passed generator | Deterministic | OK |

### 3.2 PyTorch Geometric Operations

The codebase uses PyG `HeteroData` for graph storage. Key PyG operations:

| Operation | Deterministic? | Notes |
|-----------|----------------|-------|
| `HeteroData` construction | Yes | No randomness |
| Edge index slicing | Yes | Tensor indexing is deterministic |
| `LinkNeighborLoader` | Depends on config | Not used in current training loop |

**Assessment**: No PyG-specific non-deterministic operations are used in the
current training path.

---

## 4. Test Coverage

The test file `tests/test_reproducibility.py` provides comprehensive coverage:

### 4.1 Test Matrix

| Test Class | Test Method | Covers | Expected |
|------------|-------------|--------|----------|
| `TestSetSeedCoverage` | `test_python_random` | Python `random` module | Same seed -> same sequence |
| | `test_numpy_random` | NumPy RNG | Same seed -> same sequence |
| | `test_torch_random` | PyTorch CPU RNG | Same seed -> same sequence |
| | `test_torch_randperm` | Permutation (splits) | Same seed -> same permutation |
| | `test_torch_randint` | Integer sampling | Same seed -> same integers |
| `TestSplitReproducibility` | `test_same_seed_same_split` | `random_link_split` determinism | Identical splits |
| | `test_different_seed_different_split` | Seed sensitivity | Different splits |
| | `test_stratified_split_reproducibility` | `relation_stratified_split` determinism | Identical splits |
| | `test_split_uses_torch_generator` | Generator isolation | No global RNG dependency |
| `TestNegativeSamplingReproducibility` | `test_same_seed_same_negative_samples` | Sampler determinism | Identical negatives |
| | `test_different_seed_different_negative_samples` | Seed sensitivity | Different negatives |
| | `test_sample_flat_reproducibility` | Flat output determinism | Identical flat triples |
| `TestModelInitReproducibility` | `test_same_seed_same_model_init` | TransE/DistMult/RotatE | Identical weights |
| | `test_different_seed_different_model_init` | Seed sensitivity | Different weights |
| `TestTrainingReproducibility` | `test_same_seed_same_training_loss` | TransE full training | Identical loss sequences |
| | `test_different_seed_different_training_loss` | Seed sensitivity | Different losses |
| | `test_training_reproducibility_distmult` | DistMult + BCE loss | Identical loss sequences |
| `TestDeterministicMode` | `test_deterministic_mode_enables_settings` | `set_deterministic(True)` | Correct flags |
| | `test_deterministic_mode_disables_settings` | `set_deterministic(False)` | Flags reverted |
| | `test_deterministic_mode_idempotent` | Double enable | No breakage |
| `TestSeedWorker` | `test_seed_worker_sets_random_state` | Worker RNG isolation | Correct derived seed |
| `TestCheckpointReproducibility` | `test_checkpoint_roundtrip` | Save/load fidelity | Exact parameter match |
| | `test_checkpoint_training_continuation` | Resume from checkpoint | Consistent losses |
| `TestRotatEReproducibility` | `test_rotate_init_reproducibility` | Complex number init | Identical phases |
| | `test_rotate_score_reproducibility` | Score computation | Deterministic scores |
| `TestEndToEndReproducibility` | `test_full_pipeline_deterministic` | Complete pipeline | Identical outputs |

### 4.2 Coverage Assessment

| Component | Tests | Coverage |
|-----------|-------|----------|
| `reproducibility.py::set_seed` | 5 direct + used in 15+ tests | Complete |
| `reproducibility.py::set_deterministic` | 3 direct | Complete |
| `reproducibility.py::seed_worker` | 1 direct | Complete |
| `splits.py::random_link_split` | 4 direct | Complete |
| `splits.py::relation_stratified_split` | 1 direct | Adequate |
| `sampler.py::TypeConstrainedNegativeSampler` | 3 direct | Complete |
| `models/*.py` init | 3 (parametrized over 3 models) | Complete |
| `trainer.py::Trainer.fit` | 3 direct | Complete |
| `trainer.py` checkpoint | 2 direct | Complete |
| End-to-end pipeline | 1 direct | Complete |

**Total**: 27 test methods across 10 test classes.

---

## 5. Multi-GPU Reproducibility Notes

Multi-GPU training introduces additional sources of non-determinism that are
**out of scope** for the current CPU-focused test suite but should be
documented:

### 5.1 Known Issues

1. **Floating-point reduction order**: `torch.distributed` operations like
   `all_reduce` may use different reduction trees on different GPU counts,
   leading to floating-point differences.

2. **DataParallel scatter/gather**: `torch.nn.DataParallel` splits batches
   across GPUs, and the per-GPU gradients are summed in non-deterministic
   order unless `use_deterministic_algorithms(True)` is enabled.

3. **NCCL non-determinism**: NCCL collective operations are not guaranteed
   to be deterministic across runs even with the same configuration.

4. **Different GPU counts**: Changing the number of GPUs changes the effective
   batch size and gradient accumulation, producing different optimisation
   trajectories even with identical seeds.

### 5.2 Mitigation Recommendations

- Always report the exact GPU count and model in reproducibility claims.
- Use `DistributedDataParallel` rather than `DataParallel` for better control.
- Consider gradient accumulation to normalise effective batch size across
  different GPU configurations.
- Accept float-level tolerance (1e-5 to 1e-3) for multi-GPU comparisons
  rather than requiring bit-exact equality.

---

## 6. Findings and Recommendations

### 6.1 Findings Summary

| ID | Severity | Finding | File | Line(s) |
|----|----------|---------|------|---------|
| F1 | Low | `_simple_negative_scores()` uses global RNG instead of explicit generator | `trainer.py` | 309-312 |
| F2 | Low | RNG states not saved in checkpoint | `trainer.py` | 213-222 |
| F3 | Info | `CUBLAS_WORKSPACE_CONFIG` not unset when `set_deterministic(False)` | `reproducibility.py` | 56-57 |
| F4 | Info | No multi-GPU reproducibility tests | N/A | N/A |

### 6.2 Detailed Recommendations

#### R1: Thread explicit generator through trainer fallback negative sampling (Low priority)

In `Trainer._simple_negative_scores()`, replace:
```python
rand_idx = torch.randint(0, num_dst, (edge_index.size(1),), device=self.device)
```
with a generator-based version. This would make the trainer fully independent
of global RNG state for negative sampling.

#### R2: Save RNG states in checkpoints (Medium priority)

For perfect mid-training resumption reproducibility, save and restore all RNG
states (Python, NumPy, PyTorch CPU, PyTorch CUDA) in `save_checkpoint()` and
`load_checkpoint()`. This is especially important for long training runs
where interruptions are expected.

#### R3: Document multi-GPU reproducibility limitations (Low priority)

Add a section to the training documentation explaining:
- Multi-GPU training is not guaranteed to be bit-identical across runs.
- Changing GPU count changes results.
- Recommended tolerances for comparing multi-GPU results.

#### R4: Clean up CUBLAS_WORKSPACE_CONFIG on disable (Cosmetic)

In `set_deterministic(False)`, consider unsetting the environment variable:
```python
if not enabled:
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
```

---

## 7. Conclusion

The glycoMusubi embedding pipeline implements a robust and comprehensive
reproducibility infrastructure. The `set_seed()` function covers all four
critical RNG sources (Python, NumPy, PyTorch CPU, PyTorch CUDA). Data
splitting and negative sampling correctly use local `torch.Generator`
instances, making them independent of global state. The `set_deterministic()`
function enables PyTorch's strictest deterministic mode including cuDNN and
cuBLAS controls.

The test suite (`tests/test_reproducibility.py`) provides 27 tests across
10 classes, covering every component in the reproducibility chain from
individual RNG sources through end-to-end pipeline determinism.

All findings are low-severity or informational. The codebase meets
Nature-level reproducibility standards for CPU-based experiments. Multi-GPU
reproducibility should be documented as a known limitation rather than a
strict guarantee.

**Overall Verdict**: **PASS**
