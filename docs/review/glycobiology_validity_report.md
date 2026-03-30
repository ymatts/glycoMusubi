# Glycobiology Validity Report

**Reviewer**: Glycobiology Domain Expert (validate-glycobio)
**Date**: 2026-02-13
**Scope**: Task #13 (R3) -- Glycobiology domain validity of glycoMusubi v0.1 pipeline
**Status**: COMPLETE -- 3 critical bugs found, 1 schema inconsistency, 3 design recommendations

---

## 1. Executive Summary

The glycoMusubi v0.1 pipeline provides a well-structured foundation for glycan-centric knowledge graph embedding. The schema design (node types, edge types, type constraints) is biologically sound and correctly models the key entities and relationships in glycobiology. The `TypeConstrainedNegativeSampler` properly enforces biologically valid negative sampling. However, the **WURCS feature extraction pipeline has critical bugs** that render the `wurcs_features` and `hybrid` encoder modes non-functional -- all glycans receive identical zero-feature vectors, collapsing structural diversity.

### Verdict by Component

| Component | Status | Severity |
|-----------|--------|----------|
| Node schema (7 types) | PASS | -- |
| Edge schema (6 relations) | PASS with note | Minor: `has_glycan` source_type inconsistency |
| Relation type constraints | PASS | -- |
| TypeConstrainedNegativeSampler | PASS | -- |
| GlycanEncoder (learnable mode) | PASS | -- |
| GlycanEncoder (wurcs_features mode) | **FAIL** | Critical: always produces zero vectors |
| GlycanEncoder (hybrid mode) | **FAIL** | Critical: WURCS component is zero |
| WURCS feature extraction | **FAIL** | Critical: 3 bugs (parsing + regex + precedence) |
| ProteinEncoder | PASS | -- |
| TextEncoder | PASS | -- |
| Core type estimation (logic) | PASS with note | Minor: GAG detection precedence |
| Monosaccharide classification (logic) | PASS with note | Minor: GlcNAc/NeuAc regex overlap |

---

## 2. Critical Bugs Found

### 2.1 BUG-WURCS-PARSE: Residue list always empty (CRITICAL)

**File**: `glycoMusubi/embedding/encoders/glycan_encoder.py`, function `_parse_wurcs_sections`, lines 79-109

**Description**: The WURCS v2.0 string format places unique residues in brackets, followed by `/res_seq/linkages`. After extracting unique residues, the parser finds the tail after the last `]`, which has the form `/res_seq/linkages` (starts with `/`). Splitting by `/` produces:

```
tail = "/1-2-3-4-2-4-2/a4-b1_b4-c1_..."
tail_sections = ["", "1-2-3-4-2-4-2", "a4-b1_b4-c1_..."]
                  ^     ^                  ^
                  [0]   [1]                [2] (DISCARDED)
```

- `tail_sections[0]` (empty string) is assigned to `res_seq` => `res_list` is always `[]`
- `tail_sections[1]` (the actual residue sequence) is assigned to `lin_section`
- `tail_sections[2]` (the actual linkage section) is discarded

**Impact**: All monosaccharide counts are zero. Total residue count is always 1. Core type estimation receives zero inputs. The entire WURCS feature extraction pipeline produces identical 24-dimensional vectors for all glycans, making `wurcs_features` and `hybrid` modes equivalent to untrained learnable embeddings.

**Fix**: Strip the leading `/` from `tail` before splitting, or use `tail_sections[1]` for res_seq and `tail_sections[2]` for lin_section:
```python
tail = tail.lstrip("/")
tail_sections = tail.split("/")
res_seq = tail_sections[0] if len(tail_sections) > 0 else ""
lin_section = tail_sections[1] if len(tail_sections) > 1 else ""
```

### 2.2 BUG-WURCS-REGEX: GlcNAc misclassified as NeuAc (CRITICAL)

**File**: `glycoMusubi/embedding/encoders/glycan_encoder.py`, `_WURCS_RESIDUE_PATTERNS`, line 52

**Description**: The NeuAc detection pattern `a2122h-1b_1-5_2\*N` is a prefix of the GlcNAc pattern `a2122h-1b_1-5_2\*NCC`. Since NeuAc patterns are checked before HexNAc patterns (by design, as per the comment "NeuAc / NeuGc must be checked before generic Hex"), a GlcNAc residue string like `a2122h-1b_1-5_2*NCC/3=O` matches the NeuAc regex first because `*N` appears within `*NCC`.

**Impact**: Every GlcNAc (N-acetylglucosamine) and GalNAc (N-acetylgalactosamine) residue is misclassified as NeuAc (sialic acid). This is biologically catastrophic because:
- N-glycan core detection fails (HexNAc count is always 0)
- Sialylation is massively over-reported
- O-glycan core detection fails (GalNAc not detected)
- N-glycan vs O-glycan classification is random

**Fix**: Make the NeuAc regex more specific by requiring the pattern to NOT be followed by `C`:
```python
(re.compile(r"a2122h-1b_1-5_2\*N(?!C)"), 3),  # NeuAc (not NCC)
```
Or reorder to check HexNAc (`*NCC`) before NeuAc (`*N`).

### 2.3 BUG-WURCS-NUMERIC: Numeric residue tokens not handled (CRITICAL)

**File**: `glycoMusubi/embedding/encoders/glycan_encoder.py`, function `_count_monosaccharides`, lines 126-141

**Description**: Real WURCS v2.0 residue sequences use 1-based numeric indices (`1-2-3-4-2-4-2`), but `_count_monosaccharides` expects letter-based tokens where the letter maps to a unique residue index (`a`=0, `b`=1, etc.). The function checks `token[0].isalpha()` and falls through to the Hex fallback for numeric tokens.

**Impact**: Even after fixing BUG-WURCS-PARSE, all residues would be counted as generic Hex because numeric tokens (`"1"`, `"2"`, etc.) fail the `isalpha()` check.

**Fix**: Handle numeric tokens by converting them to 0-based indices:
```python
if letter.isalpha():
    uidx = ord(letter.lower()) - ord("a")
elif letter.isdigit():
    uidx = int(token) - 1  # 1-based to 0-based
```

---

## 3. Schema Biological Validity

### 3.1 Node Types -- PASS

All 7 node types in `node_schema.yaml` are biologically appropriate:

| Node Type | Biological Entity | ID Pattern | Assessment |
|-----------|-------------------|------------|------------|
| glycan | Glycan structure | `G\d{5}[A-Z]{2}` (GlyTouCan) | Correct |
| protein | Non-enzyme protein | UniProt accession | Correct |
| enzyme | Glycosyltransferase | UniProt accession | Correct |
| disease | Disease condition | Free-form | Correct |
| variant | Genetic variant | Free-form | Correct |
| compound | Small molecule inhibitor | `CHEMBL\d+` | Correct |
| site | PTM site | `SITE::protein::pos::residue` | Correct, encodes position/residue |

The `site` node type is particularly well-designed, encoding protein ID, amino acid position, and residue type in the ID itself, enabling efficient glycosylation site mapping.

### 3.2 Edge Types -- PASS with note

| Relation | Source -> Target | Biological Meaning | Assessment |
|----------|------------------|-------------------|------------|
| has_glycan | protein -> glycan | Protein glycosylation | Correct |
| inhibits | compound -> enzyme | Enzyme inhibition | Correct |
| associated_with_disease | protein -> disease | Disease association | Correct |
| has_variant | protein -> variant | Genetic variant | Correct |
| has_site | protein/enzyme -> site | PTM site ownership | Correct |
| ptm_crosstalk | site -> site | PTM co-regulation | Correct |

**Note on `has_glycan` source_type**: The `edge_schema.yaml` and `relation_config.yaml` define `has_glycan` with `source_type: protein` only. However, the mini test data (`mini_edges.tsv`) contains an enzyme (Q11111) with a `has_glycan` edge to a glycan. Biologically, glycosyltransferases themselves can be glycosylated, so enzymes should be valid sources for `has_glycan`. Recommendation: add `enzyme` to `has_glycan.source_type` as a list `[protein, enzyme]`, consistent with how `has_site` handles multiple source types.

### 3.3 Ontology URIs -- PASS

The ontology URIs reference appropriate Relation Ontology (RO) terms:
- `RO_0002449` (inhibits) -- correct for enzyme inhibition
- `RO_0002180` (has_glycan, has_site) -- appropriate for modification relationships
- `RO_0002200` (associated_with_disease) -- correct for disease association
- `SO_0001060` (has_variant) -- correct for sequence variants

---

## 4. Negative Sampling Validity -- PASS

The `TypeConstrainedNegativeSampler` correctly enforces type constraints:

- **has_glycan**: Corrupted heads are restricted to protein nodes; corrupted tails to glycan nodes. This prevents biologically impossible triples like glycan-has_glycan-glycan.
- **inhibits**: Corrupted heads restricted to compound; tails to enzyme.
- **Unknown relations**: Fall back to all-node sampling (acceptable for extensibility).

The sampler's `corrupt_head_prob=0.5` default provides balanced head/tail corruption, which is standard practice for KGE training.

---

## 5. WURCS Feature Design Assessment

### 5.1 Feature Vector Design -- GOOD (once bugs are fixed)

The 24-dimensional feature vector is well-designed from a glycobiology perspective:

| Features | Index | Biological Meaning | Assessment |
|----------|-------|-------------------|------------|
| 8 monosaccharide counts | 0-7 | Hex, HexNAc, dHex, NeuAc, NeuGc, Pen, HexA, Kdn | Good coverage |
| Branching degree | 8 | Antennary complexity | Biologically important |
| Total residues | 9 | Glycan size | Basic but useful |
| 3 modification flags | 10-12 | Sulfation, phosphorylation, acetylation | Key GAG/mucin markers |
| 4 core type scores | 13-16 | N-glycan, O-glycan, GAG, other | Useful for classification |
| 3 derived ratios | 17-19 | Sialylation, fucosylation, branching | Biologically informative |
| 4 reserved | 20-23 | Future use | Good extensibility |

**Monosaccharide coverage**: The 8 classes cover the major building blocks of mammalian glycans. Missing: KDN is rare in mammals; NeuGc is absent in humans but present in other species. The classification is appropriate for the GlyTouCan database scope.

### 5.2 Core Type Estimation -- GOOD with minor issue

The heuristic for N-glycan vs O-glycan classification is reasonable:
- N-glycan: HexNAc >= 2 AND Hex >= 3 (captures Man3GlcNAc2 core)
- O-glycan: HexNAc >= 1 AND total <= 6 (captures small GalNAc-initiated structures)
- GAG: HexA >= 2 AND HexNAc >= 2 (captures alternating disaccharide repeat)

**BUG-CORE-TYPE-PRECEDENCE** (Minor): The `elif` chain means O-glycan is checked before GAG. A GAG disaccharide repeat unit with 6 total residues (HexA=3, HexNAc=3, total=6) triggers the O-glycan condition first, preventing GAG detection. Fix: check GAG condition before O-glycan, or use independent scoring instead of elif.

### 5.3 Missing Features (Recommendations)

For future versions, consider adding:
1. **Linkage-type features**: alpha1-3 vs alpha1-6 linkage distribution (important for immune recognition, e.g., alpha-Gal epitope)
2. **Antenna count**: Explicitly count bi/tri/tetra-antennary branches
3. **Lewis/blood group motif detection**: Important clinical glycan markers

---

## 6. Encoder Architecture Assessment

### 6.1 GlycanEncoder -- GOOD design, blocked by bugs

The three-mode design (learnable / wurcs_features / hybrid) is well-architected:

- **Learnable mode**: Standard embedding lookup; works correctly.
- **WURCS features mode**: Projects 24-dim biochemical features through 2-layer MLP with GELU + LayerNorm. Architecture is sound, but input is always zero due to parsing bugs.
- **Hybrid mode**: Concatenates learnable + WURCS, fuses via MLP. Good design for combining structural knowledge with learned representations.

The WURCS feature cache (`_wurcs_cache`) is a good optimization for avoiding redundant parsing.

### 6.2 ProteinEncoder -- PASS

The ESM-2 integration is biologically appropriate:
- Uses `esm2_t33_650M_UR50D` (650M parameters, well-validated for protein representation)
- Projection MLP: 1280 -> 640 -> 256 with GELU, dropout, LayerNorm
- Graceful fallback to learnable embeddings for missing ESM-2 cache files
- Mean-pooling for per-residue to sequence-level conversion

### 6.3 TextEncoder -- PASS

Hash-based learnable embeddings are a reasonable Phase 1 approach for disease/compound names. SHA-256 provides cross-platform determinism.

---

## 7. Mini-KG Test Data Assessment

The mini test data in `tests/test_data/` is biologically consistent:
- Glycan IDs follow GlyTouCan format (G#####XX)
- Site IDs encode protein-position-residue (SITE::P12345::100::N)
- Enzyme entries have UniProt-format IDs
- Edge directionality matches biological relationships

**Note**: One enzyme-glycan edge (Q11111 -> G00001AA via has_glycan) is present but not covered by the schema's strict protein-only source_type.

---

## 8. Recommended Fixes (Priority Order)

### P0 -- Critical (must fix before any WURCS-based training)

1. **Fix `_parse_wurcs_sections` tail splitting** (BUG-WURCS-PARSE)
   - File: `glycoMusubi/embedding/encoders/glycan_encoder.py:100`
   - Change: `tail = tail.lstrip("/")`

2. **Fix NeuAc regex to not match GlcNAc** (BUG-WURCS-REGEX)
   - File: `glycoMusubi/embedding/encoders/glycan_encoder.py:52`
   - Change: `a2122h-1b_1-5_2\*N` -> `a2122h-1b_1-5_2\*N(?!C)`

3. **Handle numeric residue tokens in `_count_monosaccharides`** (BUG-WURCS-NUMERIC)
   - File: `glycoMusubi/embedding/encoders/glycan_encoder.py:131-139`
   - Add: numeric token -> 0-based unique_res index mapping

### P1 -- Important (biological correctness)

4. **Fix core type elif precedence** (BUG-CORE-TYPE-PRECEDENCE)
   - Move GAG check before O-glycan check in `_estimate_core_type`

5. **Allow enzyme as has_glycan source**
   - Update `edge_schema.yaml` and `relation_config.yaml`: `has_glycan.source_type: [protein, enzyme]`

### P2 -- Enhancement

6. Add linkage-type features (alpha vs beta, position) to WURCS feature vector
7. Add explicit antenna count feature
8. Add glycan motif detection (Lewis, blood group, high-mannose marker)

---

## 9. Test Results Summary

**Test file**: `tests/test_glycobiology_validity.py`
**Result**: 52 passed, 12 xfailed (documenting known bugs)

| Test Class | Passed | XFail (Bug) | Description |
|------------|--------|-------------|-------------|
| TestWURCSParsingSections | 2 | 2 | Parsing section extraction |
| TestWURCSFeatures | 8 | 5 | End-to-end feature extraction |
| TestCountMonosaccharidesDirectly | 4 | 3 | Direct monosaccharide counting |
| TestCoreTypeEstimation | 6 | 1 | Core type heuristics |
| TestBiologicalConstraints | 11 | 0 | Schema + negative sampler |
| TestGlycanEncoderBiologicalProperties | 5 | 1 | Encoder output properties |
| TestWURCSParserInternals | 7 | 0 | Parser internals |
| TestEdgeBiologicalValidity | 6 | 0 | Mini-KG data consistency |
| **Total** | **52** | **12** | |

All xfail tests are `strict=True`, meaning they will fail if the bug is fixed (alerting that the xfail marker should be removed).

---

## 10. Conclusion

The glycoMusubi schema and type-constrained negative sampling are biologically sound and correctly model glycobiology domain constraints. The WURCS feature extraction design is thoughtful and captures key biochemical properties. However, **three critical bugs in the WURCS parsing pipeline prevent any meaningful glycan structural features from being extracted**, rendering the `wurcs_features` and `hybrid` encoder modes non-functional. The `learnable` mode works correctly but does not leverage glycan structural information.

Once the P0 fixes are applied, the WURCS feature extraction pipeline should correctly capture monosaccharide composition, modifications, branching, and core type -- enabling biologically meaningful glycan embeddings that distinguish N-glycans from O-glycans, sialylated from non-sialylated structures, and fucosylated from non-fucosylated structures.
