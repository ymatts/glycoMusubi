# R1: Schema Consistency & Data Conversion Accuracy Report

**Date**: 2026-02-13
**Reviewer**: Computational Science Expert (validate-compsci)
**Status**: PASS (with advisory findings)

---

## 1. Executive Summary

The converter (`glycoMusubi/data/converter.py`) correctly processes all 7 node types and all 6 edge types defined in the YAML schema files. TSV-to-HeteroData conversion is lossless for both nodes and edges on the mini test dataset. No critical issues were found. Two advisory-level findings are noted regarding schema coverage gaps and regex anchoring.

**Test results**: 37/37 tests pass (`tests/test_schema_consistency.py`)

---

## 2. Schema Inventory

### 2.1 Node Types

| Node Type | node_schema.yaml | entity_config.yaml | converter reachable | mini test data |
|-----------|:----------------:|:------------------:|:-------------------:|:--------------:|
| enzyme    | Yes              | Yes                | Yes                 | 2 nodes        |
| protein   | Yes              | Yes                | Yes                 | 4 nodes        |
| glycan    | Yes              | Yes                | Yes                 | 3 nodes        |
| disease   | Yes              | Yes                | Yes                 | 2 nodes        |
| variant   | Yes              | Yes                | Yes                 | 1 node         |
| compound  | Yes              | Yes                | Yes                 | 1 node         |
| site      | Yes              | **No**             | Yes                 | 2 nodes        |

**Total**: 7 node types, 15 nodes in mini dataset.

**Finding A-1 (Advisory)**: `entity_config.yaml` does not define the `site` entity type. The `site` type is defined in `node_schema.yaml` and is correctly handled by the converter via the edge schema (as target of `has_site` and endpoint of `ptm_crosstalk`). This is acceptable because `site` nodes are derived from edge data rather than a standalone source table, but should be documented.

### 2.2 Edge / Relation Types

| Relation               | edge_schema.yaml | relation_config.yaml | converter map | mini test data |
|------------------------|:----------------:|:-------------------:|:-------------:|:--------------:|
| inhibits               | Yes              | Yes                 | Yes           | 2 edges        |
| has_glycan             | Yes              | Yes                 | Yes           | 4 edges        |
| associated_with_disease| Yes              | Yes                 | Yes           | 3 edges        |
| has_variant            | Yes              | Yes                 | Yes           | 2 edges        |
| has_site               | Yes              | **No**              | Yes           | 3 edges        |
| ptm_crosstalk          | Yes              | **No**              | Yes           | 1 edge         |

**Total**: 6 edge types, 15 edges in mini dataset.

**Finding A-2 (Advisory)**: `relation_config.yaml` defines only 4 of 6 relation types (missing `has_site` and `ptm_crosstalk`). The converter correctly compensates by merging `edge_schema.yaml` as a fallback source (converter.py:84-87). This dual-source approach works but means `relation_config.yaml` is an incomplete reference. Consider adding the missing types for consistency.

---

## 3. Source/Target Type Matching

The converter's `_relation_type_map` was verified against both `edge_schema.yaml` and `relation_config.yaml`:

| Relation               | Schema source_type   | Schema target_type | Converter (src, tgt) pairs        | Match |
|------------------------|----------------------|--------------------|-----------------------------------|:-----:|
| inhibits               | compound             | enzyme             | {(compound, enzyme)}              | Yes   |
| has_glycan             | protein              | glycan             | {(protein, glycan)}               | Yes   |
| associated_with_disease| protein              | disease            | {(protein, disease)}              | Yes   |
| has_variant            | protein              | variant            | {(protein, variant)}              | Yes   |
| has_site               | [protein, enzyme]    | site               | {(protein, site), (enzyme, site)} | Yes   |
| ptm_crosstalk          | site                 | site               | {(site, site)}                    | Yes   |

The `has_site` relation correctly expands the list-valued `source_type: [protein, enzyme]` into two (source, target) pairs, handled at converter.py:96-102.

All source and target types are members of the 7 valid node types defined in `node_schema.yaml`.

---

## 4. Node ID Pattern Verification

| Node Type | id_pattern                        | Test IDs                              | Result |
|-----------|-----------------------------------|---------------------------------------|:------:|
| glycan    | `^G\d{5}[A-Z]{2}$`               | G00001AA, G00002BB, G00003CC          | Pass   |
| compound  | `^CHEMBL\d+$`                     | CHEMBL10001                           | Pass   |
| site      | `^SITE::[A-Z0-9]+-?\d*::\d+::[A-Z]$` | SITE::P12345::42::N, SITE::Q67890::100::S | Pass |
| enzyme    | UniProt regex (see below)         | O43451-1, Q9Y2C3-1                    | Pass*  |
| protein   | UniProt regex (see below)         | P12345, Q67890, P11111, Q22222        | Pass   |
| disease   | null (no pattern)                 | Free-text names                       | N/A    |
| variant   | null (no pattern)                 | Free-text identifiers                 | N/A    |

**Finding A-3 (Advisory)**: The enzyme/protein `id_pattern` regex uses `|` (alternation) without grouping the full expression:

```
^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$
```

Due to operator precedence, this is parsed as: `(^[OPQ][0-9][A-Z0-9]{3}[0-9]) | ([A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$)`. The first alternative lacks a `$` anchor and the second lacks a `^` anchor. In practice this works because `re.match()` anchors at the start, but `re.fullmatch()` would reject some valid IDs (e.g., isoform-suffixed IDs like `O43451-1` would fail on the first alternative). Recommended fix:

```
^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$
```

All node IDs from the mini dataset are correctly preserved in the converter's `build_node_mappings()` output: every `(node_id, node_type)` pair maps bijectively to a unique local integer index.

---

## 5. Metadata Extraction Accuracy

The converter's `extract_node_metadata()` correctly parses JSON metadata strings from the TSV `metadata` column. Verified:

| Node Type | Sample ID          | Metadata fields verified            | Result |
|-----------|--------------------|------------------------------------|:------:|
| enzyme    | O43451-1           | source="GlyGen"                    | Pass   |
| protein   | P12345             | source="UniProt"                   | Pass   |
| glycan    | G00001AA           | (empty `{}`)                       | Pass   |
| disease   | Diabetes mellitus  | source="UniProt"                   | Pass   |
| variant   | p.Arg123Cys        | (empty `{}`)                       | Pass   |
| compound  | CHEMBL10001        | source="ChEMBL"                    | Pass   |
| site      | SITE::P12345::42::N| source, position=42 (int), residue="N" | Pass |

**Finding A-4 (Advisory)**: The mini test data does not exercise several metadata fields defined in `node_schema.yaml`: WURCS (glycan), iupac (glycan), mass (glycan), gene_symbol (protein/enzyme), ec_number (enzyme), cazy_family (enzyme), omim_id (disease), mondo_id (disease), clinical_significance (variant). Consider enriching the mini dataset to improve test coverage of optional metadata fields.

---

## 6. Data Conversion Accuracy (TSV to HeteroData)

### 6.1 Node Count Preservation

| Node Type | TSV count | HeteroData num_nodes | Match |
|-----------|:---------:|:-------------------:|:-----:|
| protein   | 4         | 4                   | Yes   |
| enzyme    | 2         | 2                   | Yes   |
| glycan    | 3         | 3                   | Yes   |
| disease   | 2         | 2                   | Yes   |
| variant   | 1         | 1                   | Yes   |
| compound  | 1         | 1                   | Yes   |
| site      | 2         | 2                   | Yes   |
| **Total** | **15**    | **15**              | **Yes** |

### 6.2 Edge Count Preservation

| Edge Type (src, rel, dst)               | TSV count | HeteroData count | Match |
|-----------------------------------------|:---------:|:----------------:|:-----:|
| (compound, inhibits, enzyme)            | 2         | 2                | Yes   |
| (protein, has_glycan, glycan)           | 4         | 4                | Yes   |
| (protein, associated_with_disease, disease) | 3     | 3                | Yes   |
| (protein, has_variant, variant)         | 2         | 2                | Yes   |
| (protein, has_site, site)               | 2         | 2                | Yes   |
| (enzyme, has_site, site)                | 1         | 1                | Yes   |
| (site, ptm_crosstalk, site)            | 1         | 1                | Yes   |
| **Total**                               | **15**    | **15**            | **Yes** |

Zero data loss. No edges were skipped (all source/target IDs resolve to valid nodes).

### 6.3 Edge Index Validity

- All `edge_index` tensors have shape `[2, num_edges]` and dtype `torch.long`.
- All source indices are in `[0, num_src_nodes)`.
- All target indices are in `[0, num_dst_nodes)`.
- No negative indices.
- No self-loops in `ptm_crosstalk` (site->site).
- No duplicate edges in any edge type.

### 6.4 Node Feature Initialisation

- All node types initialised with Xavier-uniform features.
- Feature dimensionality consistent across all types (32 in test, 256 default).
- All values finite (no NaN/Inf).
- `x.size(0) == num_nodes` for every type.

---

## 7. Converter Robustness

Two additional robustness scenarios were tested:

1. **Dangling edge references**: When an edge references a node_id not in the nodes table, the edge is silently skipped with a warning log. The converter does not crash. Verified with a synthetic test.

2. **Unknown relation types**: When an edge has a relation type not defined in any schema, the converter falls back to using the actual node types of the endpoints (converter.py:118-121). Verified that `(protein, new_relation, glycan)` is correctly inferred.

---

## 8. Findings Summary

| ID  | Severity | Description | Recommendation |
|-----|----------|-------------|----------------|
| A-1 | Advisory | `entity_config.yaml` missing `site` entity type | Add `site` to entity_config or document the omission |
| A-2 | Advisory | `relation_config.yaml` missing `has_site` and `ptm_crosstalk` | Add missing relation types for completeness |
| A-3 | Advisory | Enzyme/protein `id_pattern` regex has incorrect alternation grouping | Wrap alternatives in `(?:...\|...)` and add isoform suffix outside |
| A-4 | Advisory | Mini test data lacks coverage of optional metadata fields (WURCS, gene_symbol, etc.) | Enrich mini_nodes.tsv with optional metadata |

No critical or major issues found.

---

## 9. Test Coverage

File: `tests/test_schema_consistency.py`

| Test Class                    | Tests | Status  |
|-------------------------------|:-----:|:-------:|
| TestNodeTypeConsistency       | 3     | 3 pass  |
| TestEdgeTypeConsistency       | 3     | 3 pass  |
| TestRelationSourceTargetMatch | 3     | 3 pass  |
| TestNodeIdPatterns            | 4     | 4 pass  |
| TestMetadataExtraction        | 7     | 7 pass  |
| TestNoDataLoss                | 6     | 6 pass  |
| TestEdgeIndexValidity         | 5     | 5 pass  |
| TestNodeFeatures              | 3     | 3 pass  |
| TestDuplicateEdges            | 1     | 1 pass  |
| TestConverterRobustness       | 2     | 2 pass  |
| **Total**                     | **37**| **37 pass** |

---

## 10. Conclusion

The data conversion layer (`KGConverter`) is schema-consistent and produces structurally valid HeteroData graphs with zero data loss. The converter's dual-source config loading strategy (edge_schema + relation_config) ensures all 6 edge types are covered despite `relation_config.yaml` defining only 4. All 7 node types are reachable through the relation type map. The advisory findings are non-blocking but should be addressed for long-term maintainability.
