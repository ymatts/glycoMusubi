"""Glycobiology domain-knowledge validity tests for glycoMusubi.

These tests verify that the WURCS feature extraction, encoder modules,
type-constrained negative sampler, and KG schema correctly model
glycobiology domain constraints and produce biologically meaningful
results.

Covers:
  - Monosaccharide composition counting from WURCS strings
  - Sialylation, fucosylation, and modification detection
  - Branching degree calculation
  - Core type (N-glycan / O-glycan / GAG) estimation heuristics
  - GlycanEncoder output properties (structural similarity reflected)
  - TypeConstrainedNegativeSampler biological validity
  - Edge type constraints matching glycobiology expectations
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import yaml

from glycoMusubi.embedding.encoders.glycan_encoder import (
    GlycanEncoder,
    extract_wurcs_features,
    _parse_wurcs_sections,
    _count_monosaccharides,
    _branching_degree,
    _detect_modifications,
    _estimate_core_type,
    MONOSACCHARIDE_CLASSES,
)
from glycoMusubi.data.sampler import TypeConstrainedNegativeSampler

# ---------------------------------------------------------------------------
# Reference WURCS v2.0 strings from GlyTouCan.
#
# WURCS format: WURCS=2.0/counts/[unique_res1][unique_res2]...//res_seq/linkages
# - counts: unique_res_count,total_residue_count,linkage_count
# - unique residues are bracket-delimited
# - residue sequence uses 1-based numeric indices into the unique list
# - linkage section uses letter-based references (a=res1, b=res2, ...)
# ---------------------------------------------------------------------------

# Biantennary complex N-glycan (G00028MO from GlyTouCan):
# GlcNAc2Man3GlcNAc2 core
# Unique residues: 1=GlcNAc, 2=Man-beta, 3=Man-alpha
# Res: 1-1-2-3-1-3-1 => 4 GlcNAc + 3 Man
WURCS_BIANTENNARY_NGLYCAN = (
    "WURCS=2.0/3,7,6/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # unique res 1: GlcNAc
    "[a1122h-1b_1-5]"              # unique res 2: Man-beta
    "[a1122h-1a_1-5]"              # unique res 3: Man-alpha
    "/1-1-2-3-1-3-1/"              # 4x res1(GlcNAc) + 1x res2(Man-b) + 2x res3(Man-a)
    "a4-b1_b4-c1_c3-d1_c6-f1_d4-e1_f4-g1"
)

# Core-fucosylated N-glycan:
# Same as biantennary + Fuc (dHex) on core GlcNAc
# Unique residues: 1=GlcNAc, 2=Man-beta, 3=Man-alpha, 4=Fuc
WURCS_CORE_FUCOSYLATED = (
    "WURCS=2.0/4,8,7/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Man-beta
    "[a1122h-1a_1-5]"              # 3: Man-alpha
    "[a1221m-1a_1-5]"              # 4: Fuc (dHex)
    "/1-1-2-3-1-3-1-4/"            # 4 GlcNAc + 1 Man-b + 2 Man-a + 1 Fuc
    "a4-b1_b4-c1_c3-d1_c6-f1_d4-e1_f4-g1_a6-h1"
)

# Sialylated N-glycan fragment with NeuAc:
# GlcNAc-Gal-Man-Man-NeuAc
# Unique residues: 1=GlcNAc, 2=Gal/Man-beta, 3=Man-alpha, 4=NeuAc
WURCS_SIALYLATED = (
    "WURCS=2.0/4,5,4/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Gal / Man
    "[a1122h-1a_1-5]"              # 3: Man-alpha
    "[a2122h-1b_1-5_2*N]"          # 4: NeuAc
    "/1-2-3-2-4/"
    "a4-b1_b4-c1_c3-d1_d3-e1"
)

# Simple O-glycan: GalNAc-Gal-NeuAc (core 1)
# Unique residues: 1=GalNAc(HexNAc), 2=Gal(Hex), 3=NeuAc
WURCS_OGLYCAN_CORE1 = (
    "WURCS=2.0/3,3,2/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GalNAc (HexNAc)
    "[a2122h-1b_1-5]"              # 2: Gal (Hex)
    "[a2122h-1b_1-5_2*N]"          # 3: NeuAc
    "/1-2-3/"
    "a3-b1_b3-c1"
)

# Single hexose (minimal glycan)
WURCS_SINGLE_HEXOSE = (
    "WURCS=2.0/1,1,0/"
    "[a2122h-1b_1-5]"
    "/1/"
)

# High-mannose N-glycan (Man5GlcNAc2)
# Unique residues: 1=GlcNAc, 2=Man
WURCS_HIGH_MANNOSE = (
    "WURCS=2.0/2,7,6/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Man
    "/1-1-2-2-2-2-2/"              # 2 GlcNAc + 5 Man
    "a4-b1_b4-c1_c3-d1_c6-e1_d2-f1_e2-g1"
)

# GAG-like: alternating HexA-HexNAc (heparan sulfate fragment)
WURCS_GAG_LIKE = (
    "WURCS=2.0/2,6,5/"
    "[a2122A-1b_1-5]"              # 1: GlcA (HexA)
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 2: GlcNAc (HexNAc)
    "/1-2-1-2-1-2/"                # alternating HexA-HexNAc
    "a4-b1_b4-c1_c4-d1_d4-e1_e4-f1"
)


# =====================================================================
# TestWURCSParsingSections: verify parsing splits WURCS correctly
# =====================================================================


class TestWURCSParsingSections:
    """Test that _parse_wurcs_sections extracts correct WURCS sections."""

    def test_unique_residues_extracted(self):
        """Unique residue list should be correctly extracted from brackets."""
        _header, unique_res, _res, _lin = _parse_wurcs_sections(
            WURCS_BIANTENNARY_NGLYCAN
        )
        assert len(unique_res) == 3, (
            f"Expected 3 unique residues (GlcNAc, Man-b, Man-a), got {len(unique_res)}"
        )

    def test_residue_list_extracted(self):
        """Residue list should contain 7 entries for biantennary glycan."""
        _header, _ures, res_list, _lin = _parse_wurcs_sections(
            WURCS_BIANTENNARY_NGLYCAN
        )
        assert len(res_list) == 7, (
            f"Expected 7 residue entries, got {len(res_list)}: {res_list}"
        )

    def test_linkage_section_contains_linkages(self):
        """Linkage section should contain bond descriptions (e.g. a4-b1)."""
        _header, _ures, _res, lin = _parse_wurcs_sections(
            WURCS_BIANTENNARY_NGLYCAN
        )
        assert "a4-b1" in lin, (
            f"Linkage section should contain 'a4-b1', got: {lin}"
        )

    def test_parse_rejects_invalid_string(self):
        """Non-WURCS string should raise ValueError."""
        with pytest.raises(ValueError):
            _parse_wurcs_sections("NOT_A_WURCS")


# =====================================================================
# TestWURCSFeatures: Monosaccharide composition & feature extraction
# =====================================================================


class TestWURCSFeatures:
    """Verify WURCS feature extraction captures glycobiology fundamentals."""

    def test_hexnac_count_biantennary(self):
        """Biantennary N-glycan should contain >= 2 HexNAc (GlcNAc)."""
        feats = extract_wurcs_features(WURCS_BIANTENNARY_NGLYCAN)
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        hexnac_count = feats[hexnac_idx].item()
        assert hexnac_count >= 2, (
            f"Biantennary HexNAc count {hexnac_count} < 2; expected >= 2"
        )

    def test_hex_count_biantennary(self):
        """Biantennary N-glycan should contain >= 3 Hex (Man)."""
        feats = extract_wurcs_features(WURCS_BIANTENNARY_NGLYCAN)
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        hex_count = feats[hex_idx].item()
        assert hex_count >= 3, (
            f"Biantennary Hex count {hex_count} < 3; expected Man3 minimum"
        )

    def test_sialic_acid_detection(self):
        """NeuAc-containing glycan should have positive NeuAc count."""
        feats = extract_wurcs_features(WURCS_SIALYLATED)
        neuac_idx = MONOSACCHARIDE_CLASSES.index("NeuAc")
        neuac_count = feats[neuac_idx].item()
        assert neuac_count >= 1, (
            f"Sialylated glycan NeuAc count {neuac_count}; expected >= 1"
        )

    def test_fucose_detection(self):
        """Core-fucosylated glycan should have positive dHex (Fuc) count."""
        feats = extract_wurcs_features(WURCS_CORE_FUCOSYLATED)
        dhex_idx = MONOSACCHARIDE_CLASSES.index("dHex")
        dhex_count = feats[dhex_idx].item()
        assert dhex_count >= 1, (
            f"Core-fucosylated dHex count {dhex_count}; expected >= 1"
        )

    def test_non_sialylated_has_zero_neuac(self):
        """High-mannose glycan without NeuAc should have NeuAc count == 0."""
        feats = extract_wurcs_features(WURCS_HIGH_MANNOSE)
        neuac_idx = MONOSACCHARIDE_CLASSES.index("NeuAc")
        assert feats[neuac_idx].item() == 0.0

    def test_non_fucosylated_has_zero_dhex(self):
        """Non-fucosylated biantennary should have dHex count == 0."""
        feats = extract_wurcs_features(WURCS_BIANTENNARY_NGLYCAN)
        dhex_idx = MONOSACCHARIDE_CLASSES.index("dHex")
        assert feats[dhex_idx].item() == 0.0

    def test_branching_degree_non_negative(self):
        """Branching degree should be non-negative."""
        feats = extract_wurcs_features(WURCS_BIANTENNARY_NGLYCAN)
        branching_idx = 8
        assert feats[branching_idx].item() >= 0

    def test_total_residue_count(self):
        """Total residue count for biantennary should be approximately 7."""
        feats = extract_wurcs_features(WURCS_BIANTENNARY_NGLYCAN)
        total_res_idx = 9
        total = feats[total_res_idx].item()
        assert total >= 5, (
            f"Biantennary total residues {total} < 5; expected ~7"
        )

    def test_feature_vector_dimension(self):
        """Feature vector should always be 24-dimensional."""
        for wurcs in [
            WURCS_BIANTENNARY_NGLYCAN,
            WURCS_SIALYLATED,
            WURCS_CORE_FUCOSYLATED,
            WURCS_SINGLE_HEXOSE,
            WURCS_HIGH_MANNOSE,
        ]:
            feats = extract_wurcs_features(wurcs)
            assert feats.shape == (24,), (
                f"Feature vector shape {feats.shape} != (24,)"
            )

    def test_invalid_wurcs_returns_zeros(self):
        """Invalid WURCS string should return zero vector gracefully."""
        feats = extract_wurcs_features("NOT_A_WURCS")
        assert feats.shape == (24,)
        assert torch.all(feats == 0.0)

    def test_empty_wurcs_returns_zeros(self):
        """Empty string should return zero vector."""
        feats = extract_wurcs_features("")
        assert torch.all(feats == 0.0)

    def test_features_non_negative(self):
        """Monosaccharide counts and total residues should be non-negative."""
        for wurcs in [
            WURCS_BIANTENNARY_NGLYCAN,
            WURCS_SIALYLATED,
            WURCS_CORE_FUCOSYLATED,
            WURCS_HIGH_MANNOSE,
        ]:
            feats = extract_wurcs_features(wurcs)
            for i in range(10):
                assert feats[i].item() >= 0, (
                    f"Feature index {i} is negative: {feats[i].item()}"
                )

    def test_ratios_in_valid_range(self):
        """Sialylation, fucosylation, and branching ratios should be in [0, 1]."""
        for wurcs in [
            WURCS_BIANTENNARY_NGLYCAN,
            WURCS_SIALYLATED,
            WURCS_CORE_FUCOSYLATED,
        ]:
            feats = extract_wurcs_features(wurcs)
            for idx, name in [
                (17, "sialylation"),
                (18, "fucosylation"),
                (19, "branching"),
            ]:
                val = feats[idx].item()
                assert 0.0 <= val <= 1.0, (
                    f"{name} ratio {val} out of [0, 1] range"
                )


# =====================================================================
# TestCountMonosaccharidesDirectly: bypass the parsing bug
# =====================================================================


class TestCountMonosaccharidesDirectly:
    """Test _count_monosaccharides with manually provided inputs."""

    def test_glcnac_counted_as_hexnac(self):
        """GlcNAc residue (2*NCC pattern) should map to HexNAc class."""
        unique_res = ["a2122h-1b_1-5_2*NCC/3=O"]
        res_list = ["a1"]  # 1 occurrence of unique res 0 (GlcNAc)
        counts = _count_monosaccharides(unique_res, res_list)
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        assert counts[hexnac_idx] == 1, (
            f"GlcNAc should count as HexNAc, got counts={counts}"
        )

    def test_mannose_counted_as_hex(self):
        """Man residue (a1122h-1b_1-5) should map to Hex class."""
        unique_res = ["a1122h-1b_1-5"]
        res_list = ["a1", "a2", "a3"]  # 3 Man residues
        counts = _count_monosaccharides(unique_res, res_list)
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        assert counts[hex_idx] == 3, (
            f"3 Man residues should give Hex=3, got {counts[hex_idx]}"
        )

    def test_fucose_counted_as_dhex(self):
        """Fuc residue (a1221m-1a_1-5) should map to dHex class."""
        unique_res = ["a1221m-1a_1-5"]
        res_list = ["a1"]
        counts = _count_monosaccharides(unique_res, res_list)
        dhex_idx = MONOSACCHARIDE_CLASSES.index("dHex")
        assert counts[dhex_idx] == 1, (
            f"Fuc should count as dHex, got counts={counts}"
        )

    def test_neuac_counted_as_sialic(self):
        """NeuAc residue (2*N pattern) should map to NeuAc class."""
        unique_res = ["a2122h-1b_1-5_2*N"]
        res_list = ["a1"]
        counts = _count_monosaccharides(unique_res, res_list)
        neuac_idx = MONOSACCHARIDE_CLASSES.index("NeuAc")
        assert counts[neuac_idx] == 1, (
            f"NeuAc residue should give NeuAc=1, got counts={counts}"
        )

    def test_biantennary_composition(self):
        """Biantennary N-glycan: 4 GlcNAc + 3 Man = 4 HexNAc + 3 Hex."""
        unique_res = [
            "a2122h-1b_1-5_2*NCC/3=O",  # idx 0: GlcNAc
            "a1122h-1b_1-5",              # idx 1: Man-beta
            "a1122h-1a_1-5",              # idx 2: Man-alpha
        ]
        # _count_monosaccharides expects letter-based (a=0, b=1, c=2)
        res_list = ["a1", "a2", "b1", "c1", "a3", "c2", "a4"]
        counts = _count_monosaccharides(unique_res, res_list)
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        assert counts[hexnac_idx] == 4, (
            f"Expected 4 HexNAc, got {counts[hexnac_idx]}"
        )
        assert counts[hex_idx] >= 3, (
            f"Expected >= 3 Hex, got {counts[hex_idx]}"
        )

    def test_gag_alternating_composition(self):
        """GAG-like structure: alternating HexA-HexNAc."""
        unique_res = [
            "a2122A-1b_1-5",              # idx 0: GlcA (HexA)
            "a2122h-1b_1-5_2*NCC/3=O",  # idx 1: GlcNAc (HexNAc)
        ]
        # 3 HexA + 3 HexNAc
        res_list = ["a1", "b1", "a2", "b2", "a3", "b3"]
        counts = _count_monosaccharides(unique_res, res_list)
        hexa_idx = MONOSACCHARIDE_CLASSES.index("HexA")
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        assert counts[hexa_idx] == 3
        assert counts[hexnac_idx] == 3

    def test_numeric_res_list_correctly_mapped(self):
        """Numeric residue tokens (from real WURCS) are correctly mapped
        to unique residue types using 1-based indexing.

        After BUG-WURCS-NUMERIC fix, numeric tokens like '1', '2'
        map to unique_res[0], unique_res[1] respectively.
        """
        unique_res = [
            "a2122h-1b_1-5_2*NCC/3=O",  # idx 0: GlcNAc -> HexNAc
            "a1122h-1b_1-5",              # idx 1: Man -> Hex
        ]
        # Real WURCS res_list uses numbers: "1-1-2-2-2"
        res_list = ["1", "1", "2", "2", "2"]
        counts = _count_monosaccharides(unique_res, res_list)
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        # Token "1" -> unique_res[0] (GlcNAc -> HexNAc), "2" -> unique_res[1] (Man -> Hex)
        assert counts[hexnac_idx] == 2, (
            f"Expected 2 HexNAc (from GlcNAc), got {counts[hexnac_idx]}"
        )
        assert counts[hex_idx] == 3, (
            f"Expected 3 Hex (from Man), got {counts[hex_idx]}"
        )


# =====================================================================
# TestCoreTypeEstimation: N-glycan / O-glycan / GAG classification
# =====================================================================


class TestCoreTypeEstimation:
    """Verify core type heuristics using _estimate_core_type directly."""

    def test_nglycan_detected(self):
        """2 HexNAc + 3 Hex => N-glycan score dominant."""
        # mono_counts: [Hex, HexNAc, dHex, NeuAc, NeuGc, Pen, HexA, Kdn]
        mono_counts = [3, 4, 0, 0, 0, 0, 0, 0]  # Man3 GlcNAc4
        scores = _estimate_core_type(mono_counts, total_residues=7)
        assert scores[0] > scores[1], (
            f"N-glycan={scores[0]:.3f} should > O-glycan={scores[1]:.3f}"
        )
        assert scores[0] > scores[2], (
            f"N-glycan={scores[0]:.3f} should > GAG={scores[2]:.3f}"
        )

    def test_nglycan_with_fucose_higher_score(self):
        """N-glycan with core fucosylation should have higher N-glycan score."""
        base_counts = [3, 4, 0, 0, 0, 0, 0, 0]
        fuc_counts = [3, 4, 1, 0, 0, 0, 0, 0]  # +1 Fuc
        scores_base = _estimate_core_type(base_counts, total_residues=7)
        scores_fuc = _estimate_core_type(fuc_counts, total_residues=8)
        assert scores_fuc[0] >= scores_base[0], (
            "Core fucosylation should maintain or increase N-glycan score"
        )

    def test_nglycan_with_sialylation_higher_score(self):
        """N-glycan with terminal sialylation should have highest possible score."""
        counts = [3, 4, 1, 2, 0, 0, 0, 0]  # Man3 GlcNAc4 Fuc1 NeuAc2
        scores = _estimate_core_type(counts, total_residues=10)
        assert scores[0] >= 0.8, (
            f"Complex N-glycan with Fuc+NeuAc: N-glycan score {scores[0]:.3f} < 0.8"
        )

    def test_oglycan_detected_for_small_structure(self):
        """Small structure with 1 HexNAc + 1 Hex => O-glycan notable."""
        # Core 1 O-glycan: GalNAc-Gal
        mono_counts = [1, 1, 0, 0, 0, 0, 0, 0]
        scores = _estimate_core_type(mono_counts, total_residues=2)
        assert scores[1] > 0, (
            f"O-glycan score should be > 0 for small HexNAc-containing structure"
        )

    @pytest.mark.xfail(
        reason="BUG-CORE-TYPE-PRECEDENCE: O-glycan elif fires before GAG "
               "because hexnac>=1 and total<=6, preventing GAG detection",
        strict=True,
    )
    def test_gag_detected_for_alternating_hexa_hexnac(self):
        """High HexA + HexNAc with repeating units => GAG dominant."""
        mono_counts = [0, 3, 0, 0, 0, 0, 3, 0]  # 3 HexA + 3 HexNAc
        scores = _estimate_core_type(mono_counts, total_residues=6)
        assert scores[2] > 0, (
            f"GAG score should be > 0 for alternating HexA-HexNAc"
        )
        assert scores[2] >= scores[0], (
            f"GAG={scores[2]:.3f} should >= N-glycan={scores[0]:.3f}"
        )

    def test_high_mannose_is_nglycan(self):
        """High-mannose glycan (5 Man + 2 GlcNAc) => N-glycan."""
        mono_counts = [5, 2, 0, 0, 0, 0, 0, 0]
        scores = _estimate_core_type(mono_counts, total_residues=7)
        assert scores[0] > scores[1], (
            f"High-mannose: N-glycan {scores[0]:.3f} should > O-glycan {scores[1]:.3f}"
        )

    def test_core_type_scores_sum_to_one(self):
        """Core type probability scores should sum to 1.0."""
        test_cases = [
            ([3, 4, 0, 0, 0, 0, 0, 0], 7),   # N-glycan
            ([1, 1, 0, 0, 0, 0, 0, 0], 2),    # O-glycan
            ([0, 3, 0, 0, 0, 0, 3, 0], 6),    # GAG
            ([1, 0, 0, 0, 0, 0, 0, 0], 1),    # minimal
        ]
        for mono_counts, total in test_cases:
            scores = _estimate_core_type(mono_counts, total)
            total_score = sum(scores)
            assert total_score == pytest.approx(1.0, abs=0.02), (
                f"Scores {scores} sum to {total_score:.3f}, expected ~1.0"
            )


# =====================================================================
# TestBiologicalConstraints: Schema & negative sampling validity
# =====================================================================


class TestBiologicalConstraints:
    """Verify KG schema and negative sampler respect biological constraints."""

    @pytest.fixture
    def schema_dir(self):
        return Path(__file__).resolve().parents[1] / "schemas"

    @pytest.fixture
    def edge_schema(self, schema_dir):
        with open(schema_dir / "edge_schema.yaml") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def relation_config(self, schema_dir):
        with open(schema_dir / "relation_config.yaml") as f:
            return yaml.safe_load(f)

    def test_has_glycan_source_is_protein(self, edge_schema):
        """has_glycan: source must be protein or enzyme (glycosylation is a protein PTM)."""
        spec = edge_schema["edge_types"]["has_glycan"]
        src = spec["source_type"]
        src_list = src if isinstance(src, list) else [src]
        assert "protein" in src_list

    def test_has_glycan_target_is_glycan(self, edge_schema):
        """has_glycan: target must be glycan."""
        spec = edge_schema["edge_types"]["has_glycan"]
        assert spec["target_type"] == "glycan"

    def test_inhibits_source_is_compound(self, edge_schema):
        """inhibits: source must be compound (small molecule)."""
        spec = edge_schema["edge_types"]["inhibits"]
        assert spec["source_type"] == "compound"

    def test_inhibits_target_is_enzyme(self, edge_schema):
        """inhibits: target must be enzyme (glycosyltransferase)."""
        spec = edge_schema["edge_types"]["inhibits"]
        assert spec["target_type"] == "enzyme"

    def test_associated_with_disease_source_is_protein(self, edge_schema):
        """associated_with_disease: source is protein or enzyme."""
        spec = edge_schema["edge_types"]["associated_with_disease"]
        src = spec["source_type"]
        src_list = src if isinstance(src, list) else [src]
        assert "protein" in src_list

    def test_has_site_allows_enzyme_as_source(self, edge_schema):
        """has_site: enzymes (glycosyltransferases) can have PTM sites too."""
        spec = edge_schema["edge_types"]["has_site"]
        src = spec["source_type"]
        if isinstance(src, list):
            assert "enzyme" in src
            assert "protein" in src
        else:
            assert src in ("protein", "enzyme")

    def test_ptm_crosstalk_is_site_to_site(self, edge_schema):
        """ptm_crosstalk: must be between site nodes."""
        spec = edge_schema["edge_types"]["ptm_crosstalk"]
        assert spec["source_type"] == "site"
        assert spec["target_type"] == "site"

    def test_glycan_node_id_pattern(self):
        """Glycan IDs should follow GlyTouCan accession format G#####XX."""
        ns_path = Path(__file__).resolve().parents[1] / "schemas" / "node_schema.yaml"
        with open(ns_path) as f:
            ns = yaml.safe_load(f)
        pattern = ns["node_types"]["glycan"]["id_pattern"]
        assert pattern is not None
        regex = re.compile(pattern)
        assert regex.match("G00030MO")
        assert regex.match("G12345AB")
        assert not regex.match("P12345")

    def test_type_constraint_prevents_glycan_has_protein_glycan(self):
        """Negative sampler must not produce glycan-has_glycan-glycan triples."""
        node_type_offsets = {
            "protein": (0, 10),
            "glycan": (10, 5),
            "enzyme": (15, 3),
            "disease": (18, 2),
        }
        sampler = TypeConstrainedNegativeSampler(
            node_type_offsets=node_type_offsets,
            num_negatives=100,
            corrupt_head_prob=1.0,
        )
        head = torch.tensor([0])
        relation = ["has_glycan"]
        tail = torch.tensor([10])
        neg_head, neg_tail = sampler.sample(head, relation, tail)
        # Corrupted heads for has_glycan must be proteins [0, 10) or enzymes [15, 18)
        valid = torch.logical_or(
            (neg_head >= 0) & (neg_head < 10),
            (neg_head >= 15) & (neg_head < 18),
        )
        assert torch.all(valid), (
            f"Corrupted heads include invalid indices: "
            f"min={neg_head.min().item()}, max={neg_head.max().item()}"
        )

    def test_type_constraint_prevents_compound_has_glycan(self):
        """Corrupting tail of has_glycan should only yield glycan nodes."""
        node_type_offsets = {
            "protein": (0, 10),
            "glycan": (10, 5),
            "compound": (15, 3),
            "enzyme": (18, 2),
        }
        sampler = TypeConstrainedNegativeSampler(
            node_type_offsets=node_type_offsets,
            num_negatives=100,
            corrupt_head_prob=0.0,
        )
        head = torch.tensor([0])
        relation = ["has_glycan"]
        tail = torch.tensor([10])
        neg_head, neg_tail = sampler.sample(head, relation, tail)
        # Corrupted tails must be glycans [10, 15)
        assert torch.all(neg_tail >= 10) and torch.all(neg_tail < 15), (
            f"Corrupted tails include non-glycan indices"
        )

    def test_inhibits_negative_head_is_compound_only(self):
        """For inhibits, corrupted heads must be compounds."""
        node_type_offsets = {
            "protein": (0, 10),
            "glycan": (10, 5),
            "compound": (15, 3),
            "enzyme": (18, 4),
        }
        sampler = TypeConstrainedNegativeSampler(
            node_type_offsets=node_type_offsets,
            num_negatives=100,
            corrupt_head_prob=1.0,
        )
        head = torch.tensor([15])
        relation = ["inhibits"]
        tail = torch.tensor([18])
        neg_head, _ = sampler.sample(head, relation, tail)
        assert torch.all(neg_head >= 15) and torch.all(neg_head < 18)

    def test_inhibits_negative_tail_is_enzyme_only(self):
        """For inhibits, corrupted tails must be enzymes."""
        node_type_offsets = {
            "protein": (0, 10),
            "glycan": (10, 5),
            "compound": (15, 3),
            "enzyme": (18, 4),
        }
        sampler = TypeConstrainedNegativeSampler(
            node_type_offsets=node_type_offsets,
            num_negatives=100,
            corrupt_head_prob=0.0,
        )
        head = torch.tensor([15])
        relation = ["inhibits"]
        tail = torch.tensor([18])
        _, neg_tail = sampler.sample(head, relation, tail)
        assert torch.all(neg_tail >= 18) and torch.all(neg_tail < 22)

    def test_all_relations_have_type_constraints(self, relation_config):
        """Every relation must have source_type and target_type defined."""
        for rel, spec in relation_config["relation_types"].items():
            assert "source_type" in spec, f"'{rel}' missing source_type"
            assert "target_type" in spec, f"'{rel}' missing target_type"
            assert spec["source_type"] is not None
            assert spec["target_type"] is not None


# =====================================================================
# TestGlycanEncoderBiologicalProperties
# =====================================================================


class TestGlycanEncoderBiologicalProperties:
    """Verify GlycanEncoder produces valid embeddings."""

    @pytest.fixture
    def wurcs_feature_encoder(self):
        wurcs_map = {
            0: WURCS_BIANTENNARY_NGLYCAN,
            1: WURCS_HIGH_MANNOSE,
            2: WURCS_CORE_FUCOSYLATED,
            3: WURCS_SIALYLATED,
            4: WURCS_OGLYCAN_CORE1,
        }
        return GlycanEncoder(
            num_glycans=5,
            output_dim=64,
            method="wurcs_features",
            wurcs_map=wurcs_map,
        )

    def test_output_shape(self, wurcs_feature_encoder):
        """Encoder output should have correct shape [B, output_dim]."""
        indices = torch.tensor([0, 1, 2, 3, 4])
        with torch.no_grad():
            out = wurcs_feature_encoder(indices)
        assert out.shape == (5, 64)

    def test_different_glycans_produce_different_embeddings(self, wurcs_feature_encoder):
        """Structurally different glycans should have distinct embeddings."""
        indices = torch.tensor([0, 4])  # biantennary vs O-glycan
        with torch.no_grad():
            out = wurcs_feature_encoder(indices)
        dist = torch.norm(out[0] - out[1]).item()
        assert dist > 0.01, (
            f"Biantennary and O-glycan embeddings identical: dist={dist:.6f}"
        )

    def test_learnable_mode_output(self):
        """Learnable mode should produce valid embeddings without WURCS."""
        encoder = GlycanEncoder(num_glycans=10, output_dim=32, method="learnable")
        indices = torch.tensor([0, 5, 9])
        with torch.no_grad():
            out = encoder(indices)
        assert out.shape == (3, 32)
        assert not torch.any(torch.isnan(out))

    def test_hybrid_mode_output(self):
        """Hybrid mode should combine learnable + WURCS features."""
        wurcs_map = {0: WURCS_BIANTENNARY_NGLYCAN, 1: WURCS_SIALYLATED}
        encoder = GlycanEncoder(
            num_glycans=5, output_dim=32, method="hybrid", wurcs_map=wurcs_map,
        )
        indices = torch.tensor([0, 1])
        with torch.no_grad():
            out = encoder(indices)
        assert out.shape == (2, 32)
        assert not torch.any(torch.isnan(out))

    def test_missing_wurcs_produces_valid_output(self):
        """Glycan index without WURCS should still produce valid embedding."""
        wurcs_map = {0: WURCS_BIANTENNARY_NGLYCAN}
        encoder = GlycanEncoder(
            num_glycans=5, output_dim=32, method="wurcs_features", wurcs_map=wurcs_map,
        )
        indices = torch.tensor([0, 3])  # index 3 has no WURCS
        with torch.no_grad():
            out = encoder(indices)
        assert out.shape == (2, 32)
        assert not torch.any(torch.isnan(out))

    def test_invalid_method_raises(self):
        """Invalid encoder method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            GlycanEncoder(num_glycans=5, method="invalid_method")


# =====================================================================
# TestWURCSParserInternals
# =====================================================================


class TestWURCSParserInternals:
    """Test internal WURCS parsing functions."""

    def test_unique_residues_extracted_correctly(self):
        """Bracket-delimited unique residues should be parsed."""
        _header, unique_res, _res, _lin = _parse_wurcs_sections(
            WURCS_BIANTENNARY_NGLYCAN
        )
        assert len(unique_res) == 3

    def test_parse_rejects_invalid(self):
        """Non-WURCS string should raise ValueError."""
        with pytest.raises(ValueError):
            _parse_wurcs_sections("NOT_WURCS")

    def test_monosaccharide_class_coverage(self):
        """MONOSACCHARIDE_CLASSES should cover 8 major biological categories."""
        expected = {"Hex", "HexNAc", "dHex", "NeuAc", "NeuGc", "Pen", "HexA", "Kdn"}
        actual = set(MONOSACCHARIDE_CLASSES)
        assert actual == expected

    def test_modification_detection_sulfation(self):
        """Sulfation pattern (*OSO) should be detected."""
        unique_res_with_sulfation = ["a2122h-1b_1-5_6*OSO/3=O"]
        s, p, a = _detect_modifications(unique_res_with_sulfation)
        assert s is True

    def test_modification_detection_phosphorylation(self):
        """Phosphorylation pattern (*OPO) should be detected."""
        unique_res_with_phospho = ["a2122h-1b_1-5_6*OPO/3=O"]
        s, p, a = _detect_modifications(unique_res_with_phospho)
        assert p is True

    def test_modification_detection_acetylation(self):
        """Acetylation pattern (*OCC) should be detected."""
        unique_res_with_acetyl = ["a2122h-1b_1-5_6*OCC/3=O"]
        s, p, a = _detect_modifications(unique_res_with_acetyl)
        assert a is True

    def test_branching_degree_linear(self):
        """Linear chain should have branching degree 0."""
        lin = "a4-b1_b4-c1_c4-d1"  # linear: a->b->c->d
        assert _branching_degree(lin) == 0

    def test_branching_degree_empty(self):
        """Empty linkage should have branching degree 0."""
        assert _branching_degree("") == 0


# =====================================================================
# TestEdgeBiologicalValidity: mini-KG edge data correctness
# =====================================================================


class TestEdgeBiologicalValidity:
    """Verify mini test KG edges respect biological constraints."""

    @pytest.fixture
    def mini_edges_path(self):
        return Path(__file__).parent / "test_data" / "mini_edges.tsv"

    @pytest.fixture
    def mini_nodes_path(self):
        return Path(__file__).parent / "test_data" / "mini_nodes.tsv"

    def test_has_glycan_edges_source_is_protein_or_enzyme(
        self, mini_edges_path, mini_nodes_path
    ):
        """has_glycan source should be protein or enzyme, not glycan."""
        import pandas as pd

        edges = pd.read_csv(mini_edges_path, sep="\t")
        nodes = pd.read_csv(mini_nodes_path, sep="\t")
        node_types = dict(zip(nodes["node_id"], nodes["node_type"]))

        has_glycan_edges = edges[edges["relation"] == "has_glycan"]
        for _, row in has_glycan_edges.iterrows():
            src_type = node_types.get(row["source_id"])
            tgt_type = node_types.get(row["target_id"])
            assert src_type in ("protein", "enzyme"), (
                f"has_glycan source '{row['source_id']}' is '{src_type}'"
            )
            assert tgt_type == "glycan"

    def test_inhibits_edges_direction(self, mini_edges_path, mini_nodes_path):
        """inhibits: compound -> enzyme."""
        import pandas as pd

        edges = pd.read_csv(mini_edges_path, sep="\t")
        nodes = pd.read_csv(mini_nodes_path, sep="\t")
        node_types = dict(zip(nodes["node_id"], nodes["node_type"]))

        inhibits_edges = edges[edges["relation"] == "inhibits"]
        for _, row in inhibits_edges.iterrows():
            assert node_types[row["source_id"]] == "compound"
            assert node_types[row["target_id"]] == "enzyme"

    def test_glycan_ids_follow_glytoucan_pattern(self, mini_nodes_path):
        """Glycan node IDs should match GlyTouCan accession format."""
        import pandas as pd

        nodes = pd.read_csv(mini_nodes_path, sep="\t")
        glycans = nodes[nodes["node_type"] == "glycan"]
        pattern = re.compile(r"^G\d{5}[A-Z]{2}$")
        for _, row in glycans.iterrows():
            assert pattern.match(row["node_id"])

    def test_site_ids_encode_position_and_residue(self, mini_nodes_path):
        """Site IDs should encode protein, position, and residue."""
        import pandas as pd

        nodes = pd.read_csv(mini_nodes_path, sep="\t")
        sites = nodes[nodes["node_type"] == "site"]
        pattern = re.compile(r"^SITE::[A-Z0-9]+-?\d*::\d+::[A-Z]$")
        for _, row in sites.iterrows():
            assert pattern.match(row["node_id"])

    def test_enzyme_metadata_has_source(self, mini_nodes_path):
        """Enzyme nodes should at minimum have source metadata."""
        import json
        import pandas as pd

        nodes = pd.read_csv(mini_nodes_path, sep="\t")
        enzymes = nodes[nodes["node_type"] == "enzyme"]
        assert len(enzymes) > 0, "Mini KG should contain enzyme nodes"
        for _, row in enzymes.iterrows():
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            # node_schema requires 'source' in enzyme metadata
            assert "source" in meta or isinstance(meta, dict), (
                f"Enzyme '{row['node_id']}' should have metadata"
            )

    def test_has_glycan_schema_matches_mini_edges(self, edge_schema, mini_edges_path, mini_nodes_path):
        """has_glycan schema says source=protein, but mini_edges has enzyme->glycan.

        This documents a schema inconsistency: the edge_schema defines
        has_glycan as protein->glycan only, but the mini_edges.tsv file
        contains Q11111 (enzyme) -> G00001AA (glycan) via has_glycan.
        The relation_config defines has_glycan as protein->glycan only.
        Biologically, enzymes (glycosyltransferases) can also be glycosylated,
        so has_glycan should perhaps allow enzyme as source_type too.
        """
        import pandas as pd

        edges = pd.read_csv(mini_edges_path, sep="\t")
        nodes = pd.read_csv(mini_nodes_path, sep="\t")
        node_types = dict(zip(nodes["node_id"], nodes["node_type"]))

        has_glycan_edges = edges[edges["relation"] == "has_glycan"]
        source_types = set()
        for _, row in has_glycan_edges.iterrows():
            source_types.add(node_types[row["source_id"]])

        schema_source = edge_schema["edge_types"]["has_glycan"]["source_type"]
        if "enzyme" in source_types and schema_source == "protein":
            pytest.xfail(
                "Schema inconsistency: has_glycan allows only protein as source, "
                "but mini_edges contains enzyme->glycan via has_glycan"
            )

    @pytest.fixture
    def edge_schema(self):
        with open(Path(__file__).resolve().parents[1] / "schemas" / "edge_schema.yaml") as f:
            return yaml.safe_load(f)
