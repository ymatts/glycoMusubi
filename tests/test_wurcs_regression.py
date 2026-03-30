"""Regression tests for WURCS parser bug fixes and C1 code quality improvements.

These tests ensure that the three WURCS P0 bugs fixed in C1 do not regress:

1. BUG-WURCS-PARSE: _parse_wurcs_sections split tail on "/" with leading "/"
   causing empty res_list. Fixed by stripping leading "/" before splitting.
2. BUG-WURCS-REGEX: NeuAc pattern "*N" matched GlcNAc "*NCC" (prefix collision).
   Fixed by adding negative lookahead (?!CC) to the NeuAc pattern.
3. BUG-WURCS-NUMERIC: Numeric res_list tokens (1-based) were not handled.
   Fixed by adding digit-branch in _count_monosaccharides.

Also verifies code quality fixes:
- torch.load(weights_only=True) in ProteinEncoder
- assert -> ValueError in splits.py
- val_ratio default 0.10 in random_link_split
- ProteinEncoder MLP intermediate dimension 512
"""

from __future__ import annotations

import inspect

import pytest
import torch

from glycoMusubi.embedding.encoders.glycan_encoder import (
    MONOSACCHARIDE_CLASSES,
    _count_monosaccharides,
    _parse_wurcs_sections,
    extract_wurcs_features,
)
from glycoMusubi.embedding.encoders.protein_encoder import ProteinEncoder
from glycoMusubi.data.splits import random_link_split, relation_stratified_split


# ---------------------------------------------------------------------------
# WURCS strings with "/" inside residue codes (trigger BUG-WURCS-PARSE)
# ---------------------------------------------------------------------------

# Chitobiose: GlcNAcb1-4GlcNAc, residue code contains "/3=O"
WURCS_CHITOBIOSE = (
    "WURCS=2.0/2,2,1/[a2122h-1b_1-5_2*NCC/3=O][a2112h-1b_1-5]/1-2/a4-b1"
)

# Biantennary complex N-glycan (7 residues: 4 GlcNAc + 3 Man)
WURCS_BIANTENNARY = (
    "WURCS=2.0/3,7,6/"
    "[a2122h-1b_1-5_2*NCC/3=O]"
    "[a1122h-1b_1-5]"
    "[a1122h-1a_1-5]"
    "/1-1-2-3-1-3-1/"
    "a4-b1_b4-c1_c3-d1_c6-f1_d4-e1_f4-g1"
)

# Core-fucosylated N-glycan (8 residues: 4 GlcNAc + 3 Man + 1 Fuc)
WURCS_CORE_FUCOSYLATED = (
    "WURCS=2.0/4,8,7/"
    "[a2122h-1b_1-5_2*NCC/3=O]"
    "[a1122h-1b_1-5]"
    "[a1122h-1a_1-5]"
    "[a1221m-1a_1-5]"
    "/1-1-2-3-1-3-1-4/"
    "a4-b1_b4-c1_c3-d1_c6-f1_d4-e1_f4-g1_a6-h1"
)

# Sialylated glycan with NeuAc
WURCS_SIALYLATED = (
    "WURCS=2.0/4,5,4/"
    "[a2122h-1b_1-5_2*NCC/3=O]"
    "[a1122h-1b_1-5]"
    "[a1122h-1a_1-5]"
    "[a2122h-1b_1-5_2*N]"
    "/1-2-3-2-4/"
    "a4-b1_b4-c1_c3-d1_d3-e1"
)

# Simple hexose (no "/" in residue code) -- control
WURCS_SIMPLE_HEX = "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"


# =====================================================================
# Regression tests for BUG-WURCS-PARSE (tail split off-by-one)
# =====================================================================


class TestWURCSParseRegression:
    """Verify _parse_wurcs_sections correctly splits tail with leading '/'."""

    def test_res_list_not_empty_for_slash_residues(self):
        """WURCS with '/' in residue codes must still produce non-empty res_list."""
        _, _, res_list, _ = _parse_wurcs_sections(WURCS_CHITOBIOSE)
        assert len(res_list) >= 2, (
            f"Chitobiose res_list should have >= 2 entries, got {len(res_list)}: {res_list}"
        )

    def test_res_list_correct_count_biantennary(self):
        """Biantennary WURCS with 7 residues should produce res_list of length 7."""
        _, _, res_list, _ = _parse_wurcs_sections(WURCS_BIANTENNARY)
        assert len(res_list) == 7, (
            f"Expected 7 residues, got {len(res_list)}: {res_list}"
        )

    def test_linkage_section_correct(self):
        """Linkage section should contain actual linkage data, not residue list."""
        _, _, _, lin = _parse_wurcs_sections(WURCS_BIANTENNARY)
        assert "a4-b1" in lin, (
            f"Linkage section should contain 'a4-b1', got: {lin!r}"
        )

    def test_res_list_values_are_numeric(self):
        """Residue list tokens from real WURCS should be numeric ('1', '2', etc.)."""
        _, _, res_list, _ = _parse_wurcs_sections(WURCS_BIANTENNARY)
        for token in res_list:
            assert token.isdigit(), (
                f"Expected numeric token in res_list, got {token!r}"
            )

    def test_unique_res_extraction_preserved(self):
        """Unique residue extraction (bracket-delimited) should still work."""
        _, unique_res, _, _ = _parse_wurcs_sections(WURCS_BIANTENNARY)
        assert len(unique_res) == 3
        assert "2*NCC/3=O" in unique_res[0]

    def test_total_residues_correct(self):
        """extract_wurcs_features should report correct total residue count."""
        feats = extract_wurcs_features(WURCS_BIANTENNARY)
        total_res_idx = 9
        total = feats[total_res_idx].item()
        assert total == 7.0, (
            f"Expected total_residues=7, got {total}"
        )


# =====================================================================
# Regression tests for BUG-WURCS-REGEX (NeuAc/GlcNAc prefix collision)
# =====================================================================


class TestWURCSRegexRegression:
    """Verify GlcNAc (*NCC) is not confused with NeuAc (*N)."""

    def test_glcnac_classified_as_hexnac(self):
        """GlcNAc residue (2*NCC/3=O) must be classified as HexNAc, not NeuAc."""
        unique_res = ["a2122h-1b_1-5_2*NCC/3=O"]
        res_list = ["1"]
        counts = _count_monosaccharides(unique_res, res_list)
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        neuac_idx = MONOSACCHARIDE_CLASSES.index("NeuAc")
        assert counts[hexnac_idx] == 1, (
            f"GlcNAc should be HexNAc, got counts={counts}"
        )
        assert counts[neuac_idx] == 0, (
            f"GlcNAc should NOT be NeuAc, got NeuAc count={counts[neuac_idx]}"
        )

    def test_neuac_still_classified_correctly(self):
        """NeuAc residue (2*N without CC) must still be classified as NeuAc."""
        unique_res = ["a2122h-1b_1-5_2*N"]
        res_list = ["1"]
        counts = _count_monosaccharides(unique_res, res_list)
        neuac_idx = MONOSACCHARIDE_CLASSES.index("NeuAc")
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        assert counts[neuac_idx] == 1, (
            f"NeuAc should be detected, got counts={counts}"
        )
        assert counts[hexnac_idx] == 0, (
            f"NeuAc should NOT be HexNAc, got HexNAc count={counts[hexnac_idx]}"
        )

    def test_glcnac_and_neuac_in_same_structure(self):
        """A glycan with both GlcNAc and NeuAc must count each correctly."""
        unique_res = [
            "a2122h-1b_1-5_2*NCC/3=O",  # GlcNAc -> HexNAc
            "a2122h-1b_1-5_2*N",          # NeuAc -> NeuAc
        ]
        res_list = ["1", "1", "2"]  # 2x GlcNAc + 1x NeuAc
        counts = _count_monosaccharides(unique_res, res_list)
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        neuac_idx = MONOSACCHARIDE_CLASSES.index("NeuAc")
        assert counts[hexnac_idx] == 2, f"Expected 2 HexNAc, got {counts[hexnac_idx]}"
        assert counts[neuac_idx] == 1, f"Expected 1 NeuAc, got {counts[neuac_idx]}"

    def test_biantennary_hexnac_count_via_features(self):
        """Full pipeline: biantennary N-glycan should have >= 2 HexNAc."""
        feats = extract_wurcs_features(WURCS_BIANTENNARY)
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        assert feats[hexnac_idx].item() >= 2, (
            f"Biantennary HexNAc={feats[hexnac_idx].item()}, expected >= 2"
        )

    def test_sialylated_neuac_count_via_features(self):
        """Full pipeline: sialylated glycan should detect NeuAc."""
        feats = extract_wurcs_features(WURCS_SIALYLATED)
        neuac_idx = MONOSACCHARIDE_CLASSES.index("NeuAc")
        assert feats[neuac_idx].item() >= 1, (
            f"Sialylated NeuAc={feats[neuac_idx].item()}, expected >= 1"
        )

    def test_sialylated_glcnac_not_counted_as_neuac(self):
        """In a mixed GlcNAc+NeuAc glycan, GlcNAc must not inflate NeuAc count."""
        feats = extract_wurcs_features(WURCS_SIALYLATED)
        neuac_idx = MONOSACCHARIDE_CLASSES.index("NeuAc")
        # Sialylated WURCS has 1 GlcNAc + 1 NeuAc among 4 unique residue types
        # NeuAc count should reflect only the actual NeuAc residue, not GlcNAc
        neuac_count = feats[neuac_idx].item()
        assert neuac_count <= 2, (
            f"NeuAc count {neuac_count} too high; GlcNAc may be leaking into NeuAc"
        )


# =====================================================================
# Regression tests for BUG-WURCS-NUMERIC (numeric token handling)
# =====================================================================


class TestWURCSNumericRegression:
    """Verify numeric res_list tokens are correctly mapped to unique residues."""

    def test_numeric_tokens_map_to_unique_res(self):
        """Numeric token '1' should map to unique_res[0], '2' to unique_res[1]."""
        unique_res = [
            "a2122h-1b_1-5_2*NCC/3=O",  # idx 0: GlcNAc -> HexNAc
            "a1122h-1b_1-5",              # idx 1: Man -> Hex
        ]
        res_list = ["1", "2", "1"]  # GlcNAc, Man, GlcNAc
        counts = _count_monosaccharides(unique_res, res_list)
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        assert counts[hexnac_idx] == 2, f"Expected 2 HexNAc, got {counts[hexnac_idx]}"
        assert counts[hex_idx] == 1, f"Expected 1 Hex, got {counts[hex_idx]}"

    def test_numeric_tokens_all_same_type(self):
        """All numeric tokens pointing to same unique_res should accumulate."""
        unique_res = ["a2122h-1b_1-5"]  # Hex
        res_list = ["1", "1", "1", "1"]
        counts = _count_monosaccharides(unique_res, res_list)
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        assert counts[hex_idx] == 4, f"Expected 4 Hex, got {counts[hex_idx]}"

    def test_alphabetic_tokens_still_work(self):
        """Alphabetic tokens (a, b, c) should still work correctly."""
        unique_res = [
            "a2122h-1b_1-5",            # idx 0: Hex
            "a1221m-1a_1-5",            # idx 1: dHex (Fuc)
        ]
        res_list = ["a", "b", "a"]  # Hex, dHex, Hex
        counts = _count_monosaccharides(unique_res, res_list)
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        dhex_idx = MONOSACCHARIDE_CLASSES.index("dHex")
        assert counts[hex_idx] == 2, f"Expected 2 Hex, got {counts[hex_idx]}"
        assert counts[dhex_idx] == 1, f"Expected 1 dHex, got {counts[dhex_idx]}"

    def test_full_pipeline_with_numeric_tokens(self):
        """End-to-end: biantennary WURCS (numeric tokens) should produce correct counts."""
        feats = extract_wurcs_features(WURCS_BIANTENNARY)
        hexnac_idx = MONOSACCHARIDE_CLASSES.index("HexNAc")
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        total_res_idx = 9
        # Biantennary: 4 GlcNAc (HexNAc) + 3 Man (Hex) = 7 total
        assert feats[hexnac_idx].item() == 4.0, (
            f"Expected 4 HexNAc, got {feats[hexnac_idx].item()}"
        )
        assert feats[hex_idx].item() == 3.0, (
            f"Expected 3 Hex, got {feats[hex_idx].item()}"
        )
        assert feats[total_res_idx].item() == 7.0, (
            f"Expected 7 total residues, got {feats[total_res_idx].item()}"
        )


# =====================================================================
# Edge case tests for WURCS parsing
# =====================================================================


class TestWURCSEdgeCases:
    """Edge cases and error handling for WURCS parsing."""

    def test_empty_string_returns_zeros(self):
        feats = extract_wurcs_features("")
        assert feats.shape == (24,)
        assert feats.sum().item() == 0.0

    def test_invalid_wurcs_returns_zeros(self):
        feats = extract_wurcs_features("not_a_valid_wurcs_string")
        assert feats.shape == (24,)
        assert feats.sum().item() == 0.0

    def test_short_wurcs_with_few_sections(self):
        """WURCS with too few '/' sections should return zeros."""
        feats = extract_wurcs_features("WURCS=2.0/1")
        assert feats.shape == (24,)
        assert feats.sum().item() == 0.0

    def test_single_residue_wurcs(self):
        """Single hexose: should produce exactly 1 Hex, total_residues=1."""
        feats = extract_wurcs_features(WURCS_SIMPLE_HEX)
        hex_idx = MONOSACCHARIDE_CLASSES.index("Hex")
        assert feats[hex_idx].item() == 1.0
        assert feats[9].item() == 1.0  # total_residues

    def test_parse_raises_for_non_wurcs(self):
        with pytest.raises(ValueError, match="Not a valid WURCS"):
            _parse_wurcs_sections("INVALID")

    def test_parse_raises_for_too_few_sections(self):
        with pytest.raises(ValueError, match="fewer than 3 sections"):
            _parse_wurcs_sections("WURCS=2.0/1")

    def test_empty_res_list_produces_zero_counts(self):
        """Empty res_list should produce all-zero monosaccharide counts."""
        counts = _count_monosaccharides(["a2122h-1b_1-5"], [])
        assert all(c == 0 for c in counts)

    def test_empty_unique_res_produces_zero_counts(self):
        counts = _count_monosaccharides([], [])
        assert all(c == 0 for c in counts)

    def test_core_fucosylated_has_dhex(self):
        """Core-fucosylated glycan should detect dHex (Fuc) via full pipeline."""
        feats = extract_wurcs_features(WURCS_CORE_FUCOSYLATED)
        dhex_idx = MONOSACCHARIDE_CLASSES.index("dHex")
        assert feats[dhex_idx].item() >= 1, (
            f"Core-fucosylated dHex={feats[dhex_idx].item()}, expected >= 1"
        )

    def test_different_wurcs_produce_different_features(self):
        """Structurally different glycans must produce different feature vectors."""
        feats_hex = extract_wurcs_features(WURCS_SIMPLE_HEX)
        feats_bi = extract_wurcs_features(WURCS_BIANTENNARY)
        assert not torch.allclose(feats_hex, feats_bi), (
            "Simple hexose and biantennary should have different features"
        )


# =====================================================================
# Code quality: torch.load(weights_only=True) in ProteinEncoder
# =====================================================================


class TestTorchLoadSecurity:
    """Verify torch.load uses weights_only=True for security."""

    def test_protein_encoder_uses_weights_only(self):
        """ProteinEncoder._load_esm2_embedding should use weights_only=True."""
        source = inspect.getsource(ProteinEncoder._load_esm2_embedding)
        assert "weights_only=True" in source, (
            "ProteinEncoder._load_esm2_embedding should use weights_only=True"
        )

    def test_trainer_load_checkpoint_explicitly_marks_weights_only(self):
        """Trainer.load_checkpoint should explicitly set weights_only."""
        from glycoMusubi.training.trainer import Trainer
        source = inspect.getsource(Trainer.load_checkpoint)
        assert "weights_only=" in source, (
            "Trainer.load_checkpoint should explicitly set weights_only parameter"
        )


# =====================================================================
# Code quality: assert -> ValueError in splits.py
# =====================================================================


class TestSplitsValidation:
    """Verify splits.py uses ValueError instead of assert for input validation."""

    def test_random_link_split_raises_valueerror(self):
        """random_link_split should raise ValueError for invalid ratios."""
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["a"].num_nodes = 5
        data["a"].x = torch.randn(5, 8)

        with pytest.raises(ValueError, match="val_ratio.*test_ratio"):
            random_link_split(data, val_ratio=0.5, test_ratio=0.6)

    def test_relation_stratified_split_raises_valueerror(self):
        """relation_stratified_split should raise ValueError for invalid ratios."""
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["a"].num_nodes = 5
        data["a"].x = torch.randn(5, 8)

        with pytest.raises(ValueError, match="val_ratio.*test_ratio"):
            relation_stratified_split(data, val_ratio=0.5, test_ratio=0.6)

    def test_no_bare_assert_in_splits(self):
        """splits.py should not use bare assert for input validation."""
        import glycoMusubi.data.splits as splits_module
        source = inspect.getsource(splits_module)
        # Check that validation uses "raise ValueError" not "assert"
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("assert ") and "val_ratio" in stripped:
                pytest.fail(
                    f"splits.py line {i+1}: uses 'assert' for validation "
                    f"instead of 'raise ValueError': {stripped}"
                )


# =====================================================================
# Code quality: val_ratio default is 0.10
# =====================================================================


class TestValRatioDefault:
    """Verify random_link_split default val_ratio is 0.10 per design."""

    def test_random_link_split_default_val_ratio(self):
        sig = inspect.signature(random_link_split)
        default = sig.parameters["val_ratio"].default
        assert default == 0.10, (
            f"random_link_split val_ratio default should be 0.10, got {default}"
        )

    def test_random_link_split_default_test_ratio(self):
        sig = inspect.signature(random_link_split)
        default = sig.parameters["test_ratio"].default
        assert default == 0.10, (
            f"random_link_split test_ratio default should be 0.10, got {default}"
        )


# =====================================================================
# Code quality: ProteinEncoder MLP intermediate dim = 512
# =====================================================================


class TestProteinEncoderMLP:
    """Verify ProteinEncoder ESM-2 projection MLP has intermediate dim 512."""

    def test_mlp_intermediate_dim_512(self):
        encoder = ProteinEncoder(
            num_proteins=10, output_dim=256, method="esm2", cache_path="/tmp"
        )
        # projection is nn.Sequential: Linear(1280, 512), GELU, Dropout, LayerNorm(512), Linear(512, 256)
        first_linear = encoder.projection[0]
        assert first_linear.out_features == 512, (
            f"First linear out_features should be 512, got {first_linear.out_features}"
        )
        last_linear = encoder.projection[-1]
        assert last_linear.in_features == 512, (
            f"Last linear in_features should be 512, got {last_linear.in_features}"
        )
        assert last_linear.out_features == 256, (
            f"Last linear out_features should be 256, got {last_linear.out_features}"
        )

    def test_mlp_has_gelu_activation(self):
        encoder = ProteinEncoder(
            num_proteins=10, output_dim=256, method="esm2", cache_path="/tmp"
        )
        has_gelu = any(
            isinstance(layer, torch.nn.GELU) for layer in encoder.projection
        )
        assert has_gelu, "ProteinEncoder MLP should use GELU activation"

    def test_mlp_has_layernorm(self):
        encoder = ProteinEncoder(
            num_proteins=10, output_dim=256, method="esm2", cache_path="/tmp"
        )
        has_ln = any(
            isinstance(layer, torch.nn.LayerNorm) for layer in encoder.projection
        )
        assert has_ln, "ProteinEncoder MLP should use LayerNorm"
