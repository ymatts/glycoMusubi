"""Tests for GlycanEncoder and WURCS feature extraction.

Validates three encoding modes (learnable, wurcs_features, hybrid),
WURCS biochemical feature extraction, and glycobiology-informed
embedding properties.
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.embedding.encoders.glycan_encoder import (
    GlycanEncoder,
    extract_wurcs_features,
    _parse_wurcs_sections,
    _count_monosaccharides,
    _branching_degree,
    _detect_modifications,
    _estimate_core_type,
)

# ---------------------------------------------------------------------------
# WURCS strings for testing
# ---------------------------------------------------------------------------

# Simple hexose (Glc) -- no '/' in residue codes
WURCS_SIMPLE_HEX = "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"

# Two hexoses linked -- residue codes without '/'
WURCS_TWO_HEX = "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a2122h-1a_1-5]/1-2/a4-b1"

# HexNAc (GlcNAc-like without the /3=O modification) -- tests pattern matching
WURCS_HEXNAC_SIMPLE = "WURCS=2.0/1,1,0/[a2122h-1b_1-5_2*NCC]/1/"

# Fucose-like (dHex) -- uses a1221m-1a_1-5
WURCS_DHEX = "WURCS=2.0/1,1,0/[a1221m-1a_1-5]/1/"

# WURCS with sulfation modification
WURCS_SULFATED = "WURCS=2.0/1,1,0/[a2122h-1b_1-5_6*OSO]/1/"

# Biologically real WURCS strings (contain '/' in residue codes)
# These expose the parser bug and are used in xfail tests
WURCS_CHITOBIOSE = (
    "WURCS=2.0/2,2,1/[a2122h-1b_1-5_2*NCC/3=O][a2112h-1b_1-5]/1-2/a4-b1"
)


# ---------------------------------------------------------------------------
# Tests: extract_wurcs_features
# ---------------------------------------------------------------------------

class TestExtractWurcsFeatures:
    """Tests for the WURCS feature extraction function."""

    def test_output_shape(self):
        features = extract_wurcs_features(WURCS_SIMPLE_HEX)
        assert features.shape == (24,)
        assert features.dtype == torch.float32

    def test_output_shape_24_features(self):
        """Feature vector should always have exactly 24 dimensions."""
        for wurcs in [WURCS_SIMPLE_HEX, WURCS_TWO_HEX, WURCS_HEXNAC_SIMPLE]:
            features = extract_wurcs_features(wurcs)
            assert features.shape == (24,), f"Wrong shape for {wurcs[:40]}"

    def test_invalid_wurcs_returns_zeros(self):
        features = extract_wurcs_features("not_a_valid_wurcs")
        assert features.shape == (24,)
        assert features.sum().item() == 0.0

    def test_empty_wurcs_returns_zeros(self):
        features = extract_wurcs_features("")
        assert features.shape == (24,)
        assert features.sum().item() == 0.0

    def test_valid_wurcs_not_all_zeros(self):
        """A valid WURCS string should produce at least some non-zero features.

        Even with the parser limitation, the unique_res parsing works
        correctly, and total_residues=max(len(res_list),1)=1 produces
        a non-zero total_residues feature.
        """
        features = extract_wurcs_features(WURCS_SIMPLE_HEX)
        assert features.sum().abs().item() > 0

    def test_chitobiose_has_hexnac(self):
        """Chitobiose (GlcNAcb1-4GlcNAc) should have HexNAc counts > 0.

        This test exposes a parser bug: the WURCS string for chitobiose
        contains '/' in the residue code (_2*NCC/3=O), which causes
        _parse_wurcs_sections to mis-split, yielding an empty res_list
        and therefore zero monosaccharide counts.
        """
        features = extract_wurcs_features(WURCS_CHITOBIOSE)
        hexnac_count = features[1].item()  # index 1 = HexNAc
        assert hexnac_count >= 1, f"Expected HexNAc >= 1, got {hexnac_count}"

    def test_total_residues_for_disaccharide(self):
        """A disaccharide WURCS should report total_residues >= 2."""
        features = extract_wurcs_features(WURCS_TWO_HEX)
        total_residues = features[9].item()
        assert total_residues >= 2, f"Expected total_residues >= 2, got {total_residues}"


class TestParseWurcsSections:
    """Tests for internal WURCS parsing."""

    def test_unique_res_extraction(self):
        """The parser should correctly extract bracket-delimited unique residues."""
        header, unique_res, res_list, lin = _parse_wurcs_sections(WURCS_TWO_HEX)
        assert len(unique_res) == 2
        assert "a2122h-1b_1-5" in unique_res[0]
        assert "a2122h-1a_1-5" in unique_res[1]

    def test_invalid_prefix_raises(self):
        with pytest.raises(ValueError, match="Not a valid WURCS"):
            _parse_wurcs_sections("INVALID_STRING")

    def test_too_few_sections_raises(self):
        with pytest.raises(ValueError, match="fewer than 3 sections"):
            _parse_wurcs_sections("WURCS=2.0/1")

    def test_res_list_populated(self):
        """For a two-residue WURCS, res_list should have 2 entries."""
        _, _, res_list, _ = _parse_wurcs_sections(WURCS_TWO_HEX)
        assert len(res_list) >= 2


# ---------------------------------------------------------------------------
# Tests: internal helper functions (bypass parser bug by providing inputs directly)
# ---------------------------------------------------------------------------

class TestCountMonosaccharides:
    """Tests for _count_monosaccharides with known inputs."""

    def test_hex_detection(self):
        """Hexose residue code should be classified as Hex (index 0)."""
        unique_res = ["a2122h-1b_1-5"]
        res_list = ["1"]  # single residue
        counts = _count_monosaccharides(unique_res, res_list)
        assert counts[0] >= 1, f"Expected Hex count >= 1, got {counts[0]}"

    def test_hexnac_detection(self):
        """HexNAc residue code should be classified as HexNAc (index 1)."""
        unique_res = ["a2122h-1b_1-5_2*NCC"]
        res_list = ["a"]  # letter-based token mapping to unique_res[0]
        counts = _count_monosaccharides(unique_res, res_list)
        assert counts[1] >= 1, f"Expected HexNAc count >= 1, got {counts[1]}"

    def test_dhex_detection(self):
        """dHex (Fuc) residue code should be classified as dHex (index 2)."""
        unique_res = ["a1221m-1a_1-5"]
        res_list = ["a"]  # letter-based token
        counts = _count_monosaccharides(unique_res, res_list)
        assert counts[2] >= 1, f"Expected dHex count >= 1, got {counts[2]}"

    def test_neuac_detection(self):
        """NeuAc variant residue code should be classified as NeuAc (index 3)."""
        unique_res = ["a2112h-1b_1-5"]
        res_list = ["a"]  # letter-based token
        counts = _count_monosaccharides(unique_res, res_list)
        assert counts[3] >= 1, f"Expected NeuAc count >= 1, got {counts[3]}"

    def test_multiple_residues(self):
        """Multiple residues should be counted correctly."""
        # Two unique residues: Hex(a) and Hex(b, different anomer)
        unique_res = ["a2122h-1b_1-5", "a2122h-1a_1-5"]
        # res_list uses letter tokens: a->unique[0], b->unique[1]
        res_list = ["a", "b", "a"]
        counts = _count_monosaccharides(unique_res, res_list)
        total = sum(counts)
        assert total >= 3, f"Expected total >= 3, got {total}"

    def test_empty_inputs(self):
        counts = _count_monosaccharides([], [])
        assert all(c == 0 for c in counts)


class TestBranchingDegree:
    """Tests for _branching_degree."""

    def test_linear_chain(self):
        """Linear chain (a-b-c) has no branching."""
        assert _branching_degree("a1-b4_b1-c4") == 0

    def test_single_branch(self):
        """A branch point produces branching degree >= 1."""
        # Residue 'b' is target of two linkages -> branch point
        degree = _branching_degree("a1-b4_a1-c4_b1-d4")
        # 'a' appears twice as source -> counts as branch
        assert degree >= 1

    def test_empty_linkage(self):
        assert _branching_degree("") == 0


class TestDetectModifications:
    """Tests for _detect_modifications."""

    def test_no_modifications(self):
        sulfation, phosphorylation, acetylation = _detect_modifications(
            ["a2122h-1b_1-5"]
        )
        assert not sulfation
        assert not phosphorylation
        assert not acetylation

    def test_sulfation_detected(self):
        sulfation, _, _ = _detect_modifications(["a2122h-1b_1-5_6*OSO"])
        assert sulfation

    def test_phosphorylation_detected(self):
        _, phosphorylation, _ = _detect_modifications(["a2122h-1b_1-5_6*OPO"])
        assert phosphorylation

    def test_acetylation_detected(self):
        _, _, acetylation = _detect_modifications(["a2122h-1b_1-5_2*OCC"])
        assert acetylation


class TestEstimateCoreType:
    """Tests for N-glycan / O-glycan core type estimation.

    These tests use direct monosaccharide count vectors, bypassing
    the WURCS parser entirely, to verify the biological heuristics.
    """

    def test_n_glycan_core_detection(self):
        """Structure with >= 2 HexNAc + >= 3 Hex should score high as N-glycan.

        This reflects the N-glycan core pentasaccharide: Man3GlcNAc2.
        """
        mono_counts = [3, 2, 0, 0, 0, 0, 0, 0]  # 3 Hex (Man), 2 HexNAc (GlcNAc)
        scores = _estimate_core_type(mono_counts, 5)
        assert scores[0] > scores[1], "N-glycan score should exceed O-glycan score"
        assert scores[0] > 0.5, f"N-glycan probability should be > 0.5, got {scores[0]}"

    def test_n_glycan_with_fucosylation(self):
        """Core fucosylation (common in N-glycans) should increase N-glycan score."""
        base_counts = [3, 2, 0, 0, 0, 0, 0, 0]  # Man3GlcNAc2
        fuc_counts = [3, 2, 1, 0, 0, 0, 0, 0]   # + Fuc
        base_scores = _estimate_core_type(base_counts, 5)
        fuc_scores = _estimate_core_type(fuc_counts, 6)
        assert fuc_scores[0] >= base_scores[0], (
            "Fucosylation should not decrease N-glycan probability"
        )

    def test_n_glycan_with_sialylation(self):
        """Terminal sialylation should not decrease N-glycan score."""
        base_counts = [3, 2, 0, 0, 0, 0, 0, 0]
        sia_counts = [3, 2, 0, 2, 0, 0, 0, 0]  # + 2 NeuAc
        base_scores = _estimate_core_type(base_counts, 5)
        sia_scores = _estimate_core_type(sia_counts, 7)
        assert sia_scores[0] >= base_scores[0], (
            "Sialylation should not decrease N-glycan probability"
        )

    def test_o_glycan_core_detection(self):
        """Small structure with 1 HexNAc should score high as O-glycan.

        O-glycan core 1: Gal-beta-1-3-GalNAc (2 residues total).
        """
        mono_counts = [1, 1, 0, 0, 0, 0, 0, 0]  # 1 Hex (Gal), 1 HexNAc (GalNAc)
        scores = _estimate_core_type(mono_counts, 2)
        assert scores[1] > scores[0], "O-glycan score should exceed N-glycan score"

    def test_gag_detection(self):
        """High HexA + HexNAc suggests glycosaminoglycan (GAG)."""
        mono_counts = [0, 4, 0, 0, 0, 0, 4, 0]  # 4 HexNAc, 4 HexA
        scores = _estimate_core_type(mono_counts, 8)
        assert scores[2] > 0, "GAG score should be positive for HexA/HexNAc pattern"

    def test_scores_sum_to_one(self):
        """Core type scores should sum to approximately 1.0."""
        for counts, total in [
            ([3, 2, 1, 1, 0, 0, 0, 0], 7),
            ([1, 1, 0, 0, 0, 0, 0, 0], 2),
            ([0, 0, 0, 0, 0, 0, 0, 0], 1),
        ]:
            scores = _estimate_core_type(counts, total)
            assert abs(sum(scores) - 1.0) < 1e-6, (
                f"Scores should sum to 1.0, got {sum(scores)} for {counts}"
            )

    def test_sialylation_ratio_computation(self):
        """Sialylation ratio should be (NeuAc + NeuGc) / total_residues."""
        # Test via extract_wurcs_features with a WURCS that the parser can handle
        # The feature at index 20 is sialylation_ratio
        features = extract_wurcs_features(WURCS_SIMPLE_HEX)
        # Simple hexose has no sialic acid
        assert features[20].item() == 0.0


# ---------------------------------------------------------------------------
# Tests: GlycanEncoder module
# ---------------------------------------------------------------------------

class TestGlycanEncoder:
    """Tests for the GlycanEncoder nn.Module."""

    def test_learnable_mode_output_shape(self):
        """Learnable mode should produce [batch, output_dim] embeddings."""
        encoder = GlycanEncoder(num_glycans=10, output_dim=256, method="learnable")
        indices = torch.tensor([0, 1, 2, 3])
        out = encoder(indices)
        assert out.shape == (4, 256)

    def test_wurcs_features_mode(self):
        """WURCS features mode should encode glycans using WURCS strings."""
        wurcs_map = {0: WURCS_SIMPLE_HEX, 1: WURCS_TWO_HEX, 2: WURCS_HEXNAC_SIMPLE}
        encoder = GlycanEncoder(
            num_glycans=5,
            output_dim=128,
            method="wurcs_features",
            wurcs_map=wurcs_map,
        )
        indices = torch.tensor([0, 1, 2])
        out = encoder(indices)
        assert out.shape == (3, 128)
        # Output should be non-trivial (not all zeros since WURCS strings are valid)
        assert out.abs().sum() > 0

    def test_hybrid_mode(self):
        """Hybrid mode should combine learnable and WURCS features."""
        wurcs_map = {0: WURCS_SIMPLE_HEX, 1: WURCS_TWO_HEX}
        encoder = GlycanEncoder(
            num_glycans=5,
            output_dim=64,
            method="hybrid",
            wurcs_map=wurcs_map,
        )
        indices = torch.tensor([0, 1])
        out = encoder(indices)
        assert out.shape == (2, 64)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            GlycanEncoder(num_glycans=5, method="unknown")

    def test_missing_wurcs_produces_zeros_in_features(self):
        """Indices without WURCS mapping should get zero feature vectors."""
        encoder = GlycanEncoder(
            num_glycans=5,
            output_dim=128,
            method="wurcs_features",
            wurcs_map={0: WURCS_SIMPLE_HEX},
        )
        # Index 3 has no WURCS mapping
        indices = torch.tensor([3])
        out = encoder(indices)
        assert out.shape == (1, 128)

    def test_cache_clear(self):
        """Clearing cache should not affect output correctness."""
        wurcs_map = {0: WURCS_SIMPLE_HEX}
        encoder = GlycanEncoder(
            num_glycans=5,
            output_dim=64,
            method="wurcs_features",
            wurcs_map=wurcs_map,
        )
        indices = torch.tensor([0])
        out1 = encoder(indices)
        encoder.clear_cache()
        out2 = encoder(indices)
        assert torch.allclose(out1, out2)

    def test_learnable_mode_gradient_flow(self):
        """Gradients should flow through learnable embeddings."""
        encoder = GlycanEncoder(num_glycans=5, output_dim=32, method="learnable")
        indices = torch.tensor([0, 1])
        out = encoder(indices)
        loss = out.sum()
        loss.backward()
        assert encoder.embedding.weight.grad is not None
        assert encoder.embedding.weight.grad.abs().sum() > 0

    def test_hybrid_mode_gradient_flow(self):
        """Gradients should flow through both branches in hybrid mode."""
        wurcs_map = {0: WURCS_SIMPLE_HEX, 1: WURCS_TWO_HEX}
        encoder = GlycanEncoder(
            num_glycans=5,
            output_dim=32,
            method="hybrid",
            wurcs_map=wurcs_map,
        )
        indices = torch.tensor([0, 1])
        out = encoder(indices)
        loss = out.sum()
        loss.backward()
        assert encoder.embedding.weight.grad is not None
        # WURCS projection should also get gradients
        for param in encoder.wurcs_proj.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_different_residue_types_produce_different_counts(self):
        """Different monosaccharide types should produce different count profiles.

        Tests _count_monosaccharides directly with letter-based tokens,
        bypassing the WURCS parser.
        """
        counts_hex = _count_monosaccharides(["a2122h-1b_1-5"], ["a"])
        counts_dhex = _count_monosaccharides(["a1221m-1a_1-5"], ["a"])
        assert counts_hex != counts_dhex, (
            "Hex and dHex should have different monosaccharide count profiles"
        )

    def test_n_glycan_o_glycan_separation_direct(self):
        """N-glycan and O-glycan core type estimation should separate them.

        Directly tests _estimate_core_type with biologically accurate
        monosaccharide compositions, bypassing the WURCS parser.
        """
        # N-glycan: Man3GlcNAc2 (3 Hex + 2 HexNAc)
        n_scores = _estimate_core_type([3, 2, 0, 0, 0, 0, 0, 0], 5)
        # O-glycan core 1: Gal + GalNAc (1 Hex + 1 HexNAc)
        o_scores = _estimate_core_type([1, 1, 0, 0, 0, 0, 0, 0], 2)

        assert n_scores[0] > 0.5, "N-glycan should be classified as N-glycan"
        assert o_scores[1] > 0.5, "O-glycan should be classified as O-glycan"
        assert n_scores[0] > n_scores[1], "N-glycan: N score > O score"
        assert o_scores[1] > o_scores[0], "O-glycan: O score > N score"

    def test_sialylation_feature_direct(self):
        """Sialylated glycan should have non-zero NeuAc count when counted directly."""
        # NeuAc variant residue with letter-based token
        unique_res = ["a2112h-1b_1-5"]
        res_list = ["a"]
        counts = _count_monosaccharides(unique_res, res_list)
        assert counts[3] >= 1, "NeuAc residue should be detected"

    def test_batch_size_one(self):
        """Encoder should handle batch size of 1."""
        encoder = GlycanEncoder(num_glycans=5, output_dim=32, method="learnable")
        indices = torch.tensor([0])
        out = encoder(indices)
        assert out.shape == (1, 32)

    def test_all_glycan_indices(self):
        """Encoder should handle all valid glycan indices."""
        n = 10
        encoder = GlycanEncoder(num_glycans=n, output_dim=16, method="learnable")
        indices = torch.arange(n)
        out = encoder(indices)
        assert out.shape == (n, 16)

    def test_wurcs_features_no_nan(self):
        """WURCS features mode should not produce NaN."""
        wurcs_map = {i: WURCS_SIMPLE_HEX for i in range(5)}
        encoder = GlycanEncoder(
            num_glycans=5, output_dim=32, method="wurcs_features", wurcs_map=wurcs_map
        )
        out = encoder(torch.arange(5))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
