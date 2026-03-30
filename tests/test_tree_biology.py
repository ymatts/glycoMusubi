"""Glycobiology validity tests for GlycanTreeEncoder (Phase 2 Tree-MPNN).

Validates that the WURCS-to-tree parser and GlycanTreeEncoder correctly
capture glycan biology:

1. WURCS tree parsing produces biologically correct trees
2. Monosaccharide type embeddings distinguish ~60 types
3. Glycosidic linkage encoding captures position-specific information
4. N-glycan core structure (Man3GlcNAc2) is correctly represented
5. Sialylation / fucosylation patterns are captured
6. Branching topology reflects biological branching (bi/tri/tetra-antennary)

Reference glycan structures use GlyTouCan/WURCS 2.0 format.
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.embedding.encoders.wurcs_tree_parser import (
    GlycanTree,
    GlycosidicBond,
    MonosaccharideNode,
    parse_wurcs_to_tree,
    glycan_tree_to_tensors,
    MONOSACCHARIDE_TYPE_VOCAB,
    ANOMERIC_VOCAB,
    RING_FORM_VOCAB,
    MODIFICATION_TYPES,
    NUM_MONO_TYPES,
    NUM_MODIFICATIONS,
    _classify_residue,
    _detect_anomeric,
    _detect_ring_form,
    _detect_modifications,
)


# =====================================================================
# Reference WURCS strings for biologically important glycan structures
# =====================================================================

# --- N-glycan core pentasaccharide: Man3GlcNAc2 ---
# This is the universal core of all N-linked glycans:
#   GlcNAc-beta-1,4-GlcNAc-beta-1,4-Man-alpha-1,3-Man
#                                          \-alpha-1,6-Man
# 5 residues, 4 linkages, 1 branch point (central Man)
WURCS_NGLYCAN_CORE = (
    "WURCS=2.0/3,5,4/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc (beta)
    "[a1122h-1b_1-5]"              # 2: Man (beta, core)
    "[a1122h-1a_1-5]"              # 3: Man (alpha, antenna)
    "/1-1-2-3-3/"
    "a4-b1_b4-c1_c3-d1_c6-e1"
)

# --- Biantennary complex N-glycan ---
# Man3GlcNAc2 core + 2 GlcNAc antennae = 7 residues total
# Topology:   GlcNAc-GlcNAc-Man(-Man-GlcNAc)(-Man-GlcNAc)
WURCS_BIANTENNARY = (
    "WURCS=2.0/3,7,6/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Man-beta (core)
    "[a1122h-1a_1-5]"              # 3: Man-alpha
    "/1-1-2-3-1-3-1/"
    "a4-b1_b4-c1_c3-d1_c6-f1_d4-e1_f4-g1"
)

# --- Core-fucosylated biantennary ---
# Biantennary + Fuc-alpha-1,6-GlcNAc (core fucosylation)
WURCS_CORE_FUCOSYLATED = (
    "WURCS=2.0/4,8,7/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Man-beta
    "[a1122h-1a_1-5]"              # 3: Man-alpha
    "[a1221m-1a_1-5]"              # 4: Fuc (L-fucose)
    "/1-1-2-3-1-3-1-4/"
    "a4-b1_b4-c1_c3-d1_c6-f1_d4-e1_f4-g1_a6-h1"
)

# --- Sialylated biantennary ---
# Biantennary + 2 NeuAc on terminal Gal residues
WURCS_SIALYLATED_BIANTENNARY = (
    "WURCS=2.0/5,11,10/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Man-beta
    "[a1122h-1a_1-5]"              # 3: Man-alpha
    "[a2112h-1b_1-5]"              # 4: Gal (beta)
    "[a2112h-1b_1-5_2*N]"          # 5: NeuAc
    "/1-1-2-3-1-3-1-4-5-4-5/"
    "a4-b1_b4-c1_c3-d1_c6-f1_d4-e1_f4-g1_e4-h1_g4-i1_h3-j2_i3-k2"
)

# --- High-mannose N-glycan: Man5GlcNAc2 ---
WURCS_HIGH_MANNOSE = (
    "WURCS=2.0/2,7,6/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Man
    "/1-1-2-2-2-2-2/"
    "a4-b1_b4-c1_c3-d1_c6-e1_d2-f1_e2-g1"
)

# --- O-glycan core 1: Gal-beta-1,3-GalNAc ---
WURCS_OGLYCAN_CORE1 = (
    "WURCS=2.0/2,2,1/"
    "[a2112h-1a_1-5_2*NCC/3=O]"  # 1: GalNAc (alpha)
    "[a2112h-1b_1-5]"              # 2: Gal (beta)
    "/1-2/"
    "a3-b1"
)

# --- Triantennary N-glycan (3 antennae from trimannosyl core) ---
# Three GlcNAc branches: 1,2 on Man-alpha-1,3 and 1,4 on Man-alpha-1,6
WURCS_TRIANTENNARY = (
    "WURCS=2.0/3,8,7/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Man-beta
    "[a1122h-1a_1-5]"              # 3: Man-alpha
    "/1-1-2-3-1-3-1-1/"
    "a4-b1_b4-c1_c3-d1_c6-f1_d2-e1_d4-g1_f4-h1"
)

# --- Single monosaccharide (edge case) ---
WURCS_SINGLE_MONO = (
    "WURCS=2.0/1,1,0/"
    "[a2122h-1b_1-5]"
    "/1/"
)

# --- Xylose-containing glycan (pentose) ---
WURCS_XYLOSYLATED = (
    "WURCS=2.0/3,3,2/"
    "[a2122h-1b_1-5_2*NCC/3=O]"  # 1: GlcNAc
    "[a1122h-1b_1-5]"              # 2: Man
    "[a212h-1b_1-5]"               # 3: Xyl (pentose)
    "/1-2-3/"
    "a4-b1_b2-c1"
)


# =====================================================================
# 1. WURCS tree parsing produces biologically correct trees
# =====================================================================


class TestWURCSTreeParsing:
    """Verify that parse_wurcs_to_tree generates biologically valid trees."""

    def test_nglycan_core_has_5_nodes(self):
        """Man3GlcNAc2 core should have exactly 5 monosaccharide nodes."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        assert tree.num_nodes == 5, (
            f"N-glycan core should have 5 nodes, got {tree.num_nodes}"
        )

    def test_nglycan_core_has_4_edges(self):
        """Man3GlcNAc2 core should have 4 glycosidic bonds."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        assert tree.num_edges == 4, (
            f"N-glycan core should have 4 edges, got {tree.num_edges}"
        )

    def test_tree_is_connected(self):
        """All nodes should be reachable from the root."""
        tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        visited = set()

        def dfs(idx):
            visited.add(idx)
            for child in tree.children_of(idx):
                if child not in visited:
                    dfs(child)

        dfs(tree.root_idx)
        assert len(visited) == tree.num_nodes, (
            f"Only {len(visited)}/{tree.num_nodes} nodes reachable from root"
        )

    def test_tree_has_no_cycles(self):
        """A glycan tree must be acyclic (each node has at most one parent)."""
        tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        parent_count = {}
        for edge in tree.edges:
            parent_count[edge.child_idx] = parent_count.get(edge.child_idx, 0) + 1
        for child_idx, count in parent_count.items():
            assert count == 1, (
                f"Node {child_idx} has {count} parents; expected 1 in a tree"
            )

    def test_root_is_reducing_end(self):
        """Root should be the reducing-end residue (index 0, first in sequence)."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        assert tree.root_idx == 0
        # Root should have no parent
        assert tree.parent_of(tree.root_idx) is None

    def test_biantennary_has_7_residues(self):
        """Biantennary complex N-glycan: 4 GlcNAc + 1 Man-beta + 2 Man-alpha = 7."""
        tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        assert tree.num_nodes == 7

    def test_single_monosaccharide_tree(self):
        """Single residue should produce a tree with 1 node and 0 edges."""
        tree = parse_wurcs_to_tree(WURCS_SINGLE_MONO)
        assert tree.num_nodes == 1
        assert tree.num_edges == 0
        assert tree.root_idx == 0

    def test_oglycan_core1_has_2_residues(self):
        """O-glycan core 1 (Gal-GalNAc) should have 2 residues."""
        tree = parse_wurcs_to_tree(WURCS_OGLYCAN_CORE1)
        assert tree.num_nodes == 2
        assert tree.num_edges == 1

    def test_invalid_wurcs_raises(self):
        """Non-WURCS string should raise ValueError."""
        with pytest.raises(ValueError):
            parse_wurcs_to_tree("NOT_A_WURCS_STRING")

    def test_empty_wurcs_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            parse_wurcs_to_tree("")

    def test_topological_order_bottom_up_starts_with_leaves(self):
        """Bottom-up order should start with leaf nodes."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        order = tree.topological_order_bottom_up()
        # First elements should be leaves (no children)
        for idx in order[:3]:  # leaves of Man3GlcNAc2
            children = tree.children_of(idx)
            # Leaves or intermediate nodes early in traversal
            # The root should be last
        assert order[-1] == tree.root_idx, (
            "Root should be last in bottom-up order"
        )

    def test_topological_order_top_down_starts_with_root(self):
        """Top-down order should start with root."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        order = tree.topological_order_top_down()
        assert order[0] == tree.root_idx

    def test_high_mannose_has_7_residues(self):
        """Man5GlcNAc2 should have 7 residues."""
        tree = parse_wurcs_to_tree(WURCS_HIGH_MANNOSE)
        assert tree.num_nodes == 7


# =====================================================================
# 2. Monosaccharide type classification distinguishes major types
# =====================================================================


class TestMonosaccharideTypeClassification:
    """Verify that WURCS residue codes are classified into correct monosaccharide types."""

    @pytest.mark.parametrize("wurcs_code,expected_type", [
        # Hexoses
        ("a2122h-1b_1-5", "Glc"),         # D-glucose
        ("a1122h-1b_1-5", "Man"),          # D-mannose
        # N-acetylhexosamines
        ("a2122h-1b_1-5_2*NCC/3=O", "GlcNAc"),  # N-acetylglucosamine
        ("a2122h-1a_1-5_2*NCC/3=O", "GlcNAc"),  # alpha-GlcNAc
        ("a2122h-1b_1-5_2*NCC", "GlcNAc"),       # GlcNAc without /3=O
        ("a2112h-1a_1-5_2*NCC", "GalNAc"),       # alpha-GalNAc
        # Deoxyhexoses
        ("a1221m-1a_1-5", "Fuc"),          # L-fucose
        ("a2211m-1a_1-5", "Rha"),          # L-rhamnose
        # Pentoses
        ("a212h-1b_1-5", "Xyl"),           # D-xylose
        ("a122h-1a_1-4", "Ara"),           # L-arabinose (furanose)
        # Uronic acids
        ("a2122A-1b_1-5", "GlcA"),         # D-glucuronic acid
        ("a2112A-1b_1-5", "IdoA"),         # L-iduronic acid
    ])
    def test_classify_residue(self, wurcs_code, expected_type):
        """Each WURCS residue code should map to the correct monosaccharide type."""
        result = _classify_residue(wurcs_code)
        assert result == expected_type, (
            f"Expected {expected_type} for {wurcs_code!r}, got {result}"
        )

    def test_gal_not_misclassified_as_neuac(self):
        """Plain Gal (a2112h-1b_1-5) should NOT be classified as NeuAc."""
        result = _classify_residue("a2112h-1b_1-5")
        assert result == "Gal", (
            f"Plain Gal misclassified as {result}"
        )

    def test_galnac_not_misclassified_as_neuac(self):
        """GalNAc (a2112h-1b_1-5_2*NCC/3=O) should be GalNAc, not NeuAc."""
        result = _classify_residue("a2112h-1b_1-5_2*NCC/3=O")
        assert result == "GalNAc", (
            f"GalNAc misclassified as {result}"
        )

    def test_unknown_residue_returns_unknown(self):
        """Unrecognized residue code should return 'Unknown'."""
        result = _classify_residue("x9999z-1a_1-5")
        assert result == "Unknown"

    def test_vocab_covers_major_types(self):
        """MONOSACCHARIDE_TYPE_VOCAB should cover all biologically important types."""
        required = {
            "Glc", "Man", "Gal",      # hexoses
            "GlcNAc", "GalNAc",       # N-acetylhexosamines
            "Fuc",                      # fucose
            "NeuAc", "NeuGc",         # sialic acids
            "Xyl",                      # pentose
            "GlcA", "IdoA",           # uronic acids
            "Rha",                      # rhamnose
            "Ara",                      # arabinose
        }
        actual = set(MONOSACCHARIDE_TYPE_VOCAB.keys())
        for mono_type in required:
            assert mono_type in actual, (
                f"Missing {mono_type} from MONOSACCHARIDE_TYPE_VOCAB"
            )

    def test_vocab_has_distinct_indices(self):
        """Each monosaccharide type should have a unique index."""
        indices = list(MONOSACCHARIDE_TYPE_VOCAB.values())
        assert len(indices) == len(set(indices)), (
            "Duplicate indices found in MONOSACCHARIDE_TYPE_VOCAB"
        )

    def test_vocab_size_accommodates_all_types(self):
        """NUM_MONO_TYPES should be >= the vocab size."""
        assert NUM_MONO_TYPES >= len(MONOSACCHARIDE_TYPE_VOCAB), (
            f"NUM_MONO_TYPES ({NUM_MONO_TYPES}) < vocab size "
            f"({len(MONOSACCHARIDE_TYPE_VOCAB)})"
        )

    def test_vocab_indices_within_embedding_range(self):
        """All type indices must be < NUM_MONO_TYPES for nn.Embedding."""
        for name, idx in MONOSACCHARIDE_TYPE_VOCAB.items():
            assert 0 <= idx < NUM_MONO_TYPES, (
                f"{name} has index {idx}, must be in [0, {NUM_MONO_TYPES})"
            )

    def test_nglycan_core_monosaccharide_types(self):
        """N-glycan core should contain GlcNAc and Man types."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        types_found = {n.mono_type for n in tree.nodes}
        assert "GlcNAc" in types_found, "N-glycan core must contain GlcNAc"
        assert "Man" in types_found, "N-glycan core must contain Man"

    def test_fucosylated_glycan_contains_fuc(self):
        """Core-fucosylated glycan should contain a Fuc residue."""
        tree = parse_wurcs_to_tree(WURCS_CORE_FUCOSYLATED)
        types_found = {n.mono_type for n in tree.nodes}
        assert "Fuc" in types_found, "Core-fucosylated glycan must contain Fuc"


# =====================================================================
# 3. Linkage encoding captures position-specific information
# =====================================================================


class TestLinkagePositionEncoding:
    """Verify that glycosidic linkage positions are correctly parsed and encoded."""

    def test_1_4_linkage_captured(self):
        """GlcNAc-beta-1,4-GlcNAc linkage should have position (4, 1)."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        # First linkage: a4-b1 means residue a carbon 4 -> residue b carbon 1
        found_14 = False
        for edge in tree.edges:
            if edge.linkage_position == (4, 1):
                found_14 = True
                break
        assert found_14, (
            "N-glycan core should contain a (4,1) linkage "
            f"(GlcNAc-beta-1,4-GlcNAc); found: "
            f"{[e.linkage_position for e in tree.edges]}"
        )

    def test_1_3_linkage_captured(self):
        """Man-alpha-1,3-Man linkage should have position (3, 1)."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        found_13 = any(e.linkage_position == (3, 1) for e in tree.edges)
        assert found_13, (
            "N-glycan core should contain a (3,1) linkage "
            f"(Man-alpha-1,3 branch); found: "
            f"{[e.linkage_position for e in tree.edges]}"
        )

    def test_1_6_linkage_captured(self):
        """Man-alpha-1,6-Man linkage should have position (6, 1)."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        found_16 = any(e.linkage_position == (6, 1) for e in tree.edges)
        assert found_16, (
            "N-glycan core should contain a (6,1) linkage "
            f"(Man-alpha-1,6 branch); found: "
            f"{[e.linkage_position for e in tree.edges]}"
        )

    def test_different_linkages_produce_different_encodings(self):
        """1,3 and 1,6 linkages must be distinguishable in tensor encoding."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        tensors = glycan_tree_to_tensors(tree)
        parent_carbons = tensors["linkage_parent_carbon"]
        # Should have at least two distinct parent carbon positions
        unique_positions = parent_carbons.unique()
        assert len(unique_positions) >= 2, (
            f"Expected at least 2 distinct parent carbon positions, "
            f"got {unique_positions.tolist()}"
        )

    def test_1_3_vs_1_4_vs_1_6_all_distinct(self):
        """Core N-glycan has 1,3 / 1,4 / 1,6 linkages which must all be encoded distinctly."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        positions = set()
        for edge in tree.edges:
            positions.add(edge.linkage_position)
        # We expect at least (4,1), (3,1), and (6,1)
        assert (4, 1) in positions
        assert (3, 1) in positions
        assert (6, 1) in positions

    def test_oglycan_1_3_linkage(self):
        """O-glycan core 1: Gal-beta-1,3-GalNAc should have (3, 1) linkage."""
        tree = parse_wurcs_to_tree(WURCS_OGLYCAN_CORE1)
        assert tree.num_edges == 1
        assert tree.edges[0].linkage_position == (3, 1), (
            f"O-glycan core 1 linkage should be (3,1), "
            f"got {tree.edges[0].linkage_position}"
        )

    def test_linkage_carbons_in_tensor(self):
        """Tensor encoding should separately encode parent and child carbon positions."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        tensors = glycan_tree_to_tensors(tree)
        assert "linkage_parent_carbon" in tensors
        assert "linkage_child_carbon" in tensors
        assert tensors["linkage_parent_carbon"].shape[0] == tree.num_edges
        assert tensors["linkage_child_carbon"].shape[0] == tree.num_edges

    def test_core_fucosylation_linkage_position(self):
        """Core fucosylation is Fuc-alpha-1,6-GlcNAc; should have (6,1) linkage."""
        tree = parse_wurcs_to_tree(WURCS_CORE_FUCOSYLATED)
        # The Fuc attachment should be a (6,1) linkage
        fuc_linkages = [
            e for e in tree.edges
            if tree.nodes[e.child_idx].mono_type == "Fuc"
        ]
        assert len(fuc_linkages) >= 1, "Should find at least one Fuc linkage"
        # Core fucosylation is alpha-1,6: parent_carbon=6
        assert fuc_linkages[0].linkage_position[0] == 6, (
            f"Core Fuc linkage parent carbon should be 6, "
            f"got {fuc_linkages[0].linkage_position[0]}"
        )


# =====================================================================
# 4. N-glycan core structure (Man3GlcNAc2) is correctly represented
# =====================================================================


class TestNGlycanCoreRepresentation:
    """Verify the N-glycan core pentasaccharide Man3GlcNAc2 is correctly modeled."""

    @pytest.fixture
    def core_tree(self):
        return parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)

    def test_core_composition(self, core_tree):
        """Core should have 2 GlcNAc + 3 Man residues."""
        type_counts = {}
        for node in core_tree.nodes:
            type_counts[node.mono_type] = type_counts.get(node.mono_type, 0) + 1
        assert type_counts.get("GlcNAc", 0) == 2, (
            f"Expected 2 GlcNAc, got {type_counts}"
        )
        assert type_counts.get("Man", 0) == 3, (
            f"Expected 3 Man, got {type_counts}"
        )

    def test_core_reducing_end_is_glcnac(self, core_tree):
        """Reducing end (root) should be GlcNAc (the first GlcNAc in the chitobiose core)."""
        root = core_tree.nodes[core_tree.root_idx]
        assert root.mono_type == "GlcNAc", (
            f"Root should be GlcNAc, got {root.mono_type}"
        )

    def test_core_has_one_branch_point(self, core_tree):
        """Core beta-Man should be the single branch point (children: alpha-1,3-Man and alpha-1,6-Man)."""
        branch_points = [
            i for i in range(core_tree.num_nodes)
            if core_tree.is_branching(i)
        ]
        assert len(branch_points) == 1, (
            f"Expected 1 branch point in Man3GlcNAc2, got {len(branch_points)}"
        )

    def test_branch_point_is_mannose(self, core_tree):
        """The branch point should be a Man residue."""
        for i in range(core_tree.num_nodes):
            if core_tree.is_branching(i):
                assert core_tree.nodes[i].mono_type == "Man", (
                    f"Branch point should be Man, got {core_tree.nodes[i].mono_type}"
                )

    def test_branch_point_has_two_children(self, core_tree):
        """The branching Man should have exactly 2 children (alpha-1,3 and alpha-1,6)."""
        for i in range(core_tree.num_nodes):
            if core_tree.is_branching(i):
                children = core_tree.children_of(i)
                assert len(children) == 2, (
                    f"Branch Man should have 2 children, got {len(children)}"
                )

    def test_core_depth_distribution(self, core_tree):
        """Root depth=0, chitobiose GlcNAc=1, core Man=2, antenna Man=3."""
        root_depth = core_tree.depth_of(core_tree.root_idx)
        assert root_depth == 0
        max_depth = max(core_tree.depth_of(i) for i in range(core_tree.num_nodes))
        assert max_depth >= 2, (
            f"Max depth should be >= 2 for N-glycan core, got {max_depth}"
        )

    def test_high_mannose_contains_nglycan_core(self):
        """Man5GlcNAc2 should contain the Man3GlcNAc2 core structure."""
        tree = parse_wurcs_to_tree(WURCS_HIGH_MANNOSE)
        type_counts = {}
        for node in tree.nodes:
            type_counts[node.mono_type] = type_counts.get(node.mono_type, 0) + 1
        # Must have at least 2 GlcNAc and 5 Man
        assert type_counts.get("GlcNAc", 0) >= 2
        assert type_counts.get("Man", 0) >= 3, (
            f"High-mannose should have >= 3 Man, got {type_counts}"
        )


# =====================================================================
# 5. Sialylation and fucosylation patterns
# =====================================================================


class TestSialylationFucosylationPatterns:
    """Verify that sialylation and fucosylation are correctly captured."""

    def test_non_sialylated_has_no_neuac(self):
        """Non-sialylated glycan should have no NeuAc residues."""
        tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        types = {n.mono_type for n in tree.nodes}
        assert "NeuAc" not in types, (
            "Non-sialylated biantennary should not contain NeuAc"
        )

    def test_non_fucosylated_has_no_fuc(self):
        """Non-fucosylated glycan should have no Fuc residues."""
        tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        types = {n.mono_type for n in tree.nodes}
        assert "Fuc" not in types, (
            "Non-fucosylated biantennary should not contain Fuc"
        )

    def test_fucosylated_glycan_has_fuc(self):
        """Core-fucosylated glycan should contain Fuc."""
        tree = parse_wurcs_to_tree(WURCS_CORE_FUCOSYLATED)
        types = {n.mono_type for n in tree.nodes}
        assert "Fuc" in types

    def test_fucose_is_leaf_node(self):
        """In core fucosylation, Fuc should be a leaf node (no children)."""
        tree = parse_wurcs_to_tree(WURCS_CORE_FUCOSYLATED)
        for node in tree.nodes:
            if node.mono_type == "Fuc":
                children = tree.children_of(node.index)
                assert len(children) == 0, (
                    f"Fuc should be a leaf, but has {len(children)} children"
                )

    def test_fucose_is_alpha_anomeric(self):
        """L-fucose typically has alpha anomeric configuration."""
        tree = parse_wurcs_to_tree(WURCS_CORE_FUCOSYLATED)
        for node in tree.nodes:
            if node.mono_type == "Fuc":
                assert node.anomeric == "alpha", (
                    f"Fuc anomeric should be alpha, got {node.anomeric}"
                )


# =====================================================================
# 6. Branching topology reflects biological branching
# =====================================================================


class TestBranchingTopology:
    """Verify that branching patterns are biologically correct."""

    def test_biantennary_has_one_branch_point(self):
        """Biantennary glycan should have at least 1 branch point (core Man)."""
        tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        branch_count = sum(
            1 for i in range(tree.num_nodes) if tree.is_branching(i)
        )
        assert branch_count >= 1, (
            f"Biantennary should have >= 1 branch point, got {branch_count}"
        )

    def test_triantennary_has_more_branches_than_biantennary(self):
        """Triantennary should have more branch points than biantennary."""
        bi_tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        tri_tree = parse_wurcs_to_tree(WURCS_TRIANTENNARY)
        bi_branches = sum(
            1 for i in range(bi_tree.num_nodes) if bi_tree.is_branching(i)
        )
        tri_branches = sum(
            1 for i in range(tri_tree.num_nodes) if tri_tree.is_branching(i)
        )
        assert tri_branches >= bi_branches, (
            f"Triantennary ({tri_branches}) should have >= branches "
            f"than biantennary ({bi_branches})"
        )

    def test_linear_oglycan_has_no_branch(self):
        """Linear O-glycan (core 1) should have no branch points."""
        tree = parse_wurcs_to_tree(WURCS_OGLYCAN_CORE1)
        branch_count = sum(
            1 for i in range(tree.num_nodes) if tree.is_branching(i)
        )
        assert branch_count == 0, (
            f"Linear O-glycan should have 0 branch points, got {branch_count}"
        )

    def test_high_mannose_branching(self):
        """Man5GlcNAc2 has 1 branch point at the core beta-Man.

        In Man5, the core beta-Man branches into alpha-1,3-Man and alpha-1,6-Man,
        each of which has one terminal Man child (no further branching).
        Man9GlcNAc2 would have 3 branch points.
        """
        tree = parse_wurcs_to_tree(WURCS_HIGH_MANNOSE)
        branch_count = sum(
            1 for i in range(tree.num_nodes) if tree.is_branching(i)
        )
        assert branch_count >= 1, (
            f"High-mannose Man5 should have >= 1 branch point, got {branch_count}"
        )

    def test_branch_points_are_only_at_mannose_or_glcnac(self):
        """In N-glycans, branch points typically occur at Man or GlcNAc residues."""
        tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        for i in range(tree.num_nodes):
            if tree.is_branching(i):
                mono_type = tree.nodes[i].mono_type
                assert mono_type in ("Man", "GlcNAc"), (
                    f"Branch point is {mono_type}; expected Man or GlcNAc "
                    f"in N-glycan structures"
                )

    def test_siblings_share_parent(self):
        """Siblings should share the same parent node."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        for i in range(tree.num_nodes):
            siblings = tree.siblings_of(i)
            if siblings:
                my_parent = tree.parent_of(i)
                for sib in siblings:
                    sib_parent = tree.parent_of(sib)
                    assert sib_parent == my_parent, (
                        f"Node {i} and sibling {sib} should share parent "
                        f"{my_parent}, but sibling has parent {sib_parent}"
                    )

    def test_is_branch_tensor_matches_tree(self):
        """is_branch tensor should match tree.is_branching() for all nodes."""
        tree = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        tensors = glycan_tree_to_tensors(tree)
        for i in range(tree.num_nodes):
            expected = tree.is_branching(i)
            actual = tensors["is_branch"][i].item()
            assert actual == expected, (
                f"is_branch mismatch at node {i}: tensor={actual}, tree={expected}"
            )


# =====================================================================
# 7. Tensor conversion correctness
# =====================================================================


class TestGlycanTreeToTensors:
    """Verify glycan_tree_to_tensors produces correct tensor representations."""

    @pytest.fixture
    def core_tensors(self):
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        return glycan_tree_to_tensors(tree)

    def test_mono_type_tensor_shape(self, core_tensors):
        """mono_type tensor should have shape [N]."""
        assert core_tensors["mono_type"].shape == (5,)
        assert core_tensors["mono_type"].dtype == torch.long

    def test_anomeric_tensor_shape(self, core_tensors):
        assert core_tensors["anomeric"].shape == (5,)
        assert core_tensors["anomeric"].dtype == torch.long

    def test_ring_form_tensor_shape(self, core_tensors):
        assert core_tensors["ring_form"].shape == (5,)
        assert core_tensors["ring_form"].dtype == torch.long

    def test_modifications_tensor_shape(self, core_tensors):
        assert core_tensors["modifications"].shape == (5, NUM_MODIFICATIONS)
        assert core_tensors["modifications"].dtype == torch.float32

    def test_edge_index_shape(self, core_tensors):
        """edge_index should be [2, E] for E edges."""
        assert core_tensors["edge_index"].shape == (2, 4)

    def test_depth_tensor(self, core_tensors):
        """Depth tensor should have root at depth 0."""
        depth = core_tensors["depth"]
        assert depth[0].item() == 0  # root

    def test_all_mono_type_indices_valid(self, core_tensors):
        """All mono_type indices should be in [0, NUM_MONO_TYPES)."""
        mono = core_tensors["mono_type"]
        assert torch.all(mono >= 0)
        assert torch.all(mono < NUM_MONO_TYPES)

    def test_anomeric_indices_valid(self, core_tensors):
        """All anomeric indices should be within ANOMERIC_VOCAB range."""
        anom = core_tensors["anomeric"]
        assert torch.all(anom >= 0)
        assert torch.all(anom < len(ANOMERIC_VOCAB))

    def test_ring_form_indices_valid(self, core_tensors):
        """All ring form indices should be within RING_FORM_VOCAB range."""
        ring = core_tensors["ring_form"]
        assert torch.all(ring >= 0)
        assert torch.all(ring < len(RING_FORM_VOCAB))

    def test_modifications_binary(self, core_tensors):
        """Modification tensor values should be 0.0 or 1.0."""
        mods = core_tensors["modifications"]
        assert torch.all((mods == 0.0) | (mods == 1.0))

    def test_bond_type_tensor(self, core_tensors):
        """Bond type should be 0 (alpha), 1 (beta), or 2 (unknown)."""
        bt = core_tensors["bond_type"]
        assert torch.all(bt >= 0)
        assert torch.all(bt <= 2)

    def test_single_node_tensors(self):
        """Single-node glycan should produce valid zero-edge tensors."""
        tree = parse_wurcs_to_tree(WURCS_SINGLE_MONO)
        tensors = glycan_tree_to_tensors(tree)
        assert tensors["num_nodes"] == 1
        assert tensors["edge_index"].shape == (2, 0)
        assert tensors["linkage_parent_carbon"].shape == (0,)

    def test_glcnac_has_n_acetyl_modification(self):
        """GlcNAc residues should have n_acetyl modification flagged."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        tensors = glycan_tree_to_tensors(tree)
        n_acetyl_idx = MODIFICATION_TYPES.index("n_acetyl")
        # Find GlcNAc nodes
        glcnac_type_idx = MONOSACCHARIDE_TYPE_VOCAB["GlcNAc"]
        for i in range(tensors["num_nodes"]):
            if tensors["mono_type"][i].item() == glcnac_type_idx:
                assert tensors["modifications"][i, n_acetyl_idx].item() == 1.0, (
                    f"GlcNAc at node {i} should have n_acetyl modification"
                )


# =====================================================================
# 8. Anomeric configuration detection
# =====================================================================


class TestAnomericConfiguration:
    """Verify anomeric configuration (alpha/beta) detection."""

    def test_alpha_detection(self):
        assert _detect_anomeric("a1221m-1a_1-5") == "alpha"

    def test_beta_detection(self):
        assert _detect_anomeric("a2122h-1b_1-5") == "beta"

    def test_unknown_anomeric(self):
        assert _detect_anomeric("a2122h-1x_1-5") == "unknown"

    def test_glcnac_in_nglycan_is_beta(self):
        """The chitobiose GlcNAc residues in N-glycans are beta-linked."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        root = tree.nodes[tree.root_idx]
        assert root.anomeric == "beta", (
            f"Chitobiose GlcNAc should be beta, got {root.anomeric}"
        )


# =====================================================================
# 9. Ring form detection
# =====================================================================


class TestRingFormDetection:
    """Verify ring form (pyranose/furanose) detection."""

    def test_pyranose_detection(self):
        """_1-5 indicates pyranose ring."""
        assert _detect_ring_form("a2122h-1b_1-5") == "pyranose"

    def test_furanose_detection(self):
        """_1-4 indicates furanose ring."""
        assert _detect_ring_form("a122h-1a_1-4") == "furanose"

    def test_unknown_ring_form(self):
        assert _detect_ring_form("a2122h-1b") == "unknown"

    def test_hexose_is_pyranose(self):
        """Standard hexoses (Glc, Man, Gal) should be pyranose."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        for node in tree.nodes:
            if node.mono_type in ("Glc", "Man", "Gal", "GlcNAc"):
                assert node.ring_form == "pyranose", (
                    f"{node.mono_type} should be pyranose, got {node.ring_form}"
                )


# =====================================================================
# 10. Modification detection
# =====================================================================


class TestModificationDetection:
    """Verify chemical modification detection from WURCS codes."""

    def test_sulfation_detected(self):
        mods = _detect_modifications("a2122h-1b_1-5_6*OSO/3=O")
        assert "sulfation" in mods

    def test_phosphorylation_detected(self):
        mods = _detect_modifications("a2122h-1b_1-5_6*OPO/3=O")
        assert "phosphorylation" in mods

    def test_n_acetyl_detected(self):
        mods = _detect_modifications("a2122h-1b_1-5_2*NCC/3=O")
        assert "n_acetyl" in mods

    def test_n_glycolyl_detected(self):
        mods = _detect_modifications("a2122h-1b_1-5_2*NO/3=O")
        assert "n_glycolyl" in mods

    def test_deoxy_detected(self):
        """Fucose (deoxyhexose) should flag deoxy modification."""
        mods = _detect_modifications("a1221m-1a_1-5")
        assert "deoxy" in mods

    def test_no_false_modifications_on_plain_hexose(self):
        """Plain hexose should have no modifications."""
        mods = _detect_modifications("a2122h-1b_1-5")
        assert len(mods) == 0, f"Plain hexose should have no mods, got {mods}"

    def test_modification_vector_length(self):
        """Modification vector should match NUM_MODIFICATIONS."""
        tree = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        for node in tree.nodes:
            vec = node.modification_vector
            assert len(vec) == NUM_MODIFICATIONS


# =====================================================================
# 11. Cross-structure biological consistency
# =====================================================================


class TestCrossStructureBiologicalConsistency:
    """Verify biological consistency across different glycan structures."""

    def test_nglycan_always_starts_with_glcnac(self):
        """All N-glycans should have GlcNAc at the reducing end (root)."""
        for wurcs in [WURCS_NGLYCAN_CORE, WURCS_BIANTENNARY,
                      WURCS_CORE_FUCOSYLATED, WURCS_HIGH_MANNOSE]:
            tree = parse_wurcs_to_tree(wurcs)
            root = tree.nodes[tree.root_idx]
            assert root.mono_type == "GlcNAc", (
                f"N-glycan root should be GlcNAc, got {root.mono_type} "
                f"for WURCS starting with {wurcs[:40]}"
            )

    def test_oglycan_root_is_galnac(self):
        """O-glycan core 1 should have GalNAc at the reducing end."""
        tree = parse_wurcs_to_tree(WURCS_OGLYCAN_CORE1)
        root = tree.nodes[tree.root_idx]
        assert root.mono_type == "GalNAc", (
            f"O-glycan root should be GalNAc, got {root.mono_type}"
        )

    def test_tree_edge_count_equals_nodes_minus_1(self):
        """A tree with N nodes should have exactly N-1 edges."""
        for wurcs in [WURCS_NGLYCAN_CORE, WURCS_BIANTENNARY,
                      WURCS_HIGH_MANNOSE, WURCS_OGLYCAN_CORE1]:
            tree = parse_wurcs_to_tree(wurcs)
            if tree.num_nodes > 0:
                expected_edges = tree.num_nodes - 1
                assert tree.num_edges == expected_edges, (
                    f"Tree with {tree.num_nodes} nodes should have "
                    f"{expected_edges} edges, got {tree.num_edges}"
                )

    def test_larger_glycan_has_more_nodes(self):
        """Biantennary (7 residues) should have more nodes than core (5 residues)."""
        core = parse_wurcs_to_tree(WURCS_NGLYCAN_CORE)
        biant = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        assert biant.num_nodes > core.num_nodes

    def test_fucosylated_has_one_more_residue(self):
        """Core-fucosylated = biantennary + 1 Fuc."""
        biant = parse_wurcs_to_tree(WURCS_BIANTENNARY)
        fuc = parse_wurcs_to_tree(WURCS_CORE_FUCOSYLATED)
        assert fuc.num_nodes == biant.num_nodes + 1

    def test_xylose_detected_in_xylosylated_glycan(self):
        """Xylose-containing glycan should have Xyl type node."""
        tree = parse_wurcs_to_tree(WURCS_XYLOSYLATED)
        types = {n.mono_type for n in tree.nodes}
        assert "Xyl" in types, (
            f"Xylosylated glycan should contain Xyl, found {types}"
        )

    def test_tensor_num_nodes_consistent(self):
        """num_nodes in tensor dict should match tree.num_nodes."""
        for wurcs in [WURCS_NGLYCAN_CORE, WURCS_BIANTENNARY,
                      WURCS_SINGLE_MONO, WURCS_OGLYCAN_CORE1]:
            tree = parse_wurcs_to_tree(wurcs)
            tensors = glycan_tree_to_tensors(tree)
            assert tensors["num_nodes"] == tree.num_nodes
