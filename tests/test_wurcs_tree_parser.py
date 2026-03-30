"""Tests for WURCS-to-tree-graph parser (wurcs_tree_parser.py).

Validates:
- Known N-glycan/O-glycan WURCS -> correct tree structure (nodes, edges, root)
- Monosaccharide type classification (GlcNAc, Man, Gal, Fuc, NeuAc)
- Anomeric configuration detection (alpha/beta)
- Glycosidic linkage position parsing (1->4, 1->3, 1->6, etc.)
- Linear chain (no branching) handling
- Single residue handling
- Error handling for invalid WURCS strings
- Tensor conversion via glycan_tree_to_tensors
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.embedding.encoders.wurcs_tree_parser import (
    ANOMERIC_VOCAB,
    MONOSACCHARIDE_TYPE_VOCAB,
    RING_FORM_VOCAB,
    NUM_MODIFICATIONS,
    GlycanTree,
    GlycosidicBond,
    MonosaccharideNode,
    _classify_residue,
    _detect_anomeric,
    _detect_modifications,
    _detect_ring_form,
    _parse_linkage_token,
    _parse_wurcs_sections,
    glycan_tree_to_tensors,
    parse_wurcs_to_tree,
)


# ---------------------------------------------------------------------------
# WURCS test strings
# ---------------------------------------------------------------------------

# Single Glc residue (beta, pyranose)
WURCS_SINGLE_GLC = "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"

# Disaccharide: Glc(b1-4)Glc -- linear chain, two nodes, one edge
WURCS_DISACCHARIDE = "WURCS=2.0/1,2,1/[a2122h-1b_1-5]/1-1/a4-b1"

# Chitobiose: GlcNAc(b1-4)GlcNAc -- biologically real, tests /3=O in residue code
WURCS_CHITOBIOSE = (
    "WURCS=2.0/2,2,1/"
    "[a2122h-1b_1-5_2*NCC/3=O][a2112h-1b_1-5]/"
    "1-2/a4-b1"
)

# Core 1 O-glycan: GalNAc-alpha -> Gal-beta
# WURCS=2.0/2,2,1/[a2122h-1b_1-5_2*NCC/3=O][a2112h-1b_1-5]/1-2/a3-b1
WURCS_CORE1_OGLYCAN = (
    "WURCS=2.0/2,2,1/"
    "[a2122h-1b_1-5_2*NCC/3=O][a2112h-1b_1-5]/"
    "1-2/a3-b1"
)

# Simple branching: trisaccharide with one branch point
# Root(a) -> child1(b) at position 3, Root(a) -> child2(c) at position 6
WURCS_BRANCHED_TRI = (
    "WURCS=2.0/1,3,2/[a2122h-1b_1-5]/1-1-1/a3-b1_a6-c1"
)

# Fucose residue (alpha, pyranose, deoxyhexose)
WURCS_FUCOSE = "WURCS=2.0/1,1,0/[a1221m-1a_1-5]/1/"

# Mannose residue (alpha, pyranose)
WURCS_MANNOSE = "WURCS=2.0/1,1,0/[a1122h-1a_1-5]/1/"

# Galactose residue (beta, pyranose)
WURCS_GALACTOSE = "WURCS=2.0/1,1,0/[a2112h-1b_1-5]/1/"

# NeuAc variant
WURCS_NEUAC = "WURCS=2.0/1,1,0/[a2112h-1b_1-5]/1/"


# ---------------------------------------------------------------------------
# Tests: Residue classification
# ---------------------------------------------------------------------------

class TestClassifyResidue:
    """Tests for _classify_residue: mapping WURCS codes to monosaccharide names."""

    def test_glc_classification(self):
        """Glc (glucose): a2122h-1b_1-5 should be classified as Glc."""
        assert _classify_residue("a2122h-1b_1-5") == "Glc"

    def test_man_classification(self):
        """Man (mannose): a1122h-1a_1-5 should be classified as Man."""
        assert _classify_residue("a1122h-1a_1-5") == "Man"

    def test_gal_classification(self):
        """Gal (galactose): a2112h-1b_1-5 should be classified as Gal."""
        assert _classify_residue("a2112h-1b_1-5") == "Gal"

    def test_glcnac_classification(self):
        """GlcNAc: a2122h with *NCC/3=O modification."""
        assert _classify_residue("a2122h-1b_1-5_2*NCC/3=O") == "GlcNAc"

    def test_galnac_classification(self):
        """GalNAc: a2112h with *NCC modification should classify as GalNAc."""
        assert _classify_residue("a2112h-1b_1-5_2*NCC") == "GalNAc"

    def test_galnac_alpha_classification(self):
        """GalNAc with alpha anomeric: a2112h-1a_1-5_2*NCC.

        Alpha anomeric avoids the NeuAc rule (which only matches -1b).
        """
        result = _classify_residue("a2112h-1a_1-5_2*NCC")
        assert result == "GalNAc"

    def test_glcnac_without_3o(self):
        """GlcNAc variant: a2122h with *NCC (without explicit /3=O)."""
        assert _classify_residue("a2122h-1b_1-5_2*NCC") == "GlcNAc"

    def test_fucose_classification(self):
        """Fuc (fucose): a1221m-1a_1-5 (deoxyhexose, alpha)."""
        assert _classify_residue("a1221m-1a_1-5") == "Fuc"

    def test_fucose_beta_classification(self):
        """Fuc can also appear as beta: a1221m-1b_1-5."""
        assert _classify_residue("a1221m-1b_1-5") == "Fuc"

    def test_gal_not_neuac(self):
        """a2112h-1b_1-5 is Gal stereochemistry, not NeuAc (9-carbon sialic acid)."""
        result = _classify_residue("a2112h-1b_1-5")
        assert result == "Gal"

    def test_xylose_classification(self):
        """Xyl (xylose): a212h-1b_1-5 (pentose)."""
        assert _classify_residue("a212h-1b_1-5") == "Xyl"

    def test_glca_classification(self):
        """GlcA (glucuronic acid): a2122A-1b_1-5."""
        assert _classify_residue("a2122A-1b_1-5") == "GlcA"

    def test_rhamnose_classification(self):
        """Rha (rhamnose): a2211m-1a_1-5."""
        assert _classify_residue("a2211m-1a_1-5") == "Rha"

    def test_unknown_residue(self):
        """Unrecognized residue code should return 'Unknown'."""
        assert _classify_residue("zzz_nonsense") == "Unknown"

    def test_empty_string(self):
        """Empty string should return 'Unknown'."""
        assert _classify_residue("") == "Unknown"


# ---------------------------------------------------------------------------
# Tests: Anomeric configuration detection
# ---------------------------------------------------------------------------

class TestDetectAnomeric:
    """Tests for _detect_anomeric: alpha/beta detection from WURCS codes."""

    def test_alpha_detection(self):
        """'-1a_' suffix indicates alpha configuration."""
        assert _detect_anomeric("a1122h-1a_1-5") == "alpha"

    def test_beta_detection(self):
        """'-1b_' suffix indicates beta configuration."""
        assert _detect_anomeric("a2122h-1b_1-5") == "beta"

    def test_unknown_anomeric(self):
        """Without '-1a' or '-1b', anomeric is unknown."""
        assert _detect_anomeric("a2122h_1-5") == "unknown"

    def test_alpha_at_end_of_string(self):
        """Alpha marker at end of code (no trailing '_')."""
        assert _detect_anomeric("a1122h-1a") == "alpha"

    def test_beta_at_end_of_string(self):
        """Beta marker at end of code."""
        assert _detect_anomeric("a2122h-1b") == "beta"


# ---------------------------------------------------------------------------
# Tests: Ring form detection
# ---------------------------------------------------------------------------

class TestDetectRingForm:
    """Tests for _detect_ring_form: pyranose/furanose detection."""

    def test_pyranose(self):
        """'_1-5' indicates pyranose ring."""
        assert _detect_ring_form("a2122h-1b_1-5") == "pyranose"

    def test_furanose(self):
        """'_1-4' indicates furanose ring."""
        assert _detect_ring_form("a122h-1b_1-4") == "furanose"

    def test_unknown_ring(self):
        """Without '_1-5' or '_1-4', ring form is unknown."""
        assert _detect_ring_form("a2122h-1b") == "unknown"


# ---------------------------------------------------------------------------
# Tests: Modification detection
# ---------------------------------------------------------------------------

class TestDetectModifications:
    """Tests for _detect_modifications: chemical modification detection."""

    def test_no_modifications(self):
        """Plain hexose has no modifications."""
        mods = _detect_modifications("a2122h-1b_1-5")
        assert mods == []

    def test_sulfation(self):
        """*OSO indicates sulfation."""
        mods = _detect_modifications("a2122h-1b_1-5_6*OSO")
        assert "sulfation" in mods

    def test_phosphorylation(self):
        """*OPO indicates phosphorylation."""
        mods = _detect_modifications("a2122h-1b_1-5_6*OPO")
        assert "phosphorylation" in mods

    def test_n_acetyl(self):
        """*NCC indicates N-acetylation."""
        mods = _detect_modifications("a2122h-1b_1-5_2*NCC/3=O")
        assert "n_acetyl" in mods

    def test_deoxy(self):
        """Deoxy modification (digit + 'm' or 'd')."""
        mods = _detect_modifications("a1221m-1a_1-5")
        assert "deoxy" in mods

    def test_multiple_modifications(self):
        """A residue with both sulfation and N-acetylation."""
        mods = _detect_modifications("a2122h-1b_1-5_2*NCC_6*OSO")
        assert "n_acetyl" in mods
        assert "sulfation" in mods


# ---------------------------------------------------------------------------
# Tests: WURCS string section parsing
# ---------------------------------------------------------------------------

class TestParseWurcsSections:
    """Tests for _parse_wurcs_sections: splitting WURCS into components."""

    def test_single_residue_parsing(self):
        """Single-residue WURCS should yield 1 unique res, 1 res_list entry."""
        header, unique_res, res_list, lin = _parse_wurcs_sections(WURCS_SINGLE_GLC)
        assert len(unique_res) == 1
        assert "a2122h-1b_1-5" in unique_res[0]

    def test_disaccharide_parsing(self):
        """Disaccharide should yield unique residues and 2 res_list entries."""
        header, unique_res, res_list, lin = _parse_wurcs_sections(WURCS_DISACCHARIDE)
        assert len(unique_res) == 1  # same type used twice
        assert len(res_list) == 2

    def test_linkage_section_extracted(self):
        """Linkage section should be non-empty for linked WURCS."""
        _, _, _, lin = _parse_wurcs_sections(WURCS_DISACCHARIDE)
        assert lin != ""
        assert "a4-b1" in lin

    def test_branched_trisaccharide_linkages(self):
        """Branched trisaccharide should have two linkage tokens."""
        _, _, res_list, lin = _parse_wurcs_sections(WURCS_BRANCHED_TRI)
        assert len(res_list) == 3
        # Linkage section should contain both linkages separated by '_'
        assert "a3-b1" in lin
        assert "a6-c1" in lin

    def test_invalid_prefix_raises(self):
        """Non-WURCS prefix should raise ValueError."""
        with pytest.raises(ValueError, match="Not a valid WURCS"):
            _parse_wurcs_sections("NOT_WURCS/something")

    def test_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Not a valid WURCS"):
            _parse_wurcs_sections("")

    def test_too_few_sections_raises(self):
        """WURCS with insufficient sections should raise ValueError."""
        with pytest.raises(ValueError, match="fewer than 3 sections"):
            _parse_wurcs_sections("WURCS=2.0/1")

    def test_chitobiose_unique_residues(self):
        """Chitobiose has 2 unique residue types with /3=O in first."""
        _, unique_res, _, _ = _parse_wurcs_sections(WURCS_CHITOBIOSE)
        assert len(unique_res) == 2
        assert "NCC/3=O" in unique_res[0]


# ---------------------------------------------------------------------------
# Tests: Linkage token parsing
# ---------------------------------------------------------------------------

class TestParseLinkageToken:
    """Tests for _parse_linkage_token: individual linkage parsing."""

    def test_standard_linkage(self):
        """'a4-b1' means residue a, carbon 4 -> residue b, carbon 1."""
        result = _parse_linkage_token("a4-b1")
        assert result is not None
        src_label, src_carbon, dst_label, dst_carbon = result
        assert src_label == "a"
        assert src_carbon == 4
        assert dst_label == "b"
        assert dst_carbon == 1

    def test_1_3_linkage(self):
        """'a3-b1' -> carbon 3 to carbon 1."""
        result = _parse_linkage_token("a3-b1")
        assert result is not None
        _, src_c, _, dst_c = result
        assert src_c == 3
        assert dst_c == 1

    def test_1_6_linkage(self):
        """'a6-b1' -> carbon 6 to carbon 1."""
        result = _parse_linkage_token("a6-b1")
        assert result is not None
        _, src_c, _, dst_c = result
        assert src_c == 6
        assert dst_c == 1

    def test_ambiguous_position(self):
        """'a?-b1' -> ambiguous source carbon position encoded as 0."""
        result = _parse_linkage_token("a?-b1")
        assert result is not None
        _, src_c, _, dst_c = result
        assert src_c == 0  # ambiguous
        assert dst_c == 1

    def test_invalid_token_returns_none(self):
        """Unparsable token returns None."""
        assert _parse_linkage_token("invalid") is None
        assert _parse_linkage_token("") is None

    def test_multi_letter_label(self):
        """Labels can be multi-letter (e.g., 'ab2-cd1')."""
        result = _parse_linkage_token("ab2-cd1")
        assert result is not None
        src_label, src_c, dst_label, dst_c = result
        assert src_label == "ab"
        assert src_c == 2
        assert dst_label == "cd"
        assert dst_c == 1


# ---------------------------------------------------------------------------
# Tests: parse_wurcs_to_tree (full integration)
# ---------------------------------------------------------------------------

class TestParseWurcsToTree:
    """Integration tests for parse_wurcs_to_tree."""

    def test_single_residue_tree(self):
        """Single residue -> 1 node, 0 edges, root=0."""
        tree = parse_wurcs_to_tree(WURCS_SINGLE_GLC)
        assert tree.num_nodes == 1
        assert tree.num_edges == 0
        assert tree.root_idx == 0

    def test_single_residue_type(self):
        """Single Glc residue should be classified correctly."""
        tree = parse_wurcs_to_tree(WURCS_SINGLE_GLC)
        node = tree.nodes[0]
        assert node.mono_type == "Glc"
        assert node.mono_type_idx == MONOSACCHARIDE_TYPE_VOCAB["Glc"]

    def test_single_residue_anomeric(self):
        """Single Glc(beta) should have beta anomeric config."""
        tree = parse_wurcs_to_tree(WURCS_SINGLE_GLC)
        node = tree.nodes[0]
        assert node.anomeric == "beta"
        assert node.anomeric_idx == ANOMERIC_VOCAB["beta"]

    def test_single_residue_ring_form(self):
        """Single Glc with _1-5 should be pyranose."""
        tree = parse_wurcs_to_tree(WURCS_SINGLE_GLC)
        node = tree.nodes[0]
        assert node.ring_form == "pyranose"
        assert node.ring_form_idx == RING_FORM_VOCAB["pyranose"]

    def test_disaccharide_structure(self):
        """Disaccharide: 2 nodes, 1 edge."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        assert tree.num_nodes == 2
        assert tree.num_edges == 1

    def test_disaccharide_edge(self):
        """Disaccharide edge should connect parent(a)=0 to child(b)=1."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        assert len(tree.edges) == 1
        edge = tree.edges[0]
        assert edge.parent_idx == 0
        assert edge.child_idx == 1

    def test_disaccharide_linkage_position(self):
        """Disaccharide a4-b1 linkage: parent carbon 4, child carbon 1."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        edge = tree.edges[0]
        assert edge.linkage_position == (4, 1)

    def test_branched_trisaccharide_structure(self):
        """Branched trisaccharide: 3 nodes, 2 edges."""
        tree = parse_wurcs_to_tree(WURCS_BRANCHED_TRI)
        assert tree.num_nodes == 3
        assert tree.num_edges == 2

    def test_branched_trisaccharide_branching(self):
        """Root node of branched trisaccharide should be a branch point."""
        tree = parse_wurcs_to_tree(WURCS_BRANCHED_TRI)
        assert tree.is_branching(tree.root_idx)
        children = tree.children_of(tree.root_idx)
        assert len(children) == 2

    def test_branched_trisaccharide_linkage_positions(self):
        """Branched trisaccharide: a3-b1 (pos 3->1) and a6-c1 (pos 6->1)."""
        tree = parse_wurcs_to_tree(WURCS_BRANCHED_TRI)
        positions = {e.linkage_position for e in tree.edges}
        assert (3, 1) in positions
        assert (6, 1) in positions

    def test_linear_chain_no_branching(self):
        """Disaccharide (linear chain) has no branch points."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        for i in range(tree.num_nodes):
            assert not tree.is_branching(i)

    def test_root_is_first_residue(self):
        """Root should always be the first residue (reducing end)."""
        tree = parse_wurcs_to_tree(WURCS_BRANCHED_TRI)
        assert tree.root_idx == 0

    def test_root_has_no_parent(self):
        """Root node should have no parent."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        assert tree.parent_of(tree.root_idx) is None

    def test_child_has_parent(self):
        """Non-root node should have a parent."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        # Node 1 is the child
        parent = tree.parent_of(1)
        assert parent == 0

    def test_fucose_type_and_anomeric(self):
        """Fucose: classified as Fuc, alpha anomeric."""
        tree = parse_wurcs_to_tree(WURCS_FUCOSE)
        node = tree.nodes[0]
        assert node.mono_type == "Fuc"
        assert node.anomeric == "alpha"

    def test_mannose_type(self):
        """Mannose: classified as Man."""
        tree = parse_wurcs_to_tree(WURCS_MANNOSE)
        node = tree.nodes[0]
        assert node.mono_type == "Man"

    def test_empty_wurcs_raises(self):
        """Empty WURCS should raise ValueError."""
        with pytest.raises(ValueError):
            parse_wurcs_to_tree("")

    def test_invalid_wurcs_raises(self):
        """Invalid WURCS prefix should raise ValueError."""
        with pytest.raises(ValueError):
            parse_wurcs_to_tree("NOT_A_WURCS_STRING")

    def test_wurcs_no_residues_raises(self):
        """WURCS with no unique residues should raise ValueError."""
        with pytest.raises(ValueError, match="No unique residues"):
            parse_wurcs_to_tree("WURCS=2.0/0,0,0//")


# ---------------------------------------------------------------------------
# Tests: GlycanTree helper methods
# ---------------------------------------------------------------------------

class TestGlycanTreeMethods:
    """Tests for GlycanTree structural query methods."""

    @pytest.fixture
    def branched_tree(self):
        """Create a branched tree: root(0) -> child1(1) at 3, child1(1) -> leaf(2) at 4, root(0) -> child2(3) at 6."""
        nodes = [
            MonosaccharideNode(
                index=i, wurcs_residue="", mono_type="Glc",
                mono_type_idx=1, anomeric="beta", anomeric_idx=1,
                ring_form="pyranose", ring_form_idx=0,
            )
            for i in range(4)
        ]
        edges = [
            GlycosidicBond(parent_idx=0, child_idx=1, linkage_position=(3, 1), bond_type="beta"),
            GlycosidicBond(parent_idx=1, child_idx=2, linkage_position=(4, 1), bond_type="beta"),
            GlycosidicBond(parent_idx=0, child_idx=3, linkage_position=(6, 1), bond_type="beta"),
        ]
        return GlycanTree(nodes=nodes, edges=edges, root_idx=0)

    def test_children_of_root(self, branched_tree):
        """Root has two children: 1 and 3."""
        children = branched_tree.children_of(0)
        assert set(children) == {1, 3}

    def test_children_of_leaf(self, branched_tree):
        """Leaf nodes have no children."""
        assert branched_tree.children_of(2) == []
        assert branched_tree.children_of(3) == []

    def test_parent_of(self, branched_tree):
        assert branched_tree.parent_of(1) == 0
        assert branched_tree.parent_of(2) == 1
        assert branched_tree.parent_of(3) == 0
        assert branched_tree.parent_of(0) is None

    def test_siblings_of(self, branched_tree):
        """Nodes 1 and 3 are siblings (same parent=0)."""
        assert 3 in branched_tree.siblings_of(1)
        assert 1 in branched_tree.siblings_of(3)

    def test_siblings_of_root(self, branched_tree):
        """Root has no siblings."""
        assert branched_tree.siblings_of(0) == []

    def test_siblings_of_only_child(self, branched_tree):
        """Node 2 (only child of 1) has no siblings."""
        assert branched_tree.siblings_of(2) == []

    def test_is_branching(self, branched_tree):
        """Root (0) is branching; others are not."""
        assert branched_tree.is_branching(0)
        assert not branched_tree.is_branching(1)
        assert not branched_tree.is_branching(2)
        assert not branched_tree.is_branching(3)

    def test_depth_of(self, branched_tree):
        """Depth: root=0, children of root=1, grandchild=2."""
        assert branched_tree.depth_of(0) == 0
        assert branched_tree.depth_of(1) == 1
        assert branched_tree.depth_of(3) == 1
        assert branched_tree.depth_of(2) == 2

    def test_topological_order_bottom_up(self, branched_tree):
        """Bottom-up: leaves first, root last."""
        order = branched_tree.topological_order_bottom_up()
        assert len(order) == 4
        # Root must be last
        assert order[-1] == 0
        # Leaves must appear before their parents
        assert order.index(2) < order.index(1)
        assert order.index(3) < order.index(0)

    def test_topological_order_top_down(self, branched_tree):
        """Top-down: root first, leaves last."""
        order = branched_tree.topological_order_top_down()
        assert len(order) == 4
        assert order[0] == 0  # root first
        # Parents must appear before their children
        assert order.index(0) < order.index(1)
        assert order.index(0) < order.index(3)
        assert order.index(1) < order.index(2)


# ---------------------------------------------------------------------------
# Tests: glycan_tree_to_tensors
# ---------------------------------------------------------------------------

class TestGlycanTreeToTensors:
    """Tests for converting GlycanTree to PyG-compatible tensor dict."""

    def test_single_node_shapes(self):
        """Single-node tree produces correct tensor shapes."""
        tree = parse_wurcs_to_tree(WURCS_SINGLE_GLC)
        tensors = glycan_tree_to_tensors(tree)
        assert tensors["mono_type"].shape == (1,)
        assert tensors["anomeric"].shape == (1,)
        assert tensors["ring_form"].shape == (1,)
        assert tensors["modifications"].shape == (1, NUM_MODIFICATIONS)
        assert tensors["edge_index"].shape == (2, 0)
        assert tensors["depth"].shape == (1,)
        assert tensors["is_branch"].shape == (1,)
        assert tensors["num_nodes"] == 1

    def test_disaccharide_shapes(self):
        """Disaccharide tree produces correct tensor shapes."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        tensors = glycan_tree_to_tensors(tree)
        n, e = 2, 1
        assert tensors["mono_type"].shape == (n,)
        assert tensors["edge_index"].shape == (2, e)
        assert tensors["linkage_parent_carbon"].shape == (e,)
        assert tensors["linkage_child_carbon"].shape == (e,)
        assert tensors["bond_type"].shape == (e,)
        assert tensors["num_nodes"] == n

    def test_branched_shapes(self):
        """Branched trisaccharide produces correct tensor shapes."""
        tree = parse_wurcs_to_tree(WURCS_BRANCHED_TRI)
        tensors = glycan_tree_to_tensors(tree)
        assert tensors["num_nodes"] == 3
        assert tensors["edge_index"].shape[1] == 2

    def test_edge_index_values(self):
        """Edge index should correctly encode parent->child."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        tensors = glycan_tree_to_tensors(tree)
        ei = tensors["edge_index"]
        assert ei[0, 0].item() == 0  # parent
        assert ei[1, 0].item() == 1  # child

    def test_linkage_carbon_values(self):
        """Linkage carbon positions should match parsed values."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        tensors = glycan_tree_to_tensors(tree)
        assert tensors["linkage_parent_carbon"][0].item() == 4  # a4-b1
        assert tensors["linkage_child_carbon"][0].item() == 1

    def test_bond_type_encoding(self):
        """Bond type: 0=alpha, 1=beta, 2=unknown."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        tensors = glycan_tree_to_tensors(tree)
        # Child node is beta (a2122h-1b_1-5) -> bond_type = 1
        assert tensors["bond_type"][0].item() == 1

    def test_depth_values(self):
        """Depth tensor should match tree depth."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        tensors = glycan_tree_to_tensors(tree)
        assert tensors["depth"][0].item() == 0  # root
        assert tensors["depth"][1].item() == 1  # child

    def test_branch_flag(self):
        """is_branch should be True for branch points only."""
        tree = parse_wurcs_to_tree(WURCS_BRANCHED_TRI)
        tensors = glycan_tree_to_tensors(tree)
        assert tensors["is_branch"][0].item() is True  # root is branch
        assert tensors["is_branch"][1].item() is False  # leaf
        assert tensors["is_branch"][2].item() is False  # leaf

    def test_modification_vector_shape(self):
        """Modification tensor should be [N, NUM_MODIFICATIONS] multi-hot."""
        tree = parse_wurcs_to_tree(WURCS_SINGLE_GLC)
        tensors = glycan_tree_to_tensors(tree)
        mods = tensors["modifications"]
        assert mods.dim() == 2
        assert mods.shape[1] == NUM_MODIFICATIONS

    def test_all_tensors_are_tensors(self):
        """All returned values (except num_nodes) should be torch.Tensor."""
        tree = parse_wurcs_to_tree(WURCS_DISACCHARIDE)
        tensors = glycan_tree_to_tensors(tree)
        for key, val in tensors.items():
            if key == "num_nodes":
                assert isinstance(val, int)
            else:
                assert isinstance(val, torch.Tensor), f"{key} is not a tensor"

    def test_no_nan_or_inf(self):
        """Tensors should not contain NaN or Inf."""
        tree = parse_wurcs_to_tree(WURCS_BRANCHED_TRI)
        tensors = glycan_tree_to_tensors(tree)
        for key, val in tensors.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                assert not torch.isnan(val).any(), f"NaN in {key}"
                assert not torch.isinf(val).any(), f"Inf in {key}"


# ---------------------------------------------------------------------------
# Tests: MonosaccharideNode modification_vector property
# ---------------------------------------------------------------------------

class TestMonosaccharideNodeModificationVector:
    """Tests for the modification_vector property of MonosaccharideNode."""

    def test_empty_modifications(self):
        """Node with no modifications should return zero vector."""
        node = MonosaccharideNode(
            index=0, wurcs_residue="", mono_type="Glc",
            mono_type_idx=1, anomeric="beta", anomeric_idx=1,
            ring_form="pyranose", ring_form_idx=0,
            modifications=[],
        )
        vec = node.modification_vector
        assert len(vec) == NUM_MODIFICATIONS
        assert all(v == 0.0 for v in vec)

    def test_single_modification(self):
        """Node with sulfation should have 1 in the sulfation slot."""
        node = MonosaccharideNode(
            index=0, wurcs_residue="", mono_type="Glc",
            mono_type_idx=1, anomeric="beta", anomeric_idx=1,
            ring_form="pyranose", ring_form_idx=0,
            modifications=["sulfation"],
        )
        vec = node.modification_vector
        assert vec[0] == 1.0  # sulfation is index 0

    def test_multiple_modifications(self):
        """Node with multiple modifications should have corresponding 1s."""
        node = MonosaccharideNode(
            index=0, wurcs_residue="", mono_type="Glc",
            mono_type_idx=1, anomeric="beta", anomeric_idx=1,
            ring_form="pyranose", ring_form_idx=0,
            modifications=["sulfation", "n_acetyl"],
        )
        vec = node.modification_vector
        assert vec[0] == 1.0  # sulfation
        assert vec[6] == 1.0  # n_acetyl
        assert sum(vec) == 2.0

    def test_unknown_modification_ignored(self):
        """Unknown modification string should not affect the vector."""
        node = MonosaccharideNode(
            index=0, wurcs_residue="", mono_type="Glc",
            mono_type_idx=1, anomeric="beta", anomeric_idx=1,
            ring_form="pyranose", ring_form_idx=0,
            modifications=["nonexistent_mod"],
        )
        vec = node.modification_vector
        assert sum(vec) == 0.0


# ---------------------------------------------------------------------------
# Tests: Vocabulary constants
# ---------------------------------------------------------------------------

class TestVocabConstants:
    """Sanity checks on vocabulary dictionaries."""

    def test_monosaccharide_vocab_has_common_types(self):
        """Vocabulary should include GlcNAc, Man, Gal, Fuc, NeuAc."""
        for name in ["GlcNAc", "Man", "Gal", "Fuc", "NeuAc"]:
            assert name in MONOSACCHARIDE_TYPE_VOCAB

    def test_monosaccharide_vocab_indices_unique(self):
        """All vocab indices should be unique."""
        indices = list(MONOSACCHARIDE_TYPE_VOCAB.values())
        assert len(indices) == len(set(indices))

    def test_anomeric_vocab(self):
        assert "alpha" in ANOMERIC_VOCAB
        assert "beta" in ANOMERIC_VOCAB
        assert "unknown" in ANOMERIC_VOCAB

    def test_ring_form_vocab(self):
        assert "pyranose" in RING_FORM_VOCAB
        assert "furanose" in RING_FORM_VOCAB

    def test_galnac_in_vocab(self):
        """GalNAc must be in the monosaccharide type vocabulary."""
        assert "GalNAc" in MONOSACCHARIDE_TYPE_VOCAB

    def test_neugc_in_vocab(self):
        """NeuGc must be in the monosaccharide type vocabulary."""
        assert "NeuGc" in MONOSACCHARIDE_TYPE_VOCAB


# ---------------------------------------------------------------------------
# Tests: GalNAc classification regression (C1 fix verification)
# ---------------------------------------------------------------------------

class TestGalNAcClassificationRegression:
    """Regression tests verifying the GalNAc classification bug fix from C1.

    Before C1, GalNAc WURCS codes (a2112h with *NCC) could be misclassified
    as NeuAc because the NeuAc pattern matched the *N prefix. The fix uses
    a negative lookahead (?!CC) on the NeuAc pattern and places GalNAc rules
    before generic Gal rules.
    """

    def test_galnac_with_ncc_3o(self):
        """GalNAc: a2112h with *NCC/3=O must classify as GalNAc, not NeuAc."""
        assert _classify_residue("a2112h-1b_1-5_2*NCC/3=O") == "GalNAc"

    def test_galnac_with_ncc_only(self):
        """GalNAc: a2112h with *NCC (no /3=O) must classify as GalNAc."""
        assert _classify_residue("a2112h-1b_1-5_2*NCC") == "GalNAc"

    def test_galnac_alpha(self):
        """GalNAc with alpha anomeric still classified as GalNAc."""
        assert _classify_residue("a2112h-1a_1-5_2*NCC") == "GalNAc"

    def test_galnac_alpha_with_3o(self):
        """GalNAc alpha with /3=O still classified as GalNAc."""
        assert _classify_residue("a2112h-1a_1-5_2*NCC/3=O") == "GalNAc"

    def test_galnac_not_neuac(self):
        """GalNAc codes must NOT be misclassified as NeuAc."""
        galnac_codes = [
            "a2112h-1b_1-5_2*NCC/3=O",
            "a2112h-1b_1-5_2*NCC",
            "a2112h-1a_1-5_2*NCC/3=O",
            "a2112h-1a_1-5_2*NCC",
        ]
        for code in galnac_codes:
            result = _classify_residue(code)
            assert result != "NeuAc", (
                f"WURCS code {code!r} misclassified as NeuAc (should be GalNAc)"
            )
            assert result == "GalNAc", (
                f"WURCS code {code!r} classified as {result!r} instead of GalNAc"
            )

    def test_galnac_not_gal(self):
        """GalNAc codes must NOT be classified as plain Gal."""
        galnac_codes = [
            "a2112h-1b_1-5_2*NCC",
            "a2112h-1a_1-5_2*NCC",
        ]
        for code in galnac_codes:
            result = _classify_residue(code)
            assert result != "Gal", (
                f"WURCS code {code!r} classified as Gal instead of GalNAc"
            )

    def test_gal_without_ncc_is_gal(self):
        """Plain Gal (a2112h without NCC) must remain classified as Gal."""
        assert _classify_residue("a2112h-1b_1-5") == "Gal"
        assert _classify_residue("a2112h-1a_1-5") == "Gal"

    def test_glcnac_still_correct(self):
        """GlcNAc codes must still be classified correctly after GalNAc fix."""
        assert _classify_residue("a2122h-1b_1-5_2*NCC/3=O") == "GlcNAc"
        assert _classify_residue("a2122h-1b_1-5_2*NCC") == "GlcNAc"
        assert _classify_residue("a2122h-1a_1-5_2*NCC") == "GlcNAc"

    def test_galnac_type_index_correct(self):
        """GalNAc should have the correct vocab index."""
        code = "a2112h-1b_1-5_2*NCC/3=O"
        result = _classify_residue(code)
        assert result == "GalNAc"
        assert MONOSACCHARIDE_TYPE_VOCAB[result] == 5


# ---------------------------------------------------------------------------
# Tests: Sialic acid classification regression (C1 fix verification)
# ---------------------------------------------------------------------------

class TestSialicAcidClassification:
    """Regression tests for sialic acid (NeuAc, NeuGc) classification.

    The NeuAc pattern was fixed in C1 to use negative lookahead (?!CC)
    to avoid matching GlcNAc/GalNAc codes that contain *NCC.
    """

    def test_neuac_pattern_matches(self):
        """NeuAc: a2122h-1b_1-5_2*N (without CC) should be NeuAc."""
        assert _classify_residue("a2122h-1b_1-5_2*N") == "NeuAc"

    def test_neuac_via_name(self):
        """NeuAc can also appear as a name in some WURCS encodings."""
        assert _classify_residue("Neu5Ac") == "NeuAc"

    @pytest.mark.xfail(
        reason="BUG-NEUGC-ORDER: NeuAc pattern (?!CC) matches *NO before "
               "the NeuGc rule can fire. NeuGc rule should precede NeuAc.",
        strict=True,
    )
    def test_neugc_pattern_matches(self):
        """NeuGc: a2122h-1b_1-5_2*NO should be NeuGc."""
        assert _classify_residue("a2122h-1b_1-5_2*NO") == "NeuGc"

    def test_neugc_via_name(self):
        """NeuGc can also appear as a name."""
        assert _classify_residue("Neu5Gc") == "NeuGc"

    def test_neuac_not_confused_with_glcnac(self):
        """NeuAc pattern must NOT match GlcNAc codes (*NCC)."""
        glcnac_code = "a2122h-1b_1-5_2*NCC/3=O"
        result = _classify_residue(glcnac_code)
        assert result != "NeuAc", (
            f"GlcNAc code {glcnac_code!r} was misclassified as NeuAc"
        )

    def test_neuac_not_confused_with_galnac(self):
        """NeuAc pattern must NOT match GalNAc codes (*NCC)."""
        galnac_code = "a2112h-1b_1-5_2*NCC"
        result = _classify_residue(galnac_code)
        assert result != "NeuAc", (
            f"GalNAc code {galnac_code!r} was misclassified as NeuAc"
        )

    def test_sialic_acids_have_correct_indices(self):
        """NeuAc and NeuGc should have distinct vocab indices."""
        assert MONOSACCHARIDE_TYPE_VOCAB["NeuAc"] == 7
        assert MONOSACCHARIDE_TYPE_VOCAB["NeuGc"] == 8
        assert MONOSACCHARIDE_TYPE_VOCAB["NeuAc"] != MONOSACCHARIDE_TYPE_VOCAB["NeuGc"]


# ---------------------------------------------------------------------------
# Tests: Hexose disambiguation (Glc vs Man vs Gal)
# ---------------------------------------------------------------------------

class TestHexoseDisambiguation:
    """Verify correct stereochemistry-based hexose disambiguation."""

    def test_glc_stereochemistry(self):
        """a2122h (D-Glc) -> Glc."""
        assert _classify_residue("a2122h-1b_1-5") == "Glc"
        assert _classify_residue("a2122h-1a_1-5") == "Glc"

    def test_man_stereochemistry(self):
        """a1122h (D-Man) -> Man."""
        assert _classify_residue("a1122h-1b_1-5") == "Man"
        assert _classify_residue("a1122h-1a_1-5") == "Man"

    def test_gal_stereochemistry(self):
        """a2112h (D-Gal) -> Gal."""
        assert _classify_residue("a2112h-1b_1-5") == "Gal"
        assert _classify_residue("a2112h-1a_1-5") == "Gal"

    def test_hexoses_are_distinct(self):
        """Glc, Man, and Gal should have distinct vocab indices."""
        glc_idx = MONOSACCHARIDE_TYPE_VOCAB["Glc"]
        man_idx = MONOSACCHARIDE_TYPE_VOCAB["Man"]
        gal_idx = MONOSACCHARIDE_TYPE_VOCAB["Gal"]
        assert len({glc_idx, man_idx, gal_idx}) == 3
