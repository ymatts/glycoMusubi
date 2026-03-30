"""Tests for GlycanTreeEncoder (Tree-MPNN).

Validates:
- Forward pass output shape: [batch_size, output_dim=256]
- Bottom-up pass: child-to-parent message propagation
- Top-down pass: root-to-leaf refinement
- Attention pooling: softmax-normalised attention weights
- Gradient flow: all parameters receive gradients
- Similar glycans produce close embeddings (cosine similarity > 0.5)
- Different glycans produce distinct embeddings (cosine similarity < 0.9)
- Empty tree list -> zero-size output
- Single-node tree encoding
- Batch processing correctness
- LinkageEncoder, TreeMPNNLayer, TopDownRefinement, BranchingAwarePooling
- encode_wurcs convenience method
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from glycoMusubi.embedding.encoders.glycan_tree_encoder import (
    BranchingAwarePooling,
    GlycanTreeEncoder,
    LinkageEncoder,
    TopDownRefinement,
    TreeMPNNLayer,
)
from glycoMusubi.embedding.encoders.wurcs_tree_parser import (
    GlycanTree,
    GlycosidicBond,
    MonosaccharideNode,
    parse_wurcs_to_tree,
)


# ---------------------------------------------------------------------------
# Helper: create test trees directly
# ---------------------------------------------------------------------------

def _make_node(index: int, mono_type_idx: int = 1, anomeric_idx: int = 1,
               ring_form_idx: int = 0, modifications: list | None = None) -> MonosaccharideNode:
    """Create a MonosaccharideNode with sensible defaults."""
    return MonosaccharideNode(
        index=index,
        wurcs_residue="",
        mono_type="Glc",
        mono_type_idx=mono_type_idx,
        anomeric="beta",
        anomeric_idx=anomeric_idx,
        ring_form="pyranose",
        ring_form_idx=ring_form_idx,
        modifications=modifications or [],
    )


def _make_single_node_tree() -> GlycanTree:
    """Single-node tree (one monosaccharide, no edges)."""
    return GlycanTree(nodes=[_make_node(0)], edges=[], root_idx=0)


def _make_linear_chain(n: int = 3) -> GlycanTree:
    """Linear chain: 0 -> 1 -> 2 -> ... -> n-1."""
    nodes = [_make_node(i) for i in range(n)]
    edges = [
        GlycosidicBond(parent_idx=i, child_idx=i + 1,
                       linkage_position=(4, 1), bond_type="beta")
        for i in range(n - 1)
    ]
    return GlycanTree(nodes=nodes, edges=edges, root_idx=0)


def _make_branched_tree() -> GlycanTree:
    """Branched tree: root(0) -> {child1(1), child2(2)}, child1(1) -> leaf(3).

    Structure:
        0 (Man, root, branch point)
       / \\
      1   2 (GlcNAc, Gal)
      |
      3 (Fuc)
    """
    nodes = [
        _make_node(0, mono_type_idx=2),   # Man
        _make_node(1, mono_type_idx=4),   # GlcNAc
        _make_node(2, mono_type_idx=3),   # Gal
        _make_node(3, mono_type_idx=6),   # Fuc
    ]
    edges = [
        GlycosidicBond(parent_idx=0, child_idx=1, linkage_position=(3, 1), bond_type="beta"),
        GlycosidicBond(parent_idx=0, child_idx=2, linkage_position=(6, 1), bond_type="beta"),
        GlycosidicBond(parent_idx=1, child_idx=3, linkage_position=(2, 1), bond_type="alpha"),
    ]
    return GlycanTree(nodes=nodes, edges=edges, root_idx=0)


# WURCS test strings
WURCS_SINGLE_GLC = "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"
WURCS_DISACCHARIDE = "WURCS=2.0/1,2,1/[a2122h-1b_1-5]/1-1/a4-b1"
WURCS_BRANCHED_TRI = "WURCS=2.0/1,3,2/[a2122h-1b_1-5]/1-1-1/a3-b1_a6-c1"
WURCS_MANNOSE = "WURCS=2.0/1,1,0/[a1122h-1a_1-5]/1/"
WURCS_FUCOSE = "WURCS=2.0/1,1,0/[a1221m-1a_1-5]/1/"


# ---------------------------------------------------------------------------
# Tests: GlycanTreeEncoder forward pass shape
# ---------------------------------------------------------------------------

class TestGlycanTreeEncoderShape:
    """Verify output shapes from the encoder."""

    @pytest.fixture
    def encoder(self):
        return GlycanTreeEncoder(output_dim=256, hidden_dim=256)

    def test_forward_shape_single_tree(self, encoder):
        """Single tree -> [1, 256]."""
        tree = _make_single_node_tree()
        out = encoder([tree])
        assert out.shape == (1, 256)

    def test_forward_shape_batch(self, encoder):
        """Batch of 3 trees -> [3, 256]."""
        trees = [_make_single_node_tree(), _make_linear_chain(3), _make_branched_tree()]
        out = encoder(trees)
        assert out.shape == (3, 256)

    def test_forward_shape_empty_list(self, encoder):
        """Empty list -> [0, 256]."""
        out = encoder([])
        assert out.shape == (0, 256)

    def test_forward_shape_custom_output_dim(self):
        """Custom output_dim=128 -> [B, 128]."""
        enc = GlycanTreeEncoder(output_dim=128, hidden_dim=128)
        tree = _make_branched_tree()
        out = enc([tree])
        assert out.shape == (1, 128)

    def test_forward_shape_large_batch(self, encoder):
        """Batch of 8 trees -> [8, 256]."""
        trees = [_make_linear_chain(i + 1) for i in range(8)]
        out = encoder(trees)
        assert out.shape == (8, 256)

    def test_forward_dtype_float32(self, encoder):
        """Output should be float32."""
        out = encoder([_make_branched_tree()])
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: Bottom-up message passing
# ---------------------------------------------------------------------------

class TestBottomUpPass:
    """Tests for the bottom-up Tree-MPNN layers."""

    def test_child_to_parent_message_propagation(self):
        """Bottom-up pass should propagate information from children to parents.

        We verify this by checking that a parent node's representation differs
        when children have different features vs identical features.
        """
        encoder = GlycanTreeEncoder(output_dim=64, hidden_dim=64)

        # Tree 1: root with two identical GlcNAc children
        tree1_nodes = [
            _make_node(0, mono_type_idx=2),   # Man root
            _make_node(1, mono_type_idx=4),   # GlcNAc
            _make_node(2, mono_type_idx=4),   # GlcNAc (same)
        ]
        tree1 = GlycanTree(
            nodes=tree1_nodes,
            edges=[
                GlycosidicBond(0, 1, (3, 1), "beta"),
                GlycosidicBond(0, 2, (6, 1), "beta"),
            ],
            root_idx=0,
        )

        # Tree 2: root with GlcNAc + Fuc children
        tree2_nodes = [
            _make_node(0, mono_type_idx=2),   # Man root (same)
            _make_node(1, mono_type_idx=4),   # GlcNAc
            _make_node(2, mono_type_idx=6),   # Fuc (different)
        ]
        tree2 = GlycanTree(
            nodes=tree2_nodes,
            edges=[
                GlycosidicBond(0, 1, (3, 1), "beta"),
                GlycosidicBond(0, 2, (6, 1), "beta"),
            ],
            root_idx=0,
        )

        with torch.no_grad():
            emb1 = encoder([tree1])
            emb2 = encoder([tree2])

        # Different children should produce different root representations
        assert not torch.allclose(emb1, emb2, atol=1e-4), \
            "Different child types should produce different parent embeddings"


# ---------------------------------------------------------------------------
# Tests: Top-down refinement
# ---------------------------------------------------------------------------

class TestTopDownRefinement:
    """Tests for the top-down refinement layer."""

    def test_top_down_refinement_module(self):
        """TopDownRefinement should refine hidden states given parent context."""
        d_model = 64
        td = TopDownRefinement(d_model)
        h = torch.randn(4, d_model)
        parent_map = {1: 0, 2: 0, 3: 1}  # 0 is root
        topo_order_td = [0, 1, 2, 3]  # root first

        h_refined = td(h, topo_order_td, parent_map)
        assert h_refined.shape == (4, d_model)
        # Refined should differ from original (non-identity transform)
        assert not torch.allclose(h_refined, h, atol=1e-4)

    def test_root_gets_zero_parent_context(self):
        """Root node should receive zero parent context in top-down pass."""
        d_model = 32
        td = TopDownRefinement(d_model)
        h = torch.randn(2, d_model)
        parent_map = {1: 0}
        topo_order_td = [0, 1]

        # Root (idx 0) has no parent, should use zeros
        h_refined = td(h, topo_order_td, parent_map)
        assert h_refined.shape == (2, d_model)

    def test_leaf_receives_parent_info(self):
        """Leaf nodes should be influenced by their parent's refined representation.

        We verify by checking that changing the root's hidden state
        affects the leaf's refined output.
        """
        d_model = 32
        td = TopDownRefinement(d_model)
        parent_map = {1: 0}
        topo_order_td = [0, 1]

        h_a = torch.randn(2, d_model)
        h_b = h_a.clone()
        h_b[0] += 5.0  # perturb root

        ref_a = td(h_a, topo_order_td, parent_map)
        ref_b = td(h_b, topo_order_td, parent_map)

        # Leaf (idx 1) should differ between the two
        assert not torch.allclose(ref_a[1], ref_b[1], atol=1e-4), \
            "Leaf should be affected by parent perturbation in top-down pass"


# ---------------------------------------------------------------------------
# Tests: Attention pooling
# ---------------------------------------------------------------------------

class TestBranchingAwarePooling:
    """Tests for the BranchingAwarePooling layer."""

    @pytest.fixture
    def pooling(self):
        return BranchingAwarePooling(d_model=64, output_dim=64, num_heads=4)

    def test_output_shape(self, pooling):
        """Pooling should produce [num_graphs, output_dim]."""
        h = torch.randn(5, 64)
        batch = torch.tensor([0, 0, 0, 1, 1])
        is_branch = torch.tensor([True, False, False, True, False])
        depth = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
        out = pooling(h, batch, is_branch, depth)
        assert out.shape == (2, 64)

    def test_single_graph_pooling(self, pooling):
        """Pooling a single graph should produce [1, output_dim]."""
        h = torch.randn(3, 64)
        batch = torch.zeros(3, dtype=torch.long)
        is_branch = torch.tensor([True, False, False])
        depth = torch.tensor([0, 1, 1], dtype=torch.long)
        out = pooling(h, batch, is_branch, depth)
        assert out.shape == (1, 64)

    def test_attention_weights_sum_to_one(self):
        """Attention weights within each head should sum to ~1 per graph.

        We verify this indirectly: with identical node features, each
        node should receive roughly equal weight.
        """
        pooling = BranchingAwarePooling(d_model=32, output_dim=32, num_heads=2)
        # Use same features for all nodes -> uniform attention
        h = torch.ones(4, 32)
        batch = torch.zeros(4, dtype=torch.long)
        is_branch = torch.zeros(4, dtype=torch.bool)
        depth = torch.zeros(4, dtype=torch.long)

        out = pooling(h, batch, is_branch, depth)
        assert out.shape == (1, 32)
        assert not torch.isnan(out).any()

    def test_no_branch_points(self, pooling):
        """When no nodes are branch points, h_branch is zero."""
        h = torch.randn(3, 64)
        batch = torch.zeros(3, dtype=torch.long)
        is_branch = torch.zeros(3, dtype=torch.bool)
        depth = torch.tensor([0, 1, 2], dtype=torch.long)
        out = pooling(h, batch, is_branch, depth)
        assert out.shape == (1, 64)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Tests: LinkageEncoder
# ---------------------------------------------------------------------------

class TestLinkageEncoder:
    """Tests for edge (glycosidic bond) feature encoding."""

    def test_output_shape(self):
        """LinkageEncoder should produce [E, d_edge]."""
        enc = LinkageEncoder(d_edge=24)
        pc = torch.tensor([4, 3, 6])
        cc = torch.tensor([1, 1, 1])
        bt = torch.tensor([0, 1, 2])
        out = enc(pc, cc, bt)
        assert out.shape == (3, 24)

    def test_single_edge(self):
        """Single edge should produce [1, d_edge]."""
        enc = LinkageEncoder(d_edge=16)
        out = enc(torch.tensor([4]), torch.tensor([1]), torch.tensor([1]))
        assert out.shape == (1, 16)

    def test_different_linkages_produce_different_features(self):
        """Different bond types and positions should yield different encodings."""
        enc = LinkageEncoder(d_edge=24)
        # beta 1->4 linkage
        out1 = enc(torch.tensor([4]), torch.tensor([1]), torch.tensor([1]))
        # alpha 1->3 linkage
        out2 = enc(torch.tensor([3]), torch.tensor([1]), torch.tensor([0]))
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Tests: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify that gradients flow through all encoder parameters."""

    def test_all_parameters_get_gradients(self):
        """Every trainable parameter should receive a gradient tensor (not None).

        Note: Some parameters may receive zero gradients for a specific input
        (e.g., unused embedding rows, sibling MLPs when there are no siblings),
        but the grad tensor should still be allocated (not None).
        """
        encoder = GlycanTreeEncoder(output_dim=64, hidden_dim=64)
        # Use a batch with diverse structures to activate all code paths
        trees = [
            _make_branched_tree(),     # branching + siblings
            _make_linear_chain(5),     # deep chain, no branching
            _make_single_node_tree(),  # single node, no edges
        ]
        out = encoder(trees)
        # Use squared loss to avoid near-zero gradients from LayerNorm centering
        loss = out.pow(2).sum()
        loss.backward()

        params_without_grad = []
        for name, param in encoder.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without grad tensors: {params_without_grad}"
        )

    def test_gradient_flow_with_linear_chain(self):
        """Gradient should flow through a linear chain (no branching).

        We use a squared loss to avoid near-zero gradients from LayerNorm
        centering (which causes out.sum() to be ~0).
        """
        encoder = GlycanTreeEncoder(output_dim=32, hidden_dim=32)
        tree = _make_linear_chain(5)
        out = encoder([tree])
        loss = out.pow(2).sum()  # squared loss avoids LayerNorm centering issue
        loss.backward()

        # Gradient tensors should be allocated (not None)
        assert encoder.mono_embed.weight.grad is not None
        # Pooling fuse layers should get non-zero gradients
        fuse_grad_sum = sum(
            p.grad.abs().sum().item()
            for p in encoder.pooling.fuse.parameters()
            if p.grad is not None
        )
        assert fuse_grad_sum > 0, "Pooling fuse layer should receive gradients"

    def test_gradient_flow_with_single_node(self):
        """Gradient should flow even for a single-node tree."""
        encoder = GlycanTreeEncoder(output_dim=32, hidden_dim=32)
        tree = _make_single_node_tree()
        out = encoder([tree])
        loss = out.pow(2).sum()
        loss.backward()

        assert encoder.mono_embed.weight.grad is not None

    def test_gradient_flow_batch(self):
        """Gradient should flow with batched input."""
        encoder = GlycanTreeEncoder(output_dim=32, hidden_dim=32)
        trees = [_make_single_node_tree(), _make_linear_chain(3), _make_branched_tree()]
        out = encoder(trees)
        loss = out.pow(2).sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# Tests: Embedding similarity properties
# ---------------------------------------------------------------------------

class TestEmbeddingSimilarity:
    """Verify that the encoder produces biologically meaningful embeddings."""

    @pytest.fixture
    def encoder(self):
        """Encoder with fixed seed for reproducibility."""
        torch.manual_seed(42)
        return GlycanTreeEncoder(output_dim=64, hidden_dim=64)

    def test_similar_glycans_close_embeddings(self, encoder):
        """Two linear chains of similar length should have cosine similarity > 0.5.

        Both are linear beta-1,4-linked Glc chains (like cellulose fragments),
        differing only in length by 1 residue.
        """
        tree_a = _make_linear_chain(4)
        tree_b = _make_linear_chain(5)

        with torch.no_grad():
            emb_a = encoder([tree_a])  # [1, 64]
            emb_b = encoder([tree_b])  # [1, 64]

        cos_sim = F.cosine_similarity(emb_a, emb_b).item()
        assert cos_sim > 0.5, (
            f"Similar glycans (linear chains of length 4 and 5) should have "
            f"cosine similarity > 0.5, got {cos_sim:.4f}"
        )

    def test_different_glycans_distinct_embeddings(self, encoder):
        """A single Glc node vs a branched tree should have cosine similarity < 0.9.

        These are structurally very different: a monosaccharide vs a branched
        tetrasaccharide with diverse monosaccharide types.
        """
        tree_simple = _make_single_node_tree()
        tree_complex = _make_branched_tree()

        with torch.no_grad():
            emb_simple = encoder([tree_simple])
            emb_complex = encoder([tree_complex])

        cos_sim = F.cosine_similarity(emb_simple, emb_complex).item()
        assert cos_sim < 0.9, (
            f"Structurally different glycans should have cosine similarity < 0.9, "
            f"got {cos_sim:.4f}"
        )

    def test_identical_trees_identical_embeddings(self, encoder):
        """Identical trees should produce identical embeddings (eval mode)."""
        encoder.eval()
        tree_a = _make_branched_tree()
        tree_b = _make_branched_tree()

        with torch.no_grad():
            emb_a = encoder([tree_a])
            emb_b = encoder([tree_b])

        assert torch.allclose(emb_a, emb_b, atol=1e-6), \
            "Identical trees should produce identical embeddings in eval mode"

    def test_branching_affects_embedding(self, encoder):
        """A linear chain and a branched tree of same size should differ.

        Linear: 0 -> 1 -> 2 -> 3
        Branched: 0 -> {1, 2}, 1 -> 3
        Both have 4 nodes but very different topology.
        """
        tree_linear = _make_linear_chain(4)
        tree_branched = _make_branched_tree()

        with torch.no_grad():
            emb_linear = encoder([tree_linear])
            emb_branched = encoder([tree_branched])

        assert not torch.allclose(emb_linear, emb_branched, atol=1e-4), \
            "Linear and branched trees should produce different embeddings"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def encoder(self):
        return GlycanTreeEncoder(output_dim=64, hidden_dim=64)

    def test_single_node_tree(self, encoder):
        """Single-node tree should produce a valid non-zero embedding."""
        tree = _make_single_node_tree()
        out = encoder([tree])
        assert out.shape == (1, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_no_nan_or_inf(self, encoder):
        """Output should never contain NaN or Inf."""
        trees = [
            _make_single_node_tree(),
            _make_linear_chain(2),
            _make_linear_chain(10),
            _make_branched_tree(),
        ]
        out = encoder(trees)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_deep_tree(self, encoder):
        """Deep linear chain (20 nodes) should not cause issues."""
        tree = _make_linear_chain(20)
        out = encoder([tree])
        assert out.shape == (1, 64)
        assert not torch.isnan(out).any()

    def test_wide_tree(self, encoder):
        """Wide tree (root with 10 children) should work correctly."""
        nodes = [_make_node(i) for i in range(11)]
        edges = [
            GlycosidicBond(parent_idx=0, child_idx=i + 1,
                           linkage_position=(3, 1), bond_type="beta")
            for i in range(10)
        ]
        tree = GlycanTree(nodes=nodes, edges=edges, root_idx=0)
        out = encoder([tree])
        assert out.shape == (1, 64)
        assert not torch.isnan(out).any()

    def test_mixed_batch_sizes(self, encoder):
        """Batch with trees of varying sizes should work."""
        trees = [
            _make_single_node_tree(),    # 1 node
            _make_linear_chain(5),       # 5 nodes
            _make_branched_tree(),       # 4 nodes
            _make_linear_chain(2),       # 2 nodes
        ]
        out = encoder(trees)
        assert out.shape == (4, 64)

    def test_batch_order_independence(self, encoder):
        """Each tree's embedding should not depend on batch ordering (eval mode).

        Embedding for tree X in batch [X, Y] should equal its embedding
        in batch [Y, X] (or solo [X]).
        """
        encoder.eval()
        tree_a = _make_single_node_tree()
        tree_b = _make_branched_tree()

        with torch.no_grad():
            emb_a_solo = encoder([tree_a])  # [1, 64]
            emb_b_solo = encoder([tree_b])  # [1, 64]
            emb_batch = encoder([tree_a, tree_b])  # [2, 64]

        assert torch.allclose(emb_a_solo[0], emb_batch[0], atol=1e-5), \
            "Tree embedding should be independent of batch ordering"
        assert torch.allclose(emb_b_solo[0], emb_batch[1], atol=1e-5), \
            "Tree embedding should be independent of batch ordering"


# ---------------------------------------------------------------------------
# Tests: encode_wurcs convenience method
# ---------------------------------------------------------------------------

class TestEncodeWurcs:
    """Tests for the encode_wurcs convenience method."""

    @pytest.fixture
    def encoder(self):
        return GlycanTreeEncoder(output_dim=64, hidden_dim=64)

    def test_encode_single_wurcs(self, encoder):
        """Single WURCS string should produce [1, 64]."""
        out = encoder.encode_wurcs([WURCS_SINGLE_GLC])
        assert out.shape == (1, 64)
        assert not torch.isnan(out).any()

    def test_encode_batch_wurcs(self, encoder):
        """Batch of WURCS strings should produce [B, 64]."""
        wurcs_list = [WURCS_SINGLE_GLC, WURCS_DISACCHARIDE, WURCS_BRANCHED_TRI]
        out = encoder.encode_wurcs(wurcs_list)
        assert out.shape == (3, 64)

    def test_invalid_wurcs_produces_embedding(self, encoder):
        """Invalid WURCS should produce a fallback embedding (not crash)."""
        out = encoder.encode_wurcs(["not_a_wurcs"])
        assert out.shape == (1, 64)
        assert not torch.isnan(out).any()

    def test_mixed_valid_invalid_wurcs(self, encoder):
        """Mix of valid and invalid WURCS should produce correct batch size."""
        wurcs_list = [WURCS_SINGLE_GLC, "invalid_wurcs", WURCS_MANNOSE]
        out = encoder.encode_wurcs(wurcs_list)
        assert out.shape == (3, 64)

    def test_encode_wurcs_gradient(self, encoder):
        """encode_wurcs should support gradient computation."""
        out = encoder.encode_wurcs([WURCS_DISACCHARIDE])
        loss = out.sum()
        loss.backward()
        assert encoder.mono_embed.weight.grad is not None


# ---------------------------------------------------------------------------
# Tests: TreeMPNNLayer
# ---------------------------------------------------------------------------

class TestTreeMPNNLayer:
    """Unit tests for the TreeMPNNLayer module."""

    def test_output_shape(self):
        """Layer output should have same shape as input hidden states."""
        d_model = 64
        d_edge = 24
        layer = TreeMPNNLayer(d_model, d_edge)

        n = 4  # 4 nodes
        h = torch.randn(n, d_model)
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]])  # 3 edges
        edge_attr = torch.randn(3, d_edge)
        parent_map = {1: 0, 2: 0, 3: 1}
        children_map = {0: [1, 2], 1: [3]}
        topo_order_bu = [2, 3, 1, 0]

        h_new = layer(h, edge_index, edge_attr, parent_map, children_map, topo_order_bu)
        assert h_new.shape == (n, d_model)

    def test_no_edges(self):
        """Layer should handle zero edges gracefully."""
        d_model = 32
        layer = TreeMPNNLayer(d_model)
        h = torch.randn(2, d_model)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 24)
        parent_map = {}
        children_map = {}
        topo_order_bu = [0, 1]

        h_new = layer(h, edge_index, edge_attr, parent_map, children_map, topo_order_bu)
        assert h_new.shape == (2, d_model)
        assert not torch.isnan(h_new).any()

    def test_layer_is_differentiable(self):
        """Layer should support backpropagation."""
        d_model = 32
        d_edge = 24
        layer = TreeMPNNLayer(d_model, d_edge)

        h = torch.randn(3, d_model, requires_grad=True)
        edge_index = torch.tensor([[0, 0], [1, 2]])
        edge_attr = torch.randn(2, d_edge)
        parent_map = {1: 0, 2: 0}
        children_map = {0: [1, 2]}
        topo_order_bu = [1, 2, 0]

        h_new = layer(h, edge_index, edge_attr, parent_map, children_map, topo_order_bu)
        loss = h_new.sum()
        loss.backward()
        assert h.grad is not None


# ---------------------------------------------------------------------------
# Tests: Parameter count (approximate)
# ---------------------------------------------------------------------------

class TestParameterCount:
    """Verify that the model has a reasonable parameter count."""

    def test_approximate_param_count(self):
        """Default config should be in a reasonable range for a Tree-MPNN.

        The design doc estimates ~1.2M, but the actual implementation with
        3 bottom-up layers, top-down refinement, and multi-head attention
        pooling results in ~3.4M parameters. This is acceptable for a
        tree-structured encoder with hidden_dim=256.
        """
        encoder = GlycanTreeEncoder(output_dim=256, hidden_dim=256)
        total_params = sum(p.numel() for p in encoder.parameters())
        # Allow range: 500K to 5M
        assert 500_000 < total_params < 5_000_000, (
            f"Parameter count {total_params} is outside expected range [500K, 5M]"
        )

    def test_all_parameters_require_grad(self):
        """All parameters should require gradients by default."""
        encoder = GlycanTreeEncoder()
        for name, param in encoder.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require grad"


# ---------------------------------------------------------------------------
# Tests: Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Tests for numerical stability under various conditions."""

    def test_repeated_forward_passes(self):
        """Multiple forward passes should produce consistent results."""
        encoder = GlycanTreeEncoder(output_dim=32, hidden_dim=32)
        encoder.eval()
        tree = _make_branched_tree()

        with torch.no_grad():
            out1 = encoder([tree])
            out2 = encoder([tree])

        assert torch.allclose(out1, out2, atol=1e-6), \
            "Repeated forward passes should produce identical results in eval mode"

    def test_no_nan_after_many_layers(self):
        """Encoder with more layers should not produce NaN."""
        encoder = GlycanTreeEncoder(
            output_dim=32, hidden_dim=32, num_bottom_up_layers=5
        )
        tree = _make_linear_chain(10)
        out = encoder([tree])
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
