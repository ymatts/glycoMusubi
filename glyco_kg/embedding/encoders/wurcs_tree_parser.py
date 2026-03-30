"""WURCS-to-tree-graph parser for the GlycanTreeEncoder.

Converts WURCS 2.0 strings into :class:`GlycanTree` structures suitable
for tree-structured message passing.  Each monosaccharide residue becomes
a node and each glycosidic linkage becomes a directed edge (parent -> child).

WURCS 2.0 format overview::

    WURCS=2.0/<counts>/<unique_residues>/<residue_sequence>/<linkages>

* **Unique residues** are enclosed in square brackets (``[...]``).
* **Residue sequence** maps position labels (``a``, ``b``, ...) to
  unique-residue indices (1-based).
* **Linkages** describe glycosidic bonds, e.g. ``a2-b1`` means
  "residue *a*, carbon 2 -> residue *b*, carbon 1".

The reducing-end residue (first in the sequence) is treated as the root.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

MONOSACCHARIDE_TYPE_VOCAB: Dict[str, int] = {
    # Common monosaccharide classes mapped to integer indices.
    # Index 0 is reserved for "unknown".
    "Unknown": 0,
    "Glc": 1,
    "Man": 2,
    "Gal": 3,
    "GlcNAc": 4,
    "GalNAc": 5,
    "Fuc": 6,
    "NeuAc": 7,
    "NeuGc": 8,
    "Xyl": 9,
    "GlcA": 10,
    "IdoA": 11,
    "Rha": 12,
    "Ara": 13,
    "Kdn": 14,
    "Hex": 15,
    "HexNAc": 16,
    "dHex": 17,
    "Pen": 18,
    "HexA": 19,
}

# Maximum vocab index (for nn.Embedding sizing)
NUM_MONO_TYPES: int = 64

ANOMERIC_VOCAB: Dict[str, int] = {
    "alpha": 0,
    "beta": 1,
    "unknown": 2,
}

RING_FORM_VOCAB: Dict[str, int] = {
    "pyranose": 0,
    "furanose": 1,
    "open": 2,
    "unknown": 3,
}

# Known modification types (multi-hot encoding order)
MODIFICATION_TYPES: List[str] = [
    "sulfation",
    "phosphorylation",
    "acetylation",
    "methylation",
    "deoxy",
    "amino",
    "n_acetyl",
    "n_glycolyl",
]
NUM_MODIFICATIONS: int = len(MODIFICATION_TYPES)


@dataclass
class MonosaccharideNode:
    """A node in the glycan tree (one monosaccharide residue)."""

    index: int
    wurcs_residue: str  # raw WURCS unique-residue code
    mono_type: str  # human-readable class, e.g. "GlcNAc"
    mono_type_idx: int  # integer index into MONOSACCHARIDE_TYPE_VOCAB
    anomeric: str  # "alpha", "beta", "unknown"
    anomeric_idx: int
    ring_form: str  # "pyranose", "furanose", "open", "unknown"
    ring_form_idx: int
    modifications: List[str] = field(default_factory=list)

    @property
    def modification_vector(self) -> List[float]:
        """Multi-hot vector of length :data:`NUM_MODIFICATIONS`."""
        vec = [0.0] * NUM_MODIFICATIONS
        for mod in self.modifications:
            if mod in MODIFICATION_TYPES:
                vec[MODIFICATION_TYPES.index(mod)] = 1.0
        return vec


@dataclass
class GlycosidicBond:
    """An edge in the glycan tree (one glycosidic linkage)."""

    parent_idx: int
    child_idx: int
    linkage_position: Tuple[int, int]  # (parent_carbon, child_carbon)
    bond_type: str  # "alpha", "beta", "unknown"


@dataclass
class GlycanTree:
    """A rooted tree representation of a glycan structure."""

    nodes: List[MonosaccharideNode]
    edges: List[GlycosidicBond]
    root_idx: int = 0

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def children_of(self, node_idx: int) -> List[int]:
        """Return indices of child nodes."""
        return [e.child_idx for e in self.edges if e.parent_idx == node_idx]

    def parent_of(self, node_idx: int) -> Optional[int]:
        """Return the parent index, or None for the root."""
        for e in self.edges:
            if e.child_idx == node_idx:
                return e.parent_idx
        return None

    def siblings_of(self, node_idx: int) -> List[int]:
        """Return indices of sibling nodes (same parent, excluding self)."""
        parent = self.parent_of(node_idx)
        if parent is None:
            return []
        return [c for c in self.children_of(parent) if c != node_idx]

    def is_branching(self, node_idx: int) -> bool:
        """True if the node has more than one child (branch point)."""
        return len(self.children_of(node_idx)) > 1

    def depth_of(self, node_idx: int) -> int:
        """Compute the depth of a node (root = 0)."""
        d = 0
        current = node_idx
        while True:
            p = self.parent_of(current)
            if p is None:
                return d
            d += 1
            current = p

    def topological_order_bottom_up(self) -> List[int]:
        """Return node indices in bottom-up order (leaves first, root last)."""
        visited: set = set()
        order: List[int] = []

        def _dfs(idx: int) -> None:
            visited.add(idx)
            for child in self.children_of(idx):
                if child not in visited:
                    _dfs(child)
            order.append(idx)

        _dfs(self.root_idx)
        return order

    def topological_order_top_down(self) -> List[int]:
        """Return node indices in top-down order (root first, leaves last)."""
        return list(reversed(self.topological_order_bottom_up()))


# -----------------------------------------------------------------------
# WURCS residue classification
# -----------------------------------------------------------------------

# Patterns to classify WURCS unique residue codes into monosaccharide types.
# Checked in order; first match wins.
_RESIDUE_CLASSIFICATION_RULES: List[Tuple[re.Pattern, str]] = [
    # Sialic acids (must precede generic Hex)
    (re.compile(r"a2122h-1b_1-5_2\*N(?!CC)"), "NeuAc"),
    (re.compile(r"a2122h-1b_1-5_2\*NO"), "NeuGc"),
    (re.compile(r"Neu5Ac", re.IGNORECASE), "NeuAc"),
    (re.compile(r"Neu5Gc", re.IGNORECASE), "NeuGc"),
    # Kdn
    (re.compile(r"a2122h-1[ab]_1-5_9\*O"), "Kdn"),
    # N-acetylhexosamines
    (re.compile(r"a2122h-1[ab]_1-5_2\*NCC/3=O"), "GlcNAc"),
    (re.compile(r"a2112h-1[ab]_1-5_2\*NCC/3=O"), "GalNAc"),
    (re.compile(r"a2122h-1[ab]_1-5_2\*NCC"), "GlcNAc"),
    (re.compile(r"a2112h-1[ab]_1-5_2\*NCC"), "GalNAc"),
    # Deoxyhexoses (Fuc, Rha)
    (re.compile(r"a1221m-1[ab]_1-5"), "Fuc"),
    (re.compile(r"a2211m-1[ab]_1-5"), "Rha"),
    # Pentoses
    (re.compile(r"a212h-1[ab]_1-5"), "Xyl"),
    (re.compile(r"a122h-1[ab]_1-4"), "Ara"),
    # Uronic acids
    (re.compile(r"a2122A-1[ab]_1-5"), "GlcA"),
    (re.compile(r"a2112A-1[ab]_1-5"), "IdoA"),
    # Generic hexoses – distinguish Glc / Man / Gal by stereochemistry
    (re.compile(r"a2122h-1[ab]_1-5"), "Glc"),  # Glc default
    (re.compile(r"a1122h-1[ab]_1-5"), "Man"),
    (re.compile(r"a2112h-1[ab]_1-5"), "Gal"),
]

_SULFATION_RE = re.compile(r"\*OSO", re.IGNORECASE)
_PHOSPHORYLATION_RE = re.compile(r"\*OPO", re.IGNORECASE)
_ACETYLATION_RE = re.compile(r"\*OCC", re.IGNORECASE)
_METHYLATION_RE = re.compile(r"\*OC(?!C)", re.IGNORECASE)
_DEOXY_RE = re.compile(r"\d[dm]")
_AMINO_RE = re.compile(r"\*N(?!CC|O)")
_N_ACETYL_RE = re.compile(r"\*NCC")
_N_GLYCOLYL_RE = re.compile(r"\*NO")


def _classify_residue(wurcs_code: str) -> str:
    """Classify a WURCS unique-residue code into a monosaccharide type name."""
    for pattern, name in _RESIDUE_CLASSIFICATION_RULES:
        if pattern.search(wurcs_code):
            return name
    return "Unknown"


def _detect_anomeric(wurcs_code: str) -> str:
    """Detect anomeric configuration from a WURCS residue code.

    In WURCS, ``-1a`` indicates alpha and ``-1b`` indicates beta.
    """
    if re.search(r"-1a[_\b]|-1a$", wurcs_code):
        return "alpha"
    if re.search(r"-1b[_\b]|-1b$", wurcs_code):
        return "beta"
    return "unknown"


def _detect_ring_form(wurcs_code: str) -> str:
    """Detect ring form from a WURCS residue code.

    ``_1-5`` -> pyranose, ``_1-4`` -> furanose.
    """
    if "_1-5" in wurcs_code:
        return "pyranose"
    if "_1-4" in wurcs_code:
        return "furanose"
    return "unknown"


def _detect_modifications(wurcs_code: str) -> List[str]:
    """Detect chemical modifications on a residue."""
    mods: List[str] = []
    if _SULFATION_RE.search(wurcs_code):
        mods.append("sulfation")
    if _PHOSPHORYLATION_RE.search(wurcs_code):
        mods.append("phosphorylation")
    if _ACETYLATION_RE.search(wurcs_code):
        mods.append("acetylation")
    if _METHYLATION_RE.search(wurcs_code):
        mods.append("methylation")
    if _DEOXY_RE.search(wurcs_code):
        mods.append("deoxy")
    if _AMINO_RE.search(wurcs_code):
        mods.append("amino")
    if _N_ACETYL_RE.search(wurcs_code):
        mods.append("n_acetyl")
    if _N_GLYCOLYL_RE.search(wurcs_code):
        mods.append("n_glycolyl")
    return mods


# -----------------------------------------------------------------------
# WURCS string parser
# -----------------------------------------------------------------------

def _parse_wurcs_sections(wurcs: str) -> Tuple[str, List[str], List[str], str]:
    """Split a WURCS=2.0 string into (header, unique_res, res_seq, linkages).

    Raises :class:`ValueError` on malformed input.
    """
    if not wurcs or not wurcs.startswith("WURCS"):
        raise ValueError(f"Not a valid WURCS string: {wurcs!r:.80}")

    parts = wurcs.split("/", 2)
    if len(parts) < 3:
        raise ValueError(f"WURCS string has fewer than 3 sections: {wurcs!r:.80}")

    rest = parts[2]

    # Extract unique residues inside brackets
    unique_res: List[str] = re.findall(r"\[([^\]]+)\]", rest)

    # Everything after the last ']' is residue-sequence / linkage
    last_bracket = rest.rfind("]")
    tail = rest[last_bracket + 1:] if last_bracket >= 0 else rest
    tail = tail.lstrip("/")

    tail_sections = tail.split("/")
    res_seq_str = tail_sections[0] if len(tail_sections) > 0 else ""
    lin_section = tail_sections[1] if len(tail_sections) > 1 else ""

    res_list = [r for r in res_seq_str.split("-") if r]

    return parts[1], unique_res, res_list, lin_section


def _parse_linkage_token(token: str) -> Optional[Tuple[str, int, str, int]]:
    """Parse a single linkage token like ``a2-b1``.

    Returns ``(src_label, src_carbon, dst_label, dst_carbon)`` or None.
    """
    # Typical: "a2-b1", "a4-b1", "a3-c1"
    # May also include "?" for ambiguous positions
    m = re.match(r"([a-zA-Z]+)(\d|\?)-([a-zA-Z]+)(\d|\?)", token)
    if not m:
        return None
    src_label = m.group(1).lower()
    src_carbon = int(m.group(2)) if m.group(2) != "?" else 0
    dst_label = m.group(3).lower()
    dst_carbon = int(m.group(4)) if m.group(4) != "?" else 0
    return src_label, src_carbon, dst_label, dst_carbon


def parse_wurcs_to_tree(wurcs: str) -> GlycanTree:
    """Convert a WURCS 2.0 string into a :class:`GlycanTree`.

    Parameters
    ----------
    wurcs:
        A WURCS 2.0 formatted string.

    Returns
    -------
    GlycanTree
        The parsed tree structure.

    Raises
    ------
    ValueError
        If the WURCS string is malformed or cannot be parsed.
    """
    _header, unique_res, res_list, lin_section = _parse_wurcs_sections(wurcs)

    if not unique_res:
        raise ValueError("No unique residues found in WURCS string")
    if not res_list:
        raise ValueError("No residue sequence found in WURCS string")

    # Map position labels (a, b, c, ...) to node indices
    label_to_idx: Dict[str, int] = {}
    nodes: List[MonosaccharideNode] = []

    for node_idx, token in enumerate(res_list):
        # The position label for linkage references is always a,b,c,...
        # derived from the sequential position in res_list.
        label = chr(ord("a") + node_idx)

        # The token itself is a 1-based index into the unique_res array.
        # e.g. "1" -> unique_res[0], "2" -> unique_res[1], "3" -> unique_res[2]
        ures_idx = 0
        if token.isdigit():
            ures_idx = int(token) - 1  # WURCS uses 1-based indexing
        elif token and token[0].isalpha():
            ures_idx = ord(token[0].lower()) - ord("a")

        label_to_idx[label] = node_idx

        # Get the unique residue code
        wurcs_code = unique_res[ures_idx] if ures_idx < len(unique_res) else ""

        mono_type = _classify_residue(wurcs_code)
        anomeric = _detect_anomeric(wurcs_code)
        ring_form = _detect_ring_form(wurcs_code)
        modifications = _detect_modifications(wurcs_code)

        mono_type_idx = MONOSACCHARIDE_TYPE_VOCAB.get(mono_type, 0)
        anomeric_idx = ANOMERIC_VOCAB.get(anomeric, 2)
        ring_form_idx = RING_FORM_VOCAB.get(ring_form, 3)

        nodes.append(
            MonosaccharideNode(
                index=node_idx,
                wurcs_residue=wurcs_code,
                mono_type=mono_type,
                mono_type_idx=mono_type_idx,
                anomeric=anomeric,
                anomeric_idx=anomeric_idx,
                ring_form=ring_form,
                ring_form_idx=ring_form_idx,
                modifications=modifications,
            )
        )

    # Parse linkages into edges
    edges: List[GlycosidicBond] = []
    if lin_section:
        linkage_tokens = lin_section.split("_")
        for token in linkage_tokens:
            parsed = _parse_linkage_token(token)
            if parsed is None:
                continue
            src_label, src_carbon, dst_label, dst_carbon = parsed

            # In WURCS linkage, the convention is:
            # src_label + src_carbon  ->  dst_label + dst_carbon
            # The source residue at src_carbon connects to the child at
            # dst_carbon (typically carbon 1 = anomeric carbon of child).
            # So src is the parent and dst is the child.
            parent = label_to_idx.get(src_label)
            child = label_to_idx.get(dst_label)

            if parent is None or child is None:
                logger.debug(
                    "Linkage references unknown label: %s->%s in %s",
                    src_label,
                    dst_label,
                    wurcs[:80],
                )
                continue

            # Determine bond type from the child's anomeric configuration
            child_node = nodes[child] if child < len(nodes) else None
            bond_type = child_node.anomeric if child_node else "unknown"

            edges.append(
                GlycosidicBond(
                    parent_idx=parent,
                    child_idx=child,
                    linkage_position=(src_carbon, dst_carbon),
                    bond_type=bond_type,
                )
            )

    # Root is the first residue in the sequence (reducing end)
    root_idx = 0

    return GlycanTree(nodes=nodes, edges=edges, root_idx=root_idx)


# -----------------------------------------------------------------------
# Conversion to PyG-compatible tensors
# -----------------------------------------------------------------------

def glycan_tree_to_tensors(
    tree: GlycanTree,
) -> Dict[str, torch.Tensor]:
    """Convert a :class:`GlycanTree` to a dictionary of tensors.

    Returns a dict with keys suitable for building a ``torch_geometric.data.Data`` object:

    * ``mono_type`` -- ``[N]`` long tensor of monosaccharide type indices
    * ``anomeric`` -- ``[N]`` long tensor of anomeric config indices
    * ``ring_form`` -- ``[N]`` long tensor of ring form indices
    * ``modifications`` -- ``[N, NUM_MODIFICATIONS]`` float tensor (multi-hot)
    * ``edge_index`` -- ``[2, E]`` long tensor (parent -> child)
    * ``linkage_parent_carbon`` -- ``[E]`` long tensor
    * ``linkage_child_carbon`` -- ``[E]`` long tensor
    * ``bond_type`` -- ``[E]`` long tensor (0=alpha, 1=beta, 2=unknown)
    * ``depth`` -- ``[N]`` long tensor of node depths
    * ``is_branch`` -- ``[N]`` bool tensor indicating branching nodes
    * ``num_nodes`` -- int
    """
    n = tree.num_nodes
    e = tree.num_edges

    mono_type = torch.zeros(n, dtype=torch.long)
    anomeric = torch.zeros(n, dtype=torch.long)
    ring_form = torch.zeros(n, dtype=torch.long)
    modifications = torch.zeros(n, NUM_MODIFICATIONS, dtype=torch.float32)
    depth = torch.zeros(n, dtype=torch.long)
    is_branch = torch.zeros(n, dtype=torch.bool)

    for node in tree.nodes:
        i = node.index
        mono_type[i] = node.mono_type_idx
        anomeric[i] = node.anomeric_idx
        ring_form[i] = node.ring_form_idx
        mod_vec = node.modification_vector
        modifications[i] = torch.tensor(mod_vec, dtype=torch.float32)
        depth[i] = tree.depth_of(i)
        is_branch[i] = tree.is_branching(i)

    if e > 0:
        edge_index = torch.zeros(2, e, dtype=torch.long)
        linkage_parent_carbon = torch.zeros(e, dtype=torch.long)
        linkage_child_carbon = torch.zeros(e, dtype=torch.long)
        bond_type_tensor = torch.zeros(e, dtype=torch.long)

        bond_type_map = {"alpha": 0, "beta": 1, "unknown": 2}

        for j, edge in enumerate(tree.edges):
            edge_index[0, j] = edge.parent_idx
            edge_index[1, j] = edge.child_idx
            linkage_parent_carbon[j] = edge.linkage_position[0]
            linkage_child_carbon[j] = edge.linkage_position[1]
            bond_type_tensor[j] = bond_type_map.get(edge.bond_type, 2)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        linkage_parent_carbon = torch.zeros(0, dtype=torch.long)
        linkage_child_carbon = torch.zeros(0, dtype=torch.long)
        bond_type_tensor = torch.zeros(0, dtype=torch.long)

    return {
        "mono_type": mono_type,
        "anomeric": anomeric,
        "ring_form": ring_form,
        "modifications": modifications,
        "edge_index": edge_index,
        "linkage_parent_carbon": linkage_parent_carbon,
        "linkage_child_carbon": linkage_child_carbon,
        "bond_type": bond_type_tensor,
        "depth": depth,
        "is_branch": is_branch,
        "num_nodes": n,
    }
