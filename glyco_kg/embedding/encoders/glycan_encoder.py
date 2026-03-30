"""Glycan encoder for knowledge graph embedding.

Encodes glycan nodes into fixed-dimensional vectors using one of three
strategies:

* **learnable** -- plain ``nn.Embedding`` lookup (default, fast).
* **wurcs_features** -- hand-crafted biochemical feature vector extracted
  from WURCS strings (monosaccharide composition, branching, modifications).
* **hybrid** -- concatenation of learnable + WURCS features projected
  through a two-layer MLP.

The WURCS feature extraction prioritises biologically meaningful
properties: sialylation, fucosylation, branching pattern, and
N-/O-glycan core structure estimation, so that structurally similar
glycans produce nearby embeddings.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WURCS feature extraction
# ---------------------------------------------------------------------------

# Common monosaccharide residue identifiers found in WURCS unique-RES section.
# Each key maps a short WURCS residue code fragment to a human-readable
# monosaccharide class.  The order here determines the feature-vector index.
MONOSACCHARIDE_CLASSES = [
    "Hex",      # generic hexose (Glc, Man, Gal)
    "HexNAc",   # N-acetylhexosamine (GlcNAc, GalNAc)
    "dHex",     # deoxyhexose (Fuc, Rha)
    "NeuAc",    # N-acetylneuraminic acid (sialic acid)
    "NeuGc",    # N-glycolylneuraminic acid
    "Pen",      # pentose (Xyl, Ara)
    "HexA",     # hexuronic acid (GlcA, IdoA)
    "Kdn",      # 2-keto-3-deoxy-D-glycero-D-galacto-nononic acid
]

# Mapping from WURCS residue-code substrings to monosaccharide class index.
# WURCS v2.0 encodes each unique residue in a bracket-delimited section.
# These patterns are intentionally broad so that common variants are caught.
_WURCS_RESIDUE_PATTERNS: list[tuple[re.Pattern, int]] = [
    # NeuAc / NeuGc must be checked before generic Hex
    (re.compile(r"a2122h-1b_1-5_2\*N(?!CC)"), 3),         # NeuAc-like
    (re.compile(r"a2122h-1b_1-5_2\*NO"), 4),             # NeuGc-like
    (re.compile(r"a2112h-1b_1-5"), 3),                    # NeuAc variant
    # HexNAc: N-acetylated hexosamine
    (re.compile(r"a2122h-1[ab]_1-5_2\*NCC"), 1),         # GlcNAc / GalNAc
    (re.compile(r"a2112h-1[ab]_1-5_2\*NCC"), 1),
    # dHex (deoxyhexose, e.g., Fuc = 6-deoxy-L-galactose)
    (re.compile(r"a1221m-1[ab]_1-5"), 2),                 # Fuc-like
    (re.compile(r"a2211m-1[ab]_1-5"), 2),                 # Rha-like
    # Pentose (Xyl, Ara)
    (re.compile(r"a212h-1[ab]_1-5"), 5),
    (re.compile(r"a122h-1[ab]_1-4"), 5),
    # HexA (uronic acid)
    (re.compile(r"a2122A-1[ab]_1-5"), 6),
    # Kdn
    (re.compile(r"a2122h-1[ab]_1-5_9\*O"), 7),
    # Generic hexose as catch-all (must be last among monosaccharide tests)
    (re.compile(r"a2122h-1[ab]_1-5"), 0),
    (re.compile(r"a2112h-1[ab]_1-5"), 0),
]

# Modification patterns
_SULFATION_PATTERN = re.compile(r"\*OSO", re.IGNORECASE)
_PHOSPHORYLATION_PATTERN = re.compile(r"\*OPO", re.IGNORECASE)
_ACETYLATION_PATTERN = re.compile(r"\*OCC", re.IGNORECASE)


def _parse_wurcs_sections(wurcs: str) -> tuple[str, list[str], list[str], str]:
    """Split a WURCS=2.0 string into its four sections.

    Returns (header, unique_res_list, res_list, lin_section).
    Raises ValueError if the string cannot be parsed.
    """
    if not wurcs or not wurcs.startswith("WURCS"):
        raise ValueError(f"Not a valid WURCS string: {wurcs!r:.80}")

    # WURCS=2.0/count_section/unique_res/res/lin
    parts = wurcs.split("/", 2)
    if len(parts) < 3:
        raise ValueError(f"WURCS string has fewer than 3 sections: {wurcs!r:.80}")

    # parts[0] = "WURCS=2.0", parts[1] = counts, parts[2] = rest
    rest = parts[2]  # unique_res/res/lin

    # Split unique residues (bracket-delimited)
    unique_res: list[str] = re.findall(r"\[([^\]]+)\]", rest)

    # After the last ']' comes the residue-sequence and linkage sections
    last_bracket = rest.rfind("]")
    tail = rest[last_bracket + 1 :] if last_bracket >= 0 else rest
    tail = tail.lstrip("/")

    tail_sections = tail.split("/")
    res_seq = tail_sections[0] if len(tail_sections) > 0 else ""
    lin_section = tail_sections[1] if len(tail_sections) > 1 else ""

    res_list = [r for r in res_seq.split("-") if r]

    return parts[1], unique_res, res_list, lin_section


def _count_monosaccharides(unique_res: list[str], res_list: list[str]) -> list[int]:
    """Count monosaccharides by class.

    Returns a list of length ``len(MONOSACCHARIDE_CLASSES)`` with counts.
    """
    # Map each unique residue index (1-based in res_list letters a,b,...) to a class
    unique_class: Dict[int, int] = {}
    for idx, ures in enumerate(unique_res):
        for pat, cls_idx in _WURCS_RESIDUE_PATTERNS:
            if pat.search(ures):
                unique_class[idx] = cls_idx
                break

    counts = [0] * len(MONOSACCHARIDE_CLASSES)

    for token in res_list:
        if not token:
            continue
        # WURCS 2.0 res_list tokens may be numeric (1-based: "1", "2", "3")
        # or alphabetic ("a1", "b2").  Handle both forms.
        if token[0].isdigit():
            # Numeric token: 1-based index into unique_res
            uidx = int(token) - 1
            cls = unique_class.get(uidx, 0)  # default to Hex
            counts[cls] += 1
        elif token[0].isalpha():
            # Alphabetic token: a->0, b->1, ...
            uidx = ord(token[0].lower()) - ord("a")
            cls = unique_class.get(uidx, 0)  # default to Hex
            counts[cls] += 1
        else:
            # Fallback: count as Hex
            counts[0] += 1

    return counts


def _branching_degree(lin_section: str) -> int:
    """Estimate the number of branch points from the linkage section.

    In WURCS linkage notation, branch points are nodes that appear as
    targets of more than one linkage.  We approximate this by counting
    target residue indices that appear more than once.
    """
    if not lin_section:
        return 0

    # Linkage entries are separated by '_'
    links = lin_section.split("_")
    target_counts: Dict[str, int] = {}
    for link in links:
        # Each linkage: e.g., "a1-b4" meaning residue a, position 1 -> residue b, position 4
        parts = link.split("-")
        if len(parts) >= 2:
            # Source residue is the part before the first '-'
            src_residue = re.match(r"([a-z]+)", parts[0])
            if src_residue:
                src = src_residue.group(1)
                target_counts[src] = target_counts.get(src, 0) + 1

    return sum(1 for c in target_counts.values() if c > 1)


def _detect_modifications(unique_res: list[str]) -> tuple[bool, bool, bool]:
    """Detect sulfation, phosphorylation, and acetylation in unique residues."""
    full = " ".join(unique_res)
    sulfation = bool(_SULFATION_PATTERN.search(full))
    phosphorylation = bool(_PHOSPHORYLATION_PATTERN.search(full))
    acetylation = bool(_ACETYLATION_PATTERN.search(full))
    return sulfation, phosphorylation, acetylation


def _estimate_core_type(
    mono_counts: list[int],
    total_residues: int,
) -> list[float]:
    """Estimate whether the glycan has an N-glycan or O-glycan core.

    Returns a 4-element list:
      [p_n_glycan, p_o_glycan, p_gag, p_other]

    This is a heuristic based on monosaccharide composition:
    - N-glycan core: typically contains >= 2 HexNAc + >= 3 Hex
    - O-glycan core: typically 1 HexNAc at reducing end, smaller structures
    - GAG (glycosaminoglycan): high HexA + HexNAc with repeating units
    """
    hex_count = mono_counts[0]      # Hex
    hexnac_count = mono_counts[1]   # HexNAc
    dhex_count = mono_counts[2]     # dHex (Fuc)
    neuac_count = mono_counts[3]    # NeuAc
    hexa_count = mono_counts[6]     # HexA

    scores = [0.0, 0.0, 0.0, 0.0]  # N-glycan, O-glycan, GAG, other

    # N-glycan heuristic: core structure Man3GlcNAc2
    if hexnac_count >= 2 and hex_count >= 3:
        scores[0] = 0.8
        if dhex_count >= 1:
            scores[0] += 0.1  # core fucosylation common in N-glycans
        if neuac_count >= 1:
            scores[0] += 0.1  # terminal sialylation common
    # O-glycan heuristic: small structure with 1 GalNAc at core
    elif hexnac_count >= 1 and total_residues <= 6:
        scores[1] = 0.7
        if hex_count >= 1:
            scores[1] += 0.15
        if neuac_count >= 1:
            scores[1] += 0.15
    # GAG heuristic: alternating HexA-HexNAc
    elif hexa_count >= 2 and hexnac_count >= 2:
        scores[2] = 0.8
    else:
        scores[3] = 1.0

    # Normalise to sum to 1
    total = sum(scores)
    if total > 0:
        scores = [s / total for s in scores]
    else:
        scores[3] = 1.0

    return scores


def extract_wurcs_features(wurcs: str) -> torch.Tensor:
    """Extract a biochemical feature vector from a WURCS string.

    The returned tensor has shape ``[feature_dim]`` where *feature_dim*
    is currently **24** (8 monosaccharide counts + 1 branching degree +
    1 total residues + 3 modification flags + 4 core-type scores +
    3 derived ratios + 4 padding for future use).

    The features are designed so that biologically similar glycan
    structures yield similar vectors (e.g., high-mannose N-glycans
    cluster together, sialylated structures form another cluster).
    """
    feature_dim = 24
    try:
        _header, unique_res, res_list, lin_section = _parse_wurcs_sections(wurcs)
    except (ValueError, IndexError) as exc:
        logger.debug("Failed to parse WURCS for feature extraction: %s", exc)
        return torch.zeros(feature_dim)

    mono_counts = _count_monosaccharides(unique_res, res_list)
    total_residues = max(len(res_list), 1)
    branching = _branching_degree(lin_section)
    sulfation, phosphorylation, acetylation = _detect_modifications(unique_res)
    core_type = _estimate_core_type(mono_counts, total_residues)

    # Derived ratios (biologically informative)
    sialylation_ratio = (mono_counts[3] + mono_counts[4]) / total_residues
    fucosylation_ratio = mono_counts[2] / total_residues
    branching_ratio = branching / max(total_residues - 1, 1)

    features = (
        [float(c) for c in mono_counts]      # 8: monosaccharide counts
        + [float(branching)]                   # 1: branching degree
        + [float(total_residues)]              # 1: total residues
        + [float(sulfation), float(phosphorylation), float(acetylation)]  # 3: mods
        + core_type                            # 4: core-type scores
        + [sialylation_ratio, fucosylation_ratio, branching_ratio]  # 3: ratios
        + [0.0] * 4                            # 4: reserved for future features
    )

    return torch.tensor(features[: feature_dim], dtype=torch.float32)


# ---------------------------------------------------------------------------
# GlycanEncoder nn.Module
# ---------------------------------------------------------------------------


class GlycanEncoder(nn.Module):
    """Encode glycan nodes into fixed-dimensional embedding vectors.

    Parameters
    ----------
    num_glycans:
        Number of glycan entities in the KG (for learnable embedding table).
    output_dim:
        Dimensionality of the output embedding vector.
    method:
        One of ``"learnable"``, ``"wurcs_features"``, ``"hybrid"``.
    wurcs_feature_dim:
        Dimensionality of the raw WURCS feature vector (default 24).
    wurcs_map:
        Optional mapping ``{glycan_index: wurcs_string}`` used by
        ``wurcs_features`` and ``hybrid`` modes.  Can be set after
        construction via :pyattr:`wurcs_map`.
    embedding_dropout_rate:
        In ``"hybrid"`` mode, probability of zeroing the learnable
        embedding during training.  Forces the fusion MLP to rely on
        WURCS features, improving inductive generalisation (default 0).
    """

    # Canonical glycan function categories for multi-hot encoding
    FUNCTION_CATEGORIES = [
        "N-linked", "O-linked", "GAG", "Glycosphingolipid",
        "Human Milk Oligosaccharide", "GPI anchor", "C-linked", "Other",
    ]

    VALID_METHODS = ("learnable", "wurcs_features", "hybrid")

    def __init__(
        self,
        num_glycans: int,
        output_dim: int = 256,
        method: str = "learnable",
        wurcs_feature_dim: int = 24,
        wurcs_map: Optional[Dict[int, str]] = None,
        embedding_dropout_rate: float = 0.0,
        function_feature_map: Optional[Dict[int, torch.Tensor]] = None,
        function_feature_dim: int = 8,
    ) -> None:
        super().__init__()
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method {method!r}; choose from {self.VALID_METHODS}"
            )

        self.num_glycans = num_glycans
        self.output_dim = output_dim
        self.method = method
        self.wurcs_feature_dim = wurcs_feature_dim
        self.wurcs_map: Dict[int, str] = wurcs_map or {}
        self.embedding_dropout_rate = embedding_dropout_rate
        self.function_feature_map: Dict[int, torch.Tensor] = function_feature_map or {}
        self.function_feature_dim = function_feature_dim

        # -- learnable embedding (used by "learnable" and "hybrid") --
        if method in ("learnable", "hybrid"):
            self.embedding = nn.Embedding(num_glycans, output_dim)
            nn.init.xavier_uniform_(self.embedding.weight)

        # -- WURCS feature projection (used by "wurcs_features" and "hybrid") --
        if method in ("wurcs_features", "hybrid"):
            self.wurcs_proj = nn.Sequential(
                nn.Linear(wurcs_feature_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, output_dim),
            )

        # -- Hybrid fusion MLP: concat(learnable, wurcs_proj) -> output_dim --
        if method == "hybrid":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim),
            )

        # Function feature gated fusion (optional)
        if function_feature_map:
            self.function_proj = nn.Sequential(
                nn.Linear(function_feature_dim, output_dim // 4),
                nn.GELU(),
                nn.Linear(output_dim // 4, output_dim),
            )
            self.function_gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid(),
            )
            logger.info(
                "GlycanEncoder: function feature branch enabled (%d glycans with labels)",
                len(function_feature_map),
            )

        # Cache for pre-computed WURCS feature tensors
        self._wurcs_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_wurcs_features(self, indices: torch.Tensor) -> torch.Tensor:
        """Return WURCS feature vectors for a batch of glycan indices.

        Missing WURCS strings produce zero vectors.
        """
        device = indices.device
        batch: list[torch.Tensor] = []
        for idx_t in indices:
            idx = idx_t.item()
            if idx in self._wurcs_cache:
                batch.append(self._wurcs_cache[idx])
            else:
                wurcs = self.wurcs_map.get(idx)
                if wurcs is not None:
                    feat = extract_wurcs_features(wurcs)
                else:
                    feat = torch.zeros(self.wurcs_feature_dim)
                self._wurcs_cache[idx] = feat
                batch.append(feat)
        return torch.stack(batch).to(device)

    def _get_function_features(self, indices: torch.Tensor) -> torch.Tensor:
        """Return function multi-hot vectors for a batch of glycan indices."""
        device = indices.device
        batch: list[torch.Tensor] = []
        for idx_t in indices:
            idx = idx_t.item()
            feat = self.function_feature_map.get(idx)
            if feat is not None:
                batch.append(feat)
            else:
                batch.append(torch.zeros(self.function_feature_dim))
        return torch.stack(batch).to(device)

    def _apply_function_gate(
        self, base_emb: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Apply gated fusion with function features if available."""
        if not hasattr(self, "function_proj") or not self.function_feature_map:
            return base_emb
        func_feats = self._get_function_features(indices.view(-1))
        func_emb = self.function_proj(func_feats)
        func_emb = func_emb.view(*base_emb.shape)
        gate_input = torch.cat([base_emb, func_emb], dim=-1)
        gate = self.function_gate(gate_input)
        return gate * base_emb + (1 - gate) * func_emb

    def clear_cache(self) -> None:
        """Clear the internal WURCS feature cache."""
        self._wurcs_cache.clear()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute glycan embeddings.

        Parameters
        ----------
        indices:
            Long tensor of glycan indices, shape ``[B]`` or ``[B, ...]``.

        Returns
        -------
        Tensor of shape ``[B, output_dim]`` (or ``[B, ..., output_dim]``).
        """
        if self.method == "learnable":
            out = self.embedding(indices)
            return self._apply_function_gate(out, indices)

        if self.method == "wurcs_features":
            flat = indices.view(-1)
            feats = self._get_wurcs_features(flat)
            out = self.wurcs_proj(feats)
            out = out.view(*indices.shape, self.output_dim)
            return self._apply_function_gate(out, indices)

        # hybrid
        emb = self.embedding(indices)
        if self.training and self.embedding_dropout_rate > 0:
            # Randomly zero the learnable part so fusion_mlp learns to
            # rely on WURCS features alone — critical for inductive use.
            mask = (
                torch.rand(emb.shape[0], 1, device=emb.device)
                > self.embedding_dropout_rate
            ).float()
            emb = emb * mask
        flat = indices.view(-1)
        feats = self._get_wurcs_features(flat)
        wurcs_emb = self.wurcs_proj(feats).view(*indices.shape, self.output_dim)
        fused = torch.cat([emb, wurcs_emb], dim=-1)
        out = self.fusion_mlp(fused)
        return self._apply_function_gate(out, indices)
