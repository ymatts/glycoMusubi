#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
location_normalizer.py

Maps UniProt free-text subcellular location annotations to standardized
compartment IDs for the glycoMusubi knowledge graph.

ID format: LOC::<snake_case>
"""

import re
from typing import Dict, List, Optional


COMPARTMENT_REGISTRY: Dict[str, str] = {
    "LOC::endoplasmic_reticulum": "Endoplasmic reticulum",
    "LOC::er_golgi_intermediate": "ER-Golgi intermediate compartment",
    "LOC::cis_golgi": "cis-Golgi network",
    "LOC::medial_golgi": "Medial-Golgi",
    "LOC::trans_golgi": "trans-Golgi",
    "LOC::trans_golgi_network": "trans-Golgi network",
    "LOC::golgi_apparatus": "Golgi apparatus",
    "LOC::cell_membrane": "Cell membrane",
    "LOC::cell_surface": "Cell surface",
    "LOC::extracellular": "Extracellular / Secreted",
    "LOC::lysosome": "Lysosome",
    "LOC::endosome": "Endosome",
    "LOC::cytoplasm": "Cytoplasm",
    "LOC::nucleus": "Nucleus",
    "LOC::mitochondria": "Mitochondria",
    "LOC::peroxisome": "Peroxisome",
}

# Ordered patterns: specific matches first, generic last.
_PATTERNS: List[tuple] = [
    (re.compile(r"ER-Golgi intermediate", re.IGNORECASE), "LOC::er_golgi_intermediate"),
    (re.compile(r"trans-Golgi network", re.IGNORECASE), "LOC::trans_golgi_network"),
    (re.compile(r"trans-Golgi", re.IGNORECASE), "LOC::trans_golgi"),
    (re.compile(r"cis-Golgi", re.IGNORECASE), "LOC::cis_golgi"),
    (re.compile(r"medial-Golgi|medial Golgi", re.IGNORECASE), "LOC::medial_golgi"),
    (re.compile(r"Golgi", re.IGNORECASE), "LOC::golgi_apparatus"),
    (re.compile(r"endoplasmic reticulum", re.IGNORECASE), "LOC::endoplasmic_reticulum"),
    (re.compile(r"cell surface", re.IGNORECASE), "LOC::cell_surface"),
    (re.compile(r"cell membrane|plasma membrane", re.IGNORECASE), "LOC::cell_membrane"),
    (re.compile(r"extracellular|secreted", re.IGNORECASE), "LOC::extracellular"),
    (re.compile(r"lysosome", re.IGNORECASE), "LOC::lysosome"),
    (re.compile(r"endosome", re.IGNORECASE), "LOC::endosome"),
    (re.compile(r"cytoplasm|cytosol", re.IGNORECASE), "LOC::cytoplasm"),
    (re.compile(r"nucleus|nuclear", re.IGNORECASE), "LOC::nucleus"),
    (re.compile(r"mitochondri", re.IGNORECASE), "LOC::mitochondria"),
    (re.compile(r"peroxisome", re.IGNORECASE), "LOC::peroxisome"),
]


def normalize_location(raw_text: str) -> Optional[str]:
    """Normalize a free-text subcellular location to a standard compartment ID.

    Args:
        raw_text: Raw location string from UniProt annotation.

    Returns:
        Compartment ID (e.g. ``"LOC::golgi_apparatus"``) or ``None`` if
        the text does not match any known compartment.
    """
    if not raw_text or not raw_text.strip():
        return None
    for pattern, loc_id in _PATTERNS:
        if pattern.search(raw_text):
            return loc_id
    return None


def normalize_locations_from_uniprot(subcellular_locations) -> List[str]:
    """Extract and normalize compartment IDs from UniProt JSON comment structures.

    Accepts either:
    - A list of dicts with ``"location"`` key (UniProt API JSON format)
    - A semicolon-separated string (TSV field)

    Returns:
        De-duplicated list of compartment IDs.
    """
    results: List[str] = []
    seen: set = set()

    if isinstance(subcellular_locations, str):
        items = [s.strip() for s in subcellular_locations.split(";") if s.strip()]
    elif isinstance(subcellular_locations, list):
        items = []
        for entry in subcellular_locations:
            if isinstance(entry, dict):
                loc_val = entry.get("location", {})
                if isinstance(loc_val, dict):
                    raw = loc_val.get("value", "")
                else:
                    raw = str(loc_val)
                if raw:
                    items.append(raw)
            else:
                items.append(str(entry))
    else:
        return results

    for raw in items:
        loc_id = normalize_location(raw)
        if loc_id and loc_id not in seen:
            seen.add(loc_id)
            results.append(loc_id)

    return results


def get_compartment_label(loc_id: str) -> str:
    """Return the human-readable label for a compartment ID.

    Args:
        loc_id: Compartment ID (e.g. ``"LOC::golgi_apparatus"``).

    Returns:
        Label string, or the ID itself if not found in the registry.
    """
    return COMPARTMENT_REGISTRY.get(loc_id, loc_id)
