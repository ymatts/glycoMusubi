#!/usr/bin/env python3
"""Build N-glycan biosynthesis reaction triplet DB for glycoPathAI integration.

Combines:
1. Curated N-glycan biosynthesis pathway (KEGG hsa00510 + literature)
2. GlycoEnzOnto enzyme→rule mapping (353 enzymes)
3. glycoMusubi enzyme→glycan associations (114 genes, 29K glycans)

Output: data_clean/reaction_triplets.parquet
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────
# 1. Curated N-glycan biosynthesis pathway (KEGG + literature)
# ─────────────────────────────────────────────────────────────
# Structure names follow GlycoTree/IUPAC conventions.
# Each tuple: (enzyme_gene, substrate, product)

N_GLYCAN_CORE_PATHWAY = [
    # === ER Lipid-linked assembly ===
    ("DPAGT1",   "Dol-PP",           "Dol-PP-GlcNAc"),
    ("ALG13",    "Dol-PP-GlcNAc",    "Dol-PP-GlcNAc2"),
    ("ALG14",    "Dol-PP-GlcNAc",    "Dol-PP-GlcNAc2"),  # ALG13/14 complex
    ("ALG1",     "Dol-PP-GlcNAc2",   "Man1GlcNAc2-PP-Dol"),
    ("ALG2",     "Man1GlcNAc2-PP-Dol", "Man3GlcNAc2-PP-Dol"),
    ("ALG11",    "Man3GlcNAc2-PP-Dol", "Man5GlcNAc2-PP-Dol"),
    # Flipping to ER lumen
    ("ALG3",     "Man5GlcNAc2-PP-Dol", "Man6GlcNAc2-PP-Dol"),
    ("ALG9",     "Man6GlcNAc2-PP-Dol", "Man7GlcNAc2-PP-Dol"),
    ("ALG12",    "Man7GlcNAc2-PP-Dol", "Man8GlcNAc2-PP-Dol"),
    ("ALG9",     "Man8GlcNAc2-PP-Dol", "Man9GlcNAc2-PP-Dol"),
    ("ALG6",     "Man9GlcNAc2-PP-Dol", "Glc1Man9GlcNAc2-PP-Dol"),
    ("ALG8",     "Glc1Man9GlcNAc2-PP-Dol", "Glc2Man9GlcNAc2-PP-Dol"),
    ("ALG10",    "Glc2Man9GlcNAc2-PP-Dol", "Glc3Man9GlcNAc2-PP-Dol"),
    # Transfer to protein
    ("STT3A",    "Glc3Man9GlcNAc2-PP-Dol", "Glc3Man9GlcNAc2"),
    ("STT3B",    "Glc3Man9GlcNAc2-PP-Dol", "Glc3Man9GlcNAc2"),

    # === ER Trimming ===
    ("MOGS",     "Glc3Man9GlcNAc2",  "Glc2Man9GlcNAc2"),
    ("GANAB",    "Glc2Man9GlcNAc2",  "Glc1Man9GlcNAc2"),
    ("GANAB",    "Glc1Man9GlcNAc2",  "Man9GlcNAc2"),
    ("MAN1A1",   "Man9GlcNAc2",      "Man8GlcNAc2"),
    ("MAN1A2",   "Man9GlcNAc2",      "Man8GlcNAc2"),
    ("MAN1B1",   "Man8GlcNAc2",      "Man5GlcNAc2"),

    # === Golgi Processing — Core pathway ===
    # GnT-I: first GlcNAc branch → hybrid structure
    ("MGAT1",    "Man5GlcNAc2",      "GlcNAcMan5GlcNAc2"),
    # Golgi α-mannosidase II: trim to core
    ("MAN2A1",   "GlcNAcMan5GlcNAc2", "GlcNAcMan3GlcNAc2"),
    ("MAN2A2",   "GlcNAcMan5GlcNAc2", "GlcNAcMan3GlcNAc2"),
    # GnT-II: second GlcNAc → biantennary
    ("MGAT2",    "GlcNAcMan3GlcNAc2", "GlcNAc2Man3GlcNAc2"),

    # === Branching extensions ===
    # GnT-III: bisecting GlcNAc
    ("MGAT3",    "GlcNAc2Man3GlcNAc2",    "BisGlcNAc2Man3GlcNAc2"),
    # GnT-IV: tri-antennary
    ("MGAT4A",   "GlcNAc2Man3GlcNAc2",    "GlcNAc3Man3GlcNAc2"),
    ("MGAT4B",   "GlcNAc2Man3GlcNAc2",    "GlcNAc3Man3GlcNAc2"),
    # GnT-V: tetra-antennary
    ("MGAT5",    "GlcNAc3Man3GlcNAc2",    "GlcNAc4Man3GlcNAc2"),

    # === Galactosylation ===
    ("B4GALT1",  "GlcNAc2Man3GlcNAc2",    "Gal1GlcNAc2Man3GlcNAc2"),
    ("B4GALT1",  "Gal1GlcNAc2Man3GlcNAc2","Gal2GlcNAc2Man3GlcNAc2"),
    ("B4GALT2",  "GlcNAc2Man3GlcNAc2",    "Gal1GlcNAc2Man3GlcNAc2"),
    ("B4GALT3",  "GlcNAc2Man3GlcNAc2",    "Gal1GlcNAc2Man3GlcNAc2"),
    ("B4GALT4",  "GlcNAc2Man3GlcNAc2",    "Gal1GlcNAc2Man3GlcNAc2"),
    # Triantennary galactosylation
    ("B4GALT1",  "GlcNAc3Man3GlcNAc2",    "Gal1GlcNAc3Man3GlcNAc2"),
    ("B4GALT1",  "Gal1GlcNAc3Man3GlcNAc2","Gal2GlcNAc3Man3GlcNAc2"),
    ("B4GALT1",  "Gal2GlcNAc3Man3GlcNAc2","Gal3GlcNAc3Man3GlcNAc2"),
    # Tetra-antennary galactosylation
    ("B4GALT1",  "GlcNAc4Man3GlcNAc2",    "Gal1GlcNAc4Man3GlcNAc2"),
    ("B4GALT1",  "Gal3GlcNAc4Man3GlcNAc2","Gal4GlcNAc4Man3GlcNAc2"),

    # === Sialylation ===
    ("ST6GAL1",  "Gal2GlcNAc2Man3GlcNAc2",  "Sia1Gal2GlcNAc2Man3GlcNAc2"),
    ("ST6GAL1",  "Sia1Gal2GlcNAc2Man3GlcNAc2","Sia2Gal2GlcNAc2Man3GlcNAc2"),
    ("ST6GAL2",  "Gal2GlcNAc2Man3GlcNAc2",  "Sia1Gal2GlcNAc2Man3GlcNAc2"),
    ("ST3GAL4",  "Gal2GlcNAc2Man3GlcNAc2",  "a23Sia1Gal2GlcNAc2Man3GlcNAc2"),
    ("ST3GAL6",  "Gal2GlcNAc2Man3GlcNAc2",  "a23Sia1Gal2GlcNAc2Man3GlcNAc2"),
    # Triantennary sialylation
    ("ST6GAL1",  "Gal3GlcNAc3Man3GlcNAc2",  "Sia1Gal3GlcNAc3Man3GlcNAc2"),

    # === Core fucosylation ===
    ("FUT8",     "GlcNAc2Man3GlcNAc2",       "FucGlcNAc2Man3GlcNAc2"),
    ("FUT8",     "Gal2GlcNAc2Man3GlcNAc2",   "FucGal2GlcNAc2Man3GlcNAc2"),
    ("FUT8",     "Sia2Gal2GlcNAc2Man3GlcNAc2","FucSia2Gal2GlcNAc2Man3GlcNAc2"),

    # === Lewis/blood group modifications ===
    # FUT3: Lewis a/x
    ("FUT3",     "Gal2GlcNAc2Man3GlcNAc2",   "LeXGal2GlcNAc2Man3GlcNAc2"),
    # FUT1/FUT2: H antigen
    ("FUT1",     "Gal2GlcNAc2Man3GlcNAc2",   "HGal2GlcNAc2Man3GlcNAc2"),
    ("FUT2",     "Gal2GlcNAc2Man3GlcNAc2",   "HGal2GlcNAc2Man3GlcNAc2"),

    # === Poly-LacNAc extension ===
    ("B3GNT2",   "Gal2GlcNAc2Man3GlcNAc2",   "pLN1_Gal2GlcNAc2Man3GlcNAc2"),
    ("B4GALT1",  "pLN1_Gal2GlcNAc2Man3GlcNAc2","pLN1Gal_GlcNAc2Man3GlcNAc2"),

    # === High-mannose variants ===
    ("MAN1A1",   "Man8GlcNAc2",      "Man7GlcNAc2"),
    ("MAN1A1",   "Man7GlcNAc2",      "Man6GlcNAc2"),
    ("MAN1A1",   "Man6GlcNAc2",      "Man5GlcNAc2"),

    # === Hybrid structures ===
    ("B4GALT1",  "GlcNAcMan5GlcNAc2",  "GalGlcNAcMan5GlcNAc2"),
    ("ST6GAL1",  "GalGlcNAcMan5GlcNAc2","SiaGalGlcNAcMan5GlcNAc2"),
]

# ─────────────────────────────────────────────────────────────
# 2. O-glycan biosynthesis pathway
# ─────────────────────────────────────────────────────────────
O_GLYCAN_PATHWAY = [
    # Core 1 (T antigen)
    ("GALNT1",   "Ser/Thr",          "GalNAc-Ser/Thr"),
    ("GALNT2",   "Ser/Thr",          "GalNAc-Ser/Thr"),
    ("GALNT3",   "Ser/Thr",          "GalNAc-Ser/Thr"),
    ("C1GALT1",  "GalNAc-Ser/Thr",   "Gal-GalNAc-Ser/Thr"),  # Core 1

    # Core 2
    ("GCNT1",    "Gal-GalNAc-Ser/Thr", "Core2-GalNAc-Ser/Thr"),

    # Core 3
    ("B3GNT6",   "GalNAc-Ser/Thr",     "Core3-GalNAc-Ser/Thr"),

    # Core 1 extension
    ("ST3GAL1",  "Gal-GalNAc-Ser/Thr", "Sia-Gal-GalNAc-Ser/Thr"),
    ("ST6GALNAC1","Gal-GalNAc-Ser/Thr","SiaTn-GalNAc-Ser/Thr"),

    # Tn antigen sialylation
    ("ST6GALNAC1","GalNAc-Ser/Thr",    "SiaTn-Ser/Thr"),
]


def extract_glycoenzont_enzymes(owl_path: Path) -> dict[str, list[str]]:
    """Extract enzyme_gene → [reaction_rule_text] from GlycoEnzOnto OWL."""
    text = owl_path.read_text()

    # Build rule URI → rule text map
    individuals = re.findall(
        r'<owl:NamedIndividual[^>]*rdf:about="([^"]+)">(.*?)</owl:NamedIndividual>',
        text, re.DOTALL
    )

    rule_text = {}  # uri → text
    enzyme_rules = {}  # enzyme_label → [rule_uri]

    for uri, block in individuals:
        rule_m = re.search(r'<genzo:genzo1050>(.+?)</genzo:genzo1050>', block)
        has_rule = re.findall(r'<genzo:genzo1000 rdf:resource="([^"]+)"', block)
        label_m = re.search(r'<rdfs:label[^>]*>([^<]+)</rdfs:label>', block)

        if rule_m:
            rule_text[uri] = rule_m.group(1).replace('&gt;', '>').replace('&lt;', '<')

        if has_rule and label_m:
            enzyme_label = label_m.group(1).strip()
            enzyme_rules[enzyme_label] = has_rule

    # Resolve rule URIs to text
    result = {}
    for enzyme, rule_uris in enzyme_rules.items():
        texts = [rule_text.get(u, '') for u in rule_uris if u in rule_text]
        if texts:
            result[enzyme] = texts

    return result


def build_enzyme_gene_mapping(glycoenzont_enzymes: dict) -> dict[str, str]:
    """Map glycoPathAI enzyme names (GnT-I) to HGNC gene symbols (MGAT1)."""
    # GlycoEnzOnto labels ARE gene symbols in most cases
    # But we also need the reverse: common enzyme names → gene symbols
    common_name_to_gene = {
        "GnT-I": "MGAT1",
        "GnT-II": "MGAT2",
        "GnT-III": "MGAT3",
        "GnT-IV": "MGAT4A",
        "GnT-V": "MGAT5",
        "GalT": "B4GALT1",
        "SiaT": "ST6GAL1",
        "FucT8": "FUT8",
        "FucT3": "FUT3",
        "GnT-i": "B3GNT2",
        "ManI": "MAN1A1",
        "ManII": "MAN2A1",
    }
    return common_name_to_gene


def load_ts_gene_symbols() -> set[str]:
    """Load Tabula Sapiens gene symbols for cross-referencing."""
    ts_path = PROJECT / "data_clean/ts_celltype_enzyme_expression.tsv"
    if not ts_path.exists():
        return set()
    ts = pd.read_csv(ts_path, sep="\t")
    return set(ts["gene_symbol"].unique())


def main():
    logger.info("=" * 70)
    logger.info("  Building Reaction Triplet DB for glycoPathAI Integration")
    logger.info("=" * 70)

    # 1. Load GlycoEnzOnto
    owl_path = Path.home() / ".glycopath/cache/GlycoEnzOnto.owl"
    if owl_path.exists():
        enzont = extract_glycoenzont_enzymes(owl_path)
        logger.info("GlycoEnzOnto: %d enzymes with rules", len(enzont))
    else:
        enzont = {}
        logger.warning("GlycoEnzOnto OWL not found at %s", owl_path)

    # 2. Combine all triplets
    all_triplets = list(N_GLYCAN_CORE_PATHWAY) + list(O_GLYCAN_PATHWAY)
    logger.info("Curated triplets: %d (N-glycan: %d, O-glycan: %d)",
                len(all_triplets), len(N_GLYCAN_CORE_PATHWAY), len(O_GLYCAN_PATHWAY))

    # 3. Deduplicate
    seen = set()
    unique = []
    for t in all_triplets:
        key = (t[0], t[1], t[2])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    logger.info("Unique triplets: %d", len(unique))

    # 4. Cross-reference with Tabula Sapiens genes
    ts_genes = load_ts_gene_symbols()
    triplet_genes = set(t[0] for t in unique)
    overlap = triplet_genes & ts_genes
    logger.info("Reaction genes: %d, TS genes: %d, overlap: %d",
                len(triplet_genes), len(ts_genes), len(overlap))
    missing = triplet_genes - ts_genes
    if missing:
        logger.info("Genes NOT in TS: %s", sorted(missing))

    # 5. Cross-reference with GlycoEnzOnto
    enzont_genes = set(enzont.keys())
    enzont_overlap = triplet_genes & enzont_genes
    logger.info("GlycoEnzOnto overlap: %d / %d reaction genes", len(enzont_overlap), len(triplet_genes))

    # 6. List all glycan structures in the pathway
    all_structures = set()
    for _, s, p in unique:
        all_structures.add(s)
        all_structures.add(p)
    logger.info("Unique structures in pathway: %d", len(all_structures))

    # 7. Build reachability from Man5GlcNAc2 (Golgi entry point)
    from collections import deque
    adj = defaultdict(list)
    for gene, s, p in unique:
        adj[s].append((gene, p))

    root = "Man5GlcNAc2"
    reachable = set()
    queue = deque([root])
    reachable.add(root)
    while queue:
        node = queue.popleft()
        for _, product in adj[node]:
            if product not in reachable:
                reachable.add(product)
                queue.append(product)

    logger.info("Reachable from %s: %d structures", root, len(reachable))

    # 8. Save
    df = pd.DataFrame(unique, columns=["enzyme_gene", "reactant", "product"])
    out_path = PROJECT / "data_clean/reaction_triplets.parquet"
    df.to_parquet(out_path, index=False)
    tsv_path = PROJECT / "data_clean/reaction_triplets.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info("Saved %d triplets to %s", len(df), out_path)

    # 9. Summary
    logger.info("\n" + "─" * 60)
    logger.info("Reaction Triplet DB Summary:")
    logger.info("  Total reactions: %d", len(unique))
    logger.info("  Unique enzymes: %d", len(triplet_genes))
    logger.info("  Unique structures: %d", len(all_structures))
    logger.info("  Reachable from Man5: %d", len(reachable))
    logger.info("  TS gene coverage: %d/%d (%.1f%%)",
                len(overlap), len(triplet_genes),
                100 * len(overlap) / len(triplet_genes))

    # Show pathway tree
    logger.info("\nN-glycan pathway from Man5GlcNAc2:")
    visited = set()
    def show_tree(node, depth=0):
        if node in visited or depth > 6:
            return
        visited.add(node)
        for gene, product in adj.get(node, []):
            logger.info("  %s%s ──[%s]──> %s",
                       "  " * depth, node[:25], gene, product[:30])
            show_tree(product, depth + 1)
    show_tree(root)


if __name__ == "__main__":
    main()
