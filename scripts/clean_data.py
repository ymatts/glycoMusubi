#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_data.py

Reads raw data, performs cleaning/normalization, and outputs intermediate tables.
Includes data validation and chunk-based loading for large files.
"""

import os
import sys
import re
import json
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

from utils.config_loader import load_config
from utils.location_normalizer import normalize_location, get_compartment_label

config = load_config()

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.raw_data)
DATA_CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.clean_data)
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.logs)

os.makedirs(DATA_CLEAN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "clean_data.log"))
    ]
)
logger = logging.getLogger(__name__)

GLYTOUCAN_ID_PATTERN = re.compile(r'^G\d{5}[A-Z]{2}$')
UNIPROT_AC_PATTERN = re.compile(r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$')
CHEMBL_ID_PATTERN = re.compile(r'^CHEMBL\d+$')

CHUNK_SIZE = 10000


@dataclass
class ValidationResult:
    """Container for data validation results."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str):
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def summary(self) -> str:
        lines = [f"Validation: {'PASSED' if self.valid else 'FAILED'}"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for e in self.errors[:10]:
                lines.append(f"  - {e}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"  - {w}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        return "\n".join(lines)


def validate_glytoucan_id(glycan_id: str) -> bool:
    """Validate GlyTouCan ID format (G + 5 digits + 2 letters)."""
    if not glycan_id or pd.isna(glycan_id):
        return False
    return bool(GLYTOUCAN_ID_PATTERN.match(str(glycan_id)))


def validate_uniprot_ac(uniprot_ac: str) -> bool:
    """Validate UniProt accession format."""
    if not uniprot_ac or pd.isna(uniprot_ac):
        return False
    return bool(UNIPROT_AC_PATTERN.match(str(uniprot_ac)))


def validate_chembl_id(chembl_id: str) -> bool:
    """Validate ChEMBL compound ID format."""
    if not chembl_id or pd.isna(chembl_id):
        return False
    return bool(CHEMBL_ID_PATTERN.match(str(chembl_id)))


def validate_raw_data(data: Dict[str, pd.DataFrame]) -> ValidationResult:
    """
    Validate raw data quality before cleaning.
    
    Checks:
    - Required columns exist
    - ID format validation
    - Null/duplicate detection
    - Data type consistency
    """
    result = ValidationResult()
    
    if "glycosyltransferase" in data:
        df = data["glycosyltransferase"]
        result.stats["glycosyltransferase_rows"] = len(df)
        
        if "uniprotkb_canonical_ac" not in df.columns:
            result.add_error("glycosyltransferase: missing 'uniprotkb_canonical_ac' column")
        else:
            null_count = df["uniprotkb_canonical_ac"].isna().sum()
            if null_count > 0:
                result.add_warning(f"glycosyltransferase: {null_count} null UniProt IDs")
            
            dup_count = df["uniprotkb_canonical_ac"].duplicated().sum()
            if dup_count > 0:
                result.add_warning(f"glycosyltransferase: {dup_count} duplicate UniProt IDs")
    
    if "glytoucan_structures" in data:
        df = data["glytoucan_structures"]
        result.stats["glytoucan_structures_rows"] = len(df)
        
        if "glycan_id" in df.columns:
            invalid_ids = df[~df["glycan_id"].apply(validate_glytoucan_id)]
            if len(invalid_ids) > 0:
                result.add_warning(f"glytoucan_structures: {len(invalid_ids)} invalid GlyTouCan IDs")
                result.stats["invalid_glytoucan_ids"] = invalid_ids["glycan_id"].tolist()[:10]
    
    if "glycan_masterlist" in data:
        df = data["glycan_masterlist"]
        result.stats["glycan_masterlist_rows"] = len(df)
        
        if "glytoucan_ac" in df.columns:
            null_count = df["glytoucan_ac"].isna().sum()
            if null_count > 0:
                result.add_warning(f"glycan_masterlist: {null_count} null GlyTouCan IDs")
    
    if "uniprot_annotations" in data:
        df = data["uniprot_annotations"]
        result.stats["uniprot_annotations_rows"] = len(df)
        
        if "uniprot_id" not in df.columns:
            result.add_error("uniprot_annotations: missing 'uniprot_id' column")
    
    if "chembl_inhibitors" in data:
        df = data["chembl_inhibitors"]
        result.stats["chembl_inhibitors_rows"] = len(df)
        
        if "compound_chembl_id" in df.columns:
            invalid_ids = df[~df["compound_chembl_id"].apply(validate_chembl_id)]
            if len(invalid_ids) > 0:
                result.add_warning(f"chembl_inhibitors: {len(invalid_ids)} invalid ChEMBL IDs")
    
    return result


def load_raw_data_chunked(file_path: str, sep: str = "\t", chunk_size: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """
    Load raw data in chunks for memory-efficient processing.
    
    Args:
        file_path: Path to the data file.
        sep: Column separator.
        chunk_size: Number of rows per chunk.
    
    Yields:
        DataFrame chunks.
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return
    
    try:
        for chunk in pd.read_csv(file_path, sep=sep, chunksize=chunk_size, on_bad_lines='skip', engine="python"):
            yield chunk
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")


def load_raw_data() -> Dict[str, Any]:
    """Load raw data files into DataFrames and JSON data."""
    paths = {
        "glycosyltransferase": os.path.join(DATA_RAW_DIR, "glygen_glycosyltransferase.csv"),
        "glytoucan_structures": os.path.join(DATA_RAW_DIR, "glytoucan_structures.tsv"),
        "uniprot_annotations": os.path.join(DATA_RAW_DIR, "uniprot_annotations.tsv"),
        "chembl_inhibitors": os.path.join(DATA_RAW_DIR, "chembl_gt_inhibitors.tsv"),
        "glycan_masterlist": os.path.join(DATA_RAW_DIR, "glycan_masterlist.csv"),
    }
    
    data = {}
    for k, v in paths.items():
        if os.path.exists(v):
            try:
                sep = "\t" if v.endswith(".tsv") else ","
                
                file_size = os.path.getsize(v)
                if file_size > 100 * 1024 * 1024:
                    logger.info(f"Loading {k} in chunks (file size: {file_size / 1024 / 1024:.1f} MB)")
                    chunks = list(load_raw_data_chunked(v, sep=sep))
                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        data[k] = df
                        logger.info(f"Loaded {k}: {len(df)} rows (chunked)")
                else:
                    df = pd.read_csv(v, sep=sep, on_bad_lines='skip', engine="python")
                    data[k] = df
                    logger.info(f"Loaded {k}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to load {v}: {e}")
        else:
            logger.warning(f"File not found: {v}")
    
    # Load Phase 2 glycan details (JSON format)
    # This is the canonical source for glycan-protein edges (replaces deprecated usecases API)
    glycan_details_path = os.path.join(DATA_RAW_DIR, "glygen_glycan_details.json")
    if os.path.exists(glycan_details_path):
        try:
            with open(glycan_details_path, 'r') as f:
                glycan_details = json.load(f)
                # Handle both old list format and new dict format
                if isinstance(glycan_details, list):
                    logger.info("Converting glycan_details from list to dict format")
                    data["glycan_details"] = {d['glycan_id']: d for d in glycan_details if 'glycan_id' in d}
                else:
                    data["glycan_details"] = glycan_details
            logger.info(f"Loaded glycan_details: {len(data['glycan_details'])} entries")
        except Exception as e:
            logger.error(f"Failed to load glycan details: {e}")
    else:
        logger.warning(f"File not found: {glycan_details_path}")
    
    return data

def build_tables(data: dict):
    """クリーンテーブルを構築する。"""

    # 1. Enzymes (from GlyGen GTs)
    # Node: enzyme_id (UniProt AC), label (Gene Name), type=enzyme
    if "glycosyltransferase" in data:
        gt_df = data["glycosyltransferase"]
        # Columns: uniprotkb_canonical_ac, gene_symbol
        if "uniprotkb_canonical_ac" in gt_df.columns:
            enzymes = gt_df[["uniprotkb_canonical_ac", "gene_symbol"]].drop_duplicates().copy()
            enzymes.columns = ["enzyme_id", "gene_symbol"]
            enzymes.to_csv(os.path.join(DATA_CLEAN_DIR, "enzymes_clean.tsv"), sep="\t", index=False)
            logger.info(f"Created enzymes_clean.tsv: {len(enzymes)} rows")
        else:
            logger.warning("Missing 'uniprotkb_canonical_ac' in GT data")

    # 2. Glycans (from Masterlist + Structures)
    # Node: glycan_id, structure (WURCS), type=glycan
    glycan_ids = set()
    structures_map = {}

    if "glytoucan_structures" in data:
        struct_df = data["glytoucan_structures"]
        if not struct_df.empty:
            structures_map = pd.Series(struct_df.structure.values, index=struct_df.glycan_id).to_dict()
            glycan_ids.update(struct_df.glycan_id.dropna())

    if "glycan_masterlist" in data:
        master_df = data["glycan_masterlist"]
        if "glytoucan_ac" in master_df.columns:
            glycan_ids.update(master_df["glytoucan_ac"].dropna())

    # Create DataFrame
    glycan_rows = []
    for gid in glycan_ids:
        glycan_rows.append({
            "glycan_id": gid,
            "structure": structures_map.get(gid, "")
        })

    if glycan_rows:
        glycans_df = pd.DataFrame(glycan_rows)
        glycans_df.to_csv(os.path.join(DATA_CLEAN_DIR, "glycans_clean.tsv"), sep="\t", index=False)
        logger.info(f"Created glycans_clean.tsv: {len(glycans_df)} rows")

    # 3. Proteins (Glycoproteins)
    # Node: protein_id (UniProt AC), label, type=protein
    # We get these from glycan_details (associated_proteins field)
    # This replaces the deprecated usecases API (glygen_glycan_protein.tsv)
    protein_ids = set()
    gp_edges = []
    gp_protein_nodes = []

    if "glycan_details" in data:
        glycan_details = data["glycan_details"]
        # Extract associated_proteins from each glycan detail
        for glycan_id, detail in glycan_details.items():
            glycan_ids.add(glycan_id)
            for protein in detail.get('associated_proteins', []):
                uniprot_id = protein.get('uniprot_canonical_ac')
                if uniprot_id:
                    gp_edges.append({
                        "glycan_id": glycan_id,
                        "protein_id": uniprot_id,
                        "gene": protein.get('gene'),
                        "protein_name": protein.get('protein_name'),
                        "source": "GlyGen"
                    })
                    gp_protein_nodes.append({
                        "protein_id": uniprot_id,
                        "gene_symbol": protein.get('gene') or protein.get('protein_name') or "",
                        "sequence": "",
                    })
                    protein_ids.add(uniprot_id)
        
        # Save edges
        if gp_edges:
            edges_gp_df = pd.DataFrame(gp_edges)
            edges_gp_df.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_glycan_protein.tsv"), sep="\t", index=False)
            logger.info(f"Created edges_glycan_protein.tsv: {len(edges_gp_df)} rows")
        else:
            logger.warning("No glycan-protein edges found in glycan_details.")
    else:
        logger.warning("No glycan_details found. Skipping glycan-protein edge generation.")

    # 4. Compounds (Inhibitors) & Edge Enzyme-Compound
    if "chembl_inhibitors" in data:
        chem_df = data["chembl_inhibitors"]
        # Columns: enzyme_uniprot_id, compound_chembl_id, compound_name, type, value, units
        if not chem_df.empty:
            # Nodes
            compounds = chem_df[["compound_chembl_id", "compound_name"]].drop_duplicates()
            compounds.columns = ["compound_id", "name"]
            compounds.to_csv(os.path.join(DATA_CLEAN_DIR, "compounds_clean.tsv"), sep="\t", index=False)
            logger.info(f"Created compounds_clean.tsv: {len(compounds)} rows")

            # Edges
            edges_ec = chem_df[["enzyme_uniprot_id", "compound_chembl_id", "type", "value", "units"]].copy()
            edges_ec.columns = ["enzyme_id", "compound_id", "type", "value", "units"]
            edges_ec.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_enzyme_compound.tsv"), sep="\t", index=False)
            logger.info(f"Created edges_enzyme_compound.tsv: {len(edges_ec)} rows")

    # 5. Diseases & Variants (from UniProt) & Edges
    if "uniprot_annotations" in data:
        up_df = data["uniprot_annotations"]
        # Columns: uniprot_id, diseases, variants, gene_symbol, sequence

        disease_nodes = []
        variant_nodes = []
        location_nodes = []
        protein_nodes = gp_protein_nodes  # Collect metadata for proteins (GTs and GPs)

        edges_pd = [] # Protein-Disease
        edges_pv = [] # Protein-Variant
        edges_protein_location = []

        for _, row in up_df.iterrows():
            uid = row.get("uniprot_id")
            gene = row.get("gene_symbol", "")
            sequence = row.get("sequence", "")
            if pd.isna(sequence):
                sequence = ""
            else:
                sequence = str(sequence).strip()

            # Update protein info (could be GT or GP)
            protein_nodes.append({"protein_id": uid, "gene_symbol": gene, "sequence": sequence})

            # Diseases
            d_str = str(row.get("diseases", ""))
            if d_str and d_str != "nan":
                for item in d_str.split(";"):
                    if "|" in item:
                        did, dname = item.split("|", 1)
                        disease_nodes.append({"disease_id": did, "name": dname})
                        edges_pd.append({"protein_id": uid, "disease_id": did})

            # Variants
            v_str = str(row.get("variants", ""))
            if v_str and v_str != "nan":
                for item in v_str.split(";"):
                    if "|" in item:
                        vid, vdesc = item.split("|", 1)
                        variant_nodes.append({"variant_id": vid, "description": vdesc})
                        edges_pv.append({"protein_id": uid, "variant_id": vid})

            # Subcellular locations
            loc_str = str(row.get("subcellular_locations", ""))
            if loc_str and loc_str != "nan":
                for raw_loc in loc_str.split(";"):
                    raw_loc = raw_loc.strip()
                    if not raw_loc:
                        continue
                    loc_id = normalize_location(raw_loc)
                    if loc_id:
                        location_nodes.append({
                            "location_id": loc_id,
                            "name": get_compartment_label(loc_id),
                        })
                        edges_protein_location.append({
                            "protein_id": uid,
                            "location_id": loc_id,
                        })

        # Save Nodes
        if disease_nodes:
            pd.DataFrame(disease_nodes).drop_duplicates("disease_id").to_csv(os.path.join(DATA_CLEAN_DIR, "diseases_clean.tsv"), sep="\t", index=False)
            logger.info(f"Created diseases_clean.tsv: {len(pd.DataFrame(disease_nodes).drop_duplicates('disease_id'))} rows")

        if variant_nodes:
            pd.DataFrame(variant_nodes).drop_duplicates("variant_id").to_csv(os.path.join(DATA_CLEAN_DIR, "variants_clean.tsv"), sep="\t", index=False)
            logger.info(f"Created variants_clean.tsv: {len(pd.DataFrame(variant_nodes).drop_duplicates('variant_id'))} rows")

        if protein_nodes:
            proteins_df = pd.DataFrame(protein_nodes)
            proteins_df["sequence"] = proteins_df["sequence"].fillna("").astype(str).str.strip()
            proteins_df["_has_sequence"] = proteins_df["sequence"].str.len() > 0
            proteins_df = (
                proteins_df
                .sort_values("_has_sequence", ascending=False)
                .drop_duplicates("protein_id")
                .drop(columns=["_has_sequence"])
            )
            proteins_df.to_csv(os.path.join(DATA_CLEAN_DIR, "proteins_clean.tsv"), sep="\t", index=False)
            logger.info(f"Created proteins_clean.tsv: {len(proteins_df)} rows")

        # Save location nodes and edges
        if location_nodes:
            loc_df = pd.DataFrame(location_nodes).drop_duplicates("location_id")
            loc_df.to_csv(os.path.join(DATA_CLEAN_DIR, "locations_clean.tsv"), sep="\t", index=False)
            logger.info(f"Created locations_clean.tsv: {len(loc_df)} rows")

        if edges_protein_location:
            epl_df = pd.DataFrame(edges_protein_location).drop_duplicates(subset=["protein_id", "location_id"])
            epl_df.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_protein_location.tsv"), sep="\t", index=False)
            logger.info(f"Created edges_protein_location.tsv: {len(epl_df)} rows")

        # Save Edges
        if edges_pd:
            pd.DataFrame(edges_pd).to_csv(os.path.join(DATA_CLEAN_DIR, "edges_protein_disease.tsv"), sep="\t", index=False)
        if edges_pv:
            pd.DataFrame(edges_pv).to_csv(os.path.join(DATA_CLEAN_DIR, "edges_protein_variant.tsv"), sep="\t", index=False)


def build_phase2_tables(data: dict):
    """
    Build Phase 2 edge tables from glycan detail data.
    
    This generates:
    - edges_glycan_enzyme.tsv: Glycan -> Enzyme relationships
    - edges_glycan_subsumption.tsv: Glycan -> Glycan hierarchy
    - edges_glycan_motif.tsv: Glycan -> Motif relationships
    - motifs_clean.tsv: Motif node table
    - reactions_clean.tsv: Reaction node table
    - edges_reaction.tsv: Reaction -> Enzyme/Glycan relationships
    - glycan_function_labels.tsv: Glycan function labels from GlyGen classification
    """
    if "glycan_details" not in data:
        logger.warning("No glycan_details found. Skipping Phase 2 table generation.")
        return
    
    glycan_details = data["glycan_details"]
    logger.info(f"Processing {len(glycan_details)} glycan details for Phase 2 tables...")
    
    # Collect edges and nodes
    edges_glycan_enzyme = []
    edges_glycan_subsumption = []
    edges_glycan_motif = []
    edges_enzyme_disease = []
    edges_enzyme_location = []
    motif_nodes = []
    reaction_nodes = []
    edges_reaction = []
    pathway_nodes = []
    glycan_function_labels = []
    missing_glycan_ids = set()
    missing_enzyme_nodes = {}
    
    # Track unique enzyme IDs for enzyme-disease edges
    enzyme_ids_from_glycans = set()
    
    # Handle both dict format (new) and list format (legacy)
    if isinstance(glycan_details, dict):
        details_iter = glycan_details.items()
    else:
        details_iter = [(d.get('glycan_id'), d) for d in glycan_details if d.get('glycan_id')]
    
    for glycan_id, detail in details_iter:
        if not glycan_id:
            continue
        
        # 1. Glycan -> Enzyme edges
        enzymes = detail.get('enzymes', [])
        for enzyme in enzymes:
            enzyme_id = enzyme.get('uniprot_canonical_ac')
            if not enzyme_id:
                continue
            
            # Only include human enzymes (tax_id 9606)
            tax_id = enzyme.get('tax_id')
            if tax_id and tax_id != 9606:
                continue
            
            enzyme_ids_from_glycans.add(enzyme_id)
            
            # Determine relationship type based on enzyme role
            # Guard against None values by defaulting to empty string
            gene = enzyme.get('gene') or ''
            protein_name = enzyme.get('protein_name') or ''
            
            # Default to produced_by, but could be refined based on enzyme type
            relation = 'produced_by'
            if 'hydrolase' in protein_name.lower() or 'ase' in gene.lower():
                relation = 'consumed_by'
            
            edges_glycan_enzyme.append({
                'glycan_id': glycan_id,
                'enzyme_id': enzyme_id,
                'relation': relation,
                'gene': gene,
                'protein_name': protein_name,
            })
            missing_enzyme_nodes.setdefault(
                enzyme_id,
                gene or protein_name or ""
            )
        
        # 2. Glycan -> Glycan subsumption edges
        subsumption = detail.get('subsumption', [])
        for sub in subsumption:
            related_id = sub.get('related_accession')
            relationship = sub.get('relationship')
            if not related_id or not relationship:
                continue
            
            # Map GlyGen relationship types to our edge types
            if relationship == 'ancestor':
                missing_glycan_ids.add(related_id)
                edges_glycan_subsumption.append({
                    'glycan_id': glycan_id,
                    'related_glycan_id': related_id,
                    'relation': 'child_of',
                })
            elif relationship == 'descendant':
                missing_glycan_ids.add(related_id)
                edges_glycan_subsumption.append({
                    'glycan_id': glycan_id,
                    'related_glycan_id': related_id,
                    'relation': 'parent_of',
                })
            elif relationship == 'subsumes':
                missing_glycan_ids.add(related_id)
                edges_glycan_subsumption.append({
                    'glycan_id': glycan_id,
                    'related_glycan_id': related_id,
                    'relation': 'subsumes',
                })
            elif relationship == 'subsumedby':
                missing_glycan_ids.add(related_id)
                edges_glycan_subsumption.append({
                    'glycan_id': glycan_id,
                    'related_glycan_id': related_id,
                    'relation': 'subsumed_by',
                })
        
        # 3. Glycan -> Motif edges
        motifs = detail.get('motifs', [])
        for motif in motifs:
            motif_id = motif.get('id') or motif.get('motif_id')
            motif_name = motif.get('name') or motif.get('motif_name', '')
            if not motif_id:
                continue
            
            motif_nodes.append({
                'motif_id': motif_id,
                'name': motif_name,
            })
            
            edges_glycan_motif.append({
                'glycan_id': glycan_id,
                'motif_id': motif_id,
                'relation': 'has_motif',
            })
        
        # 4. Extract residues for reaction information
        residues = detail.get('residues', [])
        for residue in residues:
            residue_id = residue.get('id')
            residue_name = residue.get('name', '')
            attached_by = residue.get('attachedby', '')
            parent_id = residue.get('parentid', '')
            
            if attached_by and attached_by.startswith('rxn.'):
                # This is a reaction
                reaction_id = attached_by
                enzyme_from_rxn = attached_by.replace('rxn.', '')
                
                reaction_nodes.append({
                    'reaction_id': reaction_id,
                    'residue_id': residue_id,
                    'residue_name': residue_name,
                    'glycan_id': glycan_id,
                })
                
                # Reaction -> Enzyme edge
                edges_reaction.append({
                    'reaction_id': reaction_id,
                    'target_id': enzyme_from_rxn,
                    'relation': 'catalyzed_by',
                    'target_type': 'enzyme',
                })
                missing_enzyme_nodes.setdefault(
                    enzyme_from_rxn,
                    ""
                )
                
                # Reaction -> Product Glycan edge
                edges_reaction.append({
                    'reaction_id': reaction_id,
                    'target_id': glycan_id,
                    'relation': 'has_product',
                    'target_type': 'glycan',
                })
        
        # 5. Extract classification for pathway / function labels
        classification = detail.get('classification', [])
        for cls in classification:
            cls_type = cls.get('type', {})
            cls_name = cls_type.get('name', '') if isinstance(cls_type, dict) else str(cls_type)
            cls_id = cls_type.get('id', '') if isinstance(cls_type, dict) else ''
            cls_name = str(cls_name).strip()
            cls_id = str(cls_id).strip()

            if cls_name:
                glycan_function_labels.append({
                    'glycan_id': glycan_id,
                    'function_term': cls_name,
                    'function_id': cls_id,
                    'source': 'GlyGen',
                })
            
            if cls_name and 'glycan' in cls_name.lower():
                pathway_id = f"pathway_{cls_id}" if cls_id else f"pathway_{cls_name.replace(' ', '_')}"
                pathway_nodes.append({
                    'pathway_id': pathway_id,
                    'name': cls_name,
                    'type': 'glycosylation_pathway',
                })
    
    # Save Phase 2 edge tables
    
    # Glycan -> Enzyme edges
    if edges_glycan_enzyme:
        df = pd.DataFrame(edges_glycan_enzyme).drop_duplicates(subset=['glycan_id', 'enzyme_id', 'relation'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_glycan_enzyme.tsv"), sep="\t", index=False)
        logger.info(f"Created edges_glycan_enzyme.tsv: {len(df)} rows")
    
    # Glycan -> Glycan subsumption edges
    if edges_glycan_subsumption:
        df = pd.DataFrame(edges_glycan_subsumption).drop_duplicates(subset=['glycan_id', 'related_glycan_id', 'relation'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_glycan_subsumption.tsv"), sep="\t", index=False)
        logger.info(f"Created edges_glycan_subsumption.tsv: {len(df)} rows")
    
    # Glycan -> Motif edges
    if edges_glycan_motif:
        df = pd.DataFrame(edges_glycan_motif).drop_duplicates(subset=['glycan_id', 'motif_id'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_glycan_motif.tsv"), sep="\t", index=False)
        logger.info(f"Created edges_glycan_motif.tsv: {len(df)} rows")
    
    # Motif nodes
    if motif_nodes:
        df = pd.DataFrame(motif_nodes).drop_duplicates(subset=['motif_id'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "motifs_clean.tsv"), sep="\t", index=False)
        logger.info(f"Created motifs_clean.tsv: {len(df)} rows")
    
    # Reaction nodes
    if reaction_nodes:
        df = pd.DataFrame(reaction_nodes).drop_duplicates(subset=['reaction_id'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "reactions_clean.tsv"), sep="\t", index=False)
        logger.info(f"Created reactions_clean.tsv: {len(df)} rows")
    
    # Reaction edges
    if edges_reaction:
        df = pd.DataFrame(edges_reaction).drop_duplicates(subset=['reaction_id', 'target_id', 'relation'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_reaction.tsv"), sep="\t", index=False)
        logger.info(f"Created edges_reaction.tsv: {len(df)} rows")
    
    # Pathway nodes
    if pathway_nodes:
        df = pd.DataFrame(pathway_nodes).drop_duplicates(subset=['pathway_id'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "pathways_clean.tsv"), sep="\t", index=False)
        logger.info(f"Created pathways_clean.tsv: {len(df)} rows")

    # Glycan function labels
    if glycan_function_labels:
        df = pd.DataFrame(glycan_function_labels).drop_duplicates(
            subset=['glycan_id', 'function_term']
        )
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "glycan_function_labels.tsv"), sep="\t", index=False)
        logger.info(f"Created glycan_function_labels.tsv: {len(df)} rows")
    
    # Generate enzyme-disease edges from UniProt data for enzymes found in glycan data
    if "uniprot_annotations" in data and enzyme_ids_from_glycans:
        up_df = data["uniprot_annotations"]
        for _, row in up_df.iterrows():
            uid = row.get("uniprot_id")
            if uid not in enzyme_ids_from_glycans:
                continue
            
            d_str = str(row.get("diseases", ""))
            if d_str and d_str != "nan":
                for item in d_str.split(";"):
                    if "|" in item:
                        did, dname = item.split("|", 1)
                        edges_enzyme_disease.append({
                            'enzyme_id': uid,
                            'disease_id': did,
                            'relation': 'associated_with_disease',
                        })
        
    if edges_enzyme_disease:
        df = pd.DataFrame(edges_enzyme_disease).drop_duplicates(subset=['enzyme_id', 'disease_id'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_enzyme_disease.tsv"), sep="\t", index=False)
        logger.info(f"Created edges_enzyme_disease.tsv: {len(df)} rows")

    # Generate enzyme-location edges from UniProt data for enzymes found in glycan data
    if "uniprot_annotations" in data and enzyme_ids_from_glycans:
        up_df = data["uniprot_annotations"]
        for _, row in up_df.iterrows():
            uid = row.get("uniprot_id")
            if uid not in enzyme_ids_from_glycans:
                continue

            loc_str = str(row.get("subcellular_locations", ""))
            if loc_str and loc_str != "nan":
                for raw_loc in loc_str.split(";"):
                    raw_loc = raw_loc.strip()
                    if not raw_loc:
                        continue
                    loc_id = normalize_location(raw_loc)
                    if loc_id:
                        edges_enzyme_location.append({
                            'enzyme_id': uid,
                            'location_id': loc_id,
                        })

    if edges_enzyme_location:
        df = pd.DataFrame(edges_enzyme_location).drop_duplicates(subset=['enzyme_id', 'location_id'])
        df.to_csv(os.path.join(DATA_CLEAN_DIR, "edges_enzyme_location.tsv"), sep="\t", index=False)
        logger.info(f"Created edges_enzyme_location.tsv: {len(df)} rows")

    # Ensure nodes for referenced glycans/enzyme IDs exist in base node tables.
    # This avoids invalid reference edges during KG construction.
    glycans_clean_path = os.path.join(DATA_CLEAN_DIR, "glycans_clean.tsv")
    if missing_glycan_ids and os.path.exists(glycans_clean_path):
        try:
            glycans_df = pd.read_csv(glycans_clean_path, sep="\t")
            existing_glycans = set(glycans_df["glycan_id"].astype(str))
            extra_glycans = sorted(
                str(gid) for gid in missing_glycan_ids if str(gid) not in existing_glycans
            )
            if extra_glycans:
                extra_df = pd.DataFrame(
                    [{"glycan_id": gid, "structure": ""} for gid in extra_glycans]
                )
                glycans_df = pd.concat([glycans_df, extra_df], ignore_index=True)
                glycans_df.to_csv(glycans_clean_path, sep="\t", index=False)
                logger.info(f"Added {len(extra_df)} glycan nodes referenced by phase-2 edges")
        except Exception as e:
            logger.warning(f"Failed to add missing glycans to glycans_clean.tsv: {e}")

    enzymes_clean_path = os.path.join(DATA_CLEAN_DIR, "enzymes_clean.tsv")
    if missing_enzyme_nodes and os.path.exists(enzymes_clean_path):
        try:
            enzymes_df = pd.read_csv(enzymes_clean_path, sep="\t")
            existing_enzymes = set(enzymes_df["enzyme_id"].astype(str))
            extra_rows = []
            for eid, gene_symbol in missing_enzyme_nodes.items():
                eid = str(eid)
                if eid not in existing_enzymes:
                    extra_rows.append({
                        "enzyme_id": eid,
                        "gene_symbol": str(gene_symbol or "")
                    })
            if extra_rows:
                enzymes_df = pd.concat([enzymes_df, pd.DataFrame(extra_rows)], ignore_index=True)
                enzymes_df.to_csv(enzymes_clean_path, sep="\t", index=False)
                logger.info(f"Added {len(extra_rows)} enzyme nodes referenced by phase-2 edges")
        except Exception as e:
            logger.warning(f"Failed to add missing enzymes to enzymes_clean.tsv: {e}")

    logger.info("Phase 2 table generation complete.")


def clean_uniprot_sites(input_json=None, output_tsv=None):
    """
    Load raw JSON site records from download, apply:
    - evidence filtering (config.uniprot_evidence_filter)
    - isoform canonicalization
    - residue/position validation
    - deduplication
    
    Save cleaned TSV to data_clean/uniprot_sites.tsv
    
    Args:
        input_json: Path to raw JSON site records. Defaults to data_raw/uniprot_sites_raw.json.
        output_tsv: Path to save cleaned TSV. Defaults to data_clean/uniprot_sites.tsv.
    
    Returns:
        DataFrame of cleaned sites or None if no data.
    """
    if input_json is None:
        input_json = os.path.join(DATA_RAW_DIR, "uniprot_sites_raw.json")
    if output_tsv is None:
        output_tsv = os.path.join(DATA_CLEAN_DIR, "uniprot_sites.tsv")
    
    if not config.site_data.enable_uniprot_sites:
        logger.info("UniProt sites cleaning disabled in config")
        return None
    
    if not os.path.exists(input_json):
        logger.warning(f"UniProt sites raw file not found: {input_json}")
        return None
    
    logger.info(f"Cleaning UniProt sites from {input_json}...")
    
    try:
        with open(input_json, 'r') as f:
            raw_sites = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load UniProt sites JSON: {e}")
        return None
    
    if not raw_sites:
        logger.warning("No raw UniProt sites found")
        return None
    
    from utils.sequence_tools import is_canonical_isoform, get_canonical_id
    
    evidence_filter = set(config.site_data.uniprot_evidence_filter)
    
    cleaned_sites = []
    filtered_evidence = 0
    filtered_isoform = 0
    filtered_residue = 0
    duplicates = 0
    
    seen_sites = set()
    valid_residues = set("ACDEFGHIKLMNPQRSTVWY")
    
    for site in raw_sites:
        eco_code = site.get('evidence_code', '')
        if eco_code and eco_code not in evidence_filter:
            filtered_evidence += 1
            continue
        
        uniprot_id = site.get('uniprot_id', '')
        if not uniprot_id:
            continue
        
        canonical_id = get_canonical_id(uniprot_id)
        if not is_canonical_isoform(canonical_id):
            filtered_isoform += 1
            continue
        
        residue = site.get('site_residue', '')
        if not residue or residue not in valid_residues:
            filtered_residue += 1
            continue
        
        position = site.get('site_position')
        if not position or not isinstance(position, int) or position < 1:
            filtered_residue += 1
            continue
        
        site_key = (canonical_id, position, residue)
        if site_key in seen_sites:
            duplicates += 1
            continue
        seen_sites.add(site_key)
        
        cleaned_sites.append({
            'uniprot_id': canonical_id,
            'site_position': position,
            'site_residue': residue,
            'site_type': site.get('site_type', ''),
            'evidence_code': eco_code,
            'evidence_type': site.get('evidence_type', ''),
            'source': 'UniProt'
        })
    
    logger.info(f"UniProt sites cleaning stats:")
    logger.info(f"  Input: {len(raw_sites)} raw sites")
    logger.info(f"  Filtered by evidence: {filtered_evidence}")
    logger.info(f"  Filtered by isoform: {filtered_isoform}")
    logger.info(f"  Filtered by residue/position: {filtered_residue}")
    logger.info(f"  Duplicates removed: {duplicates}")
    logger.info(f"  Output: {len(cleaned_sites)} cleaned sites")
    
    if cleaned_sites:
        df = pd.DataFrame(cleaned_sites)
        df.to_csv(output_tsv, sep="\t", index=False)
        logger.info(f"Saved cleaned UniProt sites to {output_tsv}")
        return df
    else:
        logger.warning("No cleaned UniProt sites to save")
        return None


def clean_ptmcode(input_dir=None, output_sites=None, output_edges=None):
    """
    Processes raw PTMCode data:
    - Load PTM site and edge tables
    - Filter PTMs by score >= config.ptmcode_min_score
    - Canonicalization of isoforms
    - Residue/position validation
    - Deduplication
    
    Produces:
      data_clean/ptmcode_sites.tsv
      data_clean/ptmcode_edges.tsv
    
    Args:
        input_dir: Directory containing raw PTMCode files. Defaults to data_raw/ptmcode_raw/.
        output_sites: Path to save cleaned sites TSV. Defaults to data_clean/ptmcode_sites.tsv.
        output_edges: Path to save cleaned edges TSV. Defaults to data_clean/ptmcode_edges.tsv.
    
    Returns:
        Tuple of (sites_df, edges_df) or (None, None) if no data.
    """
    if input_dir is None:
        input_dir = os.path.join(DATA_RAW_DIR, "ptmcode_raw")
    if output_sites is None:
        output_sites = os.path.join(DATA_CLEAN_DIR, "ptmcode_sites.tsv")
    if output_edges is None:
        output_edges = os.path.join(DATA_CLEAN_DIR, "ptmcode_edges.tsv")
    
    if not config.site_data.enable_ptmcode:
        logger.info("PTMCode cleaning disabled in config")
        return None, None
    
    sites_raw_path = os.path.join(input_dir, "sites_raw.tsv")
    edges_raw_path = os.path.join(input_dir, "edges_raw.tsv")
    
    if not os.path.exists(sites_raw_path):
        logger.warning(f"PTMCode sites raw file not found: {sites_raw_path}")
        return None, None
    
    if not os.path.exists(edges_raw_path):
        logger.warning(f"PTMCode edges raw file not found: {edges_raw_path}")
        return None, None
    
    logger.info(f"Cleaning PTMCode data from {input_dir}...")
    
    from utils.sequence_tools import is_canonical_isoform, get_canonical_id
    
    min_score = config.site_data.ptmcode_min_score
    valid_residues = set("ACDEFGHIKLMNPQRSTVWY")
    
    proteins_clean_path = os.path.join(DATA_CLEAN_DIR, "proteins_clean.tsv")
    allowed_proteins: set = set()
    if os.path.exists(proteins_clean_path):
        try:
            proteins_df = pd.read_csv(proteins_clean_path, sep="\t")
            if 'uniprot_id' in proteins_df.columns:
                allowed_proteins = set(proteins_df['uniprot_id'].dropna().astype(str))
            elif 'protein_id' in proteins_df.columns:
                allowed_proteins = set(proteins_df['protein_id'].dropna().astype(str))
            logger.info(f"Loaded {len(allowed_proteins)} allowed proteins from proteins_clean.tsv")
        except Exception as e:
            logger.warning(f"Failed to load proteins_clean.tsv: {e}")
    else:
        logger.warning(f"proteins_clean.tsv not found at {proteins_clean_path}, skipping protein filtering")
    
    try:
        sites_raw = pd.read_csv(sites_raw_path, sep="\t")
    except Exception as e:
        logger.error(f"Failed to load PTMCode sites: {e}")
        return None, None
    
    try:
        edges_raw = pd.read_csv(edges_raw_path, sep="\t")
    except Exception as e:
        logger.error(f"Failed to load PTMCode edges: {e}")
        return None, None
    
    cleaned_sites = []
    seen_sites = set()
    filtered_score = 0
    filtered_isoform = 0
    filtered_residue = 0
    filtered_not_in_kg = 0
    duplicates = 0
    
    for _, row in sites_raw.iterrows():
        score = row.get('score', 1.0)
        if pd.notna(score) and score < min_score:
            filtered_score += 1
            continue
        
        uniprot_id = str(row.get('uniprot_id', ''))
        if not uniprot_id or uniprot_id == 'nan':
            continue
        
        canonical_id = get_canonical_id(uniprot_id)
        if not is_canonical_isoform(canonical_id):
            filtered_isoform += 1
            continue
        
        if allowed_proteins and canonical_id not in allowed_proteins:
            filtered_not_in_kg += 1
            continue
        
        residue = str(row.get('site_residue', ''))
        if not residue or residue not in valid_residues:
            filtered_residue += 1
            continue
        
        position = row.get('site_position')
        if pd.isna(position):
            filtered_residue += 1
            continue
        position = int(position)
        if position < 1:
            filtered_residue += 1
            continue
        
        site_key = (canonical_id, position, residue)
        if site_key in seen_sites:
            duplicates += 1
            continue
        seen_sites.add(site_key)
        
        cleaned_sites.append({
            'uniprot_id': canonical_id,
            'site_position': position,
            'site_residue': residue,
            'ptm_type': row.get('ptm_type', ''),
            'score': score if pd.notna(score) else 1.0,
            'source': 'PTMCode'
        })
    
    logger.info(f"PTMCode sites cleaning stats:")
    logger.info(f"  Input: {len(sites_raw)} raw sites")
    logger.info(f"  Filtered by score: {filtered_score}")
    logger.info(f"  Filtered by isoform: {filtered_isoform}")
    logger.info(f"  Filtered by residue/position: {filtered_residue}")
    logger.info(f"  Filtered (protein not in KG): {filtered_not_in_kg}")
    logger.info(f"  Duplicates removed: {duplicates}")
    logger.info(f"  Output: {len(cleaned_sites)} cleaned sites")
    
    cleaned_edges = []
    seen_edges = set()
    edge_filtered_score = 0
    edge_filtered_site = 0
    edge_filtered_not_in_kg = 0
    edge_duplicates = 0
    
    valid_site_keys = seen_sites
    
    for _, row in edges_raw.iterrows():
        score = row.get('score', 1.0)
        if pd.notna(score) and score < min_score:
            edge_filtered_score += 1
            continue
        
        site1_uniprot = str(row.get('uniprot_id_a', row.get('site1_uniprot', '')))
        site1_position = row.get('site_position_a', row.get('site1_position'))
        site1_residue = str(row.get('residue_a', row.get('site1_residue', '')))
        site1_ptm_type = str(row.get('ptm_type_a', row.get('site1_ptm_type', '')))
        
        site2_uniprot = str(row.get('uniprot_id_b', row.get('site2_uniprot', '')))
        site2_position = row.get('site_position_b', row.get('site2_position'))
        site2_residue = str(row.get('residue_b', row.get('site2_residue', '')))
        site2_ptm_type = str(row.get('ptm_type_b', row.get('site2_ptm_type', '')))
        
        edge_type = str(row.get('edge_type', 'unknown'))
        
        if not site1_uniprot or site1_uniprot == 'nan' or not site2_uniprot or site2_uniprot == 'nan':
            edge_filtered_site += 1
            continue
        
        site1_canonical = get_canonical_id(site1_uniprot)
        site2_canonical = get_canonical_id(site2_uniprot)
        
        if allowed_proteins and (site1_canonical not in allowed_proteins or site2_canonical not in allowed_proteins):
            edge_filtered_not_in_kg += 1
            continue
        
        if pd.isna(site1_position) or pd.isna(site2_position):
            edge_filtered_site += 1
            continue
        
        site1_position = int(site1_position)
        site2_position = int(site2_position)
        
        edge_key = tuple(sorted([
            (site1_canonical, site1_position, site1_residue),
            (site2_canonical, site2_position, site2_residue)
        ]))
        
        if edge_key in seen_edges:
            edge_duplicates += 1
            continue
        seen_edges.add(edge_key)
        
        cleaned_edges.append({
            'site1_uniprot': site1_canonical,
            'site1_position': site1_position,
            'site1_residue': site1_residue,
            'site1_ptm_type': site1_ptm_type,
            'site2_uniprot': site2_canonical,
            'site2_position': site2_position,
            'site2_residue': site2_residue,
            'site2_ptm_type': site2_ptm_type,
            'score': score if pd.notna(score) else 1.0,
            'edge_type': edge_type,
            'source': 'PTMCode'
        })
    
    logger.info(f"PTMCode edges cleaning stats:")
    logger.info(f"  Input: {len(edges_raw)} raw edges")
    logger.info(f"  Filtered by score: {edge_filtered_score}")
    logger.info(f"  Filtered by invalid site: {edge_filtered_site}")
    logger.info(f"  Filtered (protein not in KG): {edge_filtered_not_in_kg}")
    logger.info(f"  Duplicates removed: {edge_duplicates}")
    logger.info(f"  Output: {len(cleaned_edges)} cleaned edges")
    
    sites_df = None
    edges_df = None
    
    if cleaned_sites:
        sites_df = pd.DataFrame(cleaned_sites)
        sites_df.to_csv(output_sites, sep="\t", index=False)
        logger.info(f"Saved cleaned PTMCode sites to {output_sites}")
    else:
        logger.warning("No cleaned PTMCode sites to save")
    
    if cleaned_edges:
        edges_df = pd.DataFrame(cleaned_edges)
        edges_df.to_csv(output_edges, sep="\t", index=False)
        logger.info(f"Saved cleaned PTMCode edges to {output_edges}")
    else:
        logger.warning("No cleaned PTMCode edges to save")
    
    return sites_df, edges_df


def main():
    """Main entry point for data cleaning pipeline."""
    start_time = datetime.now()
    logger.info(f"Starting data cleaning at {start_time}")
    
    os.makedirs(DATA_CLEAN_DIR, exist_ok=True)

    data = load_raw_data()
    
    if not data:
        logger.error("No raw data files found. Run download_data.py first.")
        return
    
    logger.info("Validating raw data...")
    validation_result = validate_raw_data(data)
    logger.info(validation_result.summary())
    
    if not validation_result.valid:
        logger.error("Raw data validation failed. Check errors above.")
        validation_report_path = os.path.join(LOG_DIR, "raw_data_validation.txt")
        with open(validation_report_path, 'w') as f:
            f.write(validation_result.summary())
            f.write("\n\nStats:\n")
            for k, v in validation_result.stats.items():
                f.write(f"  {k}: {v}\n")
        logger.info(f"Validation report saved to {validation_report_path}")
        return
    
    build_tables(data)
    
    # Phase 2: Build glycan-centric edge tables
    logger.info("Building Phase 2 glycan-centric tables...")
    build_phase2_tables(data)
    
    # Site-level PTM data cleaning
    logger.info("Cleaning site-level PTM data...")
    if config.site_data.enable_uniprot_sites:
        clean_uniprot_sites()
    
    if config.site_data.enable_ptmcode:
        clean_ptmcode()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Data cleaning completed at {end_time} (duration: {duration})")
    logger.info(f"Clean data saved to {DATA_CLEAN_DIR}")


if __name__ == "__main__":
    main()
