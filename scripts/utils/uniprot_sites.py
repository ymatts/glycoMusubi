#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
uniprot_sites.py

Module for extracting glycosylation site data from UniProt REST API.
Filters by evidence codes and extracts CARBOHYD features.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

import pandas as pd

from .api_client import APIClient
from .sequence_tools import (
    IsoformMapper,
    create_site_id,
    parse_uniprot_id,
    is_valid_position_range,
    validate_glycosylation_residue,
    VALID_GLYCOSYLATION_TYPES,
)

logger = logging.getLogger(__name__)

ALLOWED_EVIDENCE_CODES = {
    "ECO:0000269",
    "ECO:0000303",
    "ECO:0000250",
}

EVIDENCE_TYPE_MAP = {
    "ECO:0000269": "experimental",
    "ECO:0000303": "expert_curated",
    "ECO:0000250": "manual_assertion",
}

DISCARD_PATTERNS = [
    re.compile(r'\bby similarity\b', re.IGNORECASE),
    re.compile(r'\bpotential\b', re.IGNORECASE),
    re.compile(r'\bprobable\b', re.IGNORECASE),
    re.compile(r'\bpredicted\b', re.IGNORECASE),
]

GLYCO_TYPE_PATTERNS = {
    'N-linked': re.compile(r'N-linked|N-glyc', re.IGNORECASE),
    'O-linked': re.compile(r'O-linked|O-glyc|O-GlcNAc|O-GalNAc', re.IGNORECASE),
    'C-mannosylation': re.compile(r'C-mannosyl|C-linked', re.IGNORECASE),
    'GPI-anchor': re.compile(r'GPI-anchor|GPI anchor|glycosylphosphatidylinositol', re.IGNORECASE),
}


@dataclass
class GlycosylationSite:
    """Represents a glycosylation site from UniProt."""
    uniprot_id: str
    site_position: int
    site_residue: str
    site_type: str
    evidence_code: str
    evidence_type: str
    description: str = ""
    source: str = "UniProt"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'uniprot_id': self.uniprot_id,
            'site_position': self.site_position,
            'site_residue': self.site_residue,
            'site_type': self.site_type,
            'evidence_code': self.evidence_code,
            'evidence_type': self.evidence_type,
            'description': self.description,
            'source': self.source,
        }
    
    def get_site_id(self) -> str:
        """Generate standardized site ID."""
        return create_site_id(self.uniprot_id, self.site_position, self.site_residue)


@dataclass
class ExtractionStats:
    """Statistics for site extraction process."""
    total_proteins_processed: int = 0
    total_features_found: int = 0
    carbohyd_features: int = 0
    sites_extracted: int = 0
    sites_filtered_evidence: int = 0
    sites_filtered_description: int = 0
    sites_filtered_position: int = 0
    sites_filtered_residue: int = 0
    api_errors: int = 0
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"UniProt Site Extraction Summary:\n"
            f"  Proteins processed: {self.total_proteins_processed}\n"
            f"  Total features found: {self.total_features_found}\n"
            f"  CARBOHYD features: {self.carbohyd_features}\n"
            f"  Sites extracted: {self.sites_extracted}\n"
            f"  Filtered (evidence): {self.sites_filtered_evidence}\n"
            f"  Filtered (description): {self.sites_filtered_description}\n"
            f"  Filtered (position): {self.sites_filtered_position}\n"
            f"  Filtered (residue): {self.sites_filtered_residue}\n"
            f"  API errors: {self.api_errors}"
        )


def should_discard_description(description: str) -> bool:
    """
    Check if a feature description indicates it should be discarded.
    
    Args:
        description: Feature description text
    
    Returns:
        True if the description matches discard patterns
    """
    if not description:
        return False
    
    for pattern in DISCARD_PATTERNS:
        if pattern.search(description):
            return True
    
    return False


def extract_glyco_type(description: str) -> str:
    """
    Extract glycosylation type from feature description.
    
    Args:
        description: Feature description text
    
    Returns:
        Glycosylation type string
    """
    if not description:
        return "glycosylation"
    
    for glyco_type, pattern in GLYCO_TYPE_PATTERNS.items():
        if pattern.search(description):
            return glyco_type
    
    return "glycosylation"


def extract_evidence_codes(evidences: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract ECO evidence codes from UniProt evidence list.
    
    Args:
        evidences: List of evidence dictionaries from UniProt API
    
    Returns:
        Set of ECO codes
    """
    codes = set()
    
    for evidence in evidences:
        eco_id = evidence.get('evidenceCode')
        if eco_id:
            codes.add(eco_id)
    
    return codes


def has_allowed_evidence(evidences: List[Dict[str, Any]], allowed_codes: Set[str]) -> Tuple[bool, str]:
    """
    Check if any evidence code is in the allowed set.
    
    Args:
        evidences: List of evidence dictionaries
        allowed_codes: Set of allowed ECO codes
    
    Returns:
        Tuple of (has_allowed, best_evidence_code)
    """
    found_codes = extract_evidence_codes(evidences)
    
    for code in allowed_codes:
        if code in found_codes:
            return True, code
    
    return False, ""


def parse_position(location: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse position from UniProt feature location.
    
    Args:
        location: Location dictionary from UniProt API
    
    Returns:
        Tuple of (position, residue) or (None, None) if invalid
    """
    start = location.get('start', {})
    end = location.get('end', {})
    
    start_val = start.get('value')
    end_val = end.get('value')
    
    if start_val is None:
        return None, None
    
    if end_val is not None and start_val != end_val:
        return None, None
    
    if start.get('modifier') == 'UNKNOWN' or end.get('modifier') == 'UNKNOWN':
        return None, None
    
    try:
        position = int(start_val)
        if position < 1:
            return None, None
        return position, None
    except (ValueError, TypeError):
        return None, None


class UniProtSiteExtractor:
    """
    Extracts glycosylation sites from UniProt REST API.
    
    Filters by evidence codes and extracts CARBOHYD features only.
    """
    
    def __init__(
        self,
        allowed_evidence_codes: Optional[Set[str]] = None,
        base_url: str = "https://rest.uniprot.org/uniprotkb",
        max_retries: int = 3,
        timeout: int = 30,
        rate_limit: float = 10.0,
    ):
        """
        Initialize the extractor.
        
        Args:
            allowed_evidence_codes: Set of ECO codes to accept (default: experimental, curated, manual)
            base_url: UniProt REST API base URL
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            rate_limit: Maximum requests per second
        """
        self.allowed_evidence_codes = allowed_evidence_codes or ALLOWED_EVIDENCE_CODES
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.stats = ExtractionStats()
        self.isoform_mapper = IsoformMapper()
        self._sequence_cache: Dict[str, str] = {}
    
    def get_statistics(self) -> ExtractionStats:
        """Get extraction statistics."""
        return self.stats
    
    def reset_statistics(self):
        """Reset extraction statistics."""
        self.stats = ExtractionStats()
    
    def extract_sites_from_protein(
        self,
        data: Dict[str, Any],
        uniprot_id: str
    ) -> List[GlycosylationSite]:
        """
        Extract glycosylation sites from UniProt protein data.
        
        Args:
            data: UniProt API response data
            uniprot_id: UniProt accession
        
        Returns:
            List of GlycosylationSite objects
        """
        sites = []
        
        sequence = data.get('sequence', {}).get('value', '')
        if sequence:
            self._sequence_cache[uniprot_id] = sequence
            self.isoform_mapper.cache_sequence(uniprot_id, sequence)
        
        features = data.get('features', [])
        self.stats.total_features_found += len(features)
        
        for feature in features:
            feature_type = feature.get('type', '')
            
            if feature_type != 'Glycosylation':
                continue
            
            self.stats.carbohyd_features += 1
            
            description = feature.get('description', '')
            
            if should_discard_description(description):
                self.stats.sites_filtered_description += 1
                continue
            
            evidences = feature.get('evidences', [])
            has_valid_evidence, evidence_code = has_allowed_evidence(
                evidences, self.allowed_evidence_codes
            )
            
            if not has_valid_evidence:
                self.stats.sites_filtered_evidence += 1
                continue
            
            location = feature.get('location', {})
            position, _ = parse_position(location)
            
            if position is None:
                self.stats.sites_filtered_position += 1
                continue
            
            if sequence and position <= len(sequence):
                residue = sequence[position - 1]
            else:
                residue = ""
            
            glyco_type = extract_glyco_type(description)
            
            if residue and not validate_glycosylation_residue(residue, glyco_type):
                self.stats.sites_filtered_residue += 1
                continue
            
            evidence_type = EVIDENCE_TYPE_MAP.get(evidence_code, "unknown")
            
            site = GlycosylationSite(
                uniprot_id=uniprot_id,
                site_position=position,
                site_residue=residue,
                site_type=glyco_type,
                evidence_code=evidence_code,
                evidence_type=evidence_type,
                description=description,
                source="UniProt",
            )
            
            sites.append(site)
            self.stats.sites_extracted += 1
        
        return sites
    
    def fetch_and_extract(
        self,
        uniprot_ids: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[GlycosylationSite]:
        """
        Fetch UniProt data and extract glycosylation sites.
        
        Args:
            uniprot_ids: List of UniProt accessions to process
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of GlycosylationSite objects
        """
        all_sites = []
        
        with APIClient(
            max_retries=self.max_retries,
            backoff_factor=2.0,
            timeout=self.timeout,
            rate_limit=self.rate_limit
        ) as client:
            for i, uid in enumerate(uniprot_ids):
                if progress_callback:
                    progress_callback(i, len(uniprot_ids), uid)
                
                base_ac, _ = parse_uniprot_id(uid)
                url = f"{self.base_url}/{base_ac}.json"
                
                try:
                    data = client.get_json(url)
                    
                    if data is None:
                        self.stats.api_errors += 1
                        logger.warning(f"No data returned for {uid}")
                        continue
                    
                    self.stats.total_proteins_processed += 1
                    
                    sites = self.extract_sites_from_protein(data, base_ac)
                    all_sites.extend(sites)
                    
                except Exception as e:
                    self.stats.api_errors += 1
                    logger.error(f"Error fetching {uid}: {e}")
        
        return all_sites
    
    def sites_to_dataframe(self, sites: List[GlycosylationSite]) -> pd.DataFrame:
        """
        Convert list of sites to DataFrame.
        
        Args:
            sites: List of GlycosylationSite objects
        
        Returns:
            DataFrame with site data
        """
        if not sites:
            return pd.DataFrame(columns=[
                'uniprot_id', 'site_position', 'site_residue', 'site_type',
                'evidence_code', 'evidence_type', 'description', 'source'
            ])
        
        return pd.DataFrame([site.to_dict() for site in sites])
    
    def save_sites(self, sites: List[GlycosylationSite], output_path: str):
        """
        Save sites to TSV file.
        
        Args:
            sites: List of GlycosylationSite objects
            output_path: Output file path
        """
        df = self.sites_to_dataframe(sites)
        df.to_csv(output_path, sep='\t', index=False)
        logger.info(f"Saved {len(df)} sites to {output_path}")


def fetch_uniprot_sites(
    uniprot_ids: List[str],
    output_path: str,
    allowed_evidence_codes: Optional[Set[str]] = None,
    base_url: str = "https://rest.uniprot.org/uniprotkb",
    max_retries: int = 3,
    timeout: int = 30,
    rate_limit: float = 10.0,
) -> Tuple[pd.DataFrame, ExtractionStats]:
    """
    Main function to fetch and extract UniProt glycosylation sites.
    
    Args:
        uniprot_ids: List of UniProt accessions
        output_path: Output TSV file path
        allowed_evidence_codes: Set of allowed ECO codes
        base_url: UniProt API base URL
        max_retries: Maximum retry attempts
        timeout: Request timeout
        rate_limit: Requests per second limit
    
    Returns:
        Tuple of (DataFrame, ExtractionStats)
    """
    extractor = UniProtSiteExtractor(
        allowed_evidence_codes=allowed_evidence_codes,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
        rate_limit=rate_limit,
    )
    
    logger.info(f"Extracting glycosylation sites from {len(uniprot_ids)} proteins...")
    
    sites = extractor.fetch_and_extract(uniprot_ids)
    
    extractor.save_sites(sites, output_path)
    
    stats = extractor.get_statistics()
    logger.info(stats.summary())
    
    return extractor.sites_to_dataframe(sites), stats
