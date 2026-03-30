#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ptmcode_loader.py

Module for loading and processing PTMCode v3 data.
Extracts PTM site coordinates, types, and crosstalk edges.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

import pandas as pd

from .sequence_tools import (
    IsoformMapper,
    create_site_id,
    parse_uniprot_id,
    get_canonical_id,
)

logger = logging.getLogger(__name__)

PTM_TYPES = {
    'phosphorylation',
    'methylation',
    'acetylation',
    'ubiquitination',
    'sumoylation',
    'glycosylation',
}

PTM_TYPE_ALIASES = {
    'phos': 'phosphorylation',
    'phospho': 'phosphorylation',
    'meth': 'methylation',
    'methyl': 'methylation',
    'acet': 'acetylation',
    'acetyl': 'acetylation',
    'ubiq': 'ubiquitination',
    'ubiquitin': 'ubiquitination',
    'sumo': 'sumoylation',
    'glyc': 'glycosylation',
    'glyco': 'glycosylation',
}

DEFAULT_MIN_SCORE = 0.30


@dataclass
class PTMSite:
    """Represents a PTM site from PTMCode."""
    uniprot_id: str
    position: int
    residue: str
    ptm_type: str
    source: str = "PTMCode"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'uniprot_id': self.uniprot_id,
            'site_position': self.position,
            'site_residue': self.residue,
            'ptm_type': self.ptm_type,
            'source': self.source,
        }
    
    def get_site_id(self) -> str:
        """Generate standardized site ID."""
        return create_site_id(self.uniprot_id, self.position, self.residue)


@dataclass
class PTMCrosstalkEdge:
    """Represents a PTM crosstalk edge from PTMCode."""
    site1_uniprot: str
    site1_position: int
    site1_residue: str
    site1_ptm_type: str
    site2_uniprot: str
    site2_position: int
    site2_residue: str
    site2_ptm_type: str
    score: float
    source: str = "PTMCode"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'site1_id': self.get_site1_id(),
            'site2_id': self.get_site2_id(),
            'site1_uniprot': self.site1_uniprot,
            'site1_position': self.site1_position,
            'site1_residue': self.site1_residue,
            'site1_ptm_type': self.site1_ptm_type,
            'site2_uniprot': self.site2_uniprot,
            'site2_position': self.site2_position,
            'site2_residue': self.site2_residue,
            'site2_ptm_type': self.site2_ptm_type,
            'score': self.score,
            'source': self.source,
        }
    
    def get_site1_id(self) -> str:
        """Generate site ID for first site."""
        return create_site_id(self.site1_uniprot, self.site1_position, self.site1_residue)
    
    def get_site2_id(self) -> str:
        """Generate site ID for second site."""
        return create_site_id(self.site2_uniprot, self.site2_position, self.site2_residue)


@dataclass
class PTMCodeStats:
    """Statistics for PTMCode loading process."""
    total_sites_loaded: int = 0
    sites_after_filtering: int = 0
    sites_filtered_residue: int = 0
    sites_filtered_isoform: int = 0
    total_edges_loaded: int = 0
    edges_after_filtering: int = 0
    edges_filtered_score: int = 0
    edges_filtered_sites: int = 0
    unique_proteins: int = 0
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"PTMCode Loading Summary:\n"
            f"  Total sites loaded: {self.total_sites_loaded}\n"
            f"  Sites after filtering: {self.sites_after_filtering}\n"
            f"  Sites filtered (residue): {self.sites_filtered_residue}\n"
            f"  Sites filtered (isoform): {self.sites_filtered_isoform}\n"
            f"  Total edges loaded: {self.total_edges_loaded}\n"
            f"  Edges after filtering: {self.edges_after_filtering}\n"
            f"  Edges filtered (score): {self.edges_filtered_score}\n"
            f"  Edges filtered (sites): {self.edges_filtered_sites}\n"
            f"  Unique proteins: {self.unique_proteins}"
        )


def normalize_ptm_type(ptm_type: str) -> str:
    """
    Normalize PTM type string to standard form.
    
    Args:
        ptm_type: Raw PTM type string
    
    Returns:
        Normalized PTM type
    """
    if not ptm_type:
        return "unknown"
    
    ptm_lower = ptm_type.lower().strip()
    
    if ptm_lower in PTM_TYPES:
        return ptm_lower
    
    for alias, normalized in PTM_TYPE_ALIASES.items():
        if alias in ptm_lower:
            return normalized
    
    return ptm_lower


def parse_ptmcode_site_line(line: str) -> Optional[PTMSite]:
    """
    Parse a line from PTMCode sites file.
    
    Expected format varies by PTMCode version, but typically:
    UniProtID\tPosition\tResidue\tPTMType\t...
    
    Args:
        line: Tab-separated line from PTMCode file
    
    Returns:
        PTMSite object or None if parsing fails
    """
    if not line or line.startswith('#'):
        return None
    
    parts = line.strip().split('\t')
    if len(parts) < 4:
        return None
    
    try:
        uniprot_id = parts[0].strip()
        position = int(parts[1])
        residue = parts[2].strip().upper()
        ptm_type = normalize_ptm_type(parts[3])
        
        if not uniprot_id or position < 1 or len(residue) != 1:
            return None
        
        return PTMSite(
            uniprot_id=uniprot_id,
            position=position,
            residue=residue,
            ptm_type=ptm_type,
            source="PTMCode",
        )
    except (ValueError, IndexError):
        return None


def parse_ptmcode_edge_line(line: str) -> Optional[PTMCrosstalkEdge]:
    """
    Parse a line from PTMCode crosstalk edges file.
    
    Expected format:
    UniProt1\tPos1\tRes1\tType1\tUniProt2\tPos2\tRes2\tType2\tScore\t...
    
    Args:
        line: Tab-separated line from PTMCode file
    
    Returns:
        PTMCrosstalkEdge object or None if parsing fails
    """
    if not line or line.startswith('#'):
        return None
    
    parts = line.strip().split('\t')
    if len(parts) < 9:
        return None
    
    try:
        site1_uniprot = parts[0].strip()
        site1_position = int(parts[1])
        site1_residue = parts[2].strip().upper()
        site1_ptm_type = normalize_ptm_type(parts[3])
        
        site2_uniprot = parts[4].strip()
        site2_position = int(parts[5])
        site2_residue = parts[6].strip().upper()
        site2_ptm_type = normalize_ptm_type(parts[7])
        
        score = float(parts[8])
        
        if not site1_uniprot or not site2_uniprot:
            return None
        if site1_position < 1 or site2_position < 1:
            return None
        if len(site1_residue) != 1 or len(site2_residue) != 1:
            return None
        
        return PTMCrosstalkEdge(
            site1_uniprot=site1_uniprot,
            site1_position=site1_position,
            site1_residue=site1_residue,
            site1_ptm_type=site1_ptm_type,
            site2_uniprot=site2_uniprot,
            site2_position=site2_position,
            site2_residue=site2_residue,
            site2_ptm_type=site2_ptm_type,
            score=score,
            source="PTMCode",
        )
    except (ValueError, IndexError):
        return None


class PTMCodeLoader:
    """
    Loads and processes PTMCode v3 data.
    
    Handles site extraction, crosstalk edge loading, and filtering.
    """
    
    def __init__(
        self,
        min_score: float = DEFAULT_MIN_SCORE,
        normalize_isoforms: bool = True,
    ):
        """
        Initialize the loader.
        
        Args:
            min_score: Minimum confidence score for edges (default: 0.30)
            normalize_isoforms: Whether to normalize isoforms to canonical
        """
        self.min_score = min_score
        self.normalize_isoforms = normalize_isoforms
        self.stats = PTMCodeStats()
        self.isoform_mapper = IsoformMapper()
        self._valid_sites: Set[str] = set()
    
    def get_statistics(self) -> PTMCodeStats:
        """Get loading statistics."""
        return self.stats
    
    def reset_statistics(self):
        """Reset loading statistics."""
        self.stats = PTMCodeStats()
        self._valid_sites.clear()
    
    def load_sites_from_file(self, file_path: str) -> List[PTMSite]:
        """
        Load PTM sites from a file.
        
        Args:
            file_path: Path to PTMCode sites file
        
        Returns:
            List of PTMSite objects
        """
        sites = []
        
        if not os.path.exists(file_path):
            logger.warning(f"PTMCode sites file not found: {file_path}")
            return sites
        
        with open(file_path, 'r') as f:
            for line in f:
                site = parse_ptmcode_site_line(line)
                if site:
                    self.stats.total_sites_loaded += 1
                    
                    if self.normalize_isoforms:
                        result = self.isoform_mapper.map_site_to_canonical(
                            site.uniprot_id,
                            site.position,
                            site.residue
                        )
                        
                        if not result.success:
                            self.stats.sites_filtered_isoform += 1
                            continue
                        
                        site.uniprot_id = get_canonical_id(site.uniprot_id)
                        if result.canonical_position:
                            site.position = result.canonical_position
                        if result.canonical_residue:
                            site.residue = result.canonical_residue
                    
                    sites.append(site)
                    self._valid_sites.add(site.get_site_id())
        
        self.stats.sites_after_filtering = len(sites)
        
        unique_proteins = set(s.uniprot_id for s in sites)
        self.stats.unique_proteins = len(unique_proteins)
        
        logger.info(f"Loaded {len(sites)} PTM sites from {file_path}")
        return sites
    
    def load_sites_from_dataframe(self, df: pd.DataFrame) -> List[PTMSite]:
        """
        Load PTM sites from a DataFrame.
        
        Expected columns: uniprot_id, position, residue, ptm_type
        
        Args:
            df: DataFrame with PTM site data
        
        Returns:
            List of PTMSite objects
        """
        sites = []
        
        required_cols = {'uniprot_id', 'position', 'residue', 'ptm_type'}
        if not required_cols.issubset(df.columns):
            alt_cols = {'uniprot_id', 'site_position', 'site_residue', 'ptm_type'}
            if alt_cols.issubset(df.columns):
                df = df.rename(columns={
                    'site_position': 'position',
                    'site_residue': 'residue'
                })
            else:
                logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return sites
        
        for _, row in df.iterrows():
            self.stats.total_sites_loaded += 1
            
            try:
                uniprot_id = str(row['uniprot_id'])
                position = int(row['position'])
                residue = str(row['residue']).upper()
                ptm_type = normalize_ptm_type(str(row['ptm_type']))
                
                if not uniprot_id or position < 1 or len(residue) != 1:
                    self.stats.sites_filtered_residue += 1
                    continue
                
                if self.normalize_isoforms:
                    result = self.isoform_mapper.map_site_to_canonical(
                        uniprot_id, position, residue
                    )
                    
                    if not result.success:
                        self.stats.sites_filtered_isoform += 1
                        continue
                    
                    uniprot_id = get_canonical_id(uniprot_id)
                    if result.canonical_position:
                        position = result.canonical_position
                    if result.canonical_residue:
                        residue = result.canonical_residue
                
                site = PTMSite(
                    uniprot_id=uniprot_id,
                    position=position,
                    residue=residue,
                    ptm_type=ptm_type,
                    source="PTMCode",
                )
                
                sites.append(site)
                self._valid_sites.add(site.get_site_id())
                
            except (ValueError, KeyError) as e:
                logger.debug(f"Error parsing row: {e}")
                continue
        
        self.stats.sites_after_filtering = len(sites)
        
        unique_proteins = set(s.uniprot_id for s in sites)
        self.stats.unique_proteins = len(unique_proteins)
        
        logger.info(f"Loaded {len(sites)} PTM sites from DataFrame")
        return sites
    
    def load_edges_from_file(self, file_path: str) -> List[PTMCrosstalkEdge]:
        """
        Load PTM crosstalk edges from a file.
        
        Args:
            file_path: Path to PTMCode edges file
        
        Returns:
            List of PTMCrosstalkEdge objects
        """
        edges = []
        
        if not os.path.exists(file_path):
            logger.warning(f"PTMCode edges file not found: {file_path}")
            return edges
        
        with open(file_path, 'r') as f:
            for line in f:
                edge = parse_ptmcode_edge_line(line)
                if edge:
                    self.stats.total_edges_loaded += 1
                    
                    if edge.score < self.min_score:
                        self.stats.edges_filtered_score += 1
                        continue
                    
                    if self.normalize_isoforms:
                        edge.site1_uniprot = get_canonical_id(edge.site1_uniprot)
                        edge.site2_uniprot = get_canonical_id(edge.site2_uniprot)
                    
                    site1_id = edge.get_site1_id()
                    site2_id = edge.get_site2_id()
                    
                    if self._valid_sites and (site1_id not in self._valid_sites or site2_id not in self._valid_sites):
                        self.stats.edges_filtered_sites += 1
                        continue
                    
                    edges.append(edge)
        
        self.stats.edges_after_filtering = len(edges)
        
        logger.info(f"Loaded {len(edges)} PTM crosstalk edges from {file_path}")
        return edges
    
    def load_edges_from_dataframe(self, df: pd.DataFrame) -> List[PTMCrosstalkEdge]:
        """
        Load PTM crosstalk edges from a DataFrame.
        
        Args:
            df: DataFrame with edge data
        
        Returns:
            List of PTMCrosstalkEdge objects
        """
        edges = []
        
        required_cols = {
            'site1_uniprot', 'site1_position', 'site1_residue', 'site1_ptm_type',
            'site2_uniprot', 'site2_position', 'site2_residue', 'site2_ptm_type',
            'score'
        }
        
        if not required_cols.issubset(df.columns):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return edges
        
        for _, row in df.iterrows():
            self.stats.total_edges_loaded += 1
            
            try:
                score = float(row['score'])
                
                if score < self.min_score:
                    self.stats.edges_filtered_score += 1
                    continue
                
                site1_uniprot = str(row['site1_uniprot'])
                site2_uniprot = str(row['site2_uniprot'])
                
                if self.normalize_isoforms:
                    site1_uniprot = get_canonical_id(site1_uniprot)
                    site2_uniprot = get_canonical_id(site2_uniprot)
                
                edge = PTMCrosstalkEdge(
                    site1_uniprot=site1_uniprot,
                    site1_position=int(row['site1_position']),
                    site1_residue=str(row['site1_residue']).upper(),
                    site1_ptm_type=normalize_ptm_type(str(row['site1_ptm_type'])),
                    site2_uniprot=site2_uniprot,
                    site2_position=int(row['site2_position']),
                    site2_residue=str(row['site2_residue']).upper(),
                    site2_ptm_type=normalize_ptm_type(str(row['site2_ptm_type'])),
                    score=score,
                    source="PTMCode",
                )
                
                site1_id = edge.get_site1_id()
                site2_id = edge.get_site2_id()
                
                if self._valid_sites and (site1_id not in self._valid_sites or site2_id not in self._valid_sites):
                    self.stats.edges_filtered_sites += 1
                    continue
                
                edges.append(edge)
                
            except (ValueError, KeyError) as e:
                logger.debug(f"Error parsing edge row: {e}")
                continue
        
        self.stats.edges_after_filtering = len(edges)
        
        logger.info(f"Loaded {len(edges)} PTM crosstalk edges from DataFrame")
        return edges
    
    def sites_to_dataframe(self, sites: List[PTMSite]) -> pd.DataFrame:
        """
        Convert list of sites to DataFrame.
        
        Args:
            sites: List of PTMSite objects
        
        Returns:
            DataFrame with site data
        """
        if not sites:
            return pd.DataFrame(columns=[
                'uniprot_id', 'site_position', 'site_residue', 'ptm_type', 'source'
            ])
        
        return pd.DataFrame([site.to_dict() for site in sites])
    
    def edges_to_dataframe(self, edges: List[PTMCrosstalkEdge]) -> pd.DataFrame:
        """
        Convert list of edges to DataFrame.
        
        Args:
            edges: List of PTMCrosstalkEdge objects
        
        Returns:
            DataFrame with edge data
        """
        if not edges:
            return pd.DataFrame(columns=[
                'site1_id', 'site2_id',
                'site1_uniprot', 'site1_position', 'site1_residue', 'site1_ptm_type',
                'site2_uniprot', 'site2_position', 'site2_residue', 'site2_ptm_type',
                'score', 'source'
            ])
        
        return pd.DataFrame([edge.to_dict() for edge in edges])
    
    def save_sites(self, sites: List[PTMSite], output_path: str):
        """
        Save sites to TSV file.
        
        Args:
            sites: List of PTMSite objects
            output_path: Output file path
        """
        df = self.sites_to_dataframe(sites)
        df.to_csv(output_path, sep='\t', index=False)
        logger.info(f"Saved {len(df)} PTM sites to {output_path}")
    
    def save_edges(self, edges: List[PTMCrosstalkEdge], output_path: str):
        """
        Save edges to TSV file.
        
        Args:
            edges: List of PTMCrosstalkEdge objects
            output_path: Output file path
        """
        df = self.edges_to_dataframe(edges)
        df.to_csv(output_path, sep='\t', index=False)
        logger.info(f"Saved {len(df)} PTM crosstalk edges to {output_path}")


@dataclass
class PTMCode2ConversionStats:
    """Statistics for PTMCode2 conversion process."""
    lines_processed: int = 0
    lines_skipped_header: int = 0
    lines_skipped_species: int = 0
    lines_skipped_malformed: int = 0
    lines_skipped_score: int = 0
    lines_skipped_propagated: int = 0
    lines_skipped_unmapped: int = 0
    sites_extracted: int = 0
    sites_deduplicated: int = 0
    edges_within: int = 0
    edges_between: int = 0
    unique_gene_symbols: int = 0
    mapped_gene_symbols: int = 0
    
    def summary(self) -> str:
        return (
            f"PTMCode2 Conversion Summary:\n"
            f"  Lines processed: {self.lines_processed}\n"
            f"  Lines skipped (header/comment): {self.lines_skipped_header}\n"
            f"  Lines skipped (non-human species): {self.lines_skipped_species}\n"
            f"  Lines skipped (malformed): {self.lines_skipped_malformed}\n"
            f"  Lines skipped (low score): {self.lines_skipped_score}\n"
            f"  Lines skipped (propagated PTM): {self.lines_skipped_propagated}\n"
            f"  Lines skipped (unmapped gene symbol): {self.lines_skipped_unmapped}\n"
            f"  Unique gene symbols: {self.unique_gene_symbols}\n"
            f"  Mapped gene symbols: {self.mapped_gene_symbols}\n"
            f"  Sites extracted: {self.sites_extracted}\n"
            f"  Sites after deduplication: {self.sites_deduplicated}\n"
            f"  Edges (within protein): {self.edges_within}\n"
            f"  Edges (between proteins): {self.edges_between}"
        )


def _is_header_or_comment(line: str) -> bool:
    """
    Check if a line is a header or comment line.
    
    Only treats lines starting with '#' as comments.
    Empty lines are also skipped.
    """
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith('#'):
        return True
    return False


def _parse_site_token(token: str) -> Optional[Tuple[str, int]]:
    """
    Parse a residue+position token like "T6" or "S379" into (residue, position).
    
    Returns (residue, position) tuple or None if parsing fails.
    """
    if not token:
        return None
    token = token.strip()
    match = re.match(r'^([A-Za-z])(\d+)$', token)
    if not match:
        return None
    residue = match.group(1).upper()
    try:
        position = int(match.group(2))
        if position < 1:
            return None
        return (residue, position)
    except ValueError:
        return None


def _map_gene_symbols_to_uniprot(
    gene_symbols: Set[str],
    cache_path: Optional[str] = None,
    max_retries: int = 3,
    batch_size: int = 100,
) -> Dict[str, str]:
    """
    Map gene symbols to UniProt IDs using the UniProt search API.
    
    Uses the search endpoint with gene name queries, preferring reviewed
    Swiss-Prot entries over unreviewed TrEMBL entries.
    
    Uses caching to avoid repeated API calls.
    
    Args:
        gene_symbols: Set of gene symbols to map
        cache_path: Path to cache file (TSV format)
        max_retries: Maximum number of retries for API calls
        batch_size: Number of symbols per batch for progress logging
    
    Returns:
        Dictionary mapping gene symbol -> UniProt ID
    """
    import time
    try:
        import requests
    except ImportError:
        logger.error("requests library not available for UniProt ID mapping")
        return {}
    
    mapping: Dict[str, str] = {}
    failed_symbols: Set[str] = set()
    
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading gene symbol mapping cache from {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        if parts[1] == 'UNMAPPED':
                            failed_symbols.add(parts[0])
                        else:
                            mapping[parts[0]] = parts[1]
            logger.info(f"Loaded {len(mapping)} cached mappings, {len(failed_symbols)} known unmapped")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    symbols_to_map = gene_symbols - set(mapping.keys()) - failed_symbols
    if not symbols_to_map:
        logger.info("All gene symbols found in cache")
        return mapping
    
    logger.info(f"Mapping {len(symbols_to_map)} gene symbols to UniProt IDs via search API...")
    
    symbols_list = list(symbols_to_map)
    mapped_count = 0
    unmapped_count = 0
    
    for i, symbol in enumerate(symbols_list):
        if (i + 1) % batch_size == 0:
            logger.info(f"Progress: {i + 1}/{len(symbols_list)} symbols processed, {mapped_count} mapped")
        
        for attempt in range(max_retries):
            try:
                safe_symbol = symbol.replace('"', '').replace("'", "")
                url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{safe_symbol}+AND+organism_id:9606&format=json&size=10"
                
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                
                data = resp.json()
                results = data.get('results', [])
                
                if not results:
                    failed_symbols.add(symbol)
                    unmapped_count += 1
                    break
                
                reviewed_entries = [
                    r for r in results
                    if r.get('entryType') == 'UniProtKB reviewed (Swiss-Prot)'
                ]
                
                if reviewed_entries:
                    best_entry = reviewed_entries[0]
                else:
                    best_entry = results[0]
                
                uniprot_id = best_entry.get('primaryAccession', '')
                if uniprot_id:
                    mapping[symbol] = uniprot_id
                    mapped_count += 1
                else:
                    failed_symbols.add(symbol)
                    unmapped_count += 1
                
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.warning(f"Failed to map {symbol}: {e}")
                    failed_symbols.add(symbol)
                    unmapped_count += 1
        
        if i % 50 == 0:
            time.sleep(0.1)
    
    logger.info(f"Mapping complete: {mapped_count} mapped, {unmapped_count} unmapped")
    
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                f.write("# Gene symbol to UniProt ID mapping cache\n")
                f.write("# UNMAPPED entries indicate symbols that could not be mapped\n")
                for symbol, uniprot_id in sorted(mapping.items()):
                    f.write(f"{symbol}\t{uniprot_id}\n")
                for symbol in sorted(failed_symbols):
                    f.write(f"{symbol}\tUNMAPPED\n")
            logger.info(f"Saved {len(mapping)} mappings + {len(failed_symbols)} unmapped to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    return mapping


def _map_ensembl_ids_to_uniprot(
    ensembl_ids: Set[str],
    cache_path: Optional[str] = None,
    max_retries: int = 3,
    batch_size: int = 500,
) -> Dict[str, str]:
    """
    Map Ensembl Gene IDs (ENSG...) to UniProt IDs using the UniProt id-mapping API.
    
    Uses the id-mapping endpoint with from=Ensembl, which works for ENSG IDs.
    When multiple UniProt IDs are returned for a single ENSG ID, prefers
    reviewed Swiss-Prot entries.
    
    Uses caching to avoid repeated API calls.
    
    Args:
        ensembl_ids: Set of Ensembl Gene IDs to map (e.g., ENSG00000224841)
        cache_path: Path to cache file (TSV format)
        max_retries: Maximum number of retries for API calls
        batch_size: Number of IDs per batch for API calls
    
    Returns:
        Dictionary mapping Ensembl ID -> UniProt ID
    """
    import time
    try:
        import requests
    except ImportError:
        logger.error("requests library not available for UniProt ID mapping")
        return {}
    
    mapping: Dict[str, str] = {}
    failed_ids: Set[str] = set()
    
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading Ensembl ID mapping cache from {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        if parts[1] == 'UNMAPPED':
                            failed_ids.add(parts[0])
                        else:
                            mapping[parts[0]] = parts[1]
            logger.info(f"Loaded {len(mapping)} cached Ensembl mappings, {len(failed_ids)} known unmapped")
        except Exception as e:
            logger.warning(f"Failed to load Ensembl cache: {e}")
    
    ids_to_map = ensembl_ids - set(mapping.keys()) - failed_ids
    if not ids_to_map:
        logger.info("All Ensembl IDs found in cache")
        return mapping
    
    logger.info(f"Mapping {len(ids_to_map)} Ensembl IDs to UniProt IDs via id-mapping API...")
    
    ids_list = list(ids_to_map)
    mapped_count = 0
    unmapped_count = 0
    
    for batch_start in range(0, len(ids_list), batch_size):
        batch_end = min(batch_start + batch_size, len(ids_list))
        batch = ids_list[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start // batch_size + 1}/{(len(ids_list) + batch_size - 1) // batch_size} ({len(batch)} IDs)")
        
        for attempt in range(max_retries):
            try:
                submit_url = "https://rest.uniprot.org/idmapping/run"
                submit_data = {
                    "from": "Ensembl",
                    "to": "UniProtKB",
                    "ids": ",".join(batch)
                }
                submit_resp = requests.post(submit_url, data=submit_data, timeout=60)
                
                if submit_resp.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                submit_resp.raise_for_status()
                
                job_id = submit_resp.json().get("jobId")
                if not job_id:
                    logger.warning("No job ID returned from id-mapping API")
                    break
                
                time.sleep(1)
                
                for poll_attempt in range(30):
                    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
                    status_resp = requests.get(status_url, timeout=30)
                    status_data = status_resp.json()
                    
                    if "jobStatus" in status_data:
                        if status_data["jobStatus"] == "RUNNING":
                            time.sleep(1)
                            continue
                        elif status_data["jobStatus"] == "FINISHED":
                            break
                        else:
                            logger.warning(f"Unexpected job status: {status_data['jobStatus']}")
                            break
                    elif "results" in status_data or "failedIds" in status_data:
                        break
                    else:
                        time.sleep(1)
                
                results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
                results_resp = requests.get(results_url, timeout=60)
                results_resp.raise_for_status()
                results_data = results_resp.json()
                
                batch_results: Dict[str, List[str]] = {}
                for result in results_data.get("results", []):
                    from_id = result.get("from", "")
                    to_entry = result.get("to", {})
                    if isinstance(to_entry, dict):
                        to_id = to_entry.get("primaryAccession", "")
                        entry_type = to_entry.get("entryType", "")
                    else:
                        to_id = str(to_entry)
                        entry_type = ""
                    
                    if from_id and to_id:
                        if from_id not in batch_results:
                            batch_results[from_id] = []
                        batch_results[from_id].append((to_id, entry_type))
                
                for ensembl_id in batch:
                    if ensembl_id in batch_results:
                        entries = batch_results[ensembl_id]
                        reviewed = [e for e in entries if "Swiss-Prot" in e[1]]
                        if reviewed:
                            mapping[ensembl_id] = reviewed[0][0]
                        else:
                            mapping[ensembl_id] = entries[0][0]
                        mapped_count += 1
                    else:
                        failed_ids.add(ensembl_id)
                        unmapped_count += 1
                
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Batch mapping failed (attempt {attempt + 1}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Batch mapping failed after {max_retries} attempts: {e}")
                    for ensembl_id in batch:
                        if ensembl_id not in mapping:
                            failed_ids.add(ensembl_id)
                            unmapped_count += 1
        
        time.sleep(0.5)
    
    logger.info(f"Ensembl mapping complete: {mapped_count} mapped, {unmapped_count} unmapped")
    
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                f.write("# Ensembl Gene ID to UniProt ID mapping cache\n")
                f.write("# UNMAPPED entries indicate IDs that could not be mapped\n")
                for ensembl_id, uniprot_id in sorted(mapping.items()):
                    f.write(f"{ensembl_id}\t{uniprot_id}\n")
                for ensembl_id in sorted(failed_ids):
                    f.write(f"{ensembl_id}\tUNMAPPED\n")
            logger.info(f"Saved {len(mapping)} Ensembl mappings + {len(failed_ids)} unmapped to cache")
        except Exception as e:
            logger.warning(f"Failed to save Ensembl cache: {e}")
    
    return mapping


def _is_ensembl_id(identifier: str) -> bool:
    """Check if an identifier looks like an Ensembl ID (starts with ENS)."""
    return identifier.startswith('ENS')


def _parse_ptmcode2_within_line(
    parts: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Parse a line from PTMcode2_associations_within_proteins.txt.
    
    Column layout (14 columns, tab-separated):
    0: Protein (gene symbol)
    1: Species
    2: PTM1 (type)
    3: Residue1 (e.g., "T6")
    4: rRCS1 (score)
    5: Propagated1 (0/1)
    6: PTM2 (type)
    7: Residue2 (e.g., "S27")
    8: rRCS2 (score)
    9: Propagated2 (0/1)
    10-13: Evidence columns
    """
    if len(parts) < 10:
        return None
    
    try:
        gene_symbol = parts[0].strip()
        species = parts[1].strip()
        ptm_type_1 = parts[2].strip()
        site_token_1 = parts[3].strip()
        score_1 = float(parts[4].strip())
        propagated_1 = parts[5].strip() == '1'
        ptm_type_2 = parts[6].strip()
        site_token_2 = parts[7].strip()
        score_2 = float(parts[8].strip())
        propagated_2 = parts[9].strip() == '1'
        
        parsed_site_1 = _parse_site_token(site_token_1)
        parsed_site_2 = _parse_site_token(site_token_2)
        
        if not parsed_site_1 or not parsed_site_2:
            return None
        
        return {
            'gene_symbol_a': gene_symbol,
            'gene_symbol_b': gene_symbol,
            'species': species,
            'ptm_type_a': normalize_ptm_type(ptm_type_1),
            'residue_a': parsed_site_1[0],
            'position_a': parsed_site_1[1],
            'score_a': score_1,
            'propagated_a': propagated_1,
            'ptm_type_b': normalize_ptm_type(ptm_type_2),
            'residue_b': parsed_site_2[0],
            'position_b': parsed_site_2[1],
            'score_b': score_2,
            'propagated_b': propagated_2,
            'score': min(score_1, score_2),
        }
    except (ValueError, IndexError):
        return None


def _parse_ptmcode2_between_line(
    parts: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Parse a line from PTMcode2_associations_between_proteins.txt.
    
    Column layout (14 columns, tab-separated):
    0: Protein1 (gene symbol)
    1: Protein2 (gene symbol)
    2: Species
    3: PTM1 (type)
    4: Residue1 (e.g., "T6")
    5: rRCS1 (score)
    6: Propagated1 (0/1)
    7: PTM2 (type)
    8: Residue2 (e.g., "S27")
    9: rRCS2 (score)
    10: Propagated2 (0/1)
    11-13: Evidence columns
    """
    if len(parts) < 11:
        return None
    
    try:
        gene_symbol_a = parts[0].strip()
        gene_symbol_b = parts[1].strip()
        species = parts[2].strip()
        ptm_type_1 = parts[3].strip()
        site_token_1 = parts[4].strip()
        score_1 = float(parts[5].strip())
        propagated_1 = parts[6].strip() == '1'
        ptm_type_2 = parts[7].strip()
        site_token_2 = parts[8].strip()
        score_2 = float(parts[9].strip())
        propagated_2 = parts[10].strip() == '1'
        
        parsed_site_1 = _parse_site_token(site_token_1)
        parsed_site_2 = _parse_site_token(site_token_2)
        
        if not parsed_site_1 or not parsed_site_2:
            return None
        
        return {
            'gene_symbol_a': gene_symbol_a,
            'gene_symbol_b': gene_symbol_b,
            'species': species,
            'ptm_type_a': normalize_ptm_type(ptm_type_1),
            'residue_a': parsed_site_1[0],
            'position_a': parsed_site_1[1],
            'score_a': score_1,
            'propagated_a': propagated_1,
            'ptm_type_b': normalize_ptm_type(ptm_type_2),
            'residue_b': parsed_site_2[0],
            'position_b': parsed_site_2[1],
            'score_b': score_2,
            'propagated_b': propagated_2,
            'score': min(score_1, score_2),
        }
    except (ValueError, IndexError):
        return None


def convert_ptmcode2_to_raw(
    input_within: str,
    input_between: str,
    output_sites: str,
    output_edges: str,
    min_score: float = DEFAULT_MIN_SCORE,
    filter_propagated: bool = True,
    cache_dir: Optional[str] = None,
) -> PTMCode2ConversionStats:
    """
    Convert PTMCode2 raw association files to standardized TSV format.
    
    Parses PTMCode2 association files (within and between proteins) and generates:
    - sites_raw.tsv: Deduplicated PTM sites with UniProt IDs
    - edges_raw.tsv: PTM crosstalk edges with edge_type annotation
    
    This function:
    1. Filters to Homo sapiens rows only
    2. Maps gene symbols to UniProt IDs via the UniProt ID mapping API
    3. Parses residue+position tokens (e.g., "T6" -> T, 6)
    4. Optionally filters out propagated PTMs
    5. Applies minimum score threshold
    
    Uses streaming parsing to handle large files (1+ GB).
    
    Args:
        input_within: Path to PTMcode2_associations_within_proteins.txt
        input_between: Path to PTMcode2_associations_between_proteins.txt
        output_sites: Path to output sites_raw.tsv
        output_edges: Path to output edges_raw.tsv
        min_score: Minimum score threshold (rRCS) for edges (default: 0.30)
        filter_propagated: Whether to filter out propagated PTMs (default: True)
        cache_dir: Directory for caching gene symbol mappings (default: same as output)
    
    Returns:
        PTMCode2ConversionStats with conversion statistics
    """
    stats = PTMCode2ConversionStats()
    
    if cache_dir is None:
        cache_dir = os.path.dirname(output_sites)
    gene_symbol_cache_path = os.path.join(cache_dir, 'gene_symbol_to_uniprot_cache.tsv')
    ensembl_cache_path = os.path.join(cache_dir, 'ensembl_to_uniprot_cache.tsv')
    
    logger.info("Pass 1: Collecting unique identifiers from Homo sapiens rows...")
    gene_symbols: Set[str] = set()
    ensembl_ids: Set[str] = set()
    
    def collect_identifiers_within(file_path: str) -> None:
        if not os.path.exists(file_path):
            return
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if _is_header_or_comment(line):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 10:
                    continue
                species = parts[1].strip()
                if species != 'Homo sapiens':
                    continue
                identifier = parts[0].strip()
                if identifier:
                    if _is_ensembl_id(identifier):
                        ensembl_ids.add(identifier)
                    else:
                        gene_symbols.add(identifier)
    
    def collect_identifiers_between(file_path: str) -> None:
        if not os.path.exists(file_path):
            return
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if _is_header_or_comment(line):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 11:
                    continue
                species = parts[2].strip()
                if species != 'Homo sapiens':
                    continue
                identifier_a = parts[0].strip()
                identifier_b = parts[1].strip()
                if identifier_a:
                    if _is_ensembl_id(identifier_a):
                        ensembl_ids.add(identifier_a)
                    else:
                        gene_symbols.add(identifier_a)
                if identifier_b:
                    if _is_ensembl_id(identifier_b):
                        ensembl_ids.add(identifier_b)
                    else:
                        gene_symbols.add(identifier_b)
    
    collect_identifiers_within(input_within)
    collect_identifiers_between(input_between)
    
    stats.unique_gene_symbols = len(gene_symbols) + len(ensembl_ids)
    logger.info(f"Found {len(gene_symbols)} unique gene symbols and {len(ensembl_ids)} Ensembl IDs in Homo sapiens rows")
    
    id_to_uniprot: Dict[str, str] = {}
    
    if gene_symbols:
        logger.info("Mapping gene symbols to UniProt IDs via search API...")
        gene_to_uniprot = _map_gene_symbols_to_uniprot(
            gene_symbols,
            cache_path=gene_symbol_cache_path,
            max_retries=3,
            batch_size=500,
        )
        id_to_uniprot.update(gene_to_uniprot)
        logger.info(f"Mapped {len(gene_to_uniprot)} gene symbols to UniProt IDs")
    
    if ensembl_ids:
        logger.info("Mapping Ensembl IDs to UniProt IDs via id-mapping API...")
        ensembl_to_uniprot = _map_ensembl_ids_to_uniprot(
            ensembl_ids,
            cache_path=ensembl_cache_path,
            max_retries=3,
            batch_size=500,
        )
        id_to_uniprot.update(ensembl_to_uniprot)
        logger.info(f"Mapped {len(ensembl_to_uniprot)} Ensembl IDs to UniProt IDs")
    
    stats.mapped_gene_symbols = len(id_to_uniprot)
    logger.info(f"Total mapped identifiers: {len(id_to_uniprot)}")
    
    logger.info("Pass 2: Processing PTMCode2 files with UniProt mapping...")
    
    seen_sites: Set[Tuple[str, int, str, str]] = set()
    sites_list: List[Tuple[str, int, str, str]] = []
    edges_list: List[Dict[str, Any]] = []
    
    def process_within_file(file_path: str) -> None:
        if not os.path.exists(file_path):
            logger.warning(f"PTMCode2 file not found: {file_path}")
            return
        
        logger.info(f"Processing PTMCode2 within-protein file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                stats.lines_processed += 1
                
                if _is_header_or_comment(line):
                    stats.lines_skipped_header += 1
                    continue
                
                parts = line.strip().split('\t')
                parsed = _parse_ptmcode2_within_line(parts)
                
                if parsed is None:
                    stats.lines_skipped_malformed += 1
                    continue
                
                if parsed['species'] != 'Homo sapiens':
                    stats.lines_skipped_species += 1
                    continue
                
                if filter_propagated and (parsed['propagated_a'] or parsed['propagated_b']):
                    stats.lines_skipped_propagated += 1
                    continue
                
                score = parsed['score'] / 100.0
                if score < min_score:
                    stats.lines_skipped_score += 1
                    continue
                
                identifier = parsed['gene_symbol_a']
                uniprot_id = id_to_uniprot.get(identifier)
                if not uniprot_id:
                    stats.lines_skipped_unmapped += 1
                    continue
                
                site_a = (
                    uniprot_id,
                    parsed['position_a'],
                    parsed['residue_a'],
                    parsed['ptm_type_a'],
                )
                site_b = (
                    uniprot_id,
                    parsed['position_b'],
                    parsed['residue_b'],
                    parsed['ptm_type_b'],
                )
                
                if site_a not in seen_sites:
                    seen_sites.add(site_a)
                    sites_list.append(site_a)
                    stats.sites_extracted += 1
                
                if site_b not in seen_sites:
                    seen_sites.add(site_b)
                    sites_list.append(site_b)
                    stats.sites_extracted += 1
                
                edges_list.append({
                    'uniprot_id_a': uniprot_id,
                    'site_position_a': parsed['position_a'],
                    'residue_a': parsed['residue_a'],
                    'ptm_type_a': parsed['ptm_type_a'],
                    'uniprot_id_b': uniprot_id,
                    'site_position_b': parsed['position_b'],
                    'residue_b': parsed['residue_b'],
                    'ptm_type_b': parsed['ptm_type_b'],
                    'score': score,
                    'edge_type': 'within',
                    'source': 'PTMCode2',
                })
                stats.edges_within += 1
    
    def process_between_file(file_path: str) -> None:
        if not os.path.exists(file_path):
            logger.warning(f"PTMCode2 file not found: {file_path}")
            return
        
        logger.info(f"Processing PTMCode2 between-protein file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                stats.lines_processed += 1
                
                if _is_header_or_comment(line):
                    stats.lines_skipped_header += 1
                    continue
                
                parts = line.strip().split('\t')
                parsed = _parse_ptmcode2_between_line(parts)
                
                if parsed is None:
                    stats.lines_skipped_malformed += 1
                    continue
                
                if parsed['species'] != 'Homo sapiens':
                    stats.lines_skipped_species += 1
                    continue
                
                if filter_propagated and (parsed['propagated_a'] or parsed['propagated_b']):
                    stats.lines_skipped_propagated += 1
                    continue
                
                score = parsed['score'] / 100.0
                if score < min_score:
                    stats.lines_skipped_score += 1
                    continue
                
                uniprot_id_a = id_to_uniprot.get(parsed['gene_symbol_a'])
                uniprot_id_b = id_to_uniprot.get(parsed['gene_symbol_b'])
                
                if not uniprot_id_a or not uniprot_id_b:
                    stats.lines_skipped_unmapped += 1
                    continue
                
                site_a = (
                    uniprot_id_a,
                    parsed['position_a'],
                    parsed['residue_a'],
                    parsed['ptm_type_a'],
                )
                site_b = (
                    uniprot_id_b,
                    parsed['position_b'],
                    parsed['residue_b'],
                    parsed['ptm_type_b'],
                )
                
                if site_a not in seen_sites:
                    seen_sites.add(site_a)
                    sites_list.append(site_a)
                    stats.sites_extracted += 1
                
                if site_b not in seen_sites:
                    seen_sites.add(site_b)
                    sites_list.append(site_b)
                    stats.sites_extracted += 1
                
                edges_list.append({
                    'uniprot_id_a': uniprot_id_a,
                    'site_position_a': parsed['position_a'],
                    'residue_a': parsed['residue_a'],
                    'ptm_type_a': parsed['ptm_type_a'],
                    'uniprot_id_b': uniprot_id_b,
                    'site_position_b': parsed['position_b'],
                    'residue_b': parsed['residue_b'],
                    'ptm_type_b': parsed['ptm_type_b'],
                    'score': score,
                    'edge_type': 'between',
                    'source': 'PTMCode2',
                })
                stats.edges_between += 1
    
    process_within_file(input_within)
    process_between_file(input_between)
    
    stats.sites_deduplicated = len(sites_list)
    
    os.makedirs(os.path.dirname(output_sites), exist_ok=True)
    
    sites_list.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    
    with open(output_sites, 'w', encoding='utf-8') as f:
        f.write("uniprot_id\tsite_position\tsite_residue\tptm_type\tsource\n")
        for site in sites_list:
            f.write(f"{site[0]}\t{site[1]}\t{site[2]}\t{site[3]}\tPTMCode2\n")
    
    logger.info(f"Saved {len(sites_list)} sites to {output_sites}")
    
    edges_list.sort(key=lambda x: (
        x['uniprot_id_a'], x['site_position_a'],
        x['uniprot_id_b'], x['site_position_b'],
        x['edge_type']
    ))
    
    with open(output_edges, 'w', encoding='utf-8') as f:
        f.write("uniprot_id_a\tsite_position_a\tresidue_a\tptm_type_a\t")
        f.write("uniprot_id_b\tsite_position_b\tresidue_b\tptm_type_b\t")
        f.write("score\tedge_type\tsource\n")
        for edge in edges_list:
            f.write(
                f"{edge['uniprot_id_a']}\t{edge['site_position_a']}\t"
                f"{edge['residue_a']}\t{edge['ptm_type_a']}\t"
                f"{edge['uniprot_id_b']}\t{edge['site_position_b']}\t"
                f"{edge['residue_b']}\t{edge['ptm_type_b']}\t"
                f"{edge['score']}\t{edge['edge_type']}\t{edge['source']}\n"
            )
    
    logger.info(f"Saved {len(edges_list)} edges to {output_edges}")
    logger.info(stats.summary())
    
    return stats


def load_ptmcode_data(
    sites_path: Optional[str] = None,
    edges_path: Optional[str] = None,
    sites_output_path: Optional[str] = None,
    edges_output_path: Optional[str] = None,
    min_score: float = DEFAULT_MIN_SCORE,
    normalize_isoforms: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, PTMCodeStats]:
    """
    Main function to load PTMCode data.
    
    Args:
        sites_path: Path to PTMCode sites file
        edges_path: Path to PTMCode edges file
        sites_output_path: Output path for filtered sites
        edges_output_path: Output path for filtered edges
        min_score: Minimum confidence score for edges
        normalize_isoforms: Whether to normalize isoforms
    
    Returns:
        Tuple of (sites_df, edges_df, stats)
    """
    loader = PTMCodeLoader(
        min_score=min_score,
        normalize_isoforms=normalize_isoforms,
    )
    
    sites = []
    edges = []
    
    if sites_path:
        sites = loader.load_sites_from_file(sites_path)
        if sites_output_path:
            loader.save_sites(sites, sites_output_path)
    
    if edges_path:
        edges = loader.load_edges_from_file(edges_path)
        if edges_output_path:
            loader.save_edges(edges, edges_output_path)
    
    stats = loader.get_statistics()
    logger.info(stats.summary())
    
    return loader.sites_to_dataframe(sites), loader.edges_to_dataframe(edges), stats
