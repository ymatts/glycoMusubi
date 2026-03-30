#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sequence_tools.py

Utilities for isoform normalization and sequence alignment.
Maps PTM site positions from isoforms to canonical sequences.
"""

import re
import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SiteMappingResult:
    """Result of mapping a site position to canonical sequence."""
    success: bool
    canonical_position: Optional[int] = None
    canonical_residue: Optional[str] = None
    original_position: int = 0
    original_residue: str = ""
    error_message: str = ""
    
    def __bool__(self) -> bool:
        return self.success


def parse_uniprot_id(uniprot_id: str) -> Tuple[str, Optional[int]]:
    """
    Parse UniProt ID into base accession and isoform number.
    
    Args:
        uniprot_id: UniProt accession (e.g., 'P12345' or 'P12345-2')
    
    Returns:
        Tuple of (base_accession, isoform_number or None for canonical)
    """
    if not uniprot_id:
        return "", None
    
    match = re.match(r'^([A-Z0-9]+)(?:-(\d+))?$', uniprot_id.strip())
    if not match:
        return uniprot_id, None
    
    base_ac = match.group(1)
    isoform = int(match.group(2)) if match.group(2) else None
    
    return base_ac, isoform


def get_canonical_id(uniprot_id: str) -> str:
    """
    Get the canonical UniProt ID (without isoform suffix or with -1).
    
    Args:
        uniprot_id: UniProt accession
    
    Returns:
        Canonical UniProt ID
    """
    base_ac, isoform = parse_uniprot_id(uniprot_id)
    if isoform is None or isoform == 1:
        return base_ac
    return f"{base_ac}-1"


def is_canonical_isoform(uniprot_id: str) -> bool:
    """
    Check if a UniProt ID refers to the canonical isoform.
    
    Args:
        uniprot_id: UniProt accession
    
    Returns:
        True if canonical (no suffix or -1)
    """
    _, isoform = parse_uniprot_id(uniprot_id)
    return isoform is None or isoform == 1


class IsoformMapper:
    """
    Maps site positions between isoforms using UniProt sequence data.
    
    This class handles the mapping of PTM site positions from non-canonical
    isoforms to the canonical (-1) isoform using sequence alignment.
    """
    
    def __init__(self):
        self._sequence_cache: Dict[str, str] = {}
        self._isoform_cache: Dict[str, Dict[int, str]] = {}
        self._mapping_stats = {
            'total_mappings': 0,
            'successful_mappings': 0,
            'failed_mappings': 0,
            'residue_mismatches': 0,
            'ambiguous_mappings': 0,
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get mapping statistics."""
        return self._mapping_stats.copy()
    
    def cache_sequence(self, uniprot_id: str, sequence: str):
        """
        Cache a protein sequence for later use.
        
        Args:
            uniprot_id: UniProt accession
            sequence: Amino acid sequence
        """
        self._sequence_cache[uniprot_id] = sequence
    
    def cache_isoform_sequences(self, base_ac: str, isoform_sequences: Dict[int, str]):
        """
        Cache all isoform sequences for a protein.
        
        Args:
            base_ac: Base UniProt accession (without isoform suffix)
            isoform_sequences: Dict mapping isoform number to sequence
        """
        self._isoform_cache[base_ac] = isoform_sequences
    
    def get_sequence(self, uniprot_id: str) -> Optional[str]:
        """
        Get cached sequence for a UniProt ID.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            Sequence string or None if not cached
        """
        if uniprot_id in self._sequence_cache:
            return self._sequence_cache[uniprot_id]
        
        base_ac, isoform = parse_uniprot_id(uniprot_id)
        if base_ac in self._isoform_cache:
            isoform_num = isoform if isoform else 1
            return self._isoform_cache[base_ac].get(isoform_num)
        
        return None
    
    def map_site_to_canonical(
        self,
        accession: str,
        position: int,
        expected_residue: Optional[str] = None
    ) -> SiteMappingResult:
        """
        Map a site position from any isoform to the canonical sequence.
        
        This method handles the mapping of PTM site positions from non-canonical
        isoforms to the canonical (-1) isoform. For canonical isoforms, it simply
        validates the position and residue.
        
        Args:
            accession: UniProt accession (e.g., 'P12345' or 'P12345-2')
            position: 1-based position in the isoform sequence
            expected_residue: Expected amino acid at the position (optional)
        
        Returns:
            SiteMappingResult with mapping outcome
        """
        self._mapping_stats['total_mappings'] += 1
        
        base_ac, isoform = parse_uniprot_id(accession)
        
        if isoform is None or isoform == 1:
            canonical_seq = self.get_sequence(base_ac) or self.get_sequence(f"{base_ac}-1")
            
            if canonical_seq:
                if position < 1 or position > len(canonical_seq):
                    self._mapping_stats['failed_mappings'] += 1
                    return SiteMappingResult(
                        success=False,
                        original_position=position,
                        original_residue=expected_residue or "",
                        error_message=f"Position {position} out of range (1-{len(canonical_seq)})"
                    )
                
                actual_residue = canonical_seq[position - 1]
                
                if expected_residue and actual_residue != expected_residue:
                    self._mapping_stats['residue_mismatches'] += 1
                    self._mapping_stats['failed_mappings'] += 1
                    return SiteMappingResult(
                        success=False,
                        canonical_position=position,
                        canonical_residue=actual_residue,
                        original_position=position,
                        original_residue=expected_residue,
                        error_message=f"Residue mismatch: expected {expected_residue}, found {actual_residue}"
                    )
                
                self._mapping_stats['successful_mappings'] += 1
                return SiteMappingResult(
                    success=True,
                    canonical_position=position,
                    canonical_residue=actual_residue,
                    original_position=position,
                    original_residue=expected_residue or actual_residue
                )
            
            self._mapping_stats['successful_mappings'] += 1
            return SiteMappingResult(
                success=True,
                canonical_position=position,
                canonical_residue=expected_residue,
                original_position=position,
                original_residue=expected_residue or ""
            )
        
        isoform_seq = self.get_sequence(accession)
        canonical_seq = self.get_sequence(base_ac) or self.get_sequence(f"{base_ac}-1")
        
        if not isoform_seq or not canonical_seq:
            self._mapping_stats['failed_mappings'] += 1
            return SiteMappingResult(
                success=False,
                original_position=position,
                original_residue=expected_residue or "",
                error_message="Missing sequence data for isoform mapping"
            )
        
        if position < 1 or position > len(isoform_seq):
            self._mapping_stats['failed_mappings'] += 1
            return SiteMappingResult(
                success=False,
                original_position=position,
                original_residue=expected_residue or "",
                error_message=f"Position {position} out of range for isoform (1-{len(isoform_seq)})"
            )
        
        isoform_residue = isoform_seq[position - 1]
        
        if expected_residue and isoform_residue != expected_residue:
            self._mapping_stats['residue_mismatches'] += 1
            self._mapping_stats['failed_mappings'] += 1
            return SiteMappingResult(
                success=False,
                original_position=position,
                original_residue=expected_residue,
                error_message=f"Residue mismatch in isoform: expected {expected_residue}, found {isoform_residue}"
            )
        
        canonical_position = self._align_and_map_position(
            isoform_seq, canonical_seq, position
        )
        
        if canonical_position is None:
            self._mapping_stats['ambiguous_mappings'] += 1
            self._mapping_stats['failed_mappings'] += 1
            return SiteMappingResult(
                success=False,
                original_position=position,
                original_residue=isoform_residue,
                error_message="Could not map position to canonical sequence"
            )
        
        canonical_residue = canonical_seq[canonical_position - 1]
        
        if canonical_residue != isoform_residue:
            self._mapping_stats['residue_mismatches'] += 1
            self._mapping_stats['failed_mappings'] += 1
            return SiteMappingResult(
                success=False,
                canonical_position=canonical_position,
                canonical_residue=canonical_residue,
                original_position=position,
                original_residue=isoform_residue,
                error_message=f"Residue mismatch after mapping: isoform has {isoform_residue}, canonical has {canonical_residue}"
            )
        
        self._mapping_stats['successful_mappings'] += 1
        return SiteMappingResult(
            success=True,
            canonical_position=canonical_position,
            canonical_residue=canonical_residue,
            original_position=position,
            original_residue=isoform_residue
        )
    
    def _align_and_map_position(
        self,
        isoform_seq: str,
        canonical_seq: str,
        position: int
    ) -> Optional[int]:
        """
        Align isoform to canonical sequence and map position.
        
        Uses a simple alignment approach that works well for isoforms
        that differ by insertions/deletions.
        
        Args:
            isoform_seq: Isoform sequence
            canonical_seq: Canonical sequence
            position: 1-based position in isoform
        
        Returns:
            1-based position in canonical sequence, or None if unmappable
        """
        if isoform_seq == canonical_seq:
            return position
        
        target_residue = isoform_seq[position - 1]
        
        if len(isoform_seq) == len(canonical_seq):
            if canonical_seq[position - 1] == target_residue:
                return position
            return None
        
        window_size = 10
        start = max(0, position - 1 - window_size)
        end = min(len(isoform_seq), position + window_size)
        context = isoform_seq[start:end]
        
        best_match_pos = None
        best_match_score = 0
        
        for i in range(max(0, position - 1 - 50), min(len(canonical_seq), position + 50)):
            if canonical_seq[i] != target_residue:
                continue
            
            can_start = max(0, i - window_size)
            can_end = min(len(canonical_seq), i + 1 + window_size)
            can_context = canonical_seq[can_start:can_end]
            
            score = self._score_context_match(context, can_context, target_residue)
            
            if score > best_match_score:
                best_match_score = score
                best_match_pos = i + 1
        
        if best_match_score >= 0.5:
            return best_match_pos
        
        return None
    
    def _score_context_match(
        self,
        context1: str,
        context2: str,
        target_residue: str
    ) -> float:
        """
        Score how well two sequence contexts match.
        
        Args:
            context1: First context string
            context2: Second context string
            target_residue: The residue being mapped
        
        Returns:
            Match score between 0 and 1
        """
        if not context1 or not context2:
            return 0.0
        
        matches = 0
        comparisons = 0
        
        for i, c1 in enumerate(context1):
            if i < len(context2):
                comparisons += 1
                if c1 == context2[i]:
                    matches += 1
        
        if comparisons == 0:
            return 0.0
        
        return matches / comparisons


def create_site_id(uniprot_id: str, position: int, residue: str) -> str:
    """
    Create a standardized site ID.
    
    Format: SITE::<UniProtID>::<Position>::<Residue>
    
    Args:
        uniprot_id: UniProt accession (canonical form preferred)
        position: 1-based position
        residue: Single-letter amino acid code
    
    Returns:
        Standardized site ID string
    """
    base_ac, isoform = parse_uniprot_id(uniprot_id)
    if isoform is None or isoform == 1:
        canonical_id = f"{base_ac}-1"
    else:
        canonical_id = f"{base_ac}-{isoform}"
    
    return f"SITE::{canonical_id}::{position}::{residue}"


def parse_site_id(site_id: str) -> Optional[Tuple[str, int, str]]:
    """
    Parse a standardized site ID.
    
    Args:
        site_id: Site ID string (e.g., 'SITE::P12345-1::234::N')
    
    Returns:
        Tuple of (uniprot_id, position, residue) or None if invalid
    """
    if not site_id or not site_id.startswith("SITE::"):
        return None
    
    parts = site_id.split("::")
    if len(parts) != 4:
        return None
    
    try:
        uniprot_id = parts[1]
        position = int(parts[2])
        residue = parts[3]
        
        if len(residue) != 1 or not residue.isalpha():
            return None
        
        return (uniprot_id, position, residue)
    except (ValueError, IndexError):
        return None


def validate_site_id(site_id: str) -> bool:
    """
    Validate a site ID format.
    
    Args:
        site_id: Site ID string
    
    Returns:
        True if valid format
    """
    return parse_site_id(site_id) is not None


def is_valid_position_range(position_str: str) -> bool:
    """
    Check if a position string represents a valid single position (not a range).
    
    Args:
        position_str: Position string from UniProt
    
    Returns:
        True if single position, False if range or invalid
    """
    if not position_str:
        return False
    
    position_str = str(position_str).strip()
    
    if '..' in position_str or '-' in position_str or '?' in position_str:
        return False
    
    try:
        pos = int(position_str)
        return pos > 0
    except ValueError:
        return False


GLYCOSYLATION_RESIDUES = {
    'N-linked': ['N'],
    'O-linked': ['S', 'T', 'Y'],
    'C-mannosylation': ['W'],
    'GPI-anchor': ['S', 'N', 'D', 'G', 'A', 'C'],
}

VALID_GLYCOSYLATION_TYPES = {
    'N-linked',
    'O-linked', 
    'C-mannosylation',
    'GPI-anchor',
    'glycosylation',
}


def validate_glycosylation_residue(residue: str, glyco_type: str) -> bool:
    """
    Validate that a residue is appropriate for a glycosylation type.
    
    Args:
        residue: Single-letter amino acid code
        glyco_type: Type of glycosylation
    
    Returns:
        True if residue is valid for the glycosylation type
    """
    if not residue or not glyco_type:
        return False
    
    residue = residue.upper()
    glyco_type_lower = glyco_type.lower()
    
    for gtype, valid_residues in GLYCOSYLATION_RESIDUES.items():
        if gtype.lower() in glyco_type_lower:
            return residue in valid_residues
    
    return True
