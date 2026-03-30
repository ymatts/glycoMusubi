#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
glytoucan_parser.py

Parser for GlyTouCan API responses, handling SPARQList, JSON-LD, and REST v3 formats.
Extracts WURCS (Web3 Unique Representation of Carbohydrate Structures) from various response formats.

Supported API formats:
- SPARQList gtcid2seqs: Returns array with separate WURCS and GlycoCT entries
- SPARQList Glytoucan-list: Returns array of {gtcid, wurcs} objects
- JSON-LD (legacy): Nested @graph structure with WURCS predicates
- REST v3 (legacy): {structure: {wurcs: ...}} format
"""

import logging
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)


def parse_sparqlist_gtcid2seqs(data: Union[List, Dict, Any]) -> Dict[str, Optional[str]]:
    """
    Parse response from SPARQList gtcid2seqs endpoint.
    
    The endpoint returns a JSON array with separate entries for WURCS and GlycoCT:
    [
        {"id": "G00030MO", "wurcs": "WURCS=2.0/..."},
        {"id": "G00030MO", "glycoct": "RES\\n1b:..."}
    ]
    
    Args:
        data: Parsed JSON response from gtcid2seqs endpoint.
    
    Returns:
        Dictionary with 'wurcs', 'glycoct', and 'id' keys.
    """
    result = {'wurcs': None, 'glycoct': None, 'id': None}
    
    if not data:
        return result
    
    # Handle list response (expected format)
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            
            # Get ID from any entry
            if result['id'] is None and 'id' in entry:
                result['id'] = entry['id']
            
            # Extract WURCS
            if 'wurcs' in entry and entry['wurcs']:
                wurcs = entry['wurcs']
                if isinstance(wurcs, str) and wurcs.startswith('WURCS'):
                    result['wurcs'] = wurcs
            
            # Extract GlycoCT
            if 'glycoct' in entry and entry['glycoct']:
                result['glycoct'] = entry['glycoct']
    
    # Handle dict response (fallback)
    elif isinstance(data, dict):
        result['id'] = data.get('id')
        if 'wurcs' in data:
            wurcs = data['wurcs']
            if isinstance(wurcs, str) and wurcs.startswith('WURCS'):
                result['wurcs'] = wurcs
        if 'glycoct' in data:
            result['glycoct'] = data['glycoct']
    
    return result


def parse_sparqlist_glytoucan_list(data: Union[List, Any]) -> List[Dict[str, str]]:
    """
    Parse response from SPARQList Glytoucan-list endpoint.
    
    The endpoint returns a JSON array of all GlyTouCan IDs with WURCS:
    [
        {"gtcid": "G67878MX", "wurcs": "WURCS=2.0/..."},
        {"gtcid": "G02469EJ", "wurcs": "WURCS=2.0/..."},
        ...
    ]
    
    Args:
        data: Parsed JSON response from Glytoucan-list endpoint.
    
    Returns:
        List of dictionaries with 'glycan_id' and 'wurcs' keys.
    """
    results = []
    
    if not data or not isinstance(data, list):
        return results
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        gtcid = entry.get('gtcid')
        wurcs = entry.get('wurcs')
        
        if gtcid and wurcs and isinstance(wurcs, str) and wurcs.startswith('WURCS'):
            results.append({
                'glycan_id': gtcid,
                'wurcs': wurcs
            })
    
    return results


def extract_wurcs_from_jsonld(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract WURCS string from JSON-LD formatted GlyTouCan response.
    
    The GlyTouCan API returns data in JSON-LD format where the WURCS structure
    may be nested within @graph arrays or under various property names.
    
    Args:
        data: Parsed JSON response from GlyTouCan API.
    
    Returns:
        WURCS string if found, None otherwise.
    """
    if not data or not isinstance(data, dict):
        return None
    
    if 'wurcs' in data:
        wurcs = data['wurcs']
        if isinstance(wurcs, str) and wurcs.startswith('WURCS'):
            return wurcs
    
    if 'structure' in data:
        structure = data['structure']
        if isinstance(structure, str) and structure.startswith('WURCS'):
            return structure
        if isinstance(structure, dict):
            if 'wurcs' in structure:
                return structure['wurcs']
            if 'sequence' in structure:
                seq = structure['sequence']
                if isinstance(seq, str) and seq.startswith('WURCS'):
                    return seq
    
    if '@graph' in data:
        graph = data['@graph']
        if isinstance(graph, list):
            for item in graph:
                if not isinstance(item, dict):
                    continue
                
                if 'wurcs' in item:
                    return item['wurcs']
                
                wurcs_predicates = [
                    'http://www.glycoinfo.org/glyco/owl/relation#has_wurcs',
                    'glycoinfo:has_wurcs',
                    'has_wurcs',
                ]
                for predicate in wurcs_predicates:
                    if predicate in item:
                        value = item[predicate]
                        if isinstance(value, str):
                            return value
                        if isinstance(value, dict) and '@value' in value:
                            return value['@value']
                        if isinstance(value, list) and value:
                            first = value[0]
                            if isinstance(first, str):
                                return first
                            if isinstance(first, dict) and '@value' in first:
                                return first['@value']
                
                item_type = item.get('@type', '')
                if 'Saccharide' in str(item_type) or 'Glycan' in str(item_type):
                    for key, value in item.items():
                        if 'wurcs' in key.lower():
                            if isinstance(value, str):
                                return value
                            if isinstance(value, dict) and '@value' in value:
                                return value['@value']
    
    wurcs = _find_wurcs_recursive(data)
    if wurcs:
        return wurcs
    
    return None


def _find_wurcs_recursive(obj: Any, depth: int = 0, max_depth: int = 10) -> Optional[str]:
    """
    Recursively search for WURCS string in nested data structures.
    
    Args:
        obj: Object to search (dict, list, or primitive).
        depth: Current recursion depth.
        max_depth: Maximum recursion depth to prevent infinite loops.
    
    Returns:
        WURCS string if found, None otherwise.
    """
    if depth > max_depth:
        return None
    
    if isinstance(obj, str):
        if obj.startswith('WURCS='):
            return obj
        return None
    
    if isinstance(obj, dict):
        for key in ['wurcs', 'WURCS', 'structure', 'sequence']:
            if key in obj:
                value = obj[key]
                if isinstance(value, str) and value.startswith('WURCS'):
                    return value
        
        for key, value in obj.items():
            if 'wurcs' in key.lower():
                if isinstance(value, str) and value.startswith('WURCS'):
                    return value
                if isinstance(value, dict) and '@value' in value:
                    v = value['@value']
                    if isinstance(v, str) and v.startswith('WURCS'):
                        return v
        
        for value in obj.values():
            result = _find_wurcs_recursive(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(obj, list):
        for item in obj:
            result = _find_wurcs_recursive(item, depth + 1, max_depth)
            if result:
                return result
    
    return None


class GlyTouCanParser:
    """
    Parser for GlyTouCan API responses with support for multiple formats.
    
    Supports:
    - SPARQList gtcid2seqs format (new, recommended)
    - JSON-LD format (legacy)
    - REST v3 format (legacy)
    """
    
    def __init__(self):
        """Initialize the parser."""
        self.parse_attempts = 0
        self.parse_successes = 0
        self.parse_failures = 0
        self.sparqlist_successes = 0
        self.jsonld_successes = 0
    
    def parse_sparqlist_response(self, data: Any, glycan_id: str = "") -> Optional[str]:
        """
        Parse a SPARQList gtcid2seqs API response and extract WURCS.
        
        Args:
            data: Parsed JSON response (expected to be a list).
            glycan_id: Glycan ID for logging purposes.
        
        Returns:
            WURCS string if found, None otherwise.
        """
        self.parse_attempts += 1
        
        if not data:
            logger.debug(f"Empty SPARQList response for {glycan_id}")
            self.parse_failures += 1
            return None
        
        result = parse_sparqlist_gtcid2seqs(data)
        
        if result['wurcs']:
            self.parse_successes += 1
            self.sparqlist_successes += 1
            logger.debug(f"Successfully extracted WURCS from SPARQList for {glycan_id}")
            return result['wurcs']
        
        self.parse_failures += 1
        logger.debug(f"No WURCS found in SPARQList response for {glycan_id}")
        return None
    
    def parse_response(self, data: Dict[str, Any], glycan_id: str = "") -> Optional[str]:
        """
        Parse a GlyTouCan API response and extract WURCS.
        
        Args:
            data: Parsed JSON response.
            glycan_id: Glycan ID for logging purposes.
        
        Returns:
            WURCS string if found, None otherwise.
        """
        self.parse_attempts += 1
        
        if not data:
            logger.debug(f"Empty response for {glycan_id}")
            self.parse_failures += 1
            return None
        
        wurcs = extract_wurcs_from_jsonld(data)
        
        if wurcs:
            self.parse_successes += 1
            logger.debug(f"Successfully extracted WURCS for {glycan_id}")
            return wurcs
        
        self.parse_failures += 1
        logger.debug(f"No WURCS found for {glycan_id}")
        return None
    
    def parse_rest_v3_response(self, data: Dict[str, Any], glycan_id: str = "") -> Optional[str]:
        """
        Parse a REST v3 format response from GlyTouCan.
        
        Args:
            data: Parsed JSON response.
            glycan_id: Glycan ID for logging purposes.
        
        Returns:
            WURCS string if found, None otherwise.
        """
        self.parse_attempts += 1
        
        if not data:
            self.parse_failures += 1
            return None
        
        try:
            if 'structure' in data and isinstance(data['structure'], dict):
                structure = data['structure']
                if 'wurcs' in structure:
                    self.parse_successes += 1
                    return structure['wurcs']
                if 'sequence' in structure:
                    seq = structure['sequence']
                    if isinstance(seq, str) and seq.startswith('WURCS'):
                        self.parse_successes += 1
                        return seq
        except (KeyError, TypeError) as e:
            logger.debug(f"Error parsing REST v3 response for {glycan_id}: {e}")
        
        wurcs = self.parse_response(data, glycan_id)
        if wurcs:
            return wurcs
        
        self.parse_failures += 1
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get parsing statistics.
        
        Returns:
            Dictionary with parse attempts, successes, failures, and format breakdown.
        """
        return {
            'attempts': self.parse_attempts,
            'successes': self.parse_successes,
            'failures': self.parse_failures,
            'sparqlist_successes': self.sparqlist_successes,
            'jsonld_successes': self.jsonld_successes,
            'success_rate': (
                self.parse_successes / self.parse_attempts * 100
                if self.parse_attempts > 0 else 0
            ),
        }
    
    def reset_statistics(self):
        """Reset parsing statistics."""
        self.parse_attempts = 0
        self.parse_successes = 0
        self.parse_failures = 0
        self.sparqlist_successes = 0
        self.jsonld_successes = 0


def extract_glycan_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract additional metadata from GlyTouCan response.
    
    Args:
        data: Parsed JSON response.
    
    Returns:
        Dictionary with extracted metadata.
    """
    metadata = {}
    
    if '@graph' in data:
        for item in data['@graph']:
            if not isinstance(item, dict):
                continue
            
            if 'iupac' in item or 'http://www.glycoinfo.org/glyco/owl/relation#has_iupac' in item:
                iupac = item.get('iupac') or item.get('http://www.glycoinfo.org/glyco/owl/relation#has_iupac')
                if isinstance(iupac, dict) and '@value' in iupac:
                    iupac = iupac['@value']
                if isinstance(iupac, str):
                    metadata['iupac'] = iupac
            
            if 'mass' in item:
                mass = item['mass']
                if isinstance(mass, dict) and '@value' in mass:
                    mass = mass['@value']
                try:
                    metadata['mass'] = float(mass)
                except (ValueError, TypeError):
                    pass
    
    for key in ['iupac', 'mass', 'glycoct', 'smiles']:
        if key in data:
            value = data[key]
            if isinstance(value, dict) and '@value' in value:
                value = value['@value']
            metadata[key] = value
    
    return metadata
