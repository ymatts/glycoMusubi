#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for GlyTouCan parser module.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from utils.glytoucan_parser import (
    GlyTouCanParser,
    extract_wurcs_from_jsonld,
    extract_glycan_metadata,
    parse_sparqlist_gtcid2seqs,
    parse_sparqlist_glytoucan_list,
)


class TestParseSparqlistGtcid2seqs:
    """Tests for parse_sparqlist_gtcid2seqs function (new SPARQList API)."""
    
    def test_parse_standard_response(self):
        """Test parsing standard gtcid2seqs response with WURCS and GlycoCT."""
        data = [
            {"id": "G00030MO", "wurcs": "WURCS=2.0/4,7,6/[u2122h_2*NCC/3=O][a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5]/1-2-3-4-2-4-2/a4-b1_b4-c1_c3-d1_c6-f1_e1-d2|d4_g1-f2|f4"},
            {"id": "G00030MO", "glycoct": "RES\n1b:x-dglc-HEX-x:x\n2s:n-acetyl\n..."}
        ]
        result = parse_sparqlist_gtcid2seqs(data)
        assert result['id'] == "G00030MO"
        assert result['wurcs'].startswith("WURCS=2.0/")
        assert result['glycoct'] is not None
    
    def test_parse_wurcs_only_response(self):
        """Test parsing response with only WURCS entry."""
        data = [
            {"id": "G12345AB", "wurcs": "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"}
        ]
        result = parse_sparqlist_gtcid2seqs(data)
        assert result['id'] == "G12345AB"
        assert result['wurcs'] == "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"
        assert result['glycoct'] is None
    
    def test_parse_empty_response(self):
        """Test parsing empty response."""
        result = parse_sparqlist_gtcid2seqs([])
        assert result['wurcs'] is None
        assert result['glycoct'] is None
        assert result['id'] is None
    
    def test_parse_none_response(self):
        """Test parsing None response."""
        result = parse_sparqlist_gtcid2seqs(None)
        assert result['wurcs'] is None
        assert result['glycoct'] is None
    
    def test_parse_dict_fallback(self):
        """Test parsing dict response (fallback format)."""
        data = {"id": "G99999ZZ", "wurcs": "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"}
        result = parse_sparqlist_gtcid2seqs(data)
        assert result['id'] == "G99999ZZ"
        assert result['wurcs'] == "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"
    
    def test_invalid_wurcs_ignored(self):
        """Test that invalid WURCS strings are ignored."""
        data = [{"id": "G00001AA", "wurcs": "not_a_valid_wurcs"}]
        result = parse_sparqlist_gtcid2seqs(data)
        assert result['wurcs'] is None


class TestParseSparqlistGlytoucanList:
    """Tests for parse_sparqlist_glytoucan_list function."""
    
    def test_parse_standard_list(self):
        """Test parsing standard Glytoucan-list response."""
        data = [
            {"gtcid": "G67878MX", "wurcs": "WURCS=2.0/2,4,4/[h2122h][a2122h-1a_1-5]/1-2-2-2/a4-b1_b4-c1_c4-d1_c1-c4~1-6"},
            {"gtcid": "G02469EJ", "wurcs": "WURCS=2.0/6,11,10/[a2122h-1b_1-5_2*NCC/3=O]..."},
        ]
        result = parse_sparqlist_glytoucan_list(data)
        assert len(result) == 2
        assert result[0]['glycan_id'] == "G67878MX"
        assert result[0]['wurcs'].startswith("WURCS=")
        assert result[1]['glycan_id'] == "G02469EJ"
    
    def test_parse_empty_list(self):
        """Test parsing empty list."""
        result = parse_sparqlist_glytoucan_list([])
        assert result == []
    
    def test_parse_none_input(self):
        """Test parsing None input."""
        result = parse_sparqlist_glytoucan_list(None)
        assert result == []
    
    def test_skip_invalid_entries(self):
        """Test that entries with invalid WURCS are skipped."""
        data = [
            {"gtcid": "G00001AA", "wurcs": "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"},
            {"gtcid": "G00002BB", "wurcs": "invalid"},  # Should be skipped
            {"gtcid": "G00003CC"},  # Missing wurcs, should be skipped
        ]
        result = parse_sparqlist_glytoucan_list(data)
        assert len(result) == 1
        assert result[0]['glycan_id'] == "G00001AA"


class TestExtractWurcsFromJsonld:
    """Tests for extract_wurcs_from_jsonld function."""
    
    def test_extract_from_graph_array(self):
        """Test extraction from JSON-LD @graph array."""
        data = {
            "@graph": [
                {"@type": "Glycan", "wurcs": "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"},
                {"@type": "Other", "name": "test"}
            ]
        }
        result = extract_wurcs_from_jsonld(data)
        assert result == "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"
    
    def test_extract_from_nested_structure(self):
        """Test extraction from nested JSON structure."""
        data = {
            "data": {
                "glycan": {
                    "structure": {
                        "wurcs": "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a2112h-1b_1-5]/1-2/a4-b1"
                    }
                }
            }
        }
        result = extract_wurcs_from_jsonld(data)
        assert result == "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a2112h-1b_1-5]/1-2/a4-b1"
    
    def test_extract_from_direct_wurcs_key(self):
        """Test extraction when wurcs is at top level."""
        data = {"wurcs": "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"}
        result = extract_wurcs_from_jsonld(data)
        assert result == "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"
    
    def test_no_wurcs_found(self):
        """Test when no WURCS is present."""
        data = {"@graph": [{"@type": "Glycan", "name": "test"}]}
        result = extract_wurcs_from_jsonld(data)
        assert result is None
    
    def test_empty_data(self):
        """Test with empty data."""
        assert extract_wurcs_from_jsonld({}) is None
        assert extract_wurcs_from_jsonld(None) is None
    
    def test_invalid_wurcs_format(self):
        """Test that invalid WURCS format returns None (must start with WURCS)."""
        data = {"wurcs": "invalid_wurcs_string"}
        result = extract_wurcs_from_jsonld(data)
        assert result is None


class TestExtractGlycanMetadata:
    """Tests for extract_glycan_metadata function."""
    
    def test_extract_basic_metadata(self):
        """Test extraction of basic metadata fields."""
        data = {
            "@graph": [
                {
                    "@type": "Glycan",
                    "iupac": "Man(a1-3)Man",
                    "mass": 342.12
                }
            ]
        }
        result = extract_glycan_metadata(data)
        assert result.get("iupac") == "Man(a1-3)Man"
        assert result.get("mass") == 342.12
    
    def test_extract_from_top_level(self):
        """Test extraction from top-level keys."""
        data = {
            "iupac": "Gal(b1-4)GlcNAc",
            "mass": 383.35
        }
        result = extract_glycan_metadata(data)
        assert result.get("iupac") == "Gal(b1-4)GlcNAc"
        assert result.get("mass") == 383.35
    
    def test_empty_metadata(self):
        """Test with data containing no metadata."""
        data = {"@graph": [{"@type": "Other"}]}
        result = extract_glycan_metadata(data)
        assert isinstance(result, dict)


class TestGlyTouCanParser:
    """Tests for GlyTouCanParser class."""
    
    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        parser = GlyTouCanParser()
        assert parser is not None
        assert parser.parse_successes == 0
        assert parser.parse_failures == 0
        assert parser.sparqlist_successes == 0
        assert parser.jsonld_successes == 0
    
    def test_parse_sparqlist_response(self):
        """Test parsing SPARQList gtcid2seqs response."""
        parser = GlyTouCanParser()
        data = [
            {"id": "G00030MO", "wurcs": "WURCS=2.0/4,7,6/[u2122h_2*NCC/3=O][a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5]/1-2-3-4-2-4-2/a4-b1_b4-c1_c3-d1_c6-f1_e1-d2|d4_g1-f2|f4"},
            {"id": "G00030MO", "glycoct": "RES\n1b:x-dglc-HEX-x:x\n..."}
        ]
        wurcs = parser.parse_sparqlist_response(data, "G00030MO")
        assert wurcs is not None
        assert wurcs.startswith("WURCS=2.0/")
        assert parser.parse_successes == 1
        assert parser.sparqlist_successes == 1
    
    def test_parse_sparqlist_empty_response(self):
        """Test parsing empty SPARQList response."""
        parser = GlyTouCanParser()
        wurcs = parser.parse_sparqlist_response([], "G00000XX")
        assert wurcs is None
        assert parser.parse_failures == 1
    
    def test_parse_jsonld_response(self):
        """Test parsing JSON-LD format response."""
        parser = GlyTouCanParser()
        data = {
            "@graph": [
                {"@type": "Glycan", "wurcs": "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"}
            ]
        }
        wurcs = parser.parse_response(data)
        assert wurcs == "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"
        assert parser.parse_successes == 1
    
    def test_parse_rest_v3_response(self):
        """Test parsing REST v3 format response."""
        parser = GlyTouCanParser()
        data = {
            "structure": {
                "wurcs": "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a2112h-1b_1-5]/1-2/a4-b1"
            }
        }
        wurcs = parser.parse_rest_v3_response(data)
        assert wurcs == "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a2112h-1b_1-5]/1-2/a4-b1"
    
    def test_parse_missing_structure(self):
        """Test parsing response with no structure."""
        parser = GlyTouCanParser()
        data = {"@graph": [{"@type": "Glycan", "name": "test"}]}
        wurcs = parser.parse_response(data)
        assert wurcs is None
        assert parser.parse_failures == 1
    
    def test_get_statistics(self):
        """Test statistics tracking."""
        parser = GlyTouCanParser()
        
        parser.parse_response({"wurcs": "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"})
        parser.parse_response({"wurcs": "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a2112h-1b_1-5]/1-2/a4-b1"})
        parser.parse_response({"no_wurcs": True})
        
        stats = parser.get_statistics()
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["success_rate"] == pytest.approx(66.67, rel=0.1)
    
    def test_parse_none_data(self):
        """Test parsing None data."""
        parser = GlyTouCanParser()
        wurcs = parser.parse_response(None)
        assert wurcs is None


class TestWurcsValidation:
    """Tests for WURCS format validation."""
    
    def test_valid_wurcs_format(self):
        """Test that valid WURCS strings are recognized."""
        valid_wurcs = [
            "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/",
            "WURCS=2.0/2,2,1/[a2122h-1b_1-5][a2112h-1b_1-5]/1-2/a4-b1",
            "WURCS=2.0/3,3,2/[a2122h-1b_1-5][a2112h-1b_1-5][a1122h-1b_1-5]/1-2-3/a4-b1_b4-c1",
        ]
        
        for wurcs in valid_wurcs:
            data = {"wurcs": wurcs}
            result = extract_wurcs_from_jsonld(data)
            assert result == wurcs
            assert result.startswith("WURCS=")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
