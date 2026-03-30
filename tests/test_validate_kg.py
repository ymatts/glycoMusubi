#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for validate_kg module - validation checks.
"""

import pytest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from validate_kg import (
    ValidationReport,
    KGValidator,
    KGAutoFixer,
    GLYTOUCAN_ID_PATTERN,
    UNIPROT_AC_PATTERN,
    CHEMBL_ID_PATTERN,
    LOCATION_ID_PATTERN,
)


class TestValidationReport:
    """Tests for ValidationReport class."""
    
    def test_empty_report_is_valid(self):
        """Test that empty report is valid."""
        report = ValidationReport()
        assert report.is_valid
        assert len(report.errors) == 0
        assert len(report.warnings) == 0
    
    def test_add_error_makes_invalid(self):
        """Test that adding error makes report invalid."""
        report = ValidationReport()
        report.add_error("Test error")
        assert not report.is_valid
        assert len(report.errors) == 1
    
    def test_add_warning_keeps_valid(self):
        """Test that adding warning keeps report valid."""
        report = ValidationReport()
        report.add_warning("Test warning")
        assert report.is_valid
        assert len(report.warnings) == 1
    
    def test_add_info(self):
        """Test adding info messages."""
        report = ValidationReport()
        report.add_info("Test info")
        assert report.is_valid
        assert len(report.info) == 1
    
    def test_add_fix(self):
        """Test adding fix messages."""
        report = ValidationReport()
        report.add_fix("Applied fix")
        assert len(report.fixes_applied) == 1
    
    def test_to_text_format(self):
        """Test text report generation."""
        report = ValidationReport()
        report.add_error("Error 1")
        report.add_warning("Warning 1")
        report.add_info("Info 1")
        report.statistics["total_nodes"] = 100
        
        text = report.to_text()
        assert "KNOWLEDGE GRAPH VALIDATION REPORT" in text
        assert "FAILED" in text
        assert "Error 1" in text
        assert "Warning 1" in text
        assert "Info 1" in text
        assert "total_nodes" in text


class TestIDPatterns:
    """Tests for ID validation patterns."""
    
    def test_valid_glytoucan_ids(self):
        """Test valid GlyTouCan ID patterns."""
        valid_ids = ["G00001AB", "G12345XY", "G99999ZZ"]
        for gid in valid_ids:
            assert GLYTOUCAN_ID_PATTERN.match(gid), f"{gid} should be valid"
    
    def test_invalid_glytoucan_ids(self):
        """Test invalid GlyTouCan ID patterns."""
        invalid_ids = ["G0001AB", "G123456AB", "G00001ab", "X00001AB", "G00001A"]
        for gid in invalid_ids:
            assert not GLYTOUCAN_ID_PATTERN.match(gid), f"{gid} should be invalid"
    
    def test_valid_uniprot_acs(self):
        """Test valid UniProt AC patterns."""
        valid_acs = ["P12345", "Q9Y6K9", "A0A024R1R8", "P12345-1"]
        for ac in valid_acs:
            assert UNIPROT_AC_PATTERN.match(ac), f"{ac} should be valid"
    
    def test_invalid_uniprot_acs(self):
        """Test invalid UniProt AC patterns."""
        invalid_acs = ["12345", "PROTEIN", "P1234"]
        for ac in invalid_acs:
            assert not UNIPROT_AC_PATTERN.match(ac), f"{ac} should be invalid"
    
    def test_valid_chembl_ids(self):
        """Test valid ChEMBL ID patterns."""
        valid_ids = ["CHEMBL1", "CHEMBL123456", "CHEMBL999999999"]
        for cid in valid_ids:
            assert CHEMBL_ID_PATTERN.match(cid), f"{cid} should be valid"
    
    def test_invalid_chembl_ids(self):
        """Test invalid ChEMBL ID patterns."""
        invalid_ids = ["CHEMBL", "chembl123", "CHEMBL-123", "123"]
        for cid in invalid_ids:
            assert not CHEMBL_ID_PATTERN.match(cid), f"{cid} should be invalid"


class TestKGValidator:
    """Tests for KGValidator class."""
    
    @pytest.fixture
    def sample_nodes_df(self):
        """Create sample nodes DataFrame."""
        return pd.DataFrame({
            "node_id": ["N001", "N002", "N003", "N004"],
            "node_type": ["enzyme", "compound", "glycan", "disease"],
            "label": ["Enzyme A", "Compound X", "Glycan Y", "Disease Z"],
            "metadata": ["{}", "{}", "{}", "{}"]
        })
    
    @pytest.fixture
    def sample_edges_df(self):
        """Create sample edges DataFrame."""
        return pd.DataFrame({
            "source_id": ["N002", "N001"],
            "target_id": ["N001", "N003"],
            "relation": ["inhibits", "has_glycan"],
            "metadata": ["{}", "{}"]
        })
    
    def test_validator_initialization(self, sample_nodes_df, sample_edges_df):
        """Test validator initializes correctly."""
        validator = KGValidator(sample_nodes_df, sample_edges_df)
        assert validator is not None
        assert len(validator.node_ids) == 4
    
    def test_check_orphan_nodes(self, sample_nodes_df, sample_edges_df):
        """Test orphan node detection."""
        validator = KGValidator(sample_nodes_df, sample_edges_df)
        orphans = validator.check_orphan_nodes()
        
        assert "N004" in orphans
        assert "N001" not in orphans
        assert "N002" not in orphans
        assert "N003" not in orphans
    
    def test_check_duplicate_edges(self):
        """Test duplicate edge detection."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001", "N002"],
            "node_type": ["enzyme", "compound"],
            "label": ["A", "B"],
            "metadata": ["{}", "{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001", "N001", "N001"],
            "target_id": ["N002", "N002", "N002"],
            "relation": ["inhibits", "inhibits", "has_glycan"],
            "metadata": ["{}", "{}", "{}"]
        })
        
        validator = KGValidator(nodes_df, edges_df)
        duplicates = validator.check_duplicate_edges()
        
        assert len(duplicates) == 2
    
    def test_check_invalid_references(self):
        """Test invalid reference detection."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001"],
            "node_type": ["enzyme"],
            "label": ["A"],
            "metadata": ["{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001", "N999"],
            "target_id": ["N002", "N001"],
            "relation": ["inhibits", "has_glycan"],
            "metadata": ["{}", "{}"]
        })
        
        validator = KGValidator(nodes_df, edges_df)
        invalid_sources, invalid_targets = validator.check_invalid_references()
        
        assert "N999" in invalid_sources
        assert "N002" in invalid_targets
    
    def test_check_self_loops(self):
        """Test self-loop detection."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001", "N002"],
            "node_type": ["enzyme", "compound"],
            "label": ["A", "B"],
            "metadata": ["{}", "{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001", "N001"],
            "target_id": ["N001", "N002"],
            "relation": ["inhibits", "inhibits"],
            "metadata": ["{}", "{}"]
        })
        
        validator = KGValidator(nodes_df, edges_df)
        self_loops = validator.check_self_loops()
        
        assert len(self_loops) == 1
        assert self_loops[0][0] == "N001"
    
    def test_detect_cycles(self):
        """Test cycle detection."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001", "N002", "N003"],
            "node_type": ["enzyme", "enzyme", "enzyme"],
            "label": ["A", "B", "C"],
            "metadata": ["{}", "{}", "{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001", "N002", "N003"],
            "target_id": ["N002", "N003", "N001"],
            "relation": ["inhibits", "inhibits", "inhibits"],
            "metadata": ["{}", "{}", "{}"]
        })
        
        validator = KGValidator(nodes_df, edges_df)
        cycles = validator.detect_cycles("inhibits")
        
        assert len(cycles) > 0
    
    def test_compute_degree_distribution(self, sample_nodes_df, sample_edges_df):
        """Test degree distribution computation."""
        validator = KGValidator(sample_nodes_df, sample_edges_df)
        stats = validator.compute_degree_distribution()
        
        assert "in_degree" in stats
        assert "out_degree" in stats
        assert "max" in stats["in_degree"]
        assert "min" in stats["in_degree"]
        assert "avg" in stats["in_degree"]
    
    def test_find_connected_components(self, sample_nodes_df, sample_edges_df):
        """Test connected components detection."""
        validator = KGValidator(sample_nodes_df, sample_edges_df)
        components = validator.find_connected_components()
        
        assert len(components) >= 1
        total_nodes = sum(len(c) for c in components)
        assert total_nodes == len(sample_nodes_df)
    
    def test_run_all_checks(self, sample_nodes_df, sample_edges_df):
        """Test running all validation checks."""
        validator = KGValidator(sample_nodes_df, sample_edges_df)
        report = validator.run_all_checks()
        
        assert isinstance(report, ValidationReport)
        assert "total_nodes" in report.statistics
        assert "total_edges" in report.statistics
    
    def test_empty_graph(self):
        """Test validation of empty graph."""
        nodes_df = pd.DataFrame(columns=["node_id", "node_type", "label", "metadata"])
        edges_df = pd.DataFrame(columns=["source_id", "target_id", "relation", "metadata"])
        
        validator = KGValidator(nodes_df, edges_df)
        report = validator.run_all_checks()
        
        assert report.is_valid


class TestKGAutoFixer:
    """Tests for KGAutoFixer class."""
    
    def test_remove_orphan_nodes(self):
        """Test orphan node removal."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001", "N002", "N003"],
            "node_type": ["enzyme", "compound", "disease"],
            "label": ["A", "B", "C"],
            "metadata": ["{}", "{}", "{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001"],
            "target_id": ["N002"],
            "relation": ["inhibits"],
            "metadata": ["{}"]
        })
        
        report = ValidationReport()
        fixer = KGAutoFixer(nodes_df, edges_df, report)
        removed = fixer.remove_orphan_nodes()
        
        assert removed == 1
        assert len(fixer.nodes_df) == 2
        assert "N003" not in fixer.nodes_df["node_id"].values
    
    def test_remove_duplicate_edges(self):
        """Test duplicate edge removal."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001", "N002"],
            "node_type": ["enzyme", "compound"],
            "label": ["A", "B"],
            "metadata": ["{}", "{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001", "N001", "N001"],
            "target_id": ["N002", "N002", "N002"],
            "relation": ["inhibits", "inhibits", "inhibits"],
            "metadata": ["{}", "{}", "{}"]
        })
        
        report = ValidationReport()
        fixer = KGAutoFixer(nodes_df, edges_df, report)
        removed = fixer.remove_duplicate_edges()
        
        assert removed == 2
        assert len(fixer.edges_df) == 1
    
    def test_remove_self_loops(self):
        """Test self-loop removal."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001", "N002"],
            "node_type": ["enzyme", "compound"],
            "label": ["A", "B"],
            "metadata": ["{}", "{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001", "N001"],
            "target_id": ["N001", "N002"],
            "relation": ["inhibits", "inhibits"],
            "metadata": ["{}", "{}"]
        })
        
        report = ValidationReport()
        fixer = KGAutoFixer(nodes_df, edges_df, report)
        removed = fixer.remove_self_loops()
        
        assert removed == 1
        assert len(fixer.edges_df) == 1
    
    def test_remove_invalid_references(self):
        """Test invalid reference removal."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001"],
            "node_type": ["enzyme"],
            "label": ["A"],
            "metadata": ["{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001", "N999"],
            "target_id": ["N002", "N001"],
            "relation": ["inhibits", "has_glycan"],
            "metadata": ["{}", "{}"]
        })
        
        report = ValidationReport()
        fixer = KGAutoFixer(nodes_df, edges_df, report)
        removed = fixer.remove_invalid_references()
        
        assert removed == 2
        assert len(fixer.edges_df) == 0
    
    def test_apply_all_fixes(self):
        """Test applying all fixes."""
        nodes_df = pd.DataFrame({
            "node_id": ["N001", "N002", "N003"],
            "node_type": ["enzyme", "compound", "disease"],
            "label": ["A", "B", "C"],
            "metadata": ["{}", "{}", "{}"]
        })
        edges_df = pd.DataFrame({
            "source_id": ["N001", "N001", "N001"],
            "target_id": ["N002", "N002", "N001"],
            "relation": ["inhibits", "inhibits", "inhibits"],
            "metadata": ["{}", "{}", "{}"]
        })
        
        report = ValidationReport()
        fixer = KGAutoFixer(nodes_df, edges_df, report)
        fixed_nodes, fixed_edges = fixer.apply_all_fixes()
        
        assert len(fixed_edges) == 1
        assert len(fixed_nodes) == 2
        assert len(report.fixes_applied) > 0


class TestLocationIDPattern:
    """Tests for LOCATION_ID_PATTERN."""

    def test_valid_location_ids(self):
        valid = [
            "LOC::golgi_apparatus",
            "LOC::endoplasmic_reticulum",
            "LOC::cis_golgi",
            "LOC::cell_membrane",
            "LOC::extracellular",
        ]
        for lid in valid:
            assert LOCATION_ID_PATTERN.match(lid), f"{lid} should be valid"

    def test_invalid_location_ids(self):
        invalid = [
            "LOC::Golgi",         # uppercase
            "LOC::golgi-apparatus",  # hyphen
            "LOC::",               # empty suffix
            "golgi_apparatus",    # missing prefix
            "LOC::golgi apparatus",  # space
            "LOC::123",            # digits
        ]
        for lid in invalid:
            assert not LOCATION_ID_PATTERN.match(lid), f"{lid} should be invalid"


class TestValidateLocalizedInEdges:
    """Tests for validate_localized_in_edges."""

    def test_valid_edges(self):
        nodes_df = pd.DataFrame({
            "node_id": ["P12345", "LOC::golgi_apparatus"],
            "node_type": ["protein", "cellular_location"],
            "label": ["Protein A", "Golgi apparatus"],
            "metadata": ["{}", "{}"],
        })
        edges_df = pd.DataFrame({
            "source_id": ["P12345"],
            "target_id": ["LOC::golgi_apparatus"],
            "relation": ["localized_in"],
            "metadata": ["{}"],
        })
        validator = KGValidator(nodes_df, edges_df)
        warnings = validator.validate_localized_in_edges()
        assert len(warnings) == 0

    def test_invalid_source(self):
        nodes_df = pd.DataFrame({
            "node_id": ["G12345AB", "LOC::golgi_apparatus"],
            "node_type": ["glycan", "cellular_location"],
            "label": ["Glycan", "Golgi"],
            "metadata": ["{}", "{}"],
        })
        edges_df = pd.DataFrame({
            "source_id": ["G12345AB"],
            "target_id": ["LOC::golgi_apparatus"],
            "relation": ["localized_in"],
            "metadata": ["{}"],
        })
        validator = KGValidator(nodes_df, edges_df)
        warnings = validator.validate_localized_in_edges()
        assert len(warnings) == 1
        assert "non-protein/enzyme" in warnings[0]

    def test_invalid_target(self):
        nodes_df = pd.DataFrame({
            "node_id": ["P12345", "DIS001"],
            "node_type": ["protein", "disease"],
            "label": ["Protein", "Disease"],
            "metadata": ["{}", "{}"],
        })
        edges_df = pd.DataFrame({
            "source_id": ["P12345"],
            "target_id": ["DIS001"],
            "relation": ["localized_in"],
            "metadata": ["{}"],
        })
        validator = KGValidator(nodes_df, edges_df)
        warnings = validator.validate_localized_in_edges()
        assert len(warnings) == 1
        assert "non-location" in warnings[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
