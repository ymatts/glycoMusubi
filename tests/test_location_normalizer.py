#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for location_normalizer module.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from utils.location_normalizer import (
    COMPARTMENT_REGISTRY,
    normalize_location,
    normalize_locations_from_uniprot,
    get_compartment_label,
)


class TestNormalizeLocation:
    """Tests for normalize_location function."""

    def test_golgi_membrane(self):
        assert normalize_location("Golgi membrane") == "LOC::golgi_apparatus"

    def test_cis_golgi(self):
        assert normalize_location("cis-Golgi network") == "LOC::cis_golgi"

    def test_trans_golgi_network(self):
        assert normalize_location("trans-Golgi network") == "LOC::trans_golgi_network"

    def test_trans_golgi(self):
        assert normalize_location("trans-Golgi") == "LOC::trans_golgi"

    def test_medial_golgi(self):
        assert normalize_location("Medial-Golgi") == "LOC::medial_golgi"

    def test_er(self):
        assert normalize_location("Endoplasmic reticulum membrane") == "LOC::endoplasmic_reticulum"

    def test_er_golgi_intermediate(self):
        assert normalize_location("ER-Golgi intermediate compartment") == "LOC::er_golgi_intermediate"

    def test_cell_membrane(self):
        assert normalize_location("Cell membrane") == "LOC::cell_membrane"

    def test_plasma_membrane(self):
        assert normalize_location("Plasma membrane") == "LOC::cell_membrane"

    def test_cell_surface(self):
        assert normalize_location("Cell surface") == "LOC::cell_surface"

    def test_extracellular(self):
        assert normalize_location("Extracellular space") == "LOC::extracellular"

    def test_secreted(self):
        assert normalize_location("Secreted") == "LOC::extracellular"

    def test_lysosome(self):
        assert normalize_location("Lysosome") == "LOC::lysosome"

    def test_endosome(self):
        assert normalize_location("Endosome membrane") == "LOC::endosome"

    def test_cytoplasm(self):
        assert normalize_location("Cytoplasm") == "LOC::cytoplasm"

    def test_cytosol(self):
        assert normalize_location("Cytosol") == "LOC::cytoplasm"

    def test_nucleus(self):
        assert normalize_location("Nucleus") == "LOC::nucleus"

    def test_mitochondria(self):
        assert normalize_location("Mitochondrion inner membrane") == "LOC::mitochondria"

    def test_peroxisome(self):
        assert normalize_location("Peroxisome") == "LOC::peroxisome"

    def test_empty_string(self):
        assert normalize_location("") is None

    def test_whitespace_only(self):
        assert normalize_location("   ") is None

    def test_unknown_location(self):
        assert normalize_location("Some unknown place") is None

    def test_specificity_cis_over_golgi(self):
        """cis-Golgi should match cis_golgi, not golgi_apparatus."""
        assert normalize_location("cis-Golgi") == "LOC::cis_golgi"

    def test_specificity_tgn_over_trans(self):
        """trans-Golgi network should match trans_golgi_network, not trans_golgi."""
        assert normalize_location("trans-Golgi network membrane") == "LOC::trans_golgi_network"


class TestNormalizeFromUniprot:
    """Tests for normalize_locations_from_uniprot function."""

    def test_string_input(self):
        result = normalize_locations_from_uniprot("Golgi apparatus;Endoplasmic reticulum")
        assert "LOC::golgi_apparatus" in result
        assert "LOC::endoplasmic_reticulum" in result

    def test_list_dict_input(self):
        locs = [
            {"location": {"value": "Golgi apparatus"}},
            {"location": {"value": "Cytoplasm"}},
        ]
        result = normalize_locations_from_uniprot(locs)
        assert "LOC::golgi_apparatus" in result
        assert "LOC::cytoplasm" in result

    def test_deduplication(self):
        result = normalize_locations_from_uniprot("Golgi apparatus;Golgi membrane")
        assert result.count("LOC::golgi_apparatus") == 1

    def test_empty_string(self):
        assert normalize_locations_from_uniprot("") == []

    def test_empty_list(self):
        assert normalize_locations_from_uniprot([]) == []

    def test_unknown_entries_filtered(self):
        result = normalize_locations_from_uniprot("unknown;Golgi")
        assert len(result) == 1
        assert result[0] == "LOC::golgi_apparatus"


class TestCompartmentLabel:
    """Tests for get_compartment_label function."""

    def test_known_id(self):
        assert get_compartment_label("LOC::golgi_apparatus") == "Golgi apparatus"

    def test_er(self):
        assert get_compartment_label("LOC::endoplasmic_reticulum") == "Endoplasmic reticulum"

    def test_unknown_id_returns_id(self):
        assert get_compartment_label("LOC::unknown") == "LOC::unknown"


class TestRegistry:
    """Tests for registry completeness."""

    def test_registry_has_16_entries(self):
        assert len(COMPARTMENT_REGISTRY) == 16

    def test_all_ids_start_with_loc(self):
        for loc_id in COMPARTMENT_REGISTRY:
            assert loc_id.startswith("LOC::"), f"{loc_id} should start with LOC::"

    def test_all_labels_non_empty(self):
        for loc_id, label in COMPARTMENT_REGISTRY.items():
            assert label, f"Label for {loc_id} should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
