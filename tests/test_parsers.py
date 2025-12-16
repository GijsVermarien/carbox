#!/usr/bin/env python3
"""
Pytest-based tests for Carbox unified chemical network parser.

Tests UCLCHEM, UMIST, and LATENT-TGAS parsers following the
unified_parser_demo.py approach.
"""

import os

# Add project root to path
import sys
from pathlib import Path

import pytest

from carbox.parsers import UnifiedChemicalParser, parse_chemical_network

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestUnifiedParsers:
    """Test suite for unified chemical network parsers"""

    @pytest.fixture
    def parser(self):
        """Create parser instance for tests"""
        return UnifiedChemicalParser()

    @pytest.fixture
    def test_files(self):
        """Define test file paths using local test fixtures."""
        project_root = Path(__file__).parent.parent
        return {
            "uclchem": project_root / "tests" / "test_data" / "test_uclchem.csv",
            "umist": project_root / "tests" / "test_data" / "test_umist.csv",
            "latent_tgas": project_root / "data" / "simple_latent_tgas.csv",
        }

    def test_supported_formats(self, parser):
        """Test that all expected formats are supported"""
        formats = parser.list_supported_formats()
        expected_formats = ["uclchem", "umist", "latent_tgas"]

        for fmt in expected_formats:
            assert fmt in formats, f"Format {fmt} not supported"

    def test_uclchem_parser(self, test_files):
        """Test UCLCHEM parser exactly like unified_parser_demo.py"""
        filepath = test_files["uclchem"]

        # Check file exists
        assert filepath.exists(), f"UCLCHEM file not found: {filepath}"

        # Parse the network
        network = parse_chemical_network(filepath, "uclchem")

        # Verify results
        assert len(network.species) > 0, "No species parsed"
        assert len(network.reactions) > 0, "No reactions parsed"

        reaction_types = [r.__class__.__name__ for r in network.reactions[:3]]
        assert reaction_types == [
            "CRPhotoReaction",
            "UCLCHEMPhotonReaction",
            "KAReaction",
        ]

    def test_umist_parser(self, test_files):
        """Test UMIST parser exactly like unified_parser_demo.py"""
        filepath = test_files["umist"]

        # Check file exists
        assert filepath.exists(), f"UMIST file not found: {filepath}"

        # Parse the network
        network = parse_chemical_network(filepath, "umist")

        # Verify results
        assert len(network.species) > 0, "No species parsed"
        assert len(network.reactions) > 0, "No reactions parsed"

        reaction_types = [r.__class__.__name__ for r in network.reactions]
        assert reaction_types == ["KAReaction", "KAReaction"]

    def test_latent_tgas_parser(self, test_files):
        """Test LATENT-TGAS parser exactly like unified_parser_demo.py"""
        filepath = test_files["latent_tgas"]

        # Check file exists
        assert filepath.exists(), f"LATENT-TGAS file not found: {filepath}"

        # Parse the network
        network = parse_chemical_network(filepath, "latent_tgas")

        # Verify results
        assert len(network.species) > 0, "No species parsed"
        assert len(network.reactions) > 0, "No reactions parsed"

        reaction_types = {r.__class__.__name__ for r in network.reactions}
        assert {"KAReaction", "CRPReaction", "FUVReaction"} <= reaction_types

    def test_auto_detection(self, test_files):
        """Test auto-detection exactly like unified_parser_demo.py"""

        for format_type, filepath in test_files.items():
            if os.path.exists(filepath):
                # Parse without specifying format (auto-detect)
                network = parse_chemical_network(filepath)

                # Should succeed and return a valid network
                assert (
                    len(network.species) > 0
                ), f"Auto-detection failed for {filepath}: no species"
                assert (
                    len(network.reactions) > 0
                ), f"Auto-detection failed for {filepath}: no reactions"

    def test_error_handling(self):
        """Test error handling for invalid inputs"""

        # Test non-existent file
        with pytest.raises(Exception):
            parse_chemical_network("non_existent_file.csv")

        # Test unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            parse_chemical_network("data/simple_latent_tgas.csv", "invalid_format")
