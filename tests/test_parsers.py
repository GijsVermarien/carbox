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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from carbox.parsers import UnifiedChemicalParser, parse_chemical_network


class TestUnifiedParsers:
    """Test suite for unified chemical network parsers"""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance for tests"""
        return UnifiedChemicalParser()
    
    @pytest.fixture
    def test_files(self):
        """Define test file paths exactly like unified_parser_demo.py"""
        return {
            'uclchem': "uclchem/src/uclchem/reactions.csv",
            'umist': "data/umist22.csv", 
            'latent_tgas': 'data/simple_latent_tgas.csv'
        }
    
    def test_supported_formats(self, parser):
        """Test that all expected formats are supported"""
        formats = parser.list_supported_formats()
        expected_formats = ['uclchem', 'umist', 'latent_tgas']
        
        for fmt in expected_formats:
            assert fmt in formats, f"Format {fmt} not supported"
    
    def test_uclchem_parser(self, test_files):
        """Test UCLCHEM parser exactly like unified_parser_demo.py"""
        filepath = test_files['uclchem']
        
        # Check file exists
        assert os.path.exists(filepath), f"UCLCHEM file not found: {filepath}"
        
        # Parse the network
        network = parse_chemical_network(filepath, 'uclchem')
        
        # Verify results
        assert len(network.species) > 0, "No species parsed"
        assert len(network.reactions) > 0, "No reactions parsed"
        
        # Check expected numbers (from working test)
        assert len(network.species) == 163, f"Expected 163 species, got {len(network.species)}"
        assert len(network.reactions) == 2226, f"Expected 2226 reactions, got {len(network.reactions)}"
        
        # Check sample species
        species_names = [s.name for s in network.species[:5]]
        expected_species = ['C', 'C+', 'C2', 'C2+', 'C2H']
        assert species_names == expected_species, f"Species mismatch: {species_names}"
        
        # Check reaction types
        reaction_types = [r.__class__.__name__ for r in network.reactions[:5]]
        assert all(rt == 'CRPReaction' for rt in reaction_types), f"Unexpected reaction types: {reaction_types}"
    
    def test_umist_parser(self, test_files):
        """Test UMIST parser exactly like unified_parser_demo.py"""
        filepath = test_files['umist']
        
        # Check file exists
        assert os.path.exists(filepath), f"UMIST file not found: {filepath}"
        
        # Parse the network
        network = parse_chemical_network(filepath, 'umist')
        
        # Verify results
        assert len(network.species) > 0, "No species parsed"
        assert len(network.reactions) > 0, "No reactions parsed"
        
        # Check expected numbers (from working test)
        assert len(network.species) == 523, f"Expected 523 species, got {len(network.species)}"
        assert len(network.reactions) == 8765, f"Expected 8765 reactions, got {len(network.reactions)}"
        
        # Check sample species
        species_names = [s.name for s in network.species[:5]]
        expected_species = ['Al', 'Al2O', 'Al2O2', 'Al2O3', 'AlCl']
        assert species_names == expected_species, f"Species mismatch: {species_names}"
        
        # Check reaction types
        reaction_types = [r.__class__.__name__ for r in network.reactions[:5]]
        assert all(rt == 'KAReaction' for rt in reaction_types), f"Unexpected reaction types: {reaction_types}"
    
    def test_latent_tgas_parser(self, test_files):
        """Test LATENT-TGAS parser exactly like unified_parser_demo.py"""
        filepath = test_files['latent_tgas']
        
        # Check file exists
        assert os.path.exists(filepath), f"LATENT-TGAS file not found: {filepath}"
        
        # Parse the network
        network = parse_chemical_network(filepath, 'latent_tgas')
        
        # Verify results
        assert len(network.species) > 0, "No species parsed"
        assert len(network.reactions) > 0, "No reactions parsed"
        
        # Check expected numbers (from working test)
        assert len(network.species) == 16, f"Expected 16 species, got {len(network.species)}"
        assert len(network.reactions) == 25, f"Expected 25 reactions, got {len(network.reactions)}"
        
        # Check sample species
        species_names = [s.name for s in network.species[:5]]
        expected_species = ['C', 'C+', 'CO', 'CO+', 'E']
        assert species_names == expected_species, f"Species mismatch: {species_names}"
        
        # Check reaction types
        reaction_types = [r.__class__.__name__ for r in network.reactions[:5]]
        expected_types = ['CRPReaction', 'CRPReaction', 'CRPReaction', 'CRPReaction', 'FUVReaction']
        assert reaction_types == expected_types, f"Reaction type mismatch: {reaction_types}"
    
    def test_auto_detection(self, test_files):
        """Test auto-detection exactly like unified_parser_demo.py"""
        
        for format_type, filepath in test_files.items():
            if os.path.exists(filepath):
                # Parse without specifying format (auto-detect)
                network = parse_chemical_network(filepath)
                
                # Should succeed and return a valid network
                assert len(network.species) > 0, f"Auto-detection failed for {filepath}: no species"
                assert len(network.reactions) > 0, f"Auto-detection failed for {filepath}: no reactions"
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        
        # Test non-existent file
        with pytest.raises(Exception):
            parse_chemical_network("non_existent_file.csv")
        
        # Test unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            parse_chemical_network("data/simple_latent_tgas.csv", "invalid_format")