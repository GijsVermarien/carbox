from typing import Dict, Type, Optional
from .base_parser import BaseParser
from .uclchem_parser import UCLCHEMParser
from .umist_parser import UMISTParser
from .latent_tgas_parser import LatentTGASParser
from ..network import Network
import os


class UnifiedChemicalParser:
    """
    Unified interface for parsing different chemical reaction network formats.

    Supports:
    - UCLCHEM (CSV format with surface chemistry)
    - UMIST (colon-separated format)
    - LATENT-TGAS (CSV format)
    """

    def __init__(self):
        self.parsers: Dict[str, Type[BaseParser]] = {
            "uclchem": UCLCHEMParser,
            "umist": UMISTParser,
            "latent_tgas": LatentTGASParser,
        }

    def parse(
        self, filepath: str, format_type: Optional[str] = None, **kwargs
    ) -> Network:
        """
        Parse a chemical reaction network file using the appropriate parser.

        Args:
            filepath: Path to the reaction file
            format_type: Format type ('uclchem', 'umist', 'latent_tgas'). If None, auto-detect.
            **kwargs: Additional arguments for specific parsers

        Returns:
            Network: Parsed chemical reaction network
        """
        if format_type is None:
            format_type = self._detect_format(filepath)

        if format_type not in self.parsers:
            raise ValueError(
                f"Unsupported format: {format_type}. "
                f"Supported formats: {list(self.parsers.keys())}"
            )

        parser_class = self.parsers[format_type]
        parser = parser_class()

        return parser.parse_network(filepath, **kwargs)

    def _detect_format(self, filepath: str) -> str:
        """Auto-detect file format based on filename and structure"""
        filename = os.path.basename(filepath).lower()

        # Format detection heuristics
        if "uclchem" in filename or filename.endswith(".rates"):
            return "uclchem"
        elif "umist" in filename:
            return "umist"
        elif "latent" in filename or "tgas" in filename:
            return "latent_tgas"

        # Fallback to file structure detection
        try:
            with open(filepath, "r") as f:
                first_line = f.readline().strip()

                # Check for UCLCHEM CSV headers
                if "Reactant 1" in first_line and "Product 1" in first_line:
                    return "uclchem"

                # Check for UMIST colon-separated format
                if first_line.count(":") > 5:
                    return "umist"

                # Default to LATENT-TGAS CSV
                return "latent_tgas"

        except Exception:
            raise ValueError(f"Could not auto-detect format for file: {filepath}")

    def register_parser(self, format_type: str, parser_class: Type[BaseParser]):
        """Register a new parser for a specific format"""
        self.parsers[format_type] = parser_class

    def list_supported_formats(self) -> list:
        """List all supported reaction network formats"""
        return list(self.parsers.keys())


# Convenience function for direct parsing
def parse_chemical_network(
    filepath: str, format_type: Optional[str] = None, **kwargs
) -> Network:
    """
    Convenience function to parse a chemical reaction network file.

    Args:
        filepath: Path to the reaction file
        format_type: Format type ('uclchem', 'umist', 'latent_tgas'). If None, auto-detect.
        **kwargs: Additional arguments for specific parsers

    Returns:
        Network: Parsed chemical reaction network
    """
    parser = UnifiedChemicalParser()
    return parser.parse(filepath, format_type, **kwargs)
