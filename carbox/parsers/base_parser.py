from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional
from ..network import Network
from ..reactions import Reaction


class BaseParser(ABC):
    """Abstract base class for chemical network parsers"""

    def __init__(self):
        self.format_type = None

    @abstractmethod
    def parse_network(self, filepath: str) -> Network:
        """Parse a chemical network file and return a Network object"""
        pass

    @abstractmethod
    def parse_reaction(self, row) -> Optional[Reaction]:
        """Parse a single reaction row"""
        pass

    def _clean_species_name(self, name: str) -> Optional[str]:
        """Normalize species names to Carbox format"""
        if pd.isna(name) or name == "NAN" or name == "":
            return None
        return str(name).strip()

    def normalize_arrhenius_params(self, row, format_type: str) -> tuple:
        """Normalize Arrhenius parameters across formats"""
        if format_type == "latent_tgas":
            return float(row["a"]), float(row["b"]), float(row["c"])
        elif format_type == "umist":
            return float(row["alpha"]), float(row["beta"]), float(row["gamma"])
        elif format_type == "uclchem":
            return float(row["Alpha"]), float(row["Beta"]), float(row["Gamma"])
        else:
            raise ValueError(f"Unknown format: {format_type}")
