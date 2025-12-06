"""Base class Carbox parser for parsing chemical networks."""

from abc import ABC, abstractmethod

import pandas as pd

from ..network import Network
from ..reactions import Reaction


class BaseParser(ABC):
    """Abstract base class for chemical network parsers."""

    def __init__(self, format_type: str):  # noqa
        self.format_type = format_type

    @abstractmethod
    def parse_network(self, filepath: str) -> Network:
        """Parse a chemical network file and return a Network object."""

    @abstractmethod
    def parse_reaction(self, row) -> Reaction | None:
        """Parse a single reaction row."""

    def _clean_species_name(self, name: str) -> str | None:
        """Normalize species names to Carbox format."""
        if pd.isna(name) or name == "NAN" or name == "":
            return None
        return str(name).strip()

    def normalize_arrhenius_params(self, row, format_type: str) -> tuple:
        """Normalize Arrhenius parameters across formats."""
        if format_type == "latent_tgas":
            return float(row["a"]), float(row["b"]), float(row["c"])
        elif format_type == "umist":
            return float(row["alpha"]), float(row["beta"]), float(row["gamma"])
        elif format_type == "uclchem":
            return float(row["Alpha"]), float(row["Beta"]), float(row["Gamma"])
        else:
            raise ValueError(f"Unknown format: {format_type}")
