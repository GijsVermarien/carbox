"""
Top-level public API for the Carbox astrochemical kinetics framework.

The intent is to expose a small, stable surface for typical users:

- ``SimulationConfig``: dataclass configuring a single simulation.
- ``run_simulation``: high-level helper to run a network from a file.
- ``parse_chemical_network``: load reaction networks from UMIST/UCLCHEM/latent_tgas
  and other supported formats into a unified internal representation.

Example
-------
>>> from carbox import SimulationConfig, run_simulation
>>> config = SimulationConfig(number_density=1e4, temperature=50.0, t_end=1e6)
>>> results = run_simulation("data/network.csv", config, format_type="latent_tgas")
"""

from .config import SimulationConfig
from .main import run_simulation
from .network import JNetwork, Network
from .parsers import parse_chemical_network

__all__ = [
    "SimulationConfig",
    "run_simulation",
    "parse_chemical_network",
    "Network",
    "JNetwork",
]


