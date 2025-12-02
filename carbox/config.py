"""
Configuration management for Carbox simulations.

Simple dataclass-based config for chemical kinetics simulations.
Supports loading from YAML/JSON and programmatic setup.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import jax.numpy as jnp
import yaml


@dataclass
class SimulationConfig:
    """Configuration for astrochemical kinetics simulation.

    Attributes
    ----------
    Physical Parameters:
        number_density : float
            Total hydrogen number density [cm^-3]. Range: [1e2, 1e6]
        temperature : float
            Gas temperature [K]. Range: [10, 1e5]
        cr_rate : float
            Cosmic ray ionization rate [s^-1]. Range: [1e-17, 1e-14]
        fuv_field : float
            FUV radiation field (Draine units). Range: [1e0, 1e5]
        visual_extinction : float
            Visual extinction Av [mag]. Range: [0, 10]
        gas_to_dust_ratio : float
            Gas-to-dust mass ratio. Typical: 100 (= 0.01 dust/gas)

    Initial Abundances:
        initial_abundances : Dict[str, float]
            Species name -> fractional abundance (relative to number_density)
            Example: {"H2": 1.0, "O": 2e-4, "C": 1e-4}
        abundance_floor : float
            Minimum abundance for all species (numerical stability)

    Integration Parameters:
        t_start : float
            Start time [years]
        t_end : float
            End time [years]
        n_snapshots : int
            Number of output snapshots (log-spaced)
        solver : str
            Solver name: 'dopri5', 'kvaerno5', 'tsit5'
        atol : float
            Absolute tolerance
        rtol : float
            Relative tolerance
        max_steps : int
            Maximum integration steps

    Output Settings:
        output_dir : str
            Directory for output files
        save_abundances : bool
            Save abundance time series
        save_derivatives : bool
            Save dy/dt at each snapshot
        save_rates : bool
            Save reaction rates at each snapshot
        run_name : str
            Identifier for this run
    """

    # Physical parameters
    number_density: float = 1e4
    temperature: float = 50.0
    cr_rate: float = 1e-17
    fuv_field: float = 1.0
    visual_extinction: float = 2.0  # Can be overridden by self-consistent calculation
    gas_to_dust_ratio: float = 100.0

    # Cloud geometry (for photoreaction shielding and self-consistent Av)
    cloud_radius_pc: float = 1.0  # Cloud radius in parsecs
    base_av: float = 0.0  # Base Av before column density contribution
    use_self_consistent_av: bool = False  # Compute Av from column density

    # Initial abundances (fractional relative to number_density)
    initial_abundances: Dict[str, float] = field(
        default_factory=lambda: {
            "H2": 1.0,
            "O": 2e-4,
            "C": 1e-4,
        }
    )
    abundance_floor: float = 1e-30

    # Integration parameters
    t_start: float = 0.0
    t_end: float = 1e6  # years
    n_snapshots: int = 1000
    solver: str = "kvaerno5"
    atol: float = 1e-18
    rtol: float = 1e-12
    max_steps: int = 4096

    # Output settings
    output_dir: str = "output"
    save_abundances: bool = True
    save_derivatives: bool = False
    save_rates: bool = False
    run_name: str = "carbox_run"

    @classmethod
    def from_yaml(cls, filepath: str) -> "SimulationConfig":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: str) -> "SimulationConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def compute_visual_extinction(self) -> float:
        """
        Compute self-consistent visual extinction from column density.
        
        Formula: Av = base_Av + N_H / 1.6e21
        where N_H = cloud_radius_pc * number_density (converted to cm)
        
        Returns
        -------
        float
            Visual extinction [mag]
        """
        if not self.use_self_consistent_av:
            return self.visual_extinction
        
        # Convert parsec to cm: 1 pc = 3.086e18 cm
        PC_TO_CM = 3.086e18
        cloud_radius_cm = self.cloud_radius_pc * PC_TO_CM
        
        # Column density: N_H = n_H * L [cm^-2]
        column_density = cloud_radius_cm * self.number_density
        
        # Av = base_Av + N_H / 1.6e21
        av = self.base_av + column_density / 1.6e21
        
        return av

    def get_physical_params_jax(self):
        """Get JAX arrays for physical parameters (for solver args)."""
        # Compute Av (either fixed or self-consistent)
        visual_extinction = self.compute_visual_extinction()
        
        return {
            "temperature": jnp.array(self.temperature),
            "cr_rate": jnp.array(self.cr_rate),
            "fuv_field": jnp.array(self.fuv_field),
            "visual_extinction": jnp.array(visual_extinction),
        }

    def validate(self):
        """Basic validation of parameter ranges."""
        assert 1e2 <= self.number_density <= 1e8, "number_density out of physical range"
        assert 10 <= self.temperature <= 1e5, "temperature out of range"
        # assert 1e-18 <= self.cr_rate <= 1e-12, "cr_rate out of typical range"
        assert 0 <= self.visual_extinction, "visual_extinction out of range"
        assert self.t_end > self.t_start, "t_end must be > t_start"
        assert self.solver in ["dopri5", "kvaerno5", "tsit5"], (
            f"Unknown solver: {self.solver}"
        )
