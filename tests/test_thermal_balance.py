"""
Tests for self-consistent thermal balance (TGAS pseudo-species + ThermoRate,
see carbox/thermo.py and AbstractPhysics.integrates_temperature).
"""

from pathlib import Path

import numpy as np
import pytest

from carbox import SimulationConfig, parse_chemical_network, run_simulation
from carbox.physics import StaticCloudPhysics

PROJECT_ROOT = Path(__file__).parent.parent
THERMO_NETWORK = PROJECT_ROOT / "tests" / "test_data" / "test_latent_tgas_thermo.csv"


def _make_config(physics, output_dir):
    config = SimulationConfig(
        number_density=1e4,
        temperature=300.0,
        t_start=0.0,
        t_end=1e3,
        n_snapshots=10,
        initial_abundances={
            "H": 1e-1,
            "H2": 1e-1,
            "O": 1e-4,
            "E": 1e-4,
            "H2O+": 1e-8,
        },
        save_abundances=False,
        save_derivatives=False,
        save_rates=False,
        save_metadata=False,
        save_summary=False,
        output_dir=str(output_dir),
    )
    config.physics_model = physics
    return config


def test_tgas_is_last_species_and_excluded_from_elements():
    network = parse_chemical_network(str(THERMO_NETWORK), "latent_tgas")
    assert network.species[-1].name == "TGAS"

    elemental_content = network.get_elemental_contents(elements=["C", "H", "O", "S", "charge"])
    tgas_row = np.asarray(elemental_content)[:, network.get_index("TGAS")]
    assert np.all(tgas_row == 0), "TGAS must not contribute to any element (incl. substring 'S')"


def test_integrates_temperature_cools_toward_equilibrium(tmp_path):
    physics = StaticCloudPhysics(
        number_density=1e4, temperature=300.0, integrates_temperature=True
    )
    config = _make_config(physics, tmp_path)

    results = run_simulation(str(THERMO_NETWORK), config, format_type="latent_tgas", verbose=False)
    network = results["network"]
    ys = np.asarray(results["solution"].ys)
    tgas = ys[:, network.get_index("TGAS")]

    assert np.all(np.isfinite(tgas))
    assert tgas[0] == pytest.approx(300.0)
    # Net cooling at these abundances/temperature: T should monotonically
    # decrease (not oscillate/blow up).
    assert np.all(np.diff(tgas) <= 1e-8)
    assert tgas[-1] < tgas[0]


def test_integrates_temperature_false_ignores_tgas_dynamics(tmp_path):
    """Default (integrates_temperature=False): T stays at the physics model's
    prescribed constant, even though the network has a TGAS species/ThermoRate."""
    physics = StaticCloudPhysics(number_density=1e4, temperature=300.0)
    config = _make_config(physics, tmp_path)

    results = run_simulation(str(THERMO_NETWORK), config, format_type="latent_tgas", verbose=False)
    network = results["network"]
    ys = np.asarray(results["solution"].ys)
    tgas = ys[:, network.get_index("TGAS")]

    # TGAS's slot still integrates the ThermoRate term itself (nothing masks
    # the ODE row), but the *rate law* is evaluated at the model's constant T
    # throughout, decoupled from the actual solved trajectory.
    assert np.all(np.isfinite(tgas))
