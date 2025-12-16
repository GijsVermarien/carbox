"""
Minimal tests for the top-level public API exposed by the ``carbox`` package.

These tests are intended to be fast and exercise:

- Importing symbols from the package root
- Parsing a small example network
- Running a short simulation end-to-end
"""

from pathlib import Path

import pytest

# Ensure we can import the local package without installation
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from carbox import SimulationConfig, parse_chemical_network, run_simulation


@pytest.mark.parametrize("format_type", ["latent_tgas", None])
def test_public_api_minimal_run(tmp_path, format_type):
    """End-to-end smoke test of the high-level public API.

    Uses the small LATENT-TGAS example network and a very short integration
    to keep the test fast.
    """
    network_file = PROJECT_ROOT / "data" / "simple_latent_tgas.csv"
    assert network_file.exists(), f"Test network not found: {network_file}"

    # Ensure the parser is reachable from the top-level API
    if format_type is not None:
        network = parse_chemical_network(str(network_file), format_type)
    else:
        network = parse_chemical_network(str(network_file))

    assert len(network.species) > 0
    assert len(network.reactions) > 0

    # Run a very short simulation to avoid heavy compute in tests
    config = SimulationConfig(
        number_density=1e4,
        temperature=50.0,
        t_start=0.0,
        t_end=1e2,  # years
        n_snapshots=16,
        run_name="test_public_api",
        output_dir=str(tmp_path),
        save_abundances=False,
        save_derivatives=False,
        save_rates=False,
    )

    results = run_simulation(
        str(network_file),
        config,
        format_type="latent_tgas" if format_type is not None else None,
        verbose=False,
    )

    # Check structure of the returned dict
    for key in ["solution", "network", "jnetwork", "config", "computation_time"]:
        assert key in results, f"Missing key '{key}' in run_simulation results"

    solution = results["solution"]
    out_network = results["network"]

    # Basic sanity checks on outputs
    assert len(out_network.species) == len(network.species)
    assert len(out_network.reactions) == len(network.reactions)
    assert len(solution.ts) == config.n_snapshots
