#!/usr/bin/env python3
"""Benchmark the performance of batched vmap execution.

Uses a small parameter sweep to demonstrate the efficiency.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from carbox.config import SimulationConfig
from carbox.solver import solve_network_batch

# Enable JAX 64-bit
jax.config.update("jax_enable_x64", True)


def create_test_network_and_config():
    """Create a minimal test network and config for benchmarking."""
    # Use a simple test - we'll create a minimal network
    from carbox.parsers import UCLCHEMParser

    # For this benchmark, we'll use the gas phase only network
    network_file = (
        Path(__file__).parent.parent.parent / "data" / "uclchem_gas_phase_only.csv"
    )

    parser = UCLCHEMParser(
        cloud_radius_pc=1.0,
        number_density=1e4,
    )
    network = parser.parse_network(str(network_file))
    jnetwork = network.get_ode()

    # Load initial abundances from YAML
    initial_conditions_file = (
        Path(__file__).parent.parent
        / "initial_conditions"
        / "gas_phase_only_initial.yaml"
    )
    with open(initial_conditions_file) as f:
        data = yaml.safe_load(f)
    initial_abundances = data["abundances"]
    y0 = jnp.array([initial_abundances.get(sp.name, 0.0) for sp in network.species])

    # Create base config
    base_config = SimulationConfig(
        number_density=1e4,
        temperature=250.0,
        cr_rate=1e-17,  # Will be varied
        fuv_field=1.0,
        visual_extinction=2.0,
        use_self_consistent_av=False,
        t_start=0.0,
        t_end=1e5,  # Shorter for faster benchmarking
        n_snapshots=100,
        rtol=1e-9,
        atol=1e-30,
        solver="kvaerno5",
        max_steps=4096,
    )

    return network, jnetwork, y0, base_config


def benchmark_vmap(cr_rates, jnetwork, y0, base_config):
    """Run parameter sweep using batched vmap (without JIT)."""
    start_time = time.perf_counter()

    # Create parameter arrays
    cr_array = jnp.array(cr_rates)
    temp_array = jnp.full_like(cr_array, base_config.temperature)
    fuv_array = jnp.full_like(cr_array, base_config.fuv_field)
    av_array = jnp.full_like(cr_array, base_config.visual_extinction)

    t_snapshots = jnp.linspace(
        base_config.t_start, base_config.t_end, base_config.n_snapshots
    )

    # Batch solve without JIT
    solutions = solve_network_batch(
        jnetwork=jnetwork,
        y0=y0,
        t_eval=t_snapshots,
        temperatures=temp_array,
        cr_rates=cr_array,
        fuv_fields=fuv_array,
        visual_extinctions=av_array,
        solver_name=base_config.solver,
        atol=base_config.atol,
        rtol=base_config.rtol,
        max_steps=base_config.max_steps,
    )

    elapsed = time.perf_counter() - start_time

    # Process results
    results = []
    for i, cr_rate in enumerate(cr_rates):
        ys_i = solutions.ys[i]  # type:ignore
        success = ys_i is not None and len(ys_i) > 0
        results.append(
            {
                "cr_rate": float(cr_rate),
                "success": success,
                "final_abundance": float(ys_i[-1, 0]) if success else 0.0,
            }
        )

    return results, elapsed


def main():
    """The main entrypoint that benchmarks the parameter sweep efficiency."""
    print("=" * 70)
    print("Parallel Speedup Benchmark")
    print("=" * 70)

    # Create test setup
    network, jnetwork, y0, base_config = create_test_network_and_config()

    # Compute unique reactants
    unique_reactants = set()
    for reaction in network.reactions:
        unique_reactants.update(reaction.reactants)
    print(f"Number of Reactions: {len(network.reactions)}")
    print(f"Unique reactants: {len(unique_reactants)}")

    # Test parameter values (small sweep for quick benchmarking)
    cr_rates = np.logspace(-18, -16, 128)

    print(f"Testing with {len(cr_rates)} parameter values")
    print(f"Parameter range: {cr_rates[0]:.2e} to {cr_rates[-1]:.2e} s^-1")
    print(f"Time range: {base_config.t_start} to {base_config.t_end} years")
    print(f"Snapshots: {base_config.n_snapshots}")
    print()

    # Benchmark vmap
    print("Running vmap benchmark...")
    vmap_results, vmap_time = benchmark_vmap(cr_rates, jnetwork, y0, base_config)
    vmap_success = sum(1 for r in vmap_results if r["success"])
    print(f"Vmap time: {vmap_time:.2f}s")
    print(f"  Successful: {vmap_success}/{len(cr_rates)}")
    print()

    # Check results
    if vmap_success != len(cr_rates):
        raise Exception("Some solves were not successful.")

    print(f"Vmap time: {vmap_time:.2f}s")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
