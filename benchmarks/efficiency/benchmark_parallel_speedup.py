#!/usr/bin/env python3
"""Benchmark parallel speedup for parameter sweeps.

Compares the performance of:
1. Sequential for-loop execution (old approach)
2. Batched vmap execution (new approach)

Uses a small parameter sweep to demonstrate the speedup.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from carbox.config import SimulationConfig
from carbox.solver import solve_network, solve_network_batch

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

    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")

    parser = UCLCHEMParser(
        cloud_radius_pc=1.0,
        number_density=1e4,
    )
    network = parser.parse_network(str(network_file))
    jnetwork = network.get_ode()

    # Create initial conditions (simple test abundances)
    initial_abundances = {
        "H2": 1.0,
        "O": 2e-4,
        "C": 1e-4,
    }
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
        n_snapshots=50,
        rtol=1e-9,
        atol=1e-30,
        solver="kvaerno5",
        max_steps=4096,
    )

    return jnetwork, y0, base_config


def benchmark_sequential(cr_rates, jnetwork, y0, base_config):
    """Run parameter sweep using sequential for-loop (old approach)."""
    start_time = time.perf_counter()

    results = []
    for cr_rate in cr_rates:
        # Create config for this cr_rate
        config_dict = base_config.__dict__.copy()
        config_dict["cr_rate"] = cr_rate
        config = SimulationConfig(**config_dict)

        # Solve
        solution = solve_network(jnetwork, y0, config)

        # Store result (just check if successful)
        success = (
            hasattr(solution, "ys") and solution.ys is not None and len(solution.ys) > 0
        )
        results.append(
            {
                "cr_rate": float(cr_rate),
                "success": success,
                "final_abundance": float(solution.ys[-1, 0]) if success else 0.0,
            }
        )

    elapsed = time.perf_counter() - start_time
    return results, elapsed


def benchmark_batch(cr_rates, jnetwork, y0, base_config):
    """Run parameter sweep using batched vmap with JIT (new approach)."""
    # Create parameter arrays
    cr_array = jnp.array(cr_rates)
    temp_array = jnp.full_like(cr_array, base_config.temperature)
    fuv_array = jnp.full_like(cr_array, base_config.fuv_field)
    av_array = jnp.full_like(cr_array, base_config.visual_extinction)

    if base_config.t_start <= 0:
        t_start_log = -9
        t_log = jnp.logspace(
            t_start_log, jnp.log10(base_config.t_end), base_config.n_snapshots - 1
        )
        t_snapshots = jnp.concatenate([jnp.array([0.0]), t_log])
    else:
        t_log = jnp.logspace(
            jnp.log10(base_config.t_start),
            jnp.log10(base_config.t_end),
            base_config.n_snapshots - 1,
        )
        t_snapshots = jnp.concatenate([jnp.array([base_config.t_start]), t_log])

    # JIT compile the batch solver
    jit_batch_solve = jax.jit(
        solve_network_batch,
        static_argnames=["solver_name", "atol", "rtol", "max_steps"],
    )

    # Warm-up to exclude compilation time from benchmark
    _ = jit_batch_solve(
        jnetwork=jnetwork,
        y0=y0,
        t_eval=t_snapshots,
        temperatures=temp_array[:1],  # Use first element for warm-up
        cr_rates=cr_array[:1],
        fuv_fields=fuv_array[:1],
        visual_extinctions=av_array[:1],
        solver_name=base_config.solver,
        atol=base_config.atol,
        rtol=base_config.rtol,
        max_steps=base_config.max_steps,
    )

    # Now time the actual execution
    start_time = time.perf_counter()
    solutions = jit_batch_solve(
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
        ys_i = solutions.ys[i]
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
    print("=" * 70)
    print("Parallel Speedup Benchmark")
    print("=" * 70)

    # Create test setup
    try:
        jnetwork, y0, base_config = create_test_network_and_config()
        print("✓ Test setup created successfully")
    except Exception as e:
        print(f"✗ Failed to create test setup: {e}")
        return

    # Test parameter values (small sweep for quick benchmarking)
    cr_rates = np.logspace(-18, -16, 8)  # 8 values from 1e-18 to 1e-16

    print(f"Testing with {len(cr_rates)} parameter values")
    print(f"Parameter range: {cr_rates[0]:.2e} to {cr_rates[-1]:.2e} s^-1")
    print(f"Time range: {base_config.t_start} to {base_config.t_end} years")
    print(f"Snapshots: {base_config.n_snapshots}")
    print()

    # Benchmark sequential
    print("Running sequential (for-loop) benchmark...")
    seq_results, seq_time = benchmark_sequential(cr_rates, jnetwork, y0, base_config)
    seq_success = sum(1 for r in seq_results if r["success"])
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"  Successful: {seq_success}/{len(cr_rates)}")
    print()

    # Benchmark batch
    print("Running batch (vmap) benchmark...")
    batch_results, batch_time = benchmark_batch(cr_rates, jnetwork, y0, base_config)
    batch_success = sum(1 for r in batch_results if r["success"])
    print(f"Batch time: {batch_time:.2f}s")
    print(f"  Successful: {batch_success}/{len(cr_rates)}")
    print()

    # Compare results
    if seq_success == batch_success == len(cr_rates):
        print("✓ All simulations successful in both methods")
    else:
        print("⚠ Different success rates - check implementation")

    # Calculate speedup
    if batch_time > 0:
        speedup = seq_time / batch_time
        print(f"Speedup: {speedup:.1f}x")
    else:
        print("✗ Batch execution failed")

    # Verify results are similar
    max_diff = 0
    for seq_r, batch_r in zip(seq_results, batch_results):
        if seq_r["success"] and batch_r["success"]:
            diff = abs(seq_r["final_abundance"] - batch_r["final_abundance"])
            max_diff = max(max_diff, diff)

    print(f"Max abundance difference: {max_diff:.2e}")
    if max_diff < 1e-10:
        print("✓ Results are numerically identical")
    elif max_diff < 1e-6:
        print("✓ Results are very close (within numerical precision)")
    else:
        print("⚠ Results differ significantly - check implementation")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
