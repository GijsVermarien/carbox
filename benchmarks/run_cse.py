#!/usr/bin/env python3
"""
Run Carbox benchmark for a specific network.

Simplified standalone runner with hardcoded configurations.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import jax
import yaml

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.cse_physics import CSEPhysics

# Enable JAX 64-bit and NaN debugging
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)  # CRITICAL: Disable for performance


# Hardcoded physical parameters (matching UCLCHEM test case)
# PHYSICAL_PARAMS = {
#     "number_density": 1.0e4,  # cm^-3
#     "temperature": 250.0,  # K
#     "cr_rate": 1.0,  # s^-1
#     "fuv_field": 1.0,  # Habing units
#     "visual_extinction": 2.9643750143703076,  # mag (used if not self-consistent)
#     # Self-consistent Av calculation (optional)
#     "use_self_consistent_av": True,  # Enable self-consistent Av
#     "base_av": 2.0,  # Base Av before column density contribution
#     "cloud_radius_pc": 1.0,
#     "t_start": 0.0,  # years
#     "t_end": 1e2,  # years
#     "n_snapshots": 100,  # output timesteps (increased for detail)
#     "rtol": 1.0e-5,
#     "atol": 1.0e-25,
#     "solver": "kvaerno5",  # lowercase required
#     "max_steps": 65536,  # max steps, always use power of 16 (e.g., 4096, 65536)
# }

# Hardcoded physical parameters for the CSE outflow model
PHYSICAL_PARAMS = {
    "cr_rate": 1.0,  # s^-1
    "fuv_field": 1.0,  # Habing units

    # CSE Outflow Parameters
    "mdot": 1.0e-5,    # M_sun/yr
    "vexp": 15.0,      # km/s
    "tstar": 2000.0,   # K (Temperature at r_init)
    "eps": 0.7,        # Temperature power law
    "r_init": 1e16,  # cm
    "r_final": 1.1e17, # cm

    # Solver Parameters
    "n_snapshots": 100,  # output timesteps (increased for detail)
    "rtol": 1.0e-5,
    "atol": 1.0e-20,
    "solver": "kvaerno5",  # lowercase required
    "linear_solver" : "sparse",
    "max_steps": 65536,  # max steps, always use power of 16 (e.g., 4096, 65536)
}

# Species to track (filter output)
OUTPUT_SPECIES = [
    "H",
    "H2",
    "He",
    "C",
    "O",
    "N",
    "CO",
    "H2O",
    "OH",
    "CH",
    "NH3",
    "HCO+",
    "H3+",
    "e-",
]

# Network configurations
# Note: 'initial_conditions' is REQUIRED and must point to a valid YAML file
# containing fractional abundances extracted from UCLCHEM
NETWORK_CONFIGS = {
    "small_chemistry": {
        "description": "Small gas-phase chemistry (~20 species)",
        "input_file": "../data/uclchem_small_chemistry.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/small_chemistry_initial.yaml",
    },
    "gas_phase_only": {
        "description": "Gas-phase only chemistry (~183 species)",
        "input_file": "../data/uclchem_gas_phase_only.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/gas_phase_only_initial.yaml",
    },
    "gas_phase_only_cse": {
        "description": "Gas-phase only chemistry (~183 species)",
        "input_file": "../data/uclchem_gas_phase_only.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/orich_cse_uclchem.yaml",
    },
    "orich_cse": {
        "description": "UMIST Rate22 network with O-rich parent species",
        "input_file": "../data/umist22.csv",
        "input_format": "umist",
        "initial_conditions": "initial_conditions/orich_cse_umist.yaml",
    },
    "orich_cse_mini": {
        "description": "UMIST Rate22 network with O-rich parent species",
        "input_file": "../data/umist22_mini.csv",
        "input_format": "umist",
        "initial_conditions": "initial_conditions/orich_cse_umist.yaml",
    }

}


def run_carbox(network_name: str, output_dir: str = "results/carbox", n_runs: int = 1):
    """
    Run Carbox for specified network.

    Parameters
    ----------
    network_name : str
        Network name (must be in NETWORK_CONFIGS)
    output_dir : str
        Output directory
    n_runs : int
        Number of times to run the simulation (for timing benchmarks)

    Returns
    -------
    dict
        Benchmark results
    """
    if network_name not in NETWORK_CONFIGS:
        raise ValueError(
            f"Unknown network: {network_name}. Available: {list(NETWORK_CONFIGS.keys())}"
        )

    config_info = NETWORK_CONFIGS[network_name]

    print(f"\n{'=' * 70}")
    print(f"Running Carbox: {network_name}")
    print(f"{'=' * 70}")
    print(f"Description: {config_info['description']}")
    print("\nPhysical conditions:")
    print(f"  CR rate: {PHYSICAL_PARAMS['cr_rate']:.2e} s^-1")
    print(f"  FUV rate: {PHYSICAL_PARAMS['fuv_field']:.2e} s^-1")
    print(f"  Start radius: {PHYSICAL_PARAMS['r_init']:.2e} cm")
    print(f"  Final radius: {PHYSICAL_PARAMS['r_final']:.2e} cm")

    print("\nCSE Outflow Parameters:")
    print(f"  Mass-loss rate (mdot): {PHYSICAL_PARAMS['mdot']:.2e} M_sun/yr")
    print(f"  Expansion velocity (vexp): {PHYSICAL_PARAMS['vexp']:.1f} km/s")
    print(f"  Initial radius (r_init): {PHYSICAL_PARAMS['r_init']:.2e} cm")
    print(f"  Final radius (r_final): {PHYSICAL_PARAMS['r_final']:.2e} cm")
    print(f"  Initial temperature (tstar): {PHYSICAL_PARAMS['tstar']:.1f} K")
    print(f"  Temperature exponent (eps): {PHYSICAL_PARAMS['eps']:.2f}")

    print("\nNetwork:")
    print(f"  File: {config_info['input_file']}")
    print(f"  Format: {config_info['input_format']}")

    print("\nSolver settings:")
    print(f"  Solver: {PHYSICAL_PARAMS['solver']}")
    print(f"  rtol: {PHYSICAL_PARAMS['rtol']:.2e}")
    print(f"  atol: {PHYSICAL_PARAMS['atol']:.2e}")
    print(f"  max_steps: {PHYSICAL_PARAMS['max_steps']}")

    print("\nStarting integration...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load initial conditions from specified YAML file (REQUIRED)
    ic_path_str = config_info.get("initial_conditions")
    if not ic_path_str:
        raise ValueError(
            f"No initial_conditions specified in config for network '{network_name}'. "
            f"Add 'initial_conditions' key to NETWORK_CONFIGS."
        )

    ic_file = Path(__file__).parent / ic_path_str

    if not ic_file.exists():
        raise FileNotFoundError(
            f"\nInitial conditions file not found: {ic_file}\n"
            f"Expected path: {ic_path_str}\n\n"
            f"To generate initial conditions, run UCLCHEM first:\n"
            f"  python run_uclchem.py --network {network_name}\n"
            f"  python extract_uclchem_initial.py --network {network_name}\n"
        )

    # Load initial abundances (fractional)
    with open(ic_file, "r") as f:
        yaml_data = yaml.safe_load(f)
        initial_abundances = yaml_data["abundances"]

    print(f"\n✓ Loaded initial conditions from: {ic_file.name}")
    print(f"  Species: {len(initial_abundances)}")
    print("  Source: UCLCHEM extraction (fractional abundances)")

    # Initialize CSE Physics to get initial conditions and time range
    physics = CSEPhysics(
        mdot=PHYSICAL_PARAMS["mdot"],
        vexp=PHYSICAL_PARAMS["vexp"],
        t_star=PHYSICAL_PARAMS["tstar"],
        r_init=PHYSICAL_PARAMS["r_init"],
        eps=PHYSICAL_PARAMS["eps"],
    )

    # Calculate time range from radii
    SPY = 3600.0 * 24 * 365.0
    v_cgs = PHYSICAL_PARAMS["vexp"] * 1.0e5
    t_start_sec = 0.0
    t_end_sec = (PHYSICAL_PARAMS["r_final"] - PHYSICAL_PARAMS["r_init"]) / v_cgs
    t_start_yr = t_start_sec / SPY
    t_end_yr = t_end_sec / SPY
    print(f"  Time range: {t_start_yr:.2e} to {t_end_yr:.2e} years")

    # Get initial number density from the physics model at t=0
    initial_n, _, _, _ = physics.get_conditions(t_sec=t_start_sec)
    print(f"\n✓ Initial density at r_init: {initial_n:.2e} cm^-3")

    # Build SimulationConfig
    config = SimulationConfig(
        # Use initial conditions from physics model
        number_density=float(initial_n),
        temperature=PHYSICAL_PARAMS["tstar"],
        cr_rate=PHYSICAL_PARAMS["cr_rate"],
        fuv_field=PHYSICAL_PARAMS["fuv_field"],
        # Time settings
        t_start=t_start_yr,
        t_end=t_end_yr,
        n_snapshots=PHYSICAL_PARAMS["n_snapshots"],
        # Solver settings
        rtol=PHYSICAL_PARAMS["rtol"],
        atol=PHYSICAL_PARAMS["atol"],
        solver=PHYSICAL_PARAMS["solver"],
        linear_solver=PHYSICAL_PARAMS["linear_solver"],
        max_steps=PHYSICAL_PARAMS["max_steps"],
        output_dir=str(output_path),
        run_name=network_name,
        save_abundances=True,
        initial_abundances=initial_abundances,
    )

    # Attach the fully configured physics model to the config object
    config.physics_model = physics
    # Resolve input file path
    input_file = Path(__file__).parent / config_info["input_file"]
    if not input_file.exists():
        raise FileNotFoundError(f"Network file not found: {input_file}")

    # Measure compilation time on first run
    compile_start = time.perf_counter()

    try:
        # Run Carbox simulation (first run includes compilation)
        results = run_simulation(
            network_file=str(input_file),
            config=config,
            format_type=config_info["input_format"],
            verbose=(n_runs == 1),  # Only verbose on single run
        )

        compile_time = time.perf_counter() - compile_start

        # Extract info from results
        network = results["network"]
        solution = results["solution"]
        jnetwork = results["jnetwork"]
        n_species = len(network.species)
        n_reactions = len(network.reactions)

        # Get solver statistics from solution
        n_ode_steps = solution.stats["num_steps"]
        n_accepted = solution.stats["num_accepted_steps"]
        n_rejected = solution.stats["num_rejected_steps"]

        if n_runs == 1:
            print(f"\n✓ Integration complete in {compile_time:.2f}s")
            print(f"  Species: {n_species}")
            print(f"  Reactions: {n_reactions}")
            print(f"  ODE steps: {n_ode_steps}")
            print(f"  Accepted: {n_accepted}")
            print(f"  Rejected: {n_rejected}")

            # For single run, we can't separate compilation from runtime
            first_run_time = compile_time
            actual_compile_time = None
            mean_runtime = compile_time
            std_runtime = 0.0
            min_runtime = compile_time
            max_runtime = compile_time
            run_times = []
        else:
            # Store t0 (first run with compilation)
            first_run_time = compile_time

            print(f"\n✓ First run (compile + runtime): {first_run_time:.2f}s")
            print(f"  Species: {n_species}")
            print(f"  Reactions: {n_reactions}")

            # Run additional times to measure pure runtime
            print(f"\nRunning {n_runs - 1} additional iterations...")
            run_times = []

            for run_idx in range(n_runs - 1):
                print(f"  Run {run_idx + 2}/{n_runs}...", end=" ", flush=True)

                run_start = time.perf_counter()

                # Re-run solver with already compiled network
                from carbox.solver import solve_network

                # Get initial abundances
                y0 = results["solution"].ys[0]

                # Run solver (already compiled, so this measures pure runtime)
                _ = solve_network(jnetwork, y0, config)

                run_time = time.perf_counter() - run_start
                run_times.append(run_time)
                print(f"{run_time:.3f}s")

            # Calculate statistics
            import numpy as np

            mean_runtime = np.mean(run_times)
            std_runtime = np.std(run_times)
            min_runtime = min(run_times)
            max_runtime = max(run_times)

            # Calculate compilation time: t0 - avg(t1...tn)
            actual_compile_time = first_run_time - mean_runtime

            print("\n✓ All runs complete")
            print(f"  First run time (t0): {first_run_time:.3f}s")
            print(
                f"  Mean runtime (t1-t{n_runs}): {mean_runtime:.3f}s ± {std_runtime:.3f}s"
            )
            print(f"  Compilation time: {actual_compile_time:.3f}s")
            print(f"  Min/Max runtime: {min_runtime:.3f}s / {max_runtime:.3f}s")

        # Compute reaction rates at solution snapshots
        print("\nComputing reaction rates...")
        from carbox.solver import SPY, compute_reaction_rates

        jnetwork = results["jnetwork"]
        rates = compute_reaction_rates(jnetwork, solution, config)

        # Save rates to CSV
        import pandas as pd

        reaction_ids = [reaction.reaction_id for reaction in network.reactions]
        rates_df = pd.DataFrame(rates, columns=reaction_ids)
        # Add time column (convert from seconds to years)
        rates_df.insert(0, "time", solution.ts / SPY)

        rates_file = output_path / f"{network_name}_rates.csv"
        rates_df.to_csv(rates_file, index=False)

        # Save reaction metadata (types and strings) to YAML
        reaction_metadata = []
        for i, reaction in enumerate(network.reactions):
            reactants_str = " + ".join(reaction.reactants)
            products_str = " + ".join(reaction.products)
            reaction_metadata.append(
                {
                    "index": i,
                    "reaction": f"{reactants_str} -> {products_str}",
                    "type": reaction.reaction_type,
                }
            )

        reactions_yaml_file = output_path / f"{network_name}_reactions.yaml"
        with open(reactions_yaml_file, "w") as f:
            yaml.dump(reaction_metadata, f, default_flow_style=False)

        # Load abundance output to get timesteps
        abund_file = output_path / f"{network_name}_abundances.csv"
        df = pd.read_csv(abund_file)

        # Save benchmark metadata (convert all to native Python types for JSON)
        final_time = float(df["time_years"].iloc[-1]) if len(df) > 0 else 0

        # Prepare timing results
        if n_runs == 1:
            benchmark_results = {
                "network": network_name,
                "success": True,
                "time": first_run_time,
                "first_run_time": first_run_time,
                "compile_time": None,  # Can't separate with single run
                "mean_runtime": mean_runtime,
                "n_runs": 1,
                "n_timesteps": len(df),
                "n_species": int(n_species),
                "n_reactions": int(n_reactions),
                "n_ode_steps": int(n_ode_steps),
                "n_accepted": int(n_accepted),
                "n_rejected": int(n_rejected),
                "final_time": final_time,
                "output_file": str(abund_file),
                "physical_params": PHYSICAL_PARAMS,
            }
        else:
            benchmark_results = {
                "network": network_name,
                "success": True,
                "time": first_run_time,
                "first_run_time": first_run_time,
                "compile_time": actual_compile_time,
                "n_runs": n_runs,
                "run_times": run_times,
                "mean_runtime": mean_runtime,
                "std_runtime": std_runtime,
                "min_runtime": min_runtime,
                "max_runtime": max_runtime,
                "total_time": first_run_time + sum(run_times),
                "n_timesteps": len(df),
                "n_species": int(n_species),
                "n_reactions": int(n_reactions),
                "n_ode_steps": int(n_ode_steps),
                "n_accepted": int(n_accepted),
                "n_rejected": int(n_rejected),
                "final_time": final_time,
                "output_file": str(abund_file),
                "physical_params": PHYSICAL_PARAMS,
            }

        benchmark_file = output_path / f"{network_name}_benchmark.json"
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        print("\nSaved outputs:")
        print(f"  {abund_file}")
        print(f"  {rates_file}")
        print(f"  {reactions_yaml_file}")
        print(f"  {output_path / f'{network_name}_summary.txt'}")
        print(f"  {benchmark_file}")

        return benchmark_results

    except Exception as e:
        elapsed = time.perf_counter() - compile_start
        print(f"\nERROR: Carbox failed after {elapsed:.2f}s")
        print(f"  {type(e).__name__}: {e}")

        import traceback

        traceback.print_exc()

        return {
            "network": network_name,
            "success": False,
            "time": elapsed,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run Carbox benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available networks:
  small_chemistry - {NETWORK_CONFIGS["small_chemistry"]["description"]}
  gas_phase_only  - {NETWORK_CONFIGS["gas_phase_only"]["description"]}

Example:
  python run_carbox.py --network small_chemistry
        """,
    )

    parser.add_argument(
        "--network",
        required=True,
        choices=list(NETWORK_CONFIGS.keys()),
        help="Network to run",
    )
    parser.add_argument("--output", default="results/carbox", help="Output directory")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of times to run simulation (for timing benchmarks)",
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_carbox(args.network, args.output, args.n_runs)

    # Print summary
    print(f"\n{'=' * 70}")
    if results["success"]:
        print(f"✓ Carbox benchmark complete: {results['time']:.2f}s")
        sys.exit(0)
    else:
        print("✗ Carbox benchmark failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
