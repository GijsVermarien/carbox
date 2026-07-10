#!/usr/bin/env python3
"""
Run a Carbox benchmark for a specific network.

Unified runner replacing the former run_carbox.py (static-cloud) and
run_cse.py (CSE outflow) scripts. Each NETWORK_CONFIGS entry declares its
own physics model (static cloud or CSE outflow) via the "physics" key;
everything downstream (solver, output) is agnostic to which one it gets.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import pandas as pd
import yaml

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.physics import CSEPhysics, StaticCloudPhysics
from carbox.solver import SPY, compute_reaction_rates, solve_network

# Enable JAX 64-bit; NaN debugging is expensive, keep off by default
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)


# Network configurations. Each entry:
#   description, input_file, input_format, initial_conditions (required
#   YAML of fractional abundances), cr_rate, fuv_field,
#   physics: {"type": "static"|"cse", ...physics-specific params},
#   solver: {t_start, t_end (years; auto-derived for CSE from r_final if
#            omitted), n_snapshots, rtol, atol, solver, linear_solver,
#            max_steps}
NETWORK_CONFIGS = {
    "small_chemistry": {
        "description": "Small gas-phase chemistry (~20 species)",
        "input_file": "../data/uclchem_small_chemistry.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/small_chemistry_initial.yaml",
        "cr_rate": 1.0,
        "fuv_field": 1.0,
        "physics": {
            "type": "static",
            "number_density": 1.0e4,
            "temperature": 250.0,
            "use_self_consistent_av": True,
            "base_av": 2.0,
            "cloud_radius_pc": 1.0,
        },
        "solver": {
            "t_start": 0.0,
            "t_end": 5.0e6,
            "n_snapshots": 100,
            "rtol": 1.0e-9,
            "atol": 1.0e-30,
            "solver": "kvaerno5",
            "max_steps": 65536,
        },
    },
    "gas_phase_only": {
        "description": "Gas-phase only chemistry (~183 species)",
        "input_file": "../data/uclchem_gas_phase_only.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/gas_phase_only_initial.yaml",
        "cr_rate": 1.0,
        "fuv_field": 1.0,
        "physics": {
            "type": "static",
            "number_density": 1.0e4,
            "temperature": 250.0,
            "use_self_consistent_av": True,
            "base_av": 2.0,
            "cloud_radius_pc": 1.0,
        },
        "solver": {
            "t_start": 0.0,
            "t_end": 5.0e6,
            "n_snapshots": 100,
            "rtol": 1.0e-9,
            "atol": 1.0e-30,
            "solver": "kvaerno5",
            "max_steps": 65536,
        },
    },
    "gas_phase_only_cse": {
        "description": "Gas-phase only chemistry (~183 species), CSE outflow",
        "input_file": "../data/uclchem_gas_phase_only.csv",
        "input_format": "uclchem",
        "initial_conditions": "initial_conditions/orich_cse_uclchem.yaml",
        "cr_rate": 1.0,
        "fuv_field": 1.0,
        "physics": {
            "type": "cse",
            "mdot": 1.0e-5,
            "vexp": 15.0,
            "t_star": 2000.0,
            "r_init": 1.0e16,
            "r_final": 1.1e17,
            "r_star": 5.0e13,
            "eps": 0.7,
        },
        "solver": {
            "n_snapshots": 100,
            "rtol": 1.0e-5,
            "atol": 1.0e-20,
            "solver": "kvaerno5",
            "linear_solver": "sparse",
            "max_steps": 65536,
        },
    },
    "orich_cse": {
        "description": "UMIST Rate22 network with O-rich parent species, CSE outflow",
        "input_file": "../data/umist22.csv",
        "input_format": "umist",
        "initial_conditions": "initial_conditions/orich_cse_umist.yaml",
        "cr_rate": 1.0,
        "fuv_field": 1.0,
        "physics": {
            "type": "cse",
            "mdot": 1.0e-5,
            "vexp": 15.0,
            "t_star": 2000.0,
            "r_init": 1.0e16,
            "r_final": 1.1e17,
            "r_star": 5.0e13,
            "eps": 0.7,
        },
        "solver": {
            "n_snapshots": 100,
            "rtol": 1.0e-5,
            "atol": 1.0e-20,
            "solver": "kvaerno5",
            "linear_solver": "sparse",
            "max_steps": 65536,
        },
    },
    "orich_cse_mini": {
        "description": "Small UMIST subset with O-rich parent species, CSE outflow",
        "input_file": "../data/umist22_mini.csv",
        "input_format": "umist",
        "initial_conditions": "initial_conditions/orich_cse_umist.yaml",
        "cr_rate": 1.0,
        "fuv_field": 1.0,
        "physics": {
            "type": "cse",
            "mdot": 1.0e-5,
            "vexp": 15.0,
            "t_star": 2000.0,
            "r_init": 1.0e16,
            "r_final": 1.1e17,
            "r_star": 5.0e13,
            "eps": 0.7,
        },
        "solver": {
            "n_snapshots": 100,
            "rtol": 1.0e-5,
            "atol": 1.0e-20,
            "solver": "kvaerno5",
            "linear_solver": "sparse",
            "max_steps": 65536,
        },
    },
}


def build_physics(physics_spec: dict):
    """Construct the physics model for a NETWORK_CONFIGS "physics" entry."""
    physics_type = physics_spec["type"]
    # "type" selects the class below; "r_final" (CSE) is a driver-level
    # parameter used only to derive t_end, not a CSEPhysics field
    params = {k: v for k, v in physics_spec.items() if k not in ("type", "r_final")}

    if physics_type == "static":
        return StaticCloudPhysics(**params)
    elif physics_type == "cse":
        return CSEPhysics(**params)
    else:
        raise ValueError(f"Unknown physics type: {physics_type}")


def resolve_time_range(physics_spec: dict, solver_spec: dict) -> tuple:
    """Determine (t_start_years, t_end_years) for the integration.

    For CSE outflows, t_end is derived from r_final unless explicitly
    overridden in the solver spec.
    """
    t_start = solver_spec.get("t_start", 0.0)
    if "t_end" in solver_spec:
        return t_start, solver_spec["t_end"]

    if physics_spec["type"] == "cse" and "r_final" in physics_spec:
        v_cgs = physics_spec["vexp"] * 1.0e5
        t_end_sec = (physics_spec["r_final"] - physics_spec["r_init"]) / v_cgs
        return t_start, t_end_sec / SPY

    raise ValueError(
        "solver.t_end must be given explicitly unless physics is a CSE "
        "model with r_final set"
    )


def run_carbox(network_name: str, output_dir: str = "results/carbox", n_runs: int = 1):
    """
    Run Carbox for the specified network configuration.

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
    physics_spec = config_info["physics"]
    solver_spec = config_info["solver"]

    print(f"\n{'=' * 70}")
    print(f"Running Carbox: {network_name}")
    print(f"{'=' * 70}")
    print(f"Description: {config_info['description']}")
    print(f"Physics model: {physics_spec['type']}")
    print("\nNetwork:")
    print(f"  File: {config_info['input_file']}")
    print(f"  Format: {config_info['input_format']}")
    print("\nSolver settings:")
    print(f"  Solver: {solver_spec['solver']}")
    print(f"  rtol: {solver_spec['rtol']:.2e}")
    print(f"  atol: {solver_spec['atol']:.2e}")
    print(f"  max_steps: {solver_spec['max_steps']}")
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

    with open(ic_file, "r") as f:
        initial_abundances = yaml.safe_load(f)["abundances"]

    print(f"\n✓ Loaded initial conditions from: {ic_file.name}")
    print(f"  Species: {len(initial_abundances)}")

    # Build the physics model and derive the integration time range from it
    physics = build_physics(physics_spec)
    t_start_yr, t_end_yr = resolve_time_range(physics_spec, solver_spec)
    print(f"  Time range: {t_start_yr:.2e} to {t_end_yr:.2e} years")

    initial_n, initial_T, _, _ = physics.get_conditions(t_sec=t_start_yr * SPY)
    print(f"\n✓ Initial conditions: n={float(initial_n):.2e} cm^-3, T={float(initial_T):.1f} K")

    # Build SimulationConfig; physics_model takes precedence over the
    # legacy scalar fields, so number_density/temperature below are only
    # used if physics_model were absent (kept for config introspection).
    config = SimulationConfig(
        cr_rate=config_info["cr_rate"],
        fuv_field=config_info["fuv_field"],
        t_start=t_start_yr,
        t_end=t_end_yr,
        n_snapshots=solver_spec["n_snapshots"],
        rtol=solver_spec["rtol"],
        atol=solver_spec["atol"],
        solver=solver_spec["solver"],
        linear_solver=solver_spec.get("linear_solver", "lu"),
        max_steps=solver_spec["max_steps"],
        output_dir=str(output_path),
        run_name=network_name,
        save_abundances=True,
        initial_abundances=initial_abundances,
        physics_model=physics,
    )

    # Resolve input file path
    input_file = Path(__file__).parent / config_info["input_file"]
    if not input_file.exists():
        raise FileNotFoundError(f"Network file not found: {input_file}")

    compile_start = time.perf_counter()

    try:
        results = run_simulation(
            network_file=str(input_file),
            config=config,
            format_type=config_info["input_format"],
            verbose=(n_runs == 1),
        )

        compile_time = time.perf_counter() - compile_start

        network = results["network"]
        solution = results["solution"]
        jnetwork = results["jnetwork"]
        n_species = len(network.species)
        n_reactions = len(network.reactions)

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

            first_run_time = compile_time
            actual_compile_time = None
            mean_runtime = compile_time
            std_runtime = 0.0
            min_runtime = compile_time
            max_runtime = compile_time
            run_times = []
        else:
            first_run_time = compile_time

            print(f"\n✓ First run (compile + runtime): {first_run_time:.2f}s")
            print(f"  Species: {n_species}")
            print(f"  Reactions: {n_reactions}")
            print(f"\nRunning {n_runs - 1} additional iterations...")
            run_times = []

            for run_idx in range(n_runs - 1):
                print(f"  Run {run_idx + 2}/{n_runs}...", end=" ", flush=True)
                run_start = time.perf_counter()

                y0 = results["solution"].ys[0]
                _ = solve_network(jnetwork, y0, config)

                run_time = time.perf_counter() - run_start
                run_times.append(run_time)
                print(f"{run_time:.3f}s")

            import numpy as np

            mean_runtime = np.mean(run_times)
            std_runtime = np.std(run_times)
            min_runtime = min(run_times)
            max_runtime = max(run_times)
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
        rates = compute_reaction_rates(jnetwork, solution, config)

        reaction_ids = [reaction.reaction_id for reaction in network.reactions]
        rates_df = pd.DataFrame(rates, columns=reaction_ids)
        rates_df.insert(0, "time", solution.ts / SPY)

        rates_file = output_path / f"{network_name}_rates.csv"
        rates_df.to_csv(rates_file, index=False)

        # Save reaction metadata (types and strings) to YAML
        reaction_metadata = [
            {
                "index": i,
                "reaction_id": reaction.reaction_id,
                "reaction": f"{' + '.join(reaction.reactants)} -> {' + '.join(reaction.products)}",
                "type": reaction.reaction_type,
            }
            for i, reaction in enumerate(network.reactions)
        ]
        reactions_yaml_file = output_path / f"{network_name}_reactions.yaml"
        with open(reactions_yaml_file, "w") as f:
            yaml.dump(reaction_metadata, f, default_flow_style=False)

        # Load abundance output to get timesteps
        abund_file = output_path / f"{network_name}_abundances.csv"
        df = pd.read_csv(abund_file)
        final_time = float(df["time_years"].iloc[-1]) if len(df) > 0 else 0

        benchmark_results = {
            "network": network_name,
            "success": True,
            "time": first_run_time,
            "first_run_time": first_run_time,
            "compile_time": actual_compile_time,
            "mean_runtime": mean_runtime,
            "n_runs": n_runs,
            "run_times": run_times,
            "std_runtime": std_runtime,
            "min_runtime": min_runtime,
            "max_runtime": max_runtime,
            "n_timesteps": len(df),
            "n_species": int(n_species),
            "n_reactions": int(n_reactions),
            "n_ode_steps": int(n_ode_steps),
            "n_accepted": int(n_accepted),
            "n_rejected": int(n_rejected),
            "final_time": final_time,
            "output_file": str(abund_file),
            "physics": physics_spec,
            "solver": solver_spec,
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
        description="Run a Carbox benchmark (static-cloud or CSE outflow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available networks:\n"
        + "\n".join(
            f"  {name:<20} - {cfg['description']}"
            for name, cfg in NETWORK_CONFIGS.items()
        )
        + "\n\nExample:\n  python run.py --network gas_phase_only_cse",
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

    results = run_carbox(args.network, args.output, args.n_runs)

    print(f"\n{'=' * 70}")
    if results["success"]:
        print(f"✓ Carbox benchmark complete: {results['time']:.2f}s")
        sys.exit(0)
    else:
        print("✗ Carbox benchmark failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
