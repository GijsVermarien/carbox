#!/usr/bin/env python3
"""Cosmic Ray Ionization Rate Sensitivity Analysis.

Varies the cosmic ray ionization rate (ζ) over 6 orders of magnitude
to assess impact on chemical evolution.

Network: gas_phase_only (~183 species)
Zeta range: 10^-2 to 10^4 s^-1 (36 logarithmically spaced values)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbox.config import SimulationConfig  # noqa: E402
from carbox.solver import SPY, solve_network, solve_network_batch  # noqa: E402

# Enable JAX 64-bit and NaN debugging
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


# Physical parameters (matching benchmarks, except cr_rate)
PHYSICAL_PARAMS = {
    "number_density": 1.0e4,  # cm^-3
    "temperature": 250.0,  # K
    "cr_rate": 1.0,  # s^-1 (will be varied)
    "fuv_field": 1.0,  # Habing units
    "visual_extinction": 2.9643750143703076,  # mag
    "use_self_consistent_av": True,
    "base_av": 2.0,
    "cloud_radius_pc": 1.0,
    "t_start": 0.0,  # years
    "t_end": 5.0e6,  # years
    "n_snapshots": 100,
    "rtol": 1.0e-9,
    "atol": 1.0e-30,
    "solver": "kvaerno5",
    "max_steps": 65536,
}


def run_single_zeta(
    zeta_value: float,
    network_file: Path,
    initial_conditions_file: Path,
    output_dir: Path,
    verbose: bool = False,
):
    """Run simulation with specific cosmic ray ionization rate.

    Parameters
    ----------
    zeta_value : float
        Cosmic ray ionization rate [s^-1]
    network_file : Path
        Path to network CSV file
    initial_conditions_file : Path
        Path to initial conditions YAML
    output_dir : Path
        Output directory for this zeta value
    verbose : bool
        Print detailed output

    Returns:
    -------
    dict
        Results summary
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Cosmic ray rate: {zeta_value:.4e} s^-1")
        print(f"{'=' * 70}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load initial conditions
    with open(initial_conditions_file) as f:
        yaml_data = yaml.safe_load(f)
        initial_abundances = yaml_data["abundances"]

    # Build config with varied cr_rate
    config = SimulationConfig(
        number_density=PHYSICAL_PARAMS["number_density"],
        temperature=PHYSICAL_PARAMS["temperature"],
        cr_rate=zeta_value,  # Variable parameter
        fuv_field=PHYSICAL_PARAMS["fuv_field"],
        visual_extinction=PHYSICAL_PARAMS["visual_extinction"],
        use_self_consistent_av=PHYSICAL_PARAMS["use_self_consistent_av"],
        cloud_radius_pc=PHYSICAL_PARAMS["cloud_radius_pc"],
        base_av=PHYSICAL_PARAMS["base_av"],
        t_start=PHYSICAL_PARAMS["t_start"],
        t_end=PHYSICAL_PARAMS["t_end"],
        n_snapshots=PHYSICAL_PARAMS["n_snapshots"],
        rtol=PHYSICAL_PARAMS["rtol"],
        atol=PHYSICAL_PARAMS["atol"],
        solver=PHYSICAL_PARAMS["solver"],
        max_steps=PHYSICAL_PARAMS["max_steps"],
        output_dir=str(output_dir),
        run_name=f"zeta_{zeta_value:.4e}",
        save_abundances=True,
        initial_abundances=initial_abundances,
    )

    start_time = time.perf_counter()

    try:
        # Parse network using UCLCHEM parser with cloud parameters
        from carbox.parsers import UCLCHEMParser

        parser = UCLCHEMParser(
            cloud_radius_pc=config.cloud_radius_pc,
            number_density=config.number_density,
        )
        network = parser.parse_network(str(network_file))

        if verbose:
            print(f"  Total species: {len(network.species)}")
            print(f"  Total reactions: {len(network.reactions)}")

        # Compile network to JAX
        jnetwork = network.get_ode()

        # Get initial state
        y0 = jnp.array([initial_abundances.get(sp.name, 0.0) for sp in network.species])

        # Solve ODE
        solution = solve_network(jnetwork, y0, config)

        elapsed = time.perf_counter() - start_time

        # Check solution success
        if not hasattr(solution, "ys") or len(solution.ys) == 0:
            raise RuntimeError("Solver failed to produce output")

        # Save abundances
        species_names = [sp.name for sp in network.species]
        abund_df = pd.DataFrame(
            solution.ys,
            columns=species_names,
        )
        abund_df.insert(0, "time_years", solution.ts / SPY)

        abund_file = output_dir / "abundances.csv"
        abund_df.to_csv(abund_file, index=False)

        # Save metadata
        metadata = {
            "cr_rate": float(zeta_value),
            "success": True,
            "elapsed_time": elapsed,
            "n_species": len(network.species),
            "n_reactions": len(network.reactions),
            "n_timesteps": len(solution.ts),
            "n_ode_steps": int(solution.stats["num_steps"]),
            "n_accepted": int(solution.stats["num_accepted_steps"]),
            "n_rejected": int(solution.stats["num_rejected_steps"]),
            "final_time_years": float(solution.ts[-1] / SPY),
            "physical_params": PHYSICAL_PARAMS.copy(),
        }
        metadata["physical_params"]["cr_rate"] = float(zeta_value)

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            print(f"  ✓ Complete in {elapsed:.2f}s")
            print(f"  ODE steps: {metadata['n_ode_steps']}")
            print(f"  Saved: {abund_file}")

        return metadata

    except Exception as e:
        elapsed = time.perf_counter() - start_time

        if verbose:
            print(f"  ✗ Failed after {elapsed:.2f}s")
            print(f"  Error: {e}")

        import traceback

        traceback.print_exc()

        return {
            "cr_rate": float(zeta_value),
            "success": False,
            "elapsed_time": elapsed,
            "error": str(e),
        }


def run_batch_zeta(
    zeta_values: np.ndarray,
    network_file: Path,
    initial_conditions_file: Path,
    output_base: Path,
    verbose: bool = False,
):
    """Run batch simulations with multiple cosmic ray ionization rates.

    Parameters
    ----------
    zeta_values : np.ndarray
        Cosmic ray ionization rates [s^-1]
    network_file : Path
        Path to network CSV file
    initial_conditions_file : Path
        Path to initial conditions YAML
    output_base : Path
        Base output directory
    verbose : bool
        Print detailed output

    Returns:
    -------
    list[dict]
        Results summaries for each zeta value
    """
    start_time = time.perf_counter()

    try:
        # Load initial conditions
        with open(initial_conditions_file) as f:
            yaml_data = yaml.safe_load(f)
            initial_abundances = yaml_data["abundances"]

        # Parse network using UCLCHEM parser with cloud parameters
        from carbox.parsers import UCLCHEMParser

        parser = UCLCHEMParser(
            cloud_radius_pc=PHYSICAL_PARAMS["cloud_radius_pc"],
            number_density=PHYSICAL_PARAMS["number_density"],
        )
        network = parser.parse_network(str(network_file))

        if verbose:
            print(f"  Total species: {len(network.species)}")
            print(f"  Total reactions: {len(network.reactions)}")

        # Compile network to JAX
        jnetwork = network.get_ode()

        # Get initial state
        y0 = jnp.array([initial_abundances.get(sp.name, 0.0) for sp in network.species])

        # Create time evaluation points (same for all simulations)
        # Use same logic as solve_network
        t_start_sec = PHYSICAL_PARAMS["t_start"] * SPY
        t_end_sec = PHYSICAL_PARAMS["t_end"] * SPY

        if PHYSICAL_PARAMS["t_start"] <= 0:
            t_start_log = -9
            t_log = jnp.logspace(
                t_start_log,
                jnp.log10(PHYSICAL_PARAMS["t_end"]),
                PHYSICAL_PARAMS["n_snapshots"] - 1,
            )
            t_snapshots = jnp.concatenate([jnp.array([0.0]), t_log])
        else:
            t_log = jnp.logspace(
                jnp.log10(PHYSICAL_PARAMS["t_start"]),
                jnp.log10(PHYSICAL_PARAMS["t_end"]),
                PHYSICAL_PARAMS["n_snapshots"] - 1,
            )
            t_snapshots = jnp.concatenate(
                [jnp.array([PHYSICAL_PARAMS["t_start"]]), t_log]
            )

        # Create parameter arrays
        zeta_array = jnp.array(zeta_values)
        temp_array = jnp.full_like(zeta_array, PHYSICAL_PARAMS["temperature"])
        fuv_array = jnp.full_like(zeta_array, PHYSICAL_PARAMS["fuv_field"])

        # Compute visual extinction (same for all since other params are fixed)
        config_temp = SimulationConfig(
            number_density=PHYSICAL_PARAMS["number_density"],
            temperature=PHYSICAL_PARAMS["temperature"],
            cr_rate=PHYSICAL_PARAMS["cr_rate"],
            fuv_field=PHYSICAL_PARAMS["fuv_field"],
            visual_extinction=PHYSICAL_PARAMS["visual_extinction"],
            use_self_consistent_av=PHYSICAL_PARAMS["use_self_consistent_av"],
            cloud_radius_pc=PHYSICAL_PARAMS["cloud_radius_pc"],
            base_av=PHYSICAL_PARAMS["base_av"],
        )
        av_value = config_temp.compute_visual_extinction()
        av_array = jnp.full_like(zeta_array, av_value)

        # Batch solve
        solutions = solve_network_batch(
            jnetwork=jnetwork,
            y0=y0,
            t_eval=t_snapshots,
            temperatures=temp_array,
            cr_rates=zeta_array,
            fuv_fields=fuv_array,
            visual_extinctions=av_array,
            solver_name=PHYSICAL_PARAMS["solver"],
            atol=PHYSICAL_PARAMS["atol"],
            rtol=PHYSICAL_PARAMS["rtol"],
            max_steps=PHYSICAL_PARAMS["max_steps"],
        )

        elapsed = time.perf_counter() - start_time

        # Process batched results
        results = []
        species_names = [sp.name for sp in network.species]

        # solutions is a batched Solution object
        # ts: (batch_size, n_timesteps), ys: (batch_size, n_timesteps, n_species)
        # stats fields are also batched
        for i, zeta in enumerate(zeta_values):
            output_dir = output_base / f"zeta_{zeta:.4e}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract individual solution from batch
            ts_i = solutions.ts[i]  # Shape: (n_timesteps,)
            ys_i = solutions.ys[i]  # Shape: (n_timesteps, n_species)

            # Check solution success
            success = ys_i is not None and len(ys_i) > 0

            if success:
                # Save abundances
                abund_df = pd.DataFrame(ys_i, columns=species_names)
                abund_df.insert(0, "time_years", ts_i / SPY)

                abund_file = output_dir / "abundances.csv"
                abund_df.to_csv(abund_file, index=False)

                # Save metadata
                metadata = {
                    "cr_rate": float(zeta),
                    "success": True,
                    "elapsed_time": elapsed / len(zeta_values),  # Per-simulation time
                    "n_species": len(network.species),
                    "n_reactions": len(network.reactions),
                    "n_timesteps": len(ts_i),
                    "n_ode_steps": int(solutions.stats["num_steps"][i]),
                    "n_accepted": int(solutions.stats["num_accepted_steps"][i]),
                    "n_rejected": int(solutions.stats["num_rejected_steps"][i]),
                    "final_time_years": float(ts_i[-1] / SPY),
                    "physical_params": PHYSICAL_PARAMS.copy(),
                }
                metadata["physical_params"]["cr_rate"] = float(zeta)

                metadata_file = output_dir / "metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                if verbose:
                    print(f"  ✓ ζ = {zeta:.4e} complete")
                    print(f"     Saved: {abund_file}")

                results.append(metadata)
            else:
                if verbose:
                    print(f"  ✗ ζ = {zeta:.4e} failed")

                results.append(
                    {
                        "cr_rate": float(zeta),
                        "success": False,
                        "elapsed_time": elapsed / len(zeta_values),
                        "error": "Batch solver failed",
                    }
                )

        return results

    except Exception as e:
        elapsed = time.perf_counter() - start_time

        if verbose:
            print(f"  ✗ Batch failed after {elapsed:.2f}s")
            print(f"  Error: {e}")

        import traceback

        traceback.print_exc()

        # Return failure for all zeta values
        return [
            {
                "cr_rate": float(zeta),
                "success": False,
                "elapsed_time": elapsed / len(zeta_values),
                "error": str(e),
            }
            for zeta in zeta_values
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Run cosmic ray ionization rate sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        default="results_cr",
        help="Output directory (default: results_cr)",
    )
    parser.add_argument(
        "--n-zetas",
        type=int,
        default=36,
        help="Number of zeta values (default: 36)",
    )
    parser.add_argument(
        "--zeta-min",
        type=float,
        default=-2,
        help="Log10 of minimum zeta (default: -2 → 0.01 s^-1)",
    )
    parser.add_argument(
        "--zeta-max",
        type=float,
        default=4,
        help="Log10 of maximum zeta (default: 4 → 10000 s^-1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each run",
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    network_file = project_root / "data" / "uclchem_gas_phase_only.csv"
    initial_conditions_file = (
        project_root
        / "benchmarks"
        / "initial_conditions"
        / "gas_phase_only_initial.yaml"
    )
    output_base = Path(__file__).parent / args.output

    # Check inputs exist
    if not network_file.exists():
        print(f"ERROR: Network file not found: {network_file}")
        sys.exit(1)

    if not initial_conditions_file.exists():
        print(f"ERROR: Initial conditions not found: {initial_conditions_file}")
        print("\nGenerate with:")
        print("  cd benchmarks")
        print("  python run_uclchem.py --network gas_phase_only")
        print("  python extract_uclchem_initial.py --network gas_phase_only")
        sys.exit(1)

    # Generate zeta values
    zeta_values = np.logspace(args.zeta_min, args.zeta_max, args.n_zetas)

    print(f"\n{'=' * 70}")
    print("Cosmic Ray Ionization Rate Sensitivity Analysis")
    print(f"{'=' * 70}")
    print("Network: gas_phase_only")
    print(f"Zeta range: 10^{args.zeta_min} to 10^{args.zeta_max} s^-1")
    print(f"Number of values: {args.n_zetas}")
    print(f"Output directory: {output_base}")
    print("\nPhysical conditions:")
    print(f"  Density: {PHYSICAL_PARAMS['number_density']:.2e} cm^-3")
    print(f"  Temperature: {PHYSICAL_PARAMS['temperature']:.1f} K")
    print(f"  Final time: {PHYSICAL_PARAMS['t_end']:.2e} years")
    print(f"  Solver: {PHYSICAL_PARAMS['solver']}")

    # Run sensitivity analysis
    results = []
    successful = 0
    failed = 0

    print(f"\nRunning {args.n_zetas} simulations...")

    # Use batch execution for better performance
    print("Using batch execution...")
    results = run_batch_zeta(
        zeta_values,
        network_file,
        initial_conditions_file,
        output_base,
        verbose=args.verbose,
    )

    # Count successes/failures
    for result in results:
        if result["success"]:
            successful += 1
        else:
            failed += 1

    # Save summary
    summary_file = output_base / "sensitivity_summary.json"
    summary = {
        "total_runs": args.n_zetas,
        "successful": successful,
        "failed": failed,
        "zeta_min": float(args.zeta_min),
        "zeta_max": float(args.zeta_max),
        "results": results,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Create summary DataFrame
    df = pd.DataFrame(results)
    summary_csv = output_base / "sensitivity_summary.csv"
    df.to_csv(summary_csv, index=False)

    print(f"\n{'=' * 70}")
    print("Sensitivity Analysis Complete")
    print(f"{'=' * 70}")
    print(f"Successful: {successful}/{args.n_zetas}")
    print(f"Failed: {failed}/{args.n_zetas}")
    print("\nSaved summary:")
    print(f"  {summary_file}")
    print(f"  {summary_csv}")

    if successful > 0:
        print(f"\nResults saved in: {output_base}/")
        sys.exit(0)
    else:
        print("\nERROR: All simulations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
