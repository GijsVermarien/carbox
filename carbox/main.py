"""Carbox: JAX-accelerated chemical kinetics simulation framework.

Main entry point for running astrochemical reaction network simulations.

Usage
-----
From Python:
    from carbox.main import run_simulation
    from carbox.config import SimulationConfig

    config = SimulationConfig(
        number_density=1e4,
        temperature=50.0,
        t_end=1e6,
    )
    run_simulation('data/network.csv', config, format_type='latent_tgas')

From command line:
    python -m carbox.main --input data/network.csv --config config.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import jax

from .config import SimulationConfig
from .initial_conditions import (
    abundance_summary,
    initialize_abundances,
    validate_elemental_conservation,
)
from .output import (
    save_abundances,
    save_derivatives,
    save_metadata,
    save_reaction_rates,
    save_summary_report,
)
from .parsers import parse_chemical_network
from .solver import compute_derivatives, compute_reaction_rates, solve_network

# JAX configuration for numerical stability
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def run_simulation(
    network_file: str,
    config: SimulationConfig,
    format_type: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run a chemical kinetics simulation.

    Workflow:
    1. Load network from file
    2. Initialize abundance vector
    3. Compile JAX network
    4. Solve ODE system
    5. Save results

    Parameters
    ----------
    network_file : str
        Path to reaction network file
    config : SimulationConfig
        Simulation configuration
    format_type : str, optional
        Network format ('uclchem', 'umist', 'latent_tgas')
        If None, auto-detect
    verbose : bool
        Print progress messages

    Returns:
    -------
    results : dict
        Dictionary containing:
        - 'solution': Diffrax solution object
        - 'network': Reaction network
        - 'config': Configuration used
        - 'computation_time': Wall-clock time [s]

    Examples:
    --------
    >>> config = SimulationConfig(number_density=1e4, t_end=1e5)
    >>> results = run_simulation("data/network.csv", config)
    """
    start_time = datetime.now()

    if verbose:
        print("=" * 60)
        print("Carbox Chemical Kinetics Simulation")
        print("=" * 60)
        print(f"Network file: {network_file}")
        print(f"Run name: {config.run_name}")
        print()

    # Validate configuration
    if verbose:
        print("Validating configuration...")
    config.validate()

    # Step 1: Load network
    if verbose:
        print(f"Loading reaction network from {network_file}...")
    network = parse_chemical_network(network_file, format_type)
    if verbose:
        print(f"  Loaded {len(network.species)} species")
        print(f"  Loaded {len(network.reactions)} reactions")
        print()

    # Step 2: Initialize abundances
    if verbose:
        print("Initializing abundances...")
    y0 = initialize_abundances(network, config)

    if verbose:
        print(abundance_summary(network, y0, top_n=8))
        print()

        # Check elemental conservation
        elem_abundances = validate_elemental_conservation(network, y0)
        print("Initial elemental abundances:")
        for elem, abundance in elem_abundances.items():
            if elem != "charge":
                print(f"  {elem}: {abundance:.3e} cm^-3")
        print(f"  Net charge: {elem_abundances['charge']:.3e}")
        print()

    # Step 3: Compile JAX network
    if verbose:
        print("Compiling JAX network...")
    jnetwork = network.get_ode()
    if verbose:
        print("  Network compiled successfully")
        print()

    # Step 4: Solve ODE
    if verbose:
        print(f"Solving ODE system with {config.solver}...")
        print(f"  Time range: {config.t_start:.2e} - {config.t_end:.2e} years")
        print(f"  Snapshots: {config.n_snapshots}")
        print("  Compiling solver (first call)...")

    solve_start = datetime.now()
    solution = solve_network(jnetwork, y0, config)
    solve_time = (datetime.now() - solve_start).total_seconds()

    if (
        not hasattr(solution, "ys")
        or not hasattr(solution, "ts")
        or solution.ys is None
        or solution.ts is None
    ):
        raise Exception("Missing solution.ys or solution.ts.")

    if verbose:
        print(f"  Integration complete in {solve_time:.2f} seconds")
        if hasattr(solution, "stats"):
            print(
                f"  Steps: {solution.stats['num_steps']} "
                + f"(accepted: {solution.stats['num_accepted_steps']}, "
                + f"rejected: {solution.stats['num_rejected_steps']})"
            )
        print()

    # Step 5: Save results
    if verbose:
        print("Saving results...")

    computation_time = (datetime.now() - start_time).total_seconds()

    # Always save abundances
    if config.save_abundances:
        save_abundances(solution, network, config)

    # Optional: derivatives
    if config.save_derivatives:
        if verbose:
            print("  Computing derivatives...")
        derivatives = compute_derivatives(jnetwork, solution, config)
        save_derivatives(derivatives, solution.ts, network, config)

    # Optional: reaction rates
    if config.save_rates:
        if verbose:
            print("  Computing reaction rates...")
        rates = compute_reaction_rates(network, jnetwork, solution, config)
        save_reaction_rates(rates, solution.ts, network, config)

    # Save metadata and summary
    save_metadata(config, network, solution, computation_time)
    save_summary_report(solution, network, config)

    if verbose:
        print()
        print("=" * 60)
        print(f"Simulation complete! Total time: {computation_time:.2f} seconds")
        print(f"Output saved to: {config.output_dir}/")
        print("=" * 60)

    return {
        "solution": solution,
        "network": network,
        "jnetwork": jnetwork,
        "config": config,
        "computation_time": computation_time,
    }


def main():
    """Command-line interface for Carbox."""
    parser = argparse.ArgumentParser(
        description="Carbox: JAX-accelerated chemical kinetics simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python -m carbox.main --input data/network.csv

  # Use configuration file
  python -m carbox.main --input data/network.csv --config my_config.yaml

  # Specify format explicitly
  python -m carbox.main --input data/network.csv --format umist

  # Custom output directory and run name
  python -m carbox.main --input data/network.csv --output results/ --name test_run
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Path to reaction network file"
    )
    parser.add_argument("--config", "-c", help="Path to YAML/JSON configuration file")
    parser.add_argument(
        "--format",
        "-f",
        choices=["uclchem", "umist", "latent_tgas", "auto"],
        default="auto",
        help="Network file format (default: auto-detect)",
    )
    parser.add_argument("--output", "-o", help="Output directory (overrides config)")
    parser.add_argument("--name", "-n", help="Run name (overrides config)")
    parser.add_argument(
        "--solver",
        choices=["dopri5", "kvaerno5", "tsit5"],
        help="ODE solver (overrides config)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output messages"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in [".yaml", ".yml"]:
            config = SimulationConfig.from_yaml(args.config)
        elif config_path.suffix == ".json":
            config = SimulationConfig.from_json(args.config)
        else:
            print(f"Error: Unknown config format: {config_path.suffix}")
            sys.exit(1)
    else:
        config = SimulationConfig()

    # Override with command-line args
    if args.output:
        config.output_dir = args.output
    if args.name:
        config.run_name = args.name
    if args.solver:
        config.solver = args.solver

    # Determine format
    format_type = None if args.format == "auto" else args.format

    # Run simulation
    try:
        run_simulation(
            args.input, config, format_type=format_type, verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
