"""
Carbox: JAX-accelerated chemical kinetics simulation framework.

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
from typing import Optional

import jax
import jax.numpy as jnp
import diffrax as dx
from tqdm import tqdm

# JAX configuration for numerical stability
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)  # CRITICAL: Disable for performance

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
    initialize_abundance_output,
    write_abundance_snapshot,
)
from .parsers import parse_chemical_network
from .solver import compute_derivatives, compute_reaction_rates, solve_network, create_step_solver, get_time_grid


def run_simulation(
    network_file: str,
    config: SimulationConfig,
    format_type: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run a chemical kinetics simulation.

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

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'solution': Diffrax solution object
        - 'network': Reaction network
        - 'config': Configuration used
        - 'computation_time': Wall-clock time [s]

    Examples
    --------
    >>> config = SimulationConfig(number_density=1e4, t_end=1e5)
    >>> results = run_simulation('data/network.csv', config)
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
        print(f"Solving ODE system with {config.solver} (Streaming Mode)...")
        print(f"  Time range: {config.t_start:.2e} - {config.t_end:.2e} years")
        print(f"  Snapshots: {config.n_snapshots}")
        print("  Compiling step solver...")

    solve_start = datetime.now()
    
    # --- Streaming Implementation ---
    
    # 1. Prepare time grid and output file
    t_snapshots = get_time_grid(config)
    output_file = initialize_abundance_output(network, config)
    
    # 2. Create JIT-compiled stepper
    step_solver = create_step_solver(jnetwork, config)
    
    # 3. Initialize state
    y_current = y0
    t_current = t_snapshots[0]
    physics = config.physics_model
    solver_state = None
    controller_state = None
    
    # Accumulators for full solution (needed for derivatives/rates post-processing)
    ts_list = [t_current]
    ys_list = [y_current]
    total_steps = 0
    total_accepted = 0
    total_rejected = 0

    # 4. Write initial condition (t=0)
    write_abundance_snapshot(output_file, float(t_current), y_current, network, config)
    
    # 5. Loop over snapshots
    # We iterate from 0 to N-2 to go from t[i] to t[i+1]
    pbar = tqdm(range(len(t_snapshots) - 1), disable=not verbose, desc="Integration")
    
    if verbose:
        print("  Starting integration loop (first step triggers JIT compilation)...")

    for i in pbar:
        t_next = t_snapshots[i+1]
        
        # Solve step
        sol = step_solver(t_current, t_next, y_current, physics, solver_state, controller_state)
        y_next = sol.ys[0]
        stats = sol.stats
        solver_state = sol.solver_state
        controller_state = sol.controller_state
        
        # Write output immediately
        write_abundance_snapshot(output_file, float(t_next), y_next, network, config)
        
        # Update state
        y_current = y_next
        t_current = t_next
        
        # Accumulate
        ts_list.append(t_next)
        ys_list.append(y_next)
        total_steps += int(stats["num_steps"])
        total_accepted += int(stats["num_accepted_steps"])
        total_rejected += int(stats["num_rejected_steps"])

        # Update progress bar with current radius
        if physics is not None:
            _, _, _, r_curr = physics.get_conditions(t_next)
            pbar.set_postfix(r=f"{float(r_curr):.2e} cm")

    solve_time = (datetime.now() - solve_start).total_seconds()

    if verbose:
        print(f"  Integration complete in {solve_time:.2f} seconds")

    # Step 5: Save results
    if verbose:
        print("Saving results...")

    computation_time = (datetime.now() - start_time).total_seconds()

    # Reconstruct a Diffrax Solution object for compatibility with existing tools
    # Note: We stack the lists into JAX arrays
    solution = dx.Solution(
        ts=jnp.stack(ts_list),
        ys=jnp.stack(ys_list),
        stats={
            "num_steps": total_steps,
            "num_accepted_steps": total_accepted,
            "num_rejected_steps": total_rejected
        },
        t0=t_snapshots[0],
        t1=t_snapshots[-1],
        interpolation=None, # We don't have dense interpolation here
        result=jnp.array(0), # Success
        solver_state=solver_state,
        controller_state=controller_state,
        made_jump=None,
        event_mask=None,
    )


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
        rates = compute_reaction_rates(jnetwork, solution, config)
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
