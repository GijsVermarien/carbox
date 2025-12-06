"""Output management for simulation results.

Handles saving of abundance trajectories, derivatives, rates, and metadata.
"""

import json
from datetime import datetime
from pathlib import Path

import diffrax as dx
import jax.numpy as jnp
import pandas as pd

from .config import SimulationConfig
from .network import Network
from .solver import SPY


def prepare_output_directory(config: SimulationConfig) -> Path:
    """Create output directory if it doesn't exist.

    Parameters
    ----------
    config : SimulationConfig
        Configuration with output_dir

    Returns:
    -------
    output_path : Path
        Path to output directory
    """
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_abundances(
    solution: dx.Solution,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save abundance time series to CSV.

    Parameters
    ----------
    solution : dx.Solution
        Integration solution
    network : Network
        Reaction network (for species names)
    config : SimulationConfig
        Configuration

    Returns:
    -------
    filepath : Path
        Path to saved file

    Notes:
    -----
    Output format:
    - Columns: time, physical parameters, then species abundances
    - Values: fractional abundances relative to H nuclei (x_i = n_i / n_{H,nuclei})
    - n_{H,nuclei} = 2*n(H2) + n(H)
    - Physical parameters repeated for each row (for easy filtering/grouping)
    """
    output_path = prepare_output_directory(config)

    species_names = [s.name for s in network.species]

    # Calculate hydrogen nuclei density for each timestep
    # n_{H,nuclei} = 2*n(H2) + n(H)
    h2_idx = None
    h_idx = None
    for i, name in enumerate(species_names):
        if name == "H2":
            h2_idx = i
        elif name == "H":
            h_idx = i
    """
    if h2_idx is not None and h_idx is not None:
        n_h_nuclei = 2 * solution.ys[:, h2_idx] + solution.ys[:, h_idx]
    elif h2_idx is not None:
        n_h_nuclei = 2 * solution.ys[:, h2_idx]
    elif h_idx is not None:
        n_h_nuclei = solution.ys[:, h_idx]
    else:
        # Fallback to total density if no H or H2
    """
    n_h_nuclei = config.number_density

    # Create DataFrame with time and physical parameter columns
    df = pd.DataFrame(
        {
            "time_seconds": solution.ts,
            "time_years": solution.ts / SPY,  # type:ignore
            "number_density": config.number_density,
            "temperature": config.temperature,
            "cr_rate": config.cr_rate,
            "fuv_field": config.fuv_field,
            "visual_extinction": config.compute_visual_extinction(),
        }
    )

    # Add species fractional abundances (relative to H nuclei)
    for i, name in enumerate(species_names):
        df[name] = solution.ys[:, i] / n_h_nuclei  # type:ignore

    filepath = output_path / f"{config.run_name}_abundances.csv"
    df.to_csv(filepath, index=False)

    print(f"Saved abundances to: {filepath}")
    return filepath


def save_derivatives(
    derivatives: jnp.ndarray,
    times: jnp.ndarray,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save time derivatives to CSV.

    Parameters
    ----------
    derivatives : jnp.ndarray
        Time derivatives [n_snapshots, n_species]
    times : jnp.ndarray
        Time array [s]
    network : Network
        Reaction network
    config : SimulationConfig
        Configuration

    Returns:
    -------
    filepath : Path
        Path to saved file
    """
    output_path = prepare_output_directory(config)

    species_names = [s.name for s in network.species]

    # Create DataFrame with time and physical parameter columns
    df = pd.DataFrame(
        {
            "time_seconds": times,
            "time_years": times / SPY,
            "number_density": config.number_density,
            "temperature": config.temperature,
            "cr_rate": config.cr_rate,
            "fuv_field": config.fuv_field,
            "visual_extinction": config.compute_visual_extinction(),
        }
    )

    # Add derivatives
    for i, name in enumerate(species_names):
        df[f"d{name}_dt"] = derivatives[:, i]

    filepath = output_path / f"{config.run_name}_derivatives.csv"
    df.to_csv(filepath, index=False)

    print(f"Saved derivatives to: {filepath}")
    return filepath


def save_reaction_rates(
    rates: jnp.ndarray,
    times: jnp.ndarray,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save reaction rates to CSV.

    Parameters
    ----------
    rates : jnp.ndarray
        Reaction rates [n_snapshots, n_reactions]
    times : jnp.ndarray
        Time array [s]
    network : Network
        Reaction network
    config : SimulationConfig
        Configuration

    Returns:
    -------
    filepath : Path
        Path to saved file
    """
    output_path = prepare_output_directory(config)

    # Use reaction type as column names (could be more descriptive)
    reaction_names = [f"{r.reaction_type}_{i}" for i, r in enumerate(network.reactions)]

    # Create DataFrame with time and physical parameter columns
    df = pd.DataFrame(
        {
            "time_seconds": times,
            "time_years": times / SPY,
            "number_density": config.number_density,
            "temperature": config.temperature,
            "cr_rate": config.cr_rate,
            "fuv_field": config.fuv_field,
            "visual_extinction": config.visual_extinction,
        }
    )

    # Add reaction rates
    for i, name in enumerate(reaction_names):
        df[name] = rates[:, i]

    filepath = output_path / f"{config.run_name}_rates.csv"
    df.to_csv(filepath, index=False)

    print(f"Saved reaction rates to: {filepath}")
    return filepath


def save_metadata(
    config: SimulationConfig,
    network: Network,
    solution: dx.Solution,
    computation_time: float | None = None,
) -> Path:
    """Save simulation metadata to JSON.

    Parameters
    ----------
    config : SimulationConfig
        Configuration used
    network : Network
        Reaction network
    solution : dx.Solution
        Integration solution (for stats)
    computation_time : float, optional
        Wall-clock time [s]

    Returns:
    -------
    filepath : Path
        Path to saved file

    Notes:
    -----
    Metadata includes:
    - Configuration parameters
    - Network statistics (# species, # reactions)
    - Solver statistics
    - Timestamp and computation time
    """
    output_path = prepare_output_directory(config)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "run_name": config.run_name,
        "computation_time_seconds": computation_time,
        # Configuration
        "config": {
            "physical_params": {
                "number_density": config.number_density,
                "temperature": config.temperature,
                "cr_rate": config.cr_rate,
                "fuv_field": config.fuv_field,
                "visual_extinction": config.compute_visual_extinction(),
                "visual_extinction_config": config.visual_extinction,
                "use_self_consistent_av": config.use_self_consistent_av,
                "cloud_radius_pc": config.cloud_radius_pc,
                "base_av": config.base_av,
            },
            "integration": {
                "t_start": config.t_start,
                "t_end": config.t_end,
                "n_snapshots": config.n_snapshots,
                "solver": config.solver,
                "atol": config.atol,
                "rtol": config.rtol,
                "max_steps": config.max_steps,
            },
            "initial_abundances": config.initial_abundances,
        },
        # Network info
        "network": {
            "n_species": len(network.species),
            "n_reactions": len(network.reactions),
            "species_names": [s.name for s in network.species],
            "use_sparse": network.use_sparse,
            "vectorize_reactions": network.vectorize_reactions,
        },
        # Solver statistics
        "solver_stats": {
            "num_steps": int(solution.stats["num_steps"]),
            "num_accepted_steps": int(solution.stats["num_accepted_steps"]),
            "num_rejected_steps": int(solution.stats["num_rejected_steps"]),
        }
        if hasattr(solution, "stats")
        else {},
    }

    filepath = output_path / f"{config.run_name}_metadata.json"
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {filepath}")
    return filepath


def save_summary_report(
    solution: dx.Solution,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save human-readable summary report.

    Parameters
    ----------
    solution : dx.Solution
        Integration solution
    network : Network
        Reaction network
    config : SimulationConfig
        Configuration

    Returns:
    -------
    filepath : Path
        Path to saved file
    """
    output_path = prepare_output_directory(config)

    species_names = [s.name for s in network.species]

    lines = []
    lines.append("=" * 60)
    lines.append(f"Carbox Simulation Summary: {config.run_name}")
    lines.append("=" * 60)
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("Physical Parameters:")
    lines.append(f"  Total density: {config.number_density:.2e} cm^-3")
    lines.append(f"  Temperature: {config.temperature:.1f} K")
    lines.append(f"  CR ionization rate: {config.cr_rate:.2e} s^-1")
    lines.append(f"  FUV field: {config.fuv_field:.2e} Draine")
    lines.append(f"  Visual extinction: {config.visual_extinction:.1f} mag")
    lines.append("")

    lines.append("Integration:")
    lines.append(f"  Time range: {config.t_start:.2e} - {config.t_end:.2e} years")
    lines.append(f"  Snapshots: {config.n_snapshots}")
    lines.append(f"  Solver: {config.solver}")
    lines.append(f"  Tolerances: atol={config.atol:.2e}, rtol={config.rtol:.2e}")
    lines.append("")

    lines.append("Network:")
    lines.append(f"  Species: {len(network.species)}")
    lines.append(f"  Reactions: {len(network.reactions)}")
    lines.append("")

    if hasattr(solution, "stats"):
        lines.append("Solver Statistics:")
        lines.append(f"  Total steps: {solution.stats['num_steps']}")
        lines.append(f"  Accepted: {solution.stats['num_accepted_steps']}")
        lines.append(f"  Rejected: {solution.stats['num_rejected_steps']}")
        lines.append("")

    # Final abundances (top 10)
    lines.append("Final Abundances (top 10):")
    final_abundances = solution.ys[-1]  # type:ignore
    sorted_indices = jnp.argsort(final_abundances)[::-1]
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        lines.append(f"  {species_names[idx]:<10} {final_abundances[idx]:.3e} cm^-3")

    lines.append("=" * 60)

    report = "\n".join(lines)

    filepath = output_path / f"{config.run_name}_summary.txt"
    with open(filepath, "w") as f:
        f.write(report)

    print(f"Saved summary to: {filepath}")
    return filepath
