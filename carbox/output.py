"""
Output management for simulation results.

Handles saving of abundance trajectories, derivatives, rates, and metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import diffrax as dx
import jax
import jax.numpy as jnp
import pandas as pd
import csv

from .config import SimulationConfig
from .network import JNetwork, Network
from .solver import SPY


def prepare_output_directory(config: SimulationConfig) -> Path:
    """
    Create output directory if it doesn't exist.

    Parameters
    ----------
    config : SimulationConfig
        Configuration with output_dir

    Returns
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
    """
    Save abundance time series to CSV.

    Parameters
    ----------
    solution : dx.Solution
        Integration solution
    network : Network
        Reaction network (for species names)
    config : SimulationConfig
        Configuration

    Returns
    -------
    filepath : Path
        Path to saved file

    Notes
    -----
    Output format:
    - Columns: time, physical parameters, then species abundances
    - Values: fractional abundances relative to H nuclei (x_i = n_i / n_{H,nuclei})
    - n_{H,nuclei} = 2*n(H2) + n(H)
    - Physical parameters repeated for each row (for easy filtering/grouping)
    """
    output_path = prepare_output_directory(config)

    species_names = [s.name for s in network.species]

    # Physical conditions along the trajectory (n, T, av, r)
    physics = config.physics_model
    get_cond_vec = jax.vmap(physics.get_conditions)
    densities, temperatures, avs, radii = get_cond_vec(solution.ts)
    n_h_nuclei_arr = densities  # divisor for fractional output

    # 1. Start with the base physics data
    data = {
        "time_seconds": solution.ts,
        "time_years": solution.ts / SPY,
        "radius_cm": radii,
        "number_density": densities,
        "temperature": temperatures,
        "cr_rate": config.cr_rate,
        "fuv_field": config.fuv_field,
        "visual_extinction": avs,
    }

    # 2. Add all species to the dictionary
    for i, name in enumerate(species_names):
        data[name] = solution.ys[:, i] / n_h_nuclei_arr

    # 3. Create the DataFrame in one go
    df = pd.DataFrame(data)

    filepath = output_path / f"{config.run_name}_abundances.csv"
    df.to_csv(filepath, index=False)

    print(f"Saved abundances to: {filepath}")
    return filepath


def initialize_abundance_output(network: Network, config: SimulationConfig) -> Path:
    """Initialize the abundance CSV file with headers."""
    output_path = prepare_output_directory(config)
    filepath = output_path / f"{config.run_name}_abundances.csv"
    
    species_names = [s.name for s in network.species]
    header = [
        "time_seconds", "time_years", "radius_cm", "number_density", 
        "temperature", "cr_rate", "fuv_field", "visual_extinction"
    ] + species_names
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
    print(f"Initialized streaming output: {filepath}")
    return filepath


def write_abundance_snapshot(
    filepath: Path, 
    t_sec: float, 
    y: jnp.ndarray, 
    network: Network, 
    config: SimulationConfig
):
    """Append a single snapshot to the abundance CSV."""
    # Calculate physics for this timestamp
    n, T, av, r = config.physics_model.get_conditions(t_sec)
    n_h_nuclei = n

    # Convert to fractional abundances
    # y is absolute density [cm^-3], output is fractional relative to H nuclei
    frac_abundances = y / n_h_nuclei
    
    # Prepare row
    row = [
        float(t_sec),
        float(t_sec / SPY),
        float(r),
        float(n),
        float(T),
        float(config.cr_rate),
        float(config.fuv_field),
        float(av)
    ] + [float(val) for val in frac_abundances]
    
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_derivatives(
    derivatives: jnp.ndarray,
    times: jnp.ndarray,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """
    Save time derivatives to CSV.

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

    Returns
    -------
    filepath : Path
        Path to saved file
    """
    output_path = prepare_output_directory(config)

    species_names = [s.name for s in network.species]

    # Physical conditions along the trajectory
    get_cond_vec = jax.vmap(config.physics_model.get_conditions)
    n_dyn, T_dyn, av_dyn, r_dyn = get_cond_vec(times)

    # 1. Base physics data
    data = {
        "time_seconds": times,
        "time_years": times / SPY,
        "radius_cm": r_dyn,
        "number_density": n_dyn,
        "temperature": T_dyn,
        "cr_rate": config.cr_rate,
        "fuv_field": config.fuv_field,
        "visual_extinction": av_dyn,
    }

    # 2. Add all derivatives to dictionary
    for i, name in enumerate(species_names):
        data[f"d{name}_dt"] = derivatives[:, i]

    # 3. Build DataFrame
    df = pd.DataFrame(data)

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
    """
    Save reaction rates to CSV.

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

    Returns
    -------
    filepath : Path
        Path to saved file
    """
    output_path = prepare_output_directory(config)

    # Reaction rates are saved with numerical column names (0, 1, 2, ...)

    # Physical conditions along the trajectory
    get_cond_vec = jax.vmap(config.physics_model.get_conditions)
    n_dyn, T_dyn, av_dyn, r_dyn = get_cond_vec(times)

    # 1. Create the base data dictionary
    data = {
        "time_seconds": times,
        "time_years": times / SPY,
        "radius_cm": r_dyn,
        "number_density": n_dyn,
        "temperature": T_dyn,
        "cr_rate": config.cr_rate,
        "fuv_field": config.fuv_field,
        "visual_extinction": av_dyn,
    }

    # 2. Add all reaction rates to the dictionary first
    # This avoids the "fragmentation" warning completely
    for i in range(rates.shape[1]):
            data[str(i)] = rates[:, i]
            
    # 3. Create the DataFrame once
    df = pd.DataFrame(data)


    filepath = output_path / f"{config.run_name}_rates.csv"
    df.to_csv(filepath, index=False)

    print(f"Saved reaction rates to: {filepath}")
    return filepath


def save_metadata(
    config: SimulationConfig,
    network: Network,
    solution: dx.Solution,
    computation_time: Optional[float] = None,
) -> Path:
    """
    Save simulation metadata to JSON.

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

    Returns
    -------
    filepath : Path
        Path to saved file

    Notes
    -----
    Metadata includes:
    - Configuration parameters
    - Network statistics (# species, # reactions)
    - Solver statistics
    - Timestamp and computation time
    """
    output_path = prepare_output_directory(config)

    # Physical conditions at the start of the integration
    physics = config.physics_model
    n0, T0, av0, r0 = physics.get_conditions(config.t_start * SPY)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "run_name": config.run_name,
        "computation_time_seconds": computation_time,
        # Configuration
        "config": {
            "physical_params": {
                "physics_model": type(physics).__name__,
                "number_density_initial": float(n0),
                "temperature_initial": float(T0),
                "visual_extinction_initial": float(av0),
                "radius_initial_cm": float(r0),
                "cr_rate": config.cr_rate,
                "fuv_field": config.fuv_field,
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
    """
    Save human-readable summary report.

    Parameters
    ----------
    solution : dx.Solution
        Integration solution
    network : Network
        Reaction network
    config : SimulationConfig
        Configuration

    Returns
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

    physics = config.physics_model
    n0, T0, av0, _ = physics.get_conditions(config.t_start * SPY)
    lines.append(f"Physical Parameters ({type(physics).__name__}, at t_start):")
    lines.append(f"  Total density: {float(n0):.2e} cm^-3")
    lines.append(f"  Temperature: {float(T0):.1f} K")
    lines.append(f"  CR ionization rate: {config.cr_rate:.2e} s^-1")
    lines.append(f"  FUV field: {config.fuv_field:.2e} Draine")
    lines.append(f"  Visual extinction: {float(av0):.1f} mag")
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
    final_abundances = solution.ys[-1]
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