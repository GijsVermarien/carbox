"""
Compare abundance outputs from UCLCHEM and Carbox.

Generates plots and statistics for benchmarking.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "runners"))
from common import format_time, load_config


def load_results(
    results_dir: str,
    network_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load UCLCHEM and Carbox results.
    
    Returns
    -------
    uclchem_df : DataFrame
        UCLCHEM abundances with 'time' column in years
    carbox_df : DataFrame
        Carbox abundances with 'time_years' column
    """
    results_path = Path(results_dir)
    
    # Load UCLCHEM
    uclchem_file = results_path / "uclchem" / f"{network_name}_abundances.csv"
    if not uclchem_file.exists():
        raise FileNotFoundError(f"UCLCHEM results not found: {uclchem_file}")
    uclchem_df = pd.read_csv(uclchem_file)
    
    # Load Carbox
    carbox_file = results_path / "carbox" / f"{network_name}_abundances.csv"
    if not carbox_file.exists():
        raise FileNotFoundError(f"Carbox results not found: {carbox_file}")
    carbox_df = pd.read_csv(carbox_file)
    
    return uclchem_df, carbox_df


def get_common_species(
    uclchem_df: pd.DataFrame,
    carbox_df: pd.DataFrame
) -> List[str]:
    """
    Find species present in both dataframes.
    
    Parameters
    ----------
    uclchem_df : DataFrame
        UCLCHEM results (species in columns)
    carbox_df : DataFrame
        Carbox results (species in columns)
    
    Returns
    -------
    list
        Common species names (excluding time/parameter columns)
    """
    # Carbox metadata columns
    metadata_cols = {'time_seconds', 'time_years', 'number_density', 'temperature', 
                     'cr_rate', 'fuv_field', 'visual_extinction'}
    
    # UCLCHEM time column
    uclchem_species = set(uclchem_df.columns) - {'time'}
    carbox_species = set(carbox_df.columns) - metadata_cols
    
    common = sorted(uclchem_species & carbox_species)
    
    return common


def interpolate_to_common_times(
    uclchem_df: pd.DataFrame,
    carbox_df: pd.DataFrame,
    species: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate both datasets to common logarithmic time grid.
    
    Returns
    -------
    times : array
        Common time grid (years)
    uclchem_interp : array
        UCLCHEM abundances, shape (n_times, n_species)
    carbox_interp : array
        Carbox abundances, shape (n_times, n_species)
    """
    # Get time ranges
    t_min = max(uclchem_df['time'].min(), carbox_df['time_years'].min())
    t_max = min(uclchem_df['time'].max(), carbox_df['time_years'].max())
    
    # Create log-spaced grid
    times = np.logspace(np.log10(t_min), np.log10(t_max), 200)
    
    # Interpolate each species
    uclchem_interp = np.zeros((len(times), len(species)))
    carbox_interp = np.zeros((len(times), len(species)))
    
    for i, sp in enumerate(species):
        # UCLCHEM (linear interpolation in log-log space)
        uclchem_interp[:, i] = np.interp(
            np.log10(times),
            np.log10(uclchem_df['time']),
            np.log10(np.maximum(uclchem_df[sp], 1e-50))
        )
        
        # Carbox
        carbox_interp[:, i] = np.interp(
            np.log10(times),
            np.log10(carbox_df['time_years']),
            np.log10(np.maximum(carbox_df[sp], 1e-50))
        )
    
    # Convert back from log space
    uclchem_interp = 10 ** uclchem_interp
    carbox_interp = 10 ** carbox_interp
    
    return times, uclchem_interp, carbox_interp


def compute_differences(
    uclchem_abund: np.ndarray,
    carbox_abund: np.ndarray,
    threshold: float = 1e-15
) -> np.ndarray:
    """
    Compute relative differences.
    
    Parameters
    ----------
    uclchem_abund : array
        UCLCHEM abundances
    carbox_abund : array
        Carbox abundances
    threshold : float
        Minimum abundance for comparison
    
    Returns
    -------
    rel_diff : array
        Relative differences where both > threshold, nan elsewhere
    """
    # Mask where both are above threshold
    mask = (uclchem_abund > threshold) & (carbox_abund > threshold)
    
    # Compute relative difference
    rel_diff = np.full_like(uclchem_abund, np.nan)
    rel_diff[mask] = np.abs(carbox_abund[mask] - uclchem_abund[mask]) / uclchem_abund[mask]
    
    return rel_diff


def plot_comparison(
    times: np.ndarray,
    uclchem_abund: np.ndarray,
    carbox_abund: np.ndarray,
    species: List[str],
    output_file: str,
    title: str = "Abundance Comparison"
):
    """
    Create multi-panel comparison plot.
    
    Parameters
    ----------
    times : array
        Time grid (years)
    uclchem_abund : array
        UCLCHEM abundances, shape (n_times, n_species)
    carbox_abund : array
        Carbox abundances, shape (n_times, n_species)
    species : list
        Species names
    output_file : str
        Output PNG path
    title : str
        Plot title
    """
    # Select top species by final abundance
    final_abund = carbox_abund[-1, :]
    top_indices = np.argsort(final_abund)[-9:][::-1]  # Top 9
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    for i, idx in enumerate(top_indices):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        sp = species[idx]
        
        # Plot both
        ax.loglog(times, uclchem_abund[:, idx], 'b-', label='UCLCHEM', linewidth=2)
        ax.loglog(times, carbox_abund[:, idx], 'r--', label='Carbox', linewidth=2)
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(f'Abundance')
        ax.set_title(sp, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set y-limits
        y_min = min(uclchem_abund[:, idx].min(), carbox_abund[:, idx].min())
        y_max = max(uclchem_abund[:, idx].max(), carbox_abund[:, idx].max())
        ax.set_ylim(y_min * 0.5, y_max * 2)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_relative_differences(
    times: np.ndarray,
    rel_diff: np.ndarray,
    species: List[str],
    output_file: str,
    title: str = "Relative Differences"
):
    """
    Plot relative differences over time.
    
    Parameters
    ----------
    times : array
        Time grid (years)
    rel_diff : array
        Relative differences, shape (n_times, n_species)
    species : list
        Species names
    output_file : str
        Output PNG path
    title : str
        Plot title
    """
    # Compute max difference for each species
    max_diff = np.nanmax(rel_diff, axis=0)
    top_indices = np.argsort(max_diff)[-9:][::-1]  # Top 9 worst
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    for i, idx in enumerate(top_indices):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        sp = species[idx]
        
        # Plot relative difference
        valid = ~np.isnan(rel_diff[:, idx])
        if valid.any():
            ax.semilogx(times[valid], rel_diff[valid, idx] * 100, 'k-', linewidth=2)
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Relative Difference (%)')
        ax.set_title(sp, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color='orange', linestyle='--', alpha=0.5, label='1%')
        ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='10%')
        ax.legend(fontsize=8)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def generate_statistics_report(
    times: np.ndarray,
    uclchem_abund: np.ndarray,
    carbox_abund: np.ndarray,
    species: List[str],
    output_file: str
):
    """
    Generate text report with comparison statistics.
    """
    # Compute differences
    rel_diff = compute_differences(uclchem_abund, carbox_abund)
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ABUNDANCE COMPARISON STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 40 + "\n")
        
        valid_diffs = rel_diff[~np.isnan(rel_diff)]
        if len(valid_diffs) > 0:
            f.write(f"  Mean relative difference: {np.mean(valid_diffs)*100:.2f}%\n")
            f.write(f"  Median relative difference: {np.median(valid_diffs)*100:.2f}%\n")
            f.write(f"  Max relative difference: {np.max(valid_diffs)*100:.2f}%\n")
            f.write(f"  95th percentile: {np.percentile(valid_diffs, 95)*100:.2f}%\n")
        
        f.write(f"  Total comparisons: {len(valid_diffs)}\n")
        f.write(f"  Species compared: {len(species)}\n\n")
        
        # Per-species statistics
        f.write("Per-Species Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Species':<12} {'Mean %':<10} {'Max %':<10} {'Final UCLCHEM':<15} {'Final Carbox':<15}\n")
        f.write("-" * 70 + "\n")
        
        for i, sp in enumerate(species):
            sp_diffs = rel_diff[:, i]
            valid = ~np.isnan(sp_diffs)
            
            if valid.any():
                mean_diff = np.mean(sp_diffs[valid]) * 100
                max_diff = np.max(sp_diffs[valid]) * 100
            else:
                mean_diff = np.nan
                max_diff = np.nan
            
            final_ucl = uclchem_abund[-1, i]
            final_cbx = carbox_abund[-1, i]
            
            f.write(f"{sp:<12} {mean_diff:>9.2f} {max_diff:>9.2f} {final_ucl:>14.3e} {final_cbx:>14.3e}\n")
        
        f.write("\n" + "="*70 + "\n")


def compare_abundances(
    results_dir: str,
    network_name: str,
    output_dir: str = None,
    verbose: bool = True
) -> Dict:
    """
    Main comparison function.
    
    Parameters
    ----------
    results_dir : str
        Directory with UCLCHEM and Carbox results
    network_name : str
        Network name for file loading
    output_dir : str
        Output directory for comparison plots (default: results_dir/comparisons)
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Comparison statistics
    """
    if output_dir is None:
        output_dir = str(Path(results_dir) / "comparisons")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Comparing abundances: {network_name}")
        print(f"{'='*70}")
    
    # Load results
    if verbose:
        print("Loading results...")
    uclchem_df, carbox_df = load_results(results_dir, network_name)
    
    # Find common species
    species = get_common_species(uclchem_df, carbox_df)
    if verbose:
        print(f"  UCLCHEM timesteps: {len(uclchem_df)}")
        print(f"  Carbox timesteps: {len(carbox_df)}")
        print(f"  Common species: {len(species)}")
    
    # Interpolate to common grid
    if verbose:
        print("Interpolating to common time grid...")
    times, uclchem_interp, carbox_interp = interpolate_to_common_times(
        uclchem_df, carbox_df, species
    )
    
    # Generate plots
    if verbose:
        print("Generating comparison plots...")
    
    plot_comparison(
        times, uclchem_interp, carbox_interp, species,
        output_path / f"{network_name}_comparison.png",
        title=f"Abundance Comparison: {network_name}"
    )
    
    rel_diff = compute_differences(uclchem_interp, carbox_interp)
    plot_relative_differences(
        times, rel_diff, species,
        output_path / f"{network_name}_differences.png",
        title=f"Relative Differences: {network_name}"
    )
    
    # Generate statistics report
    if verbose:
        print("Generating statistics report...")
    stats_file = output_path / f"{network_name}_statistics.txt"
    generate_statistics_report(
        times, uclchem_interp, carbox_interp, species, stats_file
    )
    
    if verbose:
        print(f"\nOutputs saved to:")
        print(f"  {output_path / f'{network_name}_comparison.png'}")
        print(f"  {output_path / f'{network_name}_differences.png'}")
        print(f"  {stats_file}")
    
    # Return summary statistics
    valid_diffs = rel_diff[~np.isnan(rel_diff)]
    return {
        'n_species': len(species),
        'n_timesteps': len(times),
        'mean_rel_diff': float(np.mean(valid_diffs)) if len(valid_diffs) > 0 else np.nan,
        'max_rel_diff': float(np.max(valid_diffs)) if len(valid_diffs) > 0 else np.nan,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare UCLCHEM and Carbox abundances')
    parser.add_argument('--results', default='../results',
                       help='Results directory')
    parser.add_argument('--network', required=True,
                       help='Network name')
    parser.add_argument('--output', default=None,
                       help='Output directory (default: results/comparisons)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    stats = compare_abundances(
        args.results,
        args.network,
        args.output,
        verbose=not args.quiet
    )
    
    print(f"\n{'='*70}")
    print(f"Comparison complete")
    print(f"  Mean difference: {stats['mean_rel_diff']*100:.2f}%")
    print(f"  Max difference: {stats['max_rel_diff']*100:.2f}%")
