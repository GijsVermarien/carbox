"""
Publication-quality comparison plots for Carbox vs UCLCHEM.

Creates journal-ready figures comparing key species abundances.
Uses scienceplots for consistent styling.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import scienceplots for publication-quality styling
import scienceplots  # noqa: F401

plt.style.use('science')
print("Using scienceplots styling")


def format_species_name(species):
    """
    Format species name with proper subscripts and superscripts for display.
    Converts H2 -> H$_2$, CH3OH -> CH$_3$OH, C+ -> C$^+$, etc.
    """
    import re
    # First handle charges (+ or -) -> superscript
    formatted = re.sub(r'\+', r'$^+$', species)
    formatted = re.sub(r'\-$', r'$^-$', formatted)
    # Then handle digits -> subscript
    formatted = re.sub(r'(\d+)', r'$_{\1}$', formatted)
    return formatted


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
    
    # Load UCLCHEM abundances and physics
    uclchem_abund_file = results_path / "uclchem" / f"{network_name}_abundances.csv"
    uclchem_physics_file = results_path / "uclchem" / f"{network_name}_physics.csv"
    
    if not uclchem_abund_file.exists():
        raise FileNotFoundError(f"UCLCHEM abundances not found: {uclchem_abund_file}")
    if not uclchem_physics_file.exists():
        raise FileNotFoundError(f"UCLCHEM physics not found: {uclchem_physics_file}")
    
    uclchem_abund = pd.read_csv(uclchem_abund_file)
    uclchem_physics = pd.read_csv(uclchem_physics_file)
    
    # Merge time from physics into abundances
    uclchem_df = pd.concat([uclchem_physics[['Time']], uclchem_abund], axis=1)
    uclchem_df = uclchem_df.rename(columns={'Time': 'time'})
    
    # Load Carbox
    carbox_file = results_path / "carbox" / f"{network_name}_abundances.csv"
    if not carbox_file.exists():
        raise FileNotFoundError(f"Carbox results not found: {carbox_file}")
    carbox_df = pd.read_csv(carbox_file)
    
    return uclchem_df, carbox_df


def plot_species_comparison(
    uclchem_df: pd.DataFrame,
    carbox_df: pd.DataFrame,
    species_left: List[str],
    species_right: List[str],
    output_file: str,
    title: str = None
):
    """
    Create two-panel comparison plot for publication.
    
    Parameters
    ----------
    uclchem_df : DataFrame
        UCLCHEM results
    carbox_df : DataFrame
        Carbox results
    species_left : list
        Species for left panel
    species_right : list
        Species for right panel
    output_file : str
        Output file path
    title : str, optional
        Figure title
    """
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Color palette for different species (one color per species)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Left panel: Simple species
    for i, species in enumerate(species_left):
        if species in uclchem_df.columns and species in carbox_df.columns:
            color = colors[i]
            # UCLCHEM - solid lines
            ax1.loglog(
                uclchem_df['time'],
                uclchem_df[species],
                color=color,
                linestyle='-',
                linewidth=2.5,
                alpha=0.9,
                label=species if i == 0 else None  # Only first for legend
            )
            # Carbox - dashed lines
            ax1.loglog(
                carbox_df['time_years'],
                carbox_df[species],
                color=color,
                linestyle='--',
                linewidth=2.5,
                alpha=0.9
            )
    
    # Add custom legend entries for linestyles (only once per panel)
    from matplotlib.lines import Line2D
    
    # Species legend entries (colors only)
    species_handles = [
        Line2D([0], [0], color=colors[i], linewidth=2.5,
               label=format_species_name(species))
        for i, species in enumerate(species_left)
        if species in uclchem_df.columns and species in carbox_df.columns
    ]
    
    # Framework legend entries (linestyles only, gray color)
    framework_handles = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, 
               label='UCLCHEM'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5,
               label='Carbox')
    ]
    
    # Combine legends with separator
    all_handles = species_handles + [Line2D([0], [0], color='none')] + framework_handles
    
    ax1.set_xlabel('Time (years)', fontsize=13)
    ax1.set_ylabel('Fractional Abundance $x_i$', fontsize=13)
    ax1.set_title('Atomic network', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, which='both', linestyle=':')
    ax1.legend(handles=all_handles, fontsize=10, loc='best', 
               frameon=True, framealpha=0.95, ncol=1)
    ax1.set_xlim(1e-9, 1e7)
    
    # Right panel: Complex species
    for i, species in enumerate(species_right):
        if species in uclchem_df.columns and species in carbox_df.columns:
            color = colors[i]
            # UCLCHEM - solid lines
            ax2.loglog(
                uclchem_df['time'],
                uclchem_df[species],
                color=color,
                linestyle='-',
                linewidth=2.5,
                alpha=0.9
            )
            # Carbox - dashed lines
            ax2.loglog(
                carbox_df['time_years'],
                carbox_df[species],
                color=color,
                linestyle='--',
                linewidth=2.5,
                alpha=0.9
            )
    
    # Species legend entries for right panel
    species_handles_right = [
        Line2D([0], [0], color=colors[i], linewidth=2.5,
               label=format_species_name(species))
        for i, species in enumerate(species_right)
        if species in uclchem_df.columns and species in carbox_df.columns
    ]
    
    # Use same framework handles
    all_handles_right = species_handles_right + [Line2D([0], [0], color='none')] + framework_handles
    
    ax2.set_xlabel('Time (years)', fontsize=13)
    ax2.set_ylabel('Fractional Abundance $x_i$', fontsize=13)
    ax2.set_title('Molecular network', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, which='both', linestyle=':')
    ax2.legend(handles=all_handles_right, fontsize=10, loc='best',
               frameon=True, framealpha=0.95, ncol=1)
    ax2.set_xlim(1e-9, 1e7)
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_publication_comparison(
    results_dir: str = "results",
    network_left: str = "small_chemistry",
    network_right: str = "gas_phase_only",
    output_dir: str = None
):
    """
    Generate publication-quality comparison plots.
    
    Parameters
    ----------
    results_dir : str
        Directory containing benchmark results
    network_left : str
        Network name for left panel
    network_right : str
        Network name for right panel
    output_dir : str, optional
        Output directory (default: results/comparisons)
    """
    if output_dir is None:
        output_dir = str(Path(results_dir) / "comparisons")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Creating publication plots")
    print(f"  Left panel: {network_left}")
    print(f"  Right panel: {network_right}")
    print(f"{'='*70}")
    
    # Load data for left panel (simple species)
    print(f"\nLoading {network_left} results...")
    uclchem_left, carbox_left = load_results(results_dir, network_left)
    print(f"  UCLCHEM timesteps: {len(uclchem_left)}")
    print(f"  Carbox timesteps: {len(carbox_left)}")
    
    # Load data for right panel (complex species)
    print(f"\nLoading {network_right} results...")
    uclchem_right, carbox_right = load_results(results_dir, network_right)
    print(f"  UCLCHEM timesteps: {len(uclchem_right)}")
    print(f"  Carbox timesteps: {len(carbox_right)}")
    
    # Define species for each panel
    # Left panel: Simple species from small_chemistry
    species_left = ['H2', 'H', 'C', 'C+', 'CO', "E-"]
    
    # Right panel: Complex molecules from gas_phase_only
    species_right = ['H2CO', 'CH3OH', 'HCO', 'HCO+', 'CH3', 'CH4']
    
    # Check availability in right panel
    metadata_cols = {
        'time_seconds', 'time_years', 'number_density', 
        'temperature', 'cr_rate', 'fuv_field', 'visual_extinction'
    }
    carbox_species_right = set(carbox_right.columns) - metadata_cols
    uclchem_species_right = set(uclchem_right.columns) - {'time'}
    common_species_right = carbox_species_right & uclchem_species_right
    
    # Filter to available species
    species_right = [sp for sp in species_right if sp in common_species_right]
    
    print(f"\nLeft panel species: {species_left}")
    print(f"Right panel species: {species_right}")
    
    # Create plot with two different datasets
    print("\nGenerating publication figure...")
    output_file = output_path / "combined_publication_comparison.png"
    output_pdf = output_path / "combined_publication_comparison.pdf"
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    from matplotlib.lines import Line2D
    
    # LEFT PANEL: Simple species
    for i, species in enumerate(species_left):
        if species in uclchem_left.columns and species in carbox_left.columns:
            color = colors[i]
            ax1.loglog(uclchem_left['time'], uclchem_left[species],
                      color=color, linestyle='-', linewidth=2.5, alpha=0.9)
            ax1.loglog(carbox_left['time_years'], carbox_left[species],
                      color=color, linestyle='--', linewidth=2.5, alpha=0.9)
    
    species_handles_left = [
        Line2D([0], [0], color=colors[i], linewidth=2.5,
               label=format_species_name(species))
        for i, species in enumerate(species_left)
        if species in uclchem_left.columns and species in carbox_left.columns
    ]
    
    framework_handles = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5,
               label='UCLCHEM'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5,
               label='Carbox')
    ]
    
    all_handles_left = (species_handles_left + 
                       [Line2D([0], [0], color='none')] + 
                       framework_handles)
    
    ax1.set_xlabel('Time (years)', fontsize=13)
    ax1.set_ylabel('Fractional Abundance $x_i$', fontsize=13)
    ax1.set_title('Atomic network', fontsize=14, pad=10)
    ax1.grid(True, alpha=0.3, which='both', linestyle=':')
    ax1.legend(handles=all_handles_left, fontsize=10, loc='best',
              frameon=True, framealpha=0.95, ncol=1)
    ax1.set_xlim(1e-9, 1e7)
    
    # RIGHT PANEL: Complex species
    for i, species in enumerate(species_right):
        if species in uclchem_right.columns and species in carbox_right.columns:
            color = colors[i]
            ax2.loglog(uclchem_right['time'], uclchem_right[species],
                      color=color, linestyle='-', linewidth=2.5, alpha=0.9)
            ax2.loglog(carbox_right['time_years'], carbox_right[species],
                      color=color, linestyle='--', linewidth=2.5, alpha=0.9)
    
    species_handles_right = [
        Line2D([0], [0], color=colors[i], linewidth=2.5,
               label=format_species_name(species))
        for i, species in enumerate(species_right)
        if species in uclchem_right.columns and species in carbox_right.columns
    ]
    
    all_handles_right = (species_handles_right + 
                        [Line2D([0], [0], color='none')] + 
                        framework_handles)
    
    ax2.set_xlabel('Time (years)', fontsize=13)
    ax2.set_ylabel('Fractional Abundance $x_i$', fontsize=13)
    ax2.set_title('Molecular network', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, which='both', linestyle=':')
    ax2.legend(handles=all_handles_right, fontsize=10, loc='best',
              frameon=True, framealpha=0.95, ncol=1)
    ax2.set_xlim(1e-9, 1e7)
    
    # fig.suptitle('Carbox vs UCLCHEM: Abundance Comparison', 
    #              fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=300, bbox_inches='tight')
    fig.savefig(str(output_pdf), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"Publication plot saved to:")
    print(f"  {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create publication-quality comparison plots'
    )
    parser.add_argument(
        '--results',
        default='results',
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--left',
        default='small_chemistry',
        help='Network for left panel (default: small_chemistry)'
    )
    parser.add_argument(
        '--right',
        default='gas_phase_only',
        help='Network for right panel (default: gas_phase_only)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory (default: results/comparisons)'
    )
    
    args = parser.parse_args()
    
    plot_publication_comparison(
        results_dir=args.results,
        network_left=args.left,
        network_right=args.right,
        output_dir=args.output
    )
