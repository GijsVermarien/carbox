#!/usr/bin/env python3
"""
Compare UCLCHEM and Carbox results.

Generates plots and statistics for benchmarking.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def load_results(network_name: str, results_dir: str = 'results'):
    """Load UCLCHEM and Carbox results."""
    results_path = Path(results_dir)
    
    # Load UCLCHEM (abundances + physics files)
    uclchem_abund_file = results_path / "uclchem" / f"{network_name}_abundances.csv"
    uclchem_phys_file = results_path / "uclchem" / f"{network_name}_physics.csv"
    
    if not uclchem_abund_file.exists():
        raise FileNotFoundError(f"UCLCHEM abundances not found: {uclchem_abund_file}")
    if not uclchem_phys_file.exists():
        raise FileNotFoundError(f"UCLCHEM physics not found: {uclchem_phys_file}")
    
    uclchem_abund = pd.read_csv(uclchem_abund_file)
    uclchem_phys = pd.read_csv(uclchem_phys_file)
    
    # Merge on index and add time column (convert years to years)
    uclchem_df = uclchem_abund.copy()
    uclchem_df['time'] = uclchem_phys['Time'].values  # UCLCHEM time is in years
    
    # Add ice species totals (# = surface, @ = bulk/mantle)
    # For each gas-phase species, sum its ice components if they exist
    gas_species = [col for col in uclchem_df.columns 
                   if not col.startswith('#') and not col.startswith('@') 
                   and col not in ['time', 'BULK', 'SURFACE', 'E-']]
    
    # In case we want to plot ice species:
    # for species in gas_species:
    #     surface_col = f'#{species}'
    #     bulk_col = f'@{species}'
    #     ice_total = np.zeros(len(uclchem_df))
        
    #     if surface_col in uclchem_df.columns:
    #         ice_total += uclchem_df[surface_col].values
    #     if bulk_col in uclchem_df.columns:
    #         ice_total += uclchem_df[bulk_col].values
        
    #     # Only add if there's any ice
    #     if ice_total.sum() > 0:
    #         uclchem_df[f'{species}_ice'] = ice_total
    
    # Load Carbox
    carbox_file = results_path / "carbox" / f"{network_name}_abundances.csv"
    if not carbox_file.exists():
        raise FileNotFoundError(f"Carbox results not found: {carbox_file}")
    carbox_df = pd.read_csv(carbox_file)
    
    return uclchem_df, carbox_df


def get_common_species(uclchem_df, carbox_df):
    """Find gas-phase species present in both dataframes."""
    metadata_cols = {'time_seconds', 'time_years', 'number_density', 'temperature',
                     'cr_rate', 'fuv_field', 'visual_extinction'}

    # Get gas-phase species only (exclude ice species with _ice suffix for comparison)
    uclchem_species = set(col for col in uclchem_df.columns
                         if col != 'time' and not col.endswith('_ice'))
    carbox_species = set(carbox_df.columns) - metadata_cols

    return sorted(uclchem_species & carbox_species)


def plot_comparison(uclchem_df, carbox_df, species, output_file, title):
    """Create multi-panel comparison plot."""
    # Select top species by final abundance from Carbox
    final_abund = np.array([carbox_df[sp].iloc[-1] for sp in species])
    top_indices = np.argsort(final_abund)[-9:][::-1]  # Top 9

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    for i, idx in enumerate(top_indices):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        sp = species[idx]

        ax.loglog(uclchem_df['time'], uclchem_df[sp], 'b-',
                  label='UCLCHEM (gas)', linewidth=2)
        ax.loglog(carbox_df['time_years'], carbox_df[sp], 'r--',
                  label='Carbox', linewidth=3)

        # Plot ice if available
        ice_name = f'{sp}_ice'
        if ice_name in uclchem_df.columns:
            ax.loglog(uclchem_df['time'], uclchem_df[ice_name], 'c:',
                      label='UCLCHEM (ice)', linewidth=2, alpha=0.7)

        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Abundance')
        ax.set_title(sp, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

        # Set y-limits
        y_min = min(uclchem_df[sp].min(), carbox_df[sp].min())
        y_max = max(uclchem_df[sp].max(), carbox_df[sp].max())
        if ice_name in uclchem_df.columns:
            y_min = min(y_min, uclchem_df[ice_name].min())
            y_max = max(y_max, uclchem_df[ice_name].max())
        ax.set_ylim(max(y_min, 1e-40) * 0.5, min(y_max * 2, 1))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def compare_results(network_name: str, results_dir: str = 'results'):
    """Main comparison function."""
    print(f"\n{'='*70}")
    print(f"Comparing results: {network_name}")
    print(f"{'='*70}")
    
    # Load results
    print("Loading results...")
    uclchem_df, carbox_df = load_results(network_name, results_dir)
    
    # Select species to plot
    species = ["H", "H2", "H+", "E-", "C", "CO", "HCO", "HCO+", "H2CO", "CH3OH", "C+", "O+"]
    species = [
        s for s in species
        if s in uclchem_df.columns and s in carbox_df.columns
    ]
    
    print(f"  UCLCHEM timesteps: {len(uclchem_df)}")
    print(f"  Carbox timesteps: {len(carbox_df)}")
    print(f"  Species to plot: {len(species)}")
    
    # Generate outputs
    output_path = Path(results_dir) / "comparisons"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating comparison plot...")
    plot_comparison(
        uclchem_df, carbox_df, species,
        output_path / f"{network_name}_comparison.png",
        title=f"Abundance Comparison: {network_name}"
    )
    
    # Load benchmark results for performance report
    import json
    uclchem_bench_file = (
        Path(results_dir) / "uclchem" / f"{network_name}_benchmark.json"
    )
    carbox_bench_file = (
        Path(results_dir) / "carbox" / f"{network_name}_benchmark.json"
    )
    uclchem_bench = json.load(open(uclchem_bench_file))
    carbox_bench = json.load(open(carbox_bench_file))
    
    speedup = uclchem_bench['time'] / carbox_bench['time']
    
    perf_file = output_path / f"{network_name}_performance.md"
    with open(perf_file, 'w') as f:
        f.write(f"# Performance Comparison: {network_name}\n\n")
        f.write("## Summary\n\n")
        f.write("| Metric | UCLCHEM | Carbox | Ratio |\n")
        f.write("|--------|---------|--------|-------|\n")
        ucl_time = uclchem_bench['time']
        cbx_time = carbox_bench['time']
        f.write(
            f"| **Total Time** | {ucl_time:.2f}s | "
            f"{cbx_time:.2f}s | {speedup:.2f}× |\n"
        )
        ucl_steps = uclchem_bench['n_timesteps']
        cbx_steps = carbox_bench['n_timesteps']
        f.write(f"| **Timesteps** | {ucl_steps} | {cbx_steps} | - |\n")
        ucl_species = uclchem_bench['n_species']
        cbx_species = carbox_bench['n_species']
        f.write(f"| **Species** | {ucl_species} | {cbx_species} | - |\n")
        ode_steps = carbox_bench['n_ode_steps']
        f.write(f"| **ODE Steps** | - | {ode_steps} | - |\n")
    
    print("\nOutputs saved to:")
    print(f"  {output_path / f'{network_name}_comparison.png'}")
    print(f"  {perf_file}")
    print(f"\nSpeedup: {speedup:.2f}×")


def main():
    parser = argparse.ArgumentParser(
        description='Compare UCLCHEM and Carbox results'
    )
    parser.add_argument(
        '--network', required=True, help='Network name'
    )
    parser.add_argument(
        '--results', default='results', help='Results directory'
    )
    
    args = parser.parse_args()
    
    compare_results(args.network, args.results)
    
    print(f"\n{'='*70}")
    print("Comparison complete")


if __name__ == '__main__':
    main()
