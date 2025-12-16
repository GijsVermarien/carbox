#!/usr/bin/env python3
"""
Plot Cosmic Ray Ionization Rate Sensitivity Analysis Results

Visualizes the effect of varying cosmic ray ionization rate (ζ) on
the evolution of key species.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

# Use scienceplots for publication-quality plots

plt.style.use("science")
print("Using scienceplots styling")


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,  # Use matplotlib's mathtext, not system LaTeX
    }
)


def load_results(results_dir: Path):
    """Load sensitivity analysis results."""
    summary_file = results_dir / "sensitivity_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file, "r") as f:
        summary = json.load(f)

    # Extract successful runs
    zeta_values = []
    abundances = {}

    for result in summary["results"]:
        if not result["success"]:
            zeta_val = result["cr_rate"]
            print(f"Warning: Skipping failed run for ζ = {zeta_val}")
            continue

        zeta = result["cr_rate"]
        zeta_dir = results_dir / f"zeta_{zeta:.4e}"
        abund_file = zeta_dir / "abundances.csv"

        if not abund_file.exists():
            print(f"Warning: Abundance file not found: {abund_file}")
            continue

        df = pd.read_csv(abund_file)
        zeta_values.append(zeta)
        abundances[zeta] = df

    if len(zeta_values) == 0:
        raise ValueError("No successful runs found")

    # Sort by zeta
    zeta_values.sort()

    return zeta_values, abundances


def format_species_name(species):
    """
    Format species name with proper subscripts and superscripts for display.
    Converts H2 -> H$_2$, CH3OH -> CH$_3$OH, C+ -> C$^+$, etc.
    """
    import re

    # First handle charges (+ or -) -> superscript
    formatted = re.sub(r"\+", r"$^+$", species)
    formatted = re.sub(r"\-$", r"$^-$", formatted)
    # Then handle digits -> subscript
    formatted = re.sub(r"(\d+)", r"$_{\1}$", formatted)
    return formatted


def plot_species_evolution(zeta_values, abundances, species_list, output_file):
    """
    Plot species evolution (left) and final abundances (right).
    Left panel: Time evolution with gradient envelopes.
    Right panel: Final abundances vs ζ/ζ₀.
    """
    print(f"\nPlotting {len(species_list)} species")

    # Check which species exist in the data
    first_df = abundances[zeta_values[0]]
    available_species = [sp for sp in species_list if sp in first_df.columns]
    missing_species = [sp for sp in species_list if sp not in first_df.columns]

    if missing_species:
        print(f"  Warning: Species not found in data: {missing_species}")

    if not available_species:
        raise ValueError("None of the requested species found in data")

    print(f"  Plotting species: {available_species}")

    # Create two-panel figure matching publication format
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Color map for species
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Find zeta=1.0 or closest value for central line (this is zeta_0)
    zeta_0 = min(zeta_values, key=lambda x: abs(x - 1.0))
    print(f"  Using ζ₀ = {zeta_0:.2e} s⁻¹ for central lines")

    # Calculate zeta ratios
    zeta_ratios = [z / zeta_0 for z in zeta_values]

    # LEFT PANEL: Time evolution with gradients
    for idx, species in enumerate(available_species):
        # Get time array
        time = abundances[zeta_0]["time_years"].values

        # Collect abundances for all zeta values
        all_abundances = np.zeros((len(zeta_values), len(time)))

        for i, zeta in enumerate(zeta_values):
            df = abundances[zeta]
            all_abundances[i, :] = df[species].values

        # Compute statistics
        central_abundance = abundances[zeta_0][species].values
        min_abundance = np.min(all_abundances, axis=0)
        max_abundance = np.max(all_abundances, axis=0)

        # Plot central line
        ax1.loglog(
            time,
            central_abundance,
            color=colors[idx],
            linewidth=2.5,
            label=format_species_name(species),
            zorder=10 + idx,
        )

        # Create gradient effect using percentile bands
        n_bands = 10
        percentiles = np.linspace(0, 100, n_bands + 1)

        for band_idx in range(n_bands):
            lower_perc = percentiles[band_idx]
            upper_perc = percentiles[band_idx + 1]

            lower_band = np.percentile(all_abundances, lower_perc, axis=0)
            upper_band = np.percentile(all_abundances, upper_perc, axis=0)

            # Stronger alpha gradient
            alpha_value = 0.08 + (band_idx / n_bands) * 0.27

            ax1.fill_between(
                time,
                lower_band,
                upper_band,
                color=colors[idx],
                alpha=alpha_value,
                linewidth=0,
                zorder=idx + 0.01 * band_idx,
            )

        # Add thin edge lines to outermost boundaries
        ax1.loglog(
            time,
            min_abundance,
            color=colors[idx],
            linewidth=0.5,
            alpha=0.6,
            zorder=idx + 0.1,
        )
        ax1.loglog(
            time,
            max_abundance,
            color=colors[idx],
            linewidth=0.5,
            alpha=0.6,
            zorder=idx + 0.1,
        )

    # Format left panel
    ax1.set_xlabel("Time (years)", fontsize=13)
    ax1.set_ylabel("Fractional Abundance $x_i$", fontsize=13)
    ax1.set_xlim(1.0, 1.0e7)
    ax1.set_ylim(1.0e-20, 1.0)
    ax1.grid(True, alpha=0.3, which="both", linestyle=":")
    ax1.legend(
        loc="best",
        framealpha=0.95,
        fontsize=10,
        frameon=True,
    )

    # RIGHT PANEL: Final abundances vs zeta/zeta_0
    final_abundances = {sp: [] for sp in available_species}

    for zeta in zeta_values:
        df = abundances[zeta]
        for species in available_species:
            final_abundances[species].append(df[species].iloc[-1])

    for idx, species in enumerate(available_species):
        ax2.loglog(
            zeta_ratios,
            final_abundances[species],
            marker="o",
            markersize=5,
            linewidth=2,
            label=format_species_name(species),
            color=colors[idx],
            alpha=0.8,
        )

    # Format right panel
    ax2.set_xlabel("$\\zeta/\\zeta_0$", fontsize=13)
    ax2.set_ylabel("Final Fractional Abundance $x_i$", fontsize=13)
    ax2.grid(True, alpha=0.3, which="both", linestyle=":")
    ax2.legend(loc="best", framealpha=0.95, fontsize=10, frameon=True)
    ax2.set_ylim(1e-30, 1e0)

    # Add reference line at zeta/zeta_0 = 1
    ax2.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(
        1.0,
        ax2.get_ylim()[0] * 1.5,
        "$\\zeta_0$",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_file)
    plt.savefig(output_file.with_suffix(".pdf"))
    print(f"\n✓ Saved: {output_file}")
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CR ionization rate sensitivity analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--results",
        default="results_cr",
        help="Results directory (default: results_cr)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots_cr",
        help="Output directory for plots (default: plots_cr)",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["H", "H2", "H2CO", "CH3OH", "HCO+"],
        help="Species to plot",
    )

    args = parser.parse_args()

    # Setup paths
    results_dir = Path(__file__).parent / args.results
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print("\nRun sensitivity analysis first:")
        print("  bash run_cr_sensitivity.sh")
        sys.exit(1)

    print("=" * 70)
    print("Cosmic Ray Rate Sensitivity Analysis - Plotting")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Species: {args.species}")

    # Load results
    zeta_values, abundances = load_results(results_dir)

    # Generate combined plot (evolution + final abundances)
    print("\n" + "=" * 70)
    print("Creating combined sensitivity plot...")
    print("=" * 70)
    plot_species_evolution(
        zeta_values,
        abundances,
        args.species,
        output_dir / "cr_sensitivity_combined.png",
    )

    print("\n" + "=" * 70)
    print("✓ Plot created successfully!")
    print("=" * 70)
    print(f"\nPlot saved in: {output_dir}/")
    print("  - cr_sensitivity_combined.png")
    print("  - cr_sensitivity_combined.pdf")


if __name__ == "__main__":
    main()
