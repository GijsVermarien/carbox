"""Compare performance metrics from UCLCHEM and Carbox.

Generates timing and efficiency reports.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "runners"))
from common import format_time


def load_benchmark_results(results_dir: str, network_name: str) -> dict[str, dict]:
    """Load benchmark JSON files.

    Returns:
    -------
    dict
        Dictionary with 'uclchem' and 'carbox' keys containing benchmark results
    """
    results_path = Path(results_dir)

    # Load UCLCHEM results
    uclchem_file = results_path / "uclchem" / f"{network_name}_benchmark.json"
    if uclchem_file.exists():
        with open(uclchem_file) as f:
            uclchem_results = json.load(f)
    else:
        uclchem_results = None

    # Load Carbox results
    carbox_file = results_path / "carbox" / f"{network_name}_benchmark.json"
    if carbox_file.exists():
        with open(carbox_file) as f:
            carbox_results = json.load(f)
    else:
        carbox_results = None

    return {"uclchem": uclchem_results, "carbox": carbox_results}


def generate_performance_report(
    results_dir: str, network_name: str, output_file: str = None, verbose: bool = True
) -> dict:
    """Generate performance comparison report.

    Parameters
    ----------
    results_dir : str
        Directory with benchmark results
    network_name : str
        Network name
    output_file : str
        Output markdown file (default: comparisons/network_performance.md)
    verbose : bool
        Print progress

    Returns:
    -------
    dict
        Performance statistics
    """
    if output_file is None:
        output_path = Path(results_dir) / "comparisons"
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{network_name}_performance.md"

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Generating performance report: {network_name}")
        print(f"{'=' * 70}")

    # Load results
    results = load_benchmark_results(results_dir, network_name)

    uclchem = results["uclchem"]
    carbox = results["carbox"]

    if uclchem is None or carbox is None:
        print("ERROR: Missing benchmark results")
        return {}

    # Check success
    if not uclchem.get("success", False):
        print(f"WARNING: UCLCHEM failed - {uclchem.get('error', 'unknown error')}")
    if not carbox.get("success", False):
        print(f"WARNING: Carbox failed - {carbox.get('error', 'unknown error')}")

    # Compute statistics
    speedup = uclchem["time"] / carbox["time"] if carbox["time"] > 0 else np.nan

    # Generate markdown report
    with open(output_file, "w") as f:
        f.write(f"# Performance Comparison: {network_name}\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | UCLCHEM | Carbox | Ratio |\n")
        f.write("|--------|---------|--------|-------|\n")
        f.write(
            f"| **Total Time** | {format_time(uclchem['time'])} | {format_time(carbox['time'])} | {speedup:.2f}× |\n"
        )
        f.write(
            f"| **Timesteps** | {uclchem.get('n_timesteps', 'N/A')} | {carbox.get('n_timesteps', 'N/A')} | - |\n"
        )
        f.write(
            f"| **Species** | {uclchem.get('n_species', 'N/A')} | {carbox.get('n_species', 'N/A')} | - |\n"
        )

        if "n_reactions" in carbox:
            f.write(f"| **Reactions** | - | {carbox['n_reactions']} | - |\n")

        f.write("\n")

        # UCLCHEM details
        f.write("## UCLCHEM Details\n\n")
        f.write(f"- **Runtime**: {format_time(uclchem['time'])}\n")
        f.write(f"- **Return flag**: {uclchem.get('flag', 'N/A')}\n")
        f.write(f"- **Timesteps**: {uclchem.get('n_timesteps', 'N/A')}\n")
        f.write(f"- **Final time**: {uclchem.get('final_time', 'N/A'):.2e} years\n")
        f.write(f"- **Success**: {uclchem.get('success', False)}\n")

        if not uclchem.get("success", False):
            f.write(f"- **Error**: {uclchem.get('error', 'Unknown')}\n")

        f.write("\n")

        # Carbox details
        f.write("## Carbox Details\n\n")
        f.write(f"- **Runtime**: {format_time(carbox['time'])}\n")
        f.write(f"- **Timesteps**: {carbox.get('n_timesteps', 'N/A')}\n")
        f.write(f"- **Final time**: {carbox.get('final_time', 'N/A'):.2e} years\n")
        f.write(f"- **ODE steps**: {carbox.get('n_ode_steps', 'N/A')}\n")
        f.write(f"- **Accepted**: {carbox.get('n_accepted', 'N/A')}\n")
        f.write(f"- **Rejected**: {carbox.get('n_rejected', 'N/A')}\n")

        if "n_rejected" in carbox and "n_accepted" in carbox:
            total = carbox["n_accepted"] + carbox["n_rejected"]
            reject_rate = carbox["n_rejected"] / total * 100 if total > 0 else 0
            f.write(f"- **Rejection rate**: {reject_rate:.1f}%\n")

        f.write(f"- **Success**: {carbox.get('success', False)}\n")

        if not carbox.get("success", False):
            f.write(f"- **Error**: {carbox.get('error', 'Unknown')}\n")

        f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")

        if speedup > 1:
            f.write(
                f"Carbox is **{speedup:.2f}× faster** than UCLCHEM for this network.\n\n"
            )
        elif speedup < 1:
            f.write(
                f"UCLCHEM is **{1 / speedup:.2f}× faster** than Carbox for this network.\n\n"
            )
        else:
            f.write("Both codes have similar performance.\n\n")

        # Time per step
        if "n_ode_steps" in carbox and carbox["n_ode_steps"] > 0:
            time_per_step = carbox["time"] / carbox["n_ode_steps"] * 1000  # ms
            f.write(f"Carbox averaged **{time_per_step:.3f} ms per ODE step**.\n\n")

        # Efficiency notes
        f.write("### Notes\n\n")
        f.write("- UCLCHEM uses DVODE (variable-order Adams/BDF) solver\n")
        f.write("- Carbox uses Kvaerno5 (5th-order SDIRK) with JAX JIT compilation\n")
        f.write("- Different timestep strategies may affect comparison\n")
        f.write("- Carbox benefits from GPU acceleration (if available)\n")

    if verbose:
        print(f"\nReport saved to: {output_file}")
        print(f"\nSpeedup: {speedup:.2f}×")

    return {
        "speedup": float(speedup),
        "uclchem_time": uclchem["time"],
        "carbox_time": carbox["time"],
    }


def generate_multi_network_comparison(
    results_dir: str, network_names: list, output_file: str = None, verbose: bool = True
):
    """Generate comparison table across multiple networks.

    Parameters
    ----------
    results_dir : str
        Directory with benchmark results
    network_names : list
        List of network names to compare
    output_file : str
        Output markdown file (default: comparisons/multi_network.md)
    verbose : bool
        Print progress
    """
    if output_file is None:
        output_path = Path(results_dir) / "comparisons"
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "multi_network_comparison.md"

    if verbose:
        print(f"\n{'=' * 70}")
        print("Generating multi-network comparison")
        print(f"{'=' * 70}")

    # Collect results
    rows = []
    for network in network_names:
        results = load_benchmark_results(results_dir, network)
        uclchem = results["uclchem"]
        carbox = results["carbox"]

        if uclchem and carbox:
            speedup = (
                uclchem["time"] / carbox["time"] if carbox["time"] > 0 else float("nan")
            )
            rows.append(
                {
                    "network": network,
                    "uclchem_time": uclchem["time"],
                    "carbox_time": carbox["time"],
                    "speedup": speedup,
                    "n_species": carbox.get("n_species", "-"),
                    "n_reactions": carbox.get("n_reactions", "-"),
                }
            )

    # Generate markdown
    with open(output_file, "w") as f:
        f.write("# Multi-Network Performance Comparison\n\n")

        f.write("## Summary Table\n\n")
        f.write(
            "| Network | Species | Reactions | UCLCHEM Time | Carbox Time | Speedup |\n"
        )
        f.write(
            "|---------|---------|-----------|--------------|-------------|----------|\n"
        )

        for row in rows:
            f.write(
                f"| {row['network']} | {row['n_species']} | {row['n_reactions']} | "
                f"{format_time(row['uclchem_time'])} | {format_time(row['carbox_time'])} | "
                f"{row['speedup']:.2f}× |\n"
            )

        f.write("\n")

        # Statistics
        if rows:
            speedups = [r["speedup"] for r in rows if not pd.isna(r["speedup"])]
            f.write("## Overall Statistics\n\n")
            f.write(f"- **Mean speedup**: {sum(speedups) / len(speedups):.2f}×\n")
            f.write(f"- **Min speedup**: {min(speedups):.2f}×\n")
            f.write(f"- **Max speedup**: {max(speedups):.2f}×\n")

    if verbose:
        print(f"\nMulti-network report saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare performance metrics")
    parser.add_argument("--results", default="../results", help="Results directory")
    parser.add_argument(
        "--network", required=True, help="Network name (or comma-separated list)"
    )
    parser.add_argument("--output", default=None, help="Output file")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    # Handle multiple networks
    networks = [n.strip() for n in args.network.split(",")]

    if len(networks) == 1:
        stats = generate_performance_report(
            args.results, networks[0], args.output, verbose=not args.quiet
        )
        print(f"\n{'=' * 70}")
        print("Performance analysis complete")
        print(f"  Speedup: {stats.get('speedup', 'N/A'):.2f}×")
    else:
        generate_multi_network_comparison(
            args.results, networks, args.output, verbose=not args.quiet
        )
        print(f"\n{'=' * 70}")
        print("Multi-network comparison complete")
