#!/usr/bin/env python3
"""
Extract initial conditions from UCLCHEM output and save to YAML for Carbox.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def extract_initial_conditions(network_name: str, results_dir: str = 'results', 
                              gas_phase_only: bool = True):
    """Extract ALL initial abundances from UCLCHEM output."""
    results_path = Path(results_dir)
    
    # Load UCLCHEM abundances
    abund_file = results_path / "uclchem" / f"{network_name}_abundances.csv"
    if not abund_file.exists():
        raise FileNotFoundError(f"UCLCHEM results not found: {abund_file}")
    
    df = pd.read_csv(abund_file)
    initial_row = df.iloc[0]
    
    initial_abundances = {}
    
    for species, abundance in initial_row.items():
        # Skip time column and grain species
        if species in ['time', 'Time', 'TIME']:
            continue
            
        # Skip grain species (# = surface, @ = bulk/mantle), special columns
        if gas_phase_only:
            if species.startswith('#') or species.startswith('@') or species in ['BULK', 'SURFACE']:
                continue
        
        # Include ALL gas-phase species (even E- electrons) if they are large enough.
        if abundance > 1e-30:
            initial_abundances[species] = float(abundance)
    
    # Sort by abundance (descending)
    initial_abundances = dict(sorted(
        initial_abundances.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    return initial_abundances


def save_to_yaml(initial_abundances: dict, network_name: str, output_dir: str = 'initial_conditions'):
    """Save initial conditions to YAML file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    yaml_file = output_path / f"{network_name}_initial.yaml"
    
    # Create structured output
    output = {
        'network': network_name,
        'description': f'Initial abundances from UCLCHEM for {network_name}',
        'abundances': initial_abundances,
        'metadata': {
            'n_species': len(initial_abundances),
            'total_abundance': sum(initial_abundances.values()),
            'units': 'fractional (relative to H nuclei density)'
        }
    }
    
    with open(yaml_file, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    
    return yaml_file


def print_summary(initial_abundances: dict, network_name: str):
    """Print summary of initial conditions."""
    print(f"\n{'='*70}")
    print(f"Extracted {len(initial_abundances)} gas-phase species from UCLCHEM")
    print(f"{'='*70}")
    
    # Count non-zero abundances
    non_zero = sum(1 for x in initial_abundances.values() if x > 1e-50)
    print(f"Non-zero abundances: {non_zero}/{len(initial_abundances)}")
    print(f"Total fractional abundance: {sum(initial_abundances.values()):.6e}")
    
    print(f"\nTop 10 species:")
    for i, (species, abundance) in enumerate(initial_abundances.items()):
        if i >= 10:
            break
        print(f"  {species:<10} {abundance:>12.6e}")


def main():
    parser = argparse.ArgumentParser(description='Extract UCLCHEM initial conditions')
    parser.add_argument('--network', required=True, help='Network name')
    parser.add_argument('--results', default='results', help='Results directory')
    parser.add_argument('--output', default='initial_conditions', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Extract initial conditions
        initial_abundances = extract_initial_conditions(args.network, args.results)
        
        # Save to YAML
        yaml_file = save_to_yaml(initial_abundances, args.network, args.output)
        
        # Print summary
        print_summary(initial_abundances, args.network)
        
        print(f"\n{'='*70}")
        print(f"✓ Initial conditions saved to: {yaml_file}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
