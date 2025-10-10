#!/usr/bin/env python3
"""
Run Carbox benchmark for a specific network.

Simplified standalone runner with hardcoded configurations.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import yaml

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbox.config import SimulationConfig
from carbox.main import run_simulation

# Enable JAX 64-bit and NaN debugging
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


# Hardcoded physical parameters (matching UCLCHEM test case)
PHYSICAL_PARAMS = {
    'number_density': 1.0e4,                  # cm^-3
    'temperature': 250.0,            # K
    'cr_rate': 1.0,             # s^-1
    'fuv_field': 1.0,               # Habing units
    'visual_extinction': 2.9643750143703076,  # mag
    't_start': 0.0,                 # years
    't_end': 5.0e6,                 # years
    'n_snapshots': 500,             # output timesteps (increased for detail)
    'rtol': 1.0e-6,
    'atol': 1.0e-15,
    'solver': 'kvaerno5',  # lowercase required
    'max_steps': 50000,
}

# Species to track (filter output)
OUTPUT_SPECIES = [
    'H', 'H2', 'He', 'C', 'O', 'N',
    'CO', 'H2O', 'OH', 'CH', 'NH3', 'HCO+', 'H3+', 'e-'
]

# Network configurations
# Note: 'initial_conditions' is REQUIRED and must point to a valid YAML file
# containing fractional abundances extracted from UCLCHEM
NETWORK_CONFIGS = {
    'small_chemistry': {
        'description': 'Small gas-phase chemistry (~20 species)',
        'input_file': '../data/uclchem_small_chemistry.csv',
        'input_format': 'uclchem',
        'initial_conditions': 'initial_conditions/small_chemistry_initial.yaml',
    },
    'gas_phase_only': {
        'description': 'Gas-phase only chemistry (~183 species)',
        'input_file': '../data/uclchem_gas_phase_only.csv',
        'input_format': 'uclchem',
        'initial_conditions': 'initial_conditions/gas_phase_only_initial.yaml',
    },
}


def run_carbox(network_name: str, output_dir: str = 'results/carbox'):
    """
    Run Carbox for specified network.
    
    Parameters
    ----------
    network_name : str
        Network name (must be in NETWORK_CONFIGS)
    output_dir : str
        Output directory
    
    Returns
    -------
    dict
        Benchmark results
    """
    if network_name not in NETWORK_CONFIGS:
        raise ValueError(f"Unknown network: {network_name}. Available: {list(NETWORK_CONFIGS.keys())}")
    
    config_info = NETWORK_CONFIGS[network_name]
    
    print(f"\n{'='*70}")
    print(f"Running Carbox: {network_name}")
    print(f"{'='*70}")
    print(f"Description: {config_info['description']}")
    print(f"\nPhysical conditions:")
    print(f"  Density: {PHYSICAL_PARAMS['number_density']:.2e} cm^-3")
    print(f"  Temperature: {PHYSICAL_PARAMS['temperature']:.1f} K")
    print(f"  Final time: {PHYSICAL_PARAMS['t_end']:.2e} years")
    print(f"  CR rate: {PHYSICAL_PARAMS['cr_rate']:.2e} s^-1")
    print(f"  Av: {PHYSICAL_PARAMS['visual_extinction']:.1f} mag")
    print(f"\nNetwork:")
    print(f"  File: {config_info['input_file']}")
    print(f"  Format: {config_info['input_format']}")
    print(f"\nSolver settings:")
    print(f"  Solver: {PHYSICAL_PARAMS['solver']}")
    print(f"  rtol: {PHYSICAL_PARAMS['rtol']:.2e}")
    print(f"  atol: {PHYSICAL_PARAMS['atol']:.2e}")
    print(f"  max_steps: {PHYSICAL_PARAMS['max_steps']}")
    print(f"\nStarting integration...")
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load initial conditions from specified YAML file (REQUIRED)
    ic_path_str = config_info.get('initial_conditions')
    if not ic_path_str:
        raise ValueError(
            f"No initial_conditions specified in config for network '{network_name}'. "
            f"Add 'initial_conditions' key to NETWORK_CONFIGS."
        )
    
    ic_file = Path(__file__).parent / ic_path_str
    
    if not ic_file.exists():
        raise FileNotFoundError(
            f"\nInitial conditions file not found: {ic_file}\n"
            f"Expected path: {ic_path_str}\n\n"
            f"To generate initial conditions, run UCLCHEM first:\n"
            f"  python run_uclchem.py --network {network_name}\n"
            f"  python extract_uclchem_initial.py --network {network_name}\n"
        )
    
    # Load initial abundances (fractional)
    with open(ic_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
        initial_abundances = yaml_data['abundances']
    
    print(f"\n✓ Loaded initial conditions from: {ic_file.name}")
    print(f"  Species: {len(initial_abundances)}")
    print(f"  Source: UCLCHEM extraction (fractional abundances)")
    
    # Build SimulationConfig
    config = SimulationConfig(
        number_density=PHYSICAL_PARAMS['number_density'],
        temperature=PHYSICAL_PARAMS['temperature'],
        cr_rate=PHYSICAL_PARAMS['cr_rate'],
        fuv_field=PHYSICAL_PARAMS['fuv_field'],
        visual_extinction=PHYSICAL_PARAMS['visual_extinction'],
        t_start=PHYSICAL_PARAMS['t_start'],
        t_end=PHYSICAL_PARAMS['t_end'],
        n_snapshots=PHYSICAL_PARAMS['n_snapshots'],
        rtol=PHYSICAL_PARAMS['rtol'],
        atol=PHYSICAL_PARAMS['atol'],
        solver=PHYSICAL_PARAMS['solver'],
        max_steps=PHYSICAL_PARAMS['max_steps'],
        output_dir=str(output_path),
        run_name=network_name,
        save_abundances=True,
        initial_abundances=initial_abundances,
    )
    
    # Resolve input file path
    input_file = Path(__file__).parent / config_info['input_file']
    if not input_file.exists():
        raise FileNotFoundError(f"Network file not found: {input_file}")
    
    # Time the execution
    start_time = time.perf_counter()
    
    try:
        # Run Carbox simulation
        results = run_simulation(
            network_file=str(input_file),
            config=config,
            format_type=config_info['input_format'],
            verbose=True
        )
        
        elapsed = time.perf_counter() - start_time
        
        # Extract info from results
        network = results['network']
        solution = results['solution']
        n_species = len(network.species)
        n_reactions = len(network.reactions)
        
        # Get solver statistics from solution
        n_ode_steps = solution.stats['num_steps']
        n_accepted = solution.stats['num_accepted_steps']
        n_rejected = solution.stats['num_rejected_steps']
        
        print(f"\n✓ Integration complete in {elapsed:.2f}s")
        print(f"  Species: {n_species}")
        print(f"  Reactions: {n_reactions}")
        print(f"  ODE steps: {n_ode_steps}")
        print(f"  Accepted: {n_accepted}")
        print(f"  Rejected: {n_rejected}")
        
        # Compute reaction rates at solution snapshots
        print("\nComputing reaction rates...")
        from carbox.solver import SPY, compute_reaction_rates
        jnetwork = results['jnetwork']
        rates = compute_reaction_rates(network, jnetwork, solution, config)
        
        # Save rates to CSV
        import pandas as pd
        # Create rate dataframe with reaction names as strings
        # Format: "reactants -> products" (e.g., "H + H -> H2")
        reaction_names = []
        for reaction in network.reactions:
            reactants_str = ' + '.join(reaction.reactants)
            products_str = ' + '.join(reaction.products)
            reaction_names.append(f"{reactants_str} -> {products_str}")
        
        rates_df = pd.DataFrame(
            rates,
            columns=reaction_names
        )
        # Add time column (convert from seconds to years)
        rates_df.insert(0, 'time', solution.ts / SPY)
        
        rates_file = output_path / f"{network_name}_rates.csv"
        rates_df.to_csv(rates_file, index=False)
        
        # Save reaction metadata (types and strings) to YAML
        reaction_metadata = []
        for i, reaction in enumerate(network.reactions):
            reactants_str = ' + '.join(reaction.reactants)
            products_str = ' + '.join(reaction.products)
            reaction_metadata.append({
                'index': i,
                'reaction': f"{reactants_str} -> {products_str}",
                'type': reaction.reaction_type,
            })
        
        reactions_yaml_file = output_path / f"{network_name}_reactions.yaml"
        with open(reactions_yaml_file, 'w') as f:
            yaml.dump(reaction_metadata, f, default_flow_style=False)
        
        # Load abundance output to get timesteps
        abund_file = output_path / f"{network_name}_abundances.csv"
        df = pd.read_csv(abund_file)
        
        # Save benchmark metadata (convert all to native Python types for JSON)
        final_time = (
            float(df['time_years'].iloc[-1]) if len(df) > 0 else 0
        )
        benchmark_results = {
            'network': network_name,
            'success': True,
            'time': elapsed,
            'n_timesteps': len(df),
            'n_species': int(n_species),
            'n_reactions': int(n_reactions),
            'n_ode_steps': int(n_ode_steps),
            'n_accepted': int(n_accepted),
            'n_rejected': int(n_rejected),
            'final_time': final_time,
            'output_file': str(abund_file),
            'physical_params': PHYSICAL_PARAMS,
        }
        
        benchmark_file = output_path / f"{network_name}_benchmark.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print("\nSaved outputs:")
        print(f"  {abund_file}")
        print(f"  {rates_file}")
        print(f"  {reactions_yaml_file}")
        print(f"  {output_path / f'{network_name}_summary.txt'}")
        print(f"  {benchmark_file}")
        
        return benchmark_results
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"\nERROR: Carbox failed after {elapsed:.2f}s")
        print(f"  {type(e).__name__}: {e}")
        
        import traceback
        traceback.print_exc()
        
        return {
            'network': network_name,
            'success': False,
            'time': elapsed,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run Carbox benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available networks:
  small_chemistry - {NETWORK_CONFIGS['small_chemistry']['description']}
  gas_phase_only  - {NETWORK_CONFIGS['gas_phase_only']['description']}

Example:
  python run_carbox.py --network small_chemistry
        """
    )
    
    parser.add_argument(
        '--network', required=True,
        choices=list(NETWORK_CONFIGS.keys()),
        help='Network to run'
    )
    parser.add_argument(
        '--output', default='results/carbox',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_carbox(args.network, args.output)
    
    # Print summary
    print(f"\n{'='*70}")
    if results['success']:
        print(f"✓ Carbox benchmark complete: {results['time']:.2f}s")
        sys.exit(0)
    else:
        print("✗ Carbox benchmark failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
