#!/usr/bin/env python3
"""
Run UCLCHEM benchmark for a specific network.

Simplified standalone runner with hardcoded configurations.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add UCLCHEM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "uclchem" / "src"))

import uclchem

# Hardcoded physical parameters (static cloud test case)
PHYSICAL_PARAMS = {
    'zeta': 1.0,                 # s^-1
    'radfield': 1.0,             # In Habing units
    'initialDens': 1.0e4,        # cm^-3
    'initialTemp': 250.0,         # K
    'finalTime': 5.0e6,          # years
    'freefall': False,
    'freezeFactor': 0.0,
    'endAtFinalDensity': False,
    'writeStep': 1,              # Output every step
    'reltol': 1.0e-4,            # Relative tolerance
    'abstol_factor': 1.0e-10,    # Absolute tolerance factor
}

# Hardcoded species to track
OUTPUT_SPECIES = [
    'H', 'H2', 'He', 'C', 'O', 'N',
    'CO', 'H2O', 'OH', 'CH', 'NH3', 'HCO+', 'H3+', 'e-'
]

# Network configurations
NETWORK_CONFIGS = {
    'small_chemistry': {
        'description': 'Small gas-phase chemistry (~20 species)',
        'makerates_config': 'data/small_chemistry/user_settings.yaml',
    },
    'gas_phase_only': {
        'description': 'Gas-phase only chemistry (~183 species)',
        'makerates_config': 'data/gas_phase_only/user_settings.yaml',
    },
}


def run_uclchem(network_name: str, output_dir: str = 'results/uclchem'):
    """
    Run UCLCHEM for specified network.
    
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
    
    config = NETWORK_CONFIGS[network_name]
    
    print(f"\n{'='*70}")
    print(f"Running UCLCHEM: {network_name}")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    print(f"\nPhysical conditions:")
    print(f"  Density: {PHYSICAL_PARAMS['initialDens']:.2e} cm^-3")
    print(f"  Temperature: {PHYSICAL_PARAMS['initialTemp']:.1f} K")
    print(f"  Final time: {PHYSICAL_PARAMS['finalTime']:.2e} years")
    print(f"\nSolver settings:")
    print(f"  reltol: {PHYSICAL_PARAMS['reltol']:.2e}")
    print(f"  abstol_factor: {PHYSICAL_PARAMS['abstol_factor']:.2e}")
    print(f"\nStarting integration...")
    
    # Time the execution
    start_time = time.perf_counter()
    
    try:
        # Run UCLCHEM
        result = uclchem.model.cloud(
            param_dict=PHYSICAL_PARAMS,
            out_species=OUTPUT_SPECIES,
            return_dataframe=True,
            return_rates=True
        )
        
        # Unpack results (format may vary by UCLCHEM version)
        if len(result) == 4:
            physics_df, chemistry_df, abundances_start, flag = result
        elif len(result) == 5:
            physics_df, chemistry_df, rates_df, abundances_start, flag = result
        else:
            raise ValueError(f"Unexpected number of return values from UCLCHEM: {len(result)}")
        
        elapsed = time.perf_counter() - start_time
        
        print(f"\n✓ Integration complete in {elapsed:.2f}s")
        print(f"  Return flag: {flag}")
        print(f"  Timesteps: {len(chemistry_df)}")
        print(f"  Species tracked: {len(chemistry_df.columns) - 1}")
        
        # Determine time column name
        time_col = None
        for col in ['time', 'Time', 'TIME', 't']:
            if col in chemistry_df.columns:
                time_col = col
                break
        if time_col is None:
            # Assume first column is time
            time_col = chemistry_df.columns[0]
        
        # Check for errors
        if flag < 0:
            error_msg = uclchem.utils.check_error(flag)
            print(f"WARNING: UCLCHEM returned error flag {flag}: {error_msg}")
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        abund_file = output_path / f"{network_name}_abundances.csv"
        chemistry_df.to_csv(abund_file, index=False)
        
        physics_file = output_path / f"{network_name}_physics.csv"
        physics_df.to_csv(physics_file, index=False)
        
        rates_file = output_path / f"{network_name}_rates.csv"
        rates_df.to_csv(rates_file, index=False)
        print("Wrote rates to", rates_file)
        
        # Save benchmark metadata
        benchmark_results = {
            'network': network_name,
            'success': flag >= 0,
            'flag': int(flag),
            'time': elapsed,
            'n_timesteps': len(chemistry_df),
            'n_species': len(chemistry_df.columns) - 1,
            'final_time': float(chemistry_df[time_col].iloc[-1]) if len(chemistry_df) > 0 else 0,
            'output_file': str(abund_file),
            'physical_params': PHYSICAL_PARAMS,
        }
        
        benchmark_file = output_path / f"{network_name}_benchmark.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\nSaved outputs:")
        print(f"  {abund_file}")
        print(f"  {physics_file}")
        print(f"  {benchmark_file}")
        print(f"  {rates_file}")
        
        return benchmark_results
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"\nERROR: UCLCHEM failed after {elapsed:.2f}s")
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
        description='Run UCLCHEM benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available networks:
  small_chemistry - {NETWORK_CONFIGS['small_chemistry']['description']}
  gas_phase_only  - {NETWORK_CONFIGS['gas_phase_only']['description']}

Example:
  python run_uclchem.py --network small_chemistry
        """
    )
    
    parser.add_argument('--network', required=True,
                       choices=list(NETWORK_CONFIGS.keys()),
                       help='Network to run')
    parser.add_argument('--output', default='results/uclchem',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_uclchem(args.network, args.output)
    
    # Print summary
    print(f"\n{'='*70}")
    if results['success']:
        print(f"✓ UCLCHEM benchmark complete: {results['time']:.2f}s")
        sys.exit(0)
    else:
        print(f"✗ UCLCHEM benchmark failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
