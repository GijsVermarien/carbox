# Carbox Benchmarking Framework

Comprehensive benchmarking suite comparing **Carbox** (JAX-accelerated) against **UCLCHEM** (Fortran DVODE) for astrochemical kinetics simulations.

## 🚀 Quick Start

Run the complete benchmark suite:
```bash
cd benchmarks
./run_benchmarks.sh
```

See **[QUICKSTART.md](QUICKSTART.md)** for detailed usage instructions.

## Overview

This framework tests two network complexities:
- **small_chemistry**: ~34 species, minimal gas-phase chemistry
- **gas_phase_only**: ~183 species, complete gas-phase reactions (no grain surface)

### Test Case: Static Cloud Model

Physical conditions matching UCLCHEM validation case:
- Density: 10⁴ cm⁻³
- Temperature: 250 K (warm cloud)
- Integration time: 5 × 10⁶ years
- Cosmic ray ionization: 10⁻¹⁷ s⁻¹
- Visual extinction: 3.0 mag

## Directory Structure

```
benchmarks/
├── run_benchmarks.sh           # Main benchmark orchestrator
├── run_uclchem.py             # UCLCHEM runner
├── run_carbox.py              # Carbox runner
├── compare_results.py         # Comparison & visualization
├── extract_uclchem_initial.py # Extract initial conditions
├── configs/
│   ├── small_chemistry.yaml   # Network configuration
│   └── gas_phase_only.yaml    # Network configuration
├── initial_conditions/        # Initial abundance files
└── results/
    ├── uclchem/              # UCLCHEM outputs
    ├── carbox/               # Carbox outputs
    └── comparisons/          # Comparison plots & reports
```

## Quick Start

### 1. Ensure Environment is Active

The script automatically activates the conda environment at `../.conda/`, or you can manually activate:
```bash
conda activate /path/to/carbox/.conda
```

### 2. Run All Benchmarks

```bash
./run_benchmarks.sh
```

This processes each network sequentially:
1. Build UCLCHEM network
2. Install UCLCHEM
3. Run UCLCHEM simulation
4. Extract initial conditions from UCLCHEM results
5. Run Carbox simulation with extracted initial conditions
6. Repeat for next network
7. Generate comparison plots and reports for all networks

**Key Feature:** Carbox requires UCLCHEM's initial abundances (extracted from t=0). The initial conditions file path must be specified in the network configuration, and the benchmark will error if the file is missing.

### 3. Run Specific Network

```bash
./run_benchmarks.sh --network small_chemistry
```

### 4. Compare Existing Results

If simulations are already complete:
```bash
./run_benchmarks.sh --compare-only
```

## Usage Examples

### Run Only UCLCHEM
```bash
./run_benchmarks.sh --skip-carbox
```

### Run Only Carbox
```bash
./run_benchmarks.sh --skip-uclchem
```

**Note:** UCLCHEM is always rebuilt and reinstalled when running benchmarks (required by build system).
## Configuration Files

Network configurations are in `configs/{network}.yaml`:

```yaml
# configs/small_chemistry.yaml
network:
  name: small_chemistry
  description: "Small gas-phase chemistry (~34 species)"
  uclchem_csv: uclchem_small_chemistry.csv
  
physical:
  density: 1.0e4              # cm^-3
  temperature: 250.0          # K
  final_time: 5.0e6           # years
  cr_rate: 1.0e-17            # s^-1
  visual_extinction: 3.0      # mag
  
solver:
  uclchem_reltol: 1.0e-4
  uclchem_abstol: 1.0e-10
  carbox_solver: kvaerno5
  carbox_rtol: 1.0e-6
  carbox_atol: 1.0e-15
  max_steps: 50000
```

## Output Files

### UCLCHEM Results
- `results/uclchem/{network}_abundances.csv`: Time series of abundances
- `results/uclchem/{network}_physics.csv`: Physical conditions
- `results/uclchem/{network}_benchmark.json`: Timing and metadata

### Carbox Results
- `results/carbox/{network}_abundances.csv`: Time series with physical parameters
- `results/carbox/{network}_summary.txt`: Solver statistics
- `results/carbox/{network}_benchmark.json`: Timing and metadata

### Comparison Outputs
- `results/comparisons/{network}_comparison.png`: 9-panel abundance plots
- `results/comparisons/{network}_differences.png`: Relative differences
- `results/comparisons/{network}_statistics.txt`: Numerical comparison
- `results/comparisons/{network}_performance.md`: Timing analysis

## Interpreting Results

### Abundance Comparison Plots

The 9-panel plot shows the top 9 most abundant species:
- **Blue solid**: UCLCHEM
- **Red dashed**: Carbox

Look for:
- ✅ Curves that overlap → Good agreement
- ⚠️ Parallel curves with offset → Systematic difference
- ❌ Different slopes → Chemical evolution mismatch

### Relative Difference Plots

Shows `|Carbox - UCLCHEM| / UCLCHEM × 100%`:
- < 1%: Excellent agreement
- 1-10%: Good agreement (typical for different solvers)
- > 10%: Investigate further

### Performance Reports

Compares:
- **Total runtime**: Wall-clock time for integration
- **Speedup**: Ratio of UCLCHEM/Carbox times
- **Timesteps**: Number of output points
- **ODE steps**: Internal solver steps

Expected:
- Carbox faster for large networks (GPU acceleration)
- UCLCHEM competitive for small networks (mature Fortran code)

## Troubleshooting

### UCLCHEM Network Not Found

If you see `ERROR: UCLCHEM network not built`:
```bash
cd ../uclchem/Makerates
python MakeRates.py configs/small_gas_only.yaml
cd ../
pip install .
```

### Missing Dependencies

Carbox requires:
```bash
pip install jax jaxlib equinox diffrax pandas pyyaml matplotlib
```

UCLCHEM requires:
```bash
pip install numpy scipy matplotlib meson ninja
```

### JAX Configuration Errors

Ensure 64-bit is enabled:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

### Species Naming Mismatches

UCLCHEM and Carbox may use different naming conventions:
- UCLCHEM: `H2O`, `HCO+`, `H3O+`
- Some formats: `H2O`, `HCO_plus`, `H3O_plus`

The comparison tools automatically handle common species only.

## Advanced Usage

### Custom Solver Settings

For Carbox, edit the config YAML file:
```yaml
solver:
  carbox_solver: Tsit5  # Options: Kvaerno5, Tsit5, Dopri5, Kvaerno3
```

## Initial Conditions Workflow

For accurate benchmarking, Carbox uses UCLCHEM's initial conditions:

### 1. Extract UCLCHEM Initial Conditions
```bash
python extract_uclchem_initial.py --network small_chemistry
```

This extracts ALL gas-phase species from UCLCHEM's first timestep and saves to:
`initial_conditions/small_chemistry_initial.yaml`

### 2. Carbox Auto-Loads Initial Conditions

The `run_carbox.py` script automatically loads the YAML file if it exists:
```python
# Automatically loaded from initial_conditions/{network}_initial.yaml
initial_abundances = {...}  # H: 0.5, H2: 0.25, etc.
```

### 3. Abundance Convention

**CRITICAL**: Both codes use **fractional abundances** relative to H nuclei density:
```
x_i = n_i / n_{H,nuclei}
n_{H,nuclei} = 2*n(H2) + n(H)
```

This matches UCLCHEM's output convention for direct comparison.

## Citation

If using this benchmark framework, please cite:

**Carbox**:
- [Citation pending]

**UCLCHEM**:
- Holdship et al. (2017), AJ, 154, 38

**UMIST Database**:
- McElroy et al. (2013), A&A, 550, A36

## Contact

For issues or questions:
- Carbox: [GitHub Issues](https://github.com/user/carbox/issues)
- UCLCHEM: [GitHub Issues](https://github.com/uclchem/UCLCHEM/issues)
