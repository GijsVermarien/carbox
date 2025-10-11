#!/bin/bash
#
# Run cosmic ray ionization rate sensitivity analysis
#
# This script:
# 1. Activates the conda environment
# 2. Runs sensitivity analysis varying ζ over 6 orders of magnitude
# 3. Saves results for each zeta value
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================================================"
echo "Cosmic Ray Ionization Rate Sensitivity Analysis"
echo "========================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Activate conda environment
echo "Activating conda environment..."
conda activate "$PROJECT_ROOT/.conda"
echo "✓ Environment activated"
echo ""

# Check if initial conditions exist
IC_FILE="$PROJECT_ROOT/benchmarks/initial_conditions/gas_phase_only_initial.yaml"
if [ ! -f "$IC_FILE" ]; then
    echo "ERROR: Initial conditions not found: $IC_FILE"
    echo ""
    echo "Generate initial conditions first:"
    echo "  cd benchmarks"
    echo "  python run_uclchem.py --network gas_phase_only"
    echo "  python extract_uclchem_initial.py --network gas_phase_only"
    echo ""
    exit 1
fi

echo "✓ Initial conditions found"
echo ""

# Run sensitivity analysis
cd "$SCRIPT_DIR"
echo "Running sensitivity analysis..."
echo "  Zeta range: 10^-2 to 10^4 s^-1 (36 values)"
echo "  Network: gas_phase_only (~183 species)"
echo ""

python run_cr_sensitivity.py \
    --output results_cr \
    --n-zetas 36 \
    --zeta-min -2 \
    --zeta-max 4 \
    "$@"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Sensitivity analysis complete!"
    echo "========================================================================"
    echo "Results saved in: $SCRIPT_DIR/results_cr/"
    echo ""
    echo "To analyze results:"
    echo "  cd $SCRIPT_DIR"
    echo "  python plot_cr_sensitivity.py"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "✗ Sensitivity analysis failed"
    echo "========================================================================"
    exit 1
fi
