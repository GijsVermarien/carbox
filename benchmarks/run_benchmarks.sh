#!/bin/bash
#
# Carbox vs UCLCHEM Benchmark Suite
# 
# Workflow (per network):
#   1. Build UCLCHEM network
#   2. Install UCLCHEM (required after each build)
#   3. Run UCLCHEM simulation
#   4. Extract initial conditions from UCLCHEM results
#   5. Run Carbox simulation (using UCLCHEM initial conditions)
#   Then repeat for next network
#   Finally: Generate comparison reports for all networks
#

set -e  # Exit on any error

# ============================================================
# Setup
# ============================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default options
NETWORKS=("small_chemistry" "gas_phase_only")
SKIP_UCLCHEM=false
SKIP_CARBOX=false
COMPARE_ONLY=false
TIME_BENCHMARK=false
N_RUNS=1

# ============================================================
# Parse Arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-uclchem)
            SKIP_UCLCHEM=true
            shift
            ;;
        --skip-carbox)
            SKIP_CARBOX=true
            shift
            ;;
        --compare-only)
            COMPARE_ONLY=true
            SKIP_UCLCHEM=true
            SKIP_CARBOX=true
            shift
            ;;
        --network)
            NETWORKS=($2)
            shift 2
            ;;
        --time-benchmark)
            TIME_BENCHMARK=true
            N_RUNS=${2:-10}  # Default to 10 runs if not specified
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--skip-uclchem] [--skip-carbox] [--compare-only] [--network NAME] [--time-benchmark N_RUNS]"
            exit 1
            ;;
    esac
done

# ============================================================
# Activate Environment
# ============================================================

if [ -d "$PROJECT_ROOT/.conda" ]; then
    echo -e "${BLUE}Activating conda environment...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate "$PROJECT_ROOT/.conda"
    echo -e "${GREEN}✓ Conda environment activated${NC}"
    echo ""
else
    echo -e "${YELLOW}WARNING: Conda environment not found at $PROJECT_ROOT/.conda${NC}"
    echo -e "${YELLOW}Proceeding with system Python...${NC}"
    echo ""
fi

# ============================================================
# Banner
# ============================================================

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Carbox vs UCLCHEM Benchmark Suite${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "Networks to benchmark: ${GREEN}${NETWORKS[@]}${NC}"
echo ""

# ============================================================
# Main Loop: Process Each Network Sequentially
# ============================================================

for network in "${NETWORKS[@]}"; do
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Processing: ${network}${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    
    # --------------------------------------------------------
    # Step 1: Build UCLCHEM Network
    # --------------------------------------------------------
    
    if [[ "$COMPARE_ONLY" == false  && "$SKIP_UCLCHEM" == false ]]; then
        echo -e "${YELLOW}[1/4] Building UCLCHEM network...${NC}"
        
        cd "$PROJECT_ROOT/uclchem/Makerates"
        CONFIG_FILE="data/${network}/user_settings.yaml"
        
        if [ ! -f "$CONFIG_FILE" ]; then
            echo -e "${RED}✗ Config not found: $CONFIG_FILE${NC}"
            exit 1
        fi
        
        python MakeRates.py "$CONFIG_FILE"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Network built${NC}"
        else
            echo -e "${RED}✗ Build failed${NC}"
            exit 1
        fi
        echo ""
        
        # --------------------------------------------------------
        # Step 2: Install UCLCHEM
        # --------------------------------------------------------
        
        echo -e "${YELLOW}[2/4] Installing UCLCHEM...${NC}"
        
        cd "$PROJECT_ROOT/uclchem"
        pip install . --quiet
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ UCLCHEM installed${NC}"
        else
            echo -e "${RED}✗ Installation failed${NC}"
            exit 1
        fi
        echo ""
    fi
    
    # --------------------------------------------------------
    # Step 3: Run UCLCHEM Simulation
    # --------------------------------------------------------
    
    if [ "$SKIP_UCLCHEM" = false ]; then
        echo -e "${YELLOW}[3/5] Running UCLCHEM simulation...${NC}"
        echo ""
        
        cd "$SCRIPT_DIR"
        if [ "$TIME_BENCHMARK" = true ]; then
            python run_uclchem.py --network "$network" --n-runs "$N_RUNS"
        else
            python run_uclchem.py --network "$network"
        fi
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ UCLCHEM complete${NC}"
        else
            echo -e "${RED}✗ UCLCHEM failed${NC}"
            exit 1
        fi
        echo ""
        
        # --------------------------------------------------------
        # Step 4: Extract Initial Conditions from UCLCHEM
        # --------------------------------------------------------
        
        echo -e "${YELLOW}[4/5] Extracting UCLCHEM initial conditions...${NC}"
        
        python extract_uclchem_initial.py --network "$network"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Initial conditions extracted${NC}"
        else
            echo -e "${RED}✗ Extraction failed${NC}"
            exit 1
        fi
        echo ""
    fi
    
    # --------------------------------------------------------
    # Step 5: Run Carbox Simulation
    # --------------------------------------------------------
    
    if [ "$SKIP_CARBOX" = false ]; then
        echo -e "${YELLOW}[5/5] Running Carbox simulation...${NC}"
        echo ""
        
        cd "$SCRIPT_DIR"
        if [ "$TIME_BENCHMARK" = true ]; then
            python run_carbox.py --network "$network" --n-runs "$N_RUNS"
        else
            python run_carbox.py --network "$network"
        fi
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Carbox complete${NC}"
        else
            echo -e "${RED}✗ Carbox failed${NC}"
            exit 1
        fi
        echo ""
    fi
    
    echo -e "${GREEN}✓ ${network} complete${NC}"
    echo ""
done

# ============================================================
# Generate Comparisons
# ============================================================

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Generating Comparisons${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

cd "$SCRIPT_DIR"

for network in "${NETWORKS[@]}"; do
    echo -e "${YELLOW}Comparing: ${network}${NC}"
    
    UCLCHEM_FILE="results/uclchem/${network}_abundances.csv"
    CARBOX_FILE="results/carbox/${network}_abundances.csv"
    
    # Check if results exist
    if [ ! -f "$UCLCHEM_FILE" ]; then
        echo -e "${RED}✗ UCLCHEM results missing${NC}"
        echo ""
        continue
    fi
    
    if [ ! -f "$CARBOX_FILE" ]; then
        echo -e "${RED}✗ Carbox results missing${NC}"
        echo ""
        continue
    fi
    
    # Run comparison
    python compare_results.py --network "$network"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Comparison complete${NC}"
    else
        echo -e "${RED}✗ Comparison failed${NC}"
    fi
    echo ""
done

# ============================================================
# Summary
# ============================================================

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Benchmark Suite Complete${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

cd "$SCRIPT_DIR"

echo "Results directory: ${YELLOW}results/${NC}"
echo ""

for network in "${NETWORKS[@]}"; do
    echo -e "${BLUE}${network}:${NC}"
    
    # UCLCHEM timing
    if [ -f "results/uclchem/${network}_benchmark.json" ]; then
        TIME=$(python -c "import json; print(f\"{json.load(open('results/uclchem/${network}_benchmark.json'))['time']:.2f}s\")" 2>/dev/null || echo "?")
        echo -e "  UCLCHEM:     ${TIME}"
    fi
    
    # Carbox timing
    if [ -f "results/carbox/${network}_benchmark.json" ]; then
        TIME=$(python -c "import json; print(f\"{json.load(open('results/carbox/${network}_benchmark.json'))['time']:.2f}s\")" 2>/dev/null || echo "?")
        echo -e "  Carbox:      ${TIME}"
    fi
    
    # Speedup
    if [ -f "results/comparisons/${network}_performance.md" ]; then
        SPEEDUP=$(grep "Speedup:" "results/comparisons/${network}_performance.md" | awk '{print $2}' || echo "?")
        echo -e "  Speedup:     ${SPEEDUP}"
    fi
    
    echo ""
done

echo -e "${GREEN}✓ All benchmarks complete!${NC}"
echo ""
echo "View results:"
echo "  • Plots:      results/comparisons/*_comparison.png"
echo "  • Statistics: results/comparisons/*_statistics.txt"
echo "  • Performance: results/comparisons/*_performance.md"
