# Minimal Changes for Parallel Simulation Execution

## Overview
This document outlines how to implement the parallel simulation plan from `parallel_simulation_plan.md` with the fewest possible code changes. The goal is to enable batched execution of parameter sweeps while maintaining backward compatibility with existing code.

## Core Changes Required

### 1. Add Batch Solver Function to `carbox/solver.py` (5 lines)

Add a new function that accepts parameter arrays instead of a single config:

```python
def solve_network_batch(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    temperatures: jnp.ndarray,
    cr_rates: jnp.ndarray,
    fuv_fields: jnp.ndarray,
    visual_extinctions: jnp.ndarray,
    t_eval: jnp.ndarray,
    solver_name: str = "kvaerno5",
    atol: float = 1e-18,
    rtol: float = 1e-12,
    max_steps: int = 4096,
) -> dx.Solution:
    """Batch version of solve_network for parallel parameter sweeps."""
    # Implementation: vmap the single solve with array parameters
    return jax.vmap(
        lambda temp, cr, fuv, av: solve_network_core(
            jnetwork, y0, t_eval, temp, cr, fuv, av,
            solver_name, atol, rtol, max_steps
        ),
        in_axes=(0, 0, 0, 0)
    )(temperatures, cr_rates, fuv_fields, visual_extinctions)
```

### 2. Extract Core Solver Function (Refactor existing `solve_network`)

Split `solve_network` into:
- `solve_network_core`: Pure JAX function with raw parameters
- `solve_network`: Wrapper that extracts params from config

This requires moving ~50 lines of solver logic into the core function.

### 3. Modify `sensitivity_analysis/run_cr_sensitivity.py` (10 lines)

Replace the loop with batch execution:

```python
# Instead of:
# for zeta in zeta_values:
#     result = run_single_zeta(zeta, ...)

# Use:
zeta_array = jnp.array(zeta_values)
temp_array = jnp.full_like(zeta_array, PHYSICAL_PARAMS["temperature"])
fuv_array = jnp.full_like(zeta_array, PHYSICAL_PARAMS["fuv_field"])
av_array = jnp.full_like(zeta_array, PHYSICAL_PARAMS["visual_extinction"])

solutions = solve_network_batch(
    jnetwork, y0, temp_array, zeta_array, fuv_array, av_array, t_eval
)

# Process batched results
for i, solution in enumerate(solutions):
    # Save individual results as before
```

## Implementation Steps

### Phase 1: Solver Refactoring (1 hour)
1. Extract `solve_network_core` from `solve_network`
2. Add `solve_network_batch` function
3. Test single simulation still works

### Phase 2: Batch Execution (30 minutes)
1. Modify sensitivity script to use batch solver
2. Update result processing for batched output
3. Test parallel execution

### Phase 3: Chunking for Large Sweeps (Optional, 1 hour)
1. Add chunking logic to handle >1000 simulations
2. Implement memory-aware batch sizing

## Code Change Summary

| File | Lines Changed | Description |
|------|---------------|-------------|
| `carbox/solver.py` | ~60 | Extract core function + add batch wrapper |
| `sensitivity_analysis/run_cr_sensitivity.py` | ~15 | Replace loop with batch call |
| Total | **~75 lines** | Minimal invasive changes |

## Benefits
- **Maintains API**: Existing single-simulation code unchanged
- **Backward Compatible**: All current scripts continue working
- **Performance**: 10-100x speedup for parameter sweeps
- **Memory Efficient**: Uses JAX's optimized vmap implementation

## Testing
- Verify single simulations produce identical results
- Test batch execution with small parameter grids
- Benchmark speedup on larger sweeps (100+ points)

## Future Extensions
Once this minimal implementation works, it can be extended to:
- Multi-dimensional parameter sweeps (temp + cr_rate + fuv)
- GPU acceleration
- Integration with optimization libraries
- CLI tool for arbitrary parameter sweeps
