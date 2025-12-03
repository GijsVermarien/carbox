# High-Performance Grid Search Optimization Plan

## Executive Summary
The goal is to transition from a sequential, loop-based simulation approach to a fully vectorized, batched execution model using JAX. This will enable running thousands of chemical network simulations in parallel, reducing parameter sweep times from hours to minutes.

## Core Architecture

### 1. Vectorized Solver Kernel (`vmap`)
Instead of calling `solve_network` in a Python loop, we will expose a pure JAX kernel that accepts batches of parameters.

**Current Flow:**
```python
for params in grid:
    solve_network(params) # Re-dispatches, runs sequentially
```

**Optimized Flow:**
```python
# Shape: (batch_size, num_params)
batched_params = stack(grid)

# Single compilation, parallel execution
vmap(solve_network_core)(batched_params)
```

### 2. Chunked Execution Strategy
To prevent Out-Of-Memory (OOM) errors when running massive grids (e.g., 10,000+ points), we will implement a chunking mechanism.
- **Grid Generation**: Create the full Cartesian product of parameters.
- **Chunking**: Split the grid into manageable sub-batches (e.g., 1024 simulations per chunk).
- **Execution**: Sequentially process chunks, where each chunk is executed in parallel.

### 3. Data-Oriented Storage
Writing thousands of individual CSV files is inefficient (I/O bound). We will switch to a columnar storage format.
- **Format**: Parquet or HDF5.
- **Structure**: Single file containing results for the entire sweep, indexed by parameter IDs.

---

## Implementation Roadmap

### Phase 1: Solver Refactoring
Refactor `carbox/solver.py` to separate the configuration object from the numerical solver.
- **Task**: Extract `solve_ode_kernel` which accepts raw JAX arrays (temperature, cr_rate, etc.) instead of `SimulationConfig`.
- **Goal**: Make the solver function pure and vmap-compatible.

### Phase 2: The Batch Engine
Create a new module `carbox/batch.py` to handle grid orchestration.
- **`GridSearch` Class**:
    - Accepts parameter ranges (e.g., `cr_rate=[1e-17, 1e-16]`, `temp=[10, 20]`).
    - Generates the Cartesian product of all parameters.
- **`run_batch` Function**:
    - Compiles the `vmap`'d solver once.
    - Iterates over chunks of the parameter grid.
    - Aggregates results in memory.

### Phase 3: Efficient I/O
Implement a result writer that handles batched outputs.
- **`BatchResult` Class**:
    - Stores `ys` (abundances) with shape `(n_sims, n_steps, n_species)`.
    - Stores `ts` (time) and parameter metadata.
    - Methods to export to Parquet/HDF5.

### Phase 4: CLI & Benchmarking
- Update `run_cr_sensitivity.py` to use the new batch engine.
- Add a CLI command `carbox-sweep` for running grid searches from YAML configs.

---

## Technical Specifications

### New Function Signature
```python
def solve_chunk(jnetwork, y0, t_eval, temperatures, cr_rates, ...):
    """
    Args:
        temperatures: Array of shape (batch_size,)
        cr_rates: Array of shape (batch_size,)
    Returns:
        Solution object with shape (batch_size, n_steps, n_species)
    """
    return jax.vmap(solve_single, in_axes=(None, None, None, 0, 0, ...))(...)
```

### Memory Estimation
- **State Size**: ~200 species * 8 bytes = 1.6 KB per step.
- **Trajectory**: 1000 steps * 1.6 KB = 1.6 MB per simulation.
- **Batch of 1024**: ~1.6 GB RAM.
- **Conclusion**: A batch size of 1024-4096 is feasible on standard GPUs (16GB+ VRAM).

## Success Metrics
1.  **Speedup**: >10x on CPU, >100x on GPU for a 1000-point grid.
2.  **Scaling**: Linear scaling with compute resources until memory saturation.
3.  **Usability**: Define a sweep in <10 lines of code.
