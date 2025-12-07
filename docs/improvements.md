# Performance Improvements & Numerical Stability

## Problem: "Wild Abundances" and Slow Performance

You observed two related issues in the simulation:
1.  **Extreme Abundances:** Species reaching physically irrelevant values like `1e-80` or `1e-100`.
2.  **Slow Execution:** The solver taking a long time to complete, likely taking many small steps.

### Why this happens

**1. Stiff Systems & Timescales**
Chemical kinetic networks are "stiff" systems. This means they contain processes happening on vastly different timescales (from microseconds to millions of years). Implicit solvers like `kvaerno5` are designed to handle this, but they struggle when the system enters unphysical regimes.

**2. Floating Point Denormals & Underflow**
When abundances drop below approx `1e-30` (depending on float precision), they approach the limits of floating-point representation.
*   **Denormals:** Modern CPUs can handle numbers smaller than the standard minimum (denormals), but operations on them can be significantly slower (sometimes 100x slower) than normal floating-point operations.
*   **Numerical Noise:** At `1e-80`, the values are essentially numerical noise. However, the ODE solver doesn't know this. It attempts to accurately integrate the derivatives of these tiny numbers. If a species oscillates between `1e-80` and `1e-79`, the relative change is huge, forcing the solver to reduce its timestep drastically to maintain error tolerances (`rtol`).

**3. The Result**
The solver gets "stuck" resolving the dynamics of practically empty species, taking thousands of tiny steps to advance time, which kills performance.

## Solution 1: Imposing an Abundance Floor (The "Clamp")

To fix this, we must enforce a minimum abundance floor (e.g., `1e-30`) during the integration steps, not just at initialization.

### How it works

1.  **Configuration:** Define a floor value (e.g., `1e-30` relative to hydrogen).
2.  **Integration Step:** Inside the ODE function (the `JNetwork` call), before calculating reaction rates, we "clamp" the abundance vector:
    ```python
    # Effective abundance for rate calculation
    y_clamped = jnp.maximum(y, abundance_floor)
    ```
3.  **Result:**
    *   Reaction rates are calculated based on this floor, preventing derivatives from exploding or becoming unstable due to underflow.
    *   Species naturally tending to zero will simply sit at the floor value.
    *   The solver sees a stable system and can take larger timesteps.

### Is this optimal? (The Discontinuity Problem)

While `jnp.maximum` is simple, it introduces a **discontinuity** in the derivative at the floor value.
*   Implicit solvers (like `kvaerno5`) rely on the Jacobian matrix (derivatives of the system).
*   A sharp "kink" can confuse the solver's Newton-Raphson iteration, potentially causing convergence failures or reduced step sizes near the floor.

**Better Alternatives:**

1.  **Smooth Maximum (Softplus):** Use a differentiable approximation to the maximum function.
    ```python
    # Smooth transition prevents Jacobian discontinuity
    y_smooth = jax.nn.softplus(k * (y - floor)) / k + floor
    ```
    This keeps the gradients smooth, helping the solver.

2.  **Log-Transformation (The "Gold Standard"):**
    Instead of solving for abundance $y$, solve for $z = \ln(y)$.
    *   **Positivity:** $y = e^z$ is always positive.
    *   **Scale:** Chemical abundances span 30 orders of magnitude. Linear solvers struggle with this dynamic range. Log-space compresses this to a range of ~[-70, 0], which is much easier for floating-point arithmetic.
    *   **No Floor Needed:** You don't need an artificial floor; species just go to very negative log-values without underflowing to zero.

    *Note: This requires rewriting the ODE function to compute $d(\ln y)/dt = (1/y) \cdot dy/dt$.*

## Solution 2: Tuning Solver Tolerances

The default tolerances in `carbox` are extremely strict (`atol=1e-18`, `rtol=1e-12`).

### Optimization Strategy
For many astrophysical applications, we do not need precision down to the 18th decimal place, especially when input reaction rates have uncertainties of 20-50%.

*   **Recommendation:** Relax tolerances to `atol=1e-20` (if using floor) or `atol=1e-10` (absolute abundance) and `rtol=1e-4` or `1e-5`.
*   **Impact:** This allows the solver to take larger steps, as it doesn't need to resolve the 12th significant digit of every species.

## Solution 3: Sparse Jacobian Optimization

The current solver (`kvaerno5`) uses an implicit method, which requires inverting the Jacobian matrix ($J = \partial f / \partial y$) at every step.

*   **The Bottleneck:** For a network with $N$ species, the Jacobian is $N \times N$. Inverting a dense matrix scales as $O(N^3)$. For 500 species, this is manageable, but for larger networks (UMIST full ~6000 reactions), this becomes the dominant cost.
*   **Sparsity:** Chemical networks are naturally sparse (each species only reacts with a few others).
*   **Improvement:** Utilizing sparse matrix solvers within the Newton-Raphson iteration of the implicit solver can drastically reduce computation time from $O(N^3)$ to roughly $O(N)$.

## Implementation Checklist

To implement these improvements in Carbox:

1.  **Update `JNetwork.__call__`**: Modify `carbox/network.py` to accept `abundance_floor` as an argument and apply `jnp.maximum(abundances, abundance_floor)` before computing rates.
2.  **Update Solver**: Modify `carbox/solver.py` to extract `abundance_floor` from the `SimulationConfig` and pass it into the ODE term.
3.  **Config Update**: Ensure `SimulationConfig` defaults for `atol` and `rtol` are sensible for production runs (e.g., `1e-15` / `1e-6`).
