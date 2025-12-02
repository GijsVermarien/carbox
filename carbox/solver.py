"""
ODE solver wrapper for chemical kinetics integration.

Wraps Diffrax solvers with appropriate settings for stiff chemistry ODEs.
"""

from typing import Tuple

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp

from .config import SimulationConfig
from .network import JNetwork

# Seconds per year
SPY = 3600.0 * 24 * 365.0


def get_solver(solver_name: str):
    """Get Diffrax solver instance from name.

    Parameters
    ----------
    solver_name : str
        Solver identifier: 'dopri5', 'kvaerno5', 'tsit5'

    Returns
    -------
    solver : diffrax.AbstractSolver
        Configured solver instance

    Notes
    -----
    - dopri5: Explicit RK method, good for non-stiff
    - kvaerno5: SDIRK method, good for stiff chemistry (recommended)
    - tsit5: Explicit RK method, efficient for moderate stiffness
    """
    solvers = {
        "dopri5": dx.Dopri5,
        "kvaerno5": dx.Kvaerno5,
        "tsit5": dx.Tsit5,
    }

    if solver_name.lower() not in solvers:
        raise ValueError(
            f"Unknown solver: {solver_name}. Available: {list(solvers.keys())}"
        )

    return solvers[solver_name.lower()]()


def solve_network(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    config: SimulationConfig,
) -> dx.Solution:
    """
    Solve chemical network ODE system.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial abundance vector [cm^-3]
    config : SimulationConfig
        Configuration with solver and physical parameters

    Returns
    -------
    solution : diffrax.Solution
        Integration results with:
        - ts: time array [s]
        - ys: abundance array [n_snapshots, n_species]
        - stats: solver statistics

    Notes
    -----
    - Uses logarithmic time sampling for astrophysical timescales
    - Physical parameters passed as args to ODE function
    - JIT compiled for performance (first call compiles)
    - Stiff solver (Kvaerno5) recommended for chemistry
    """
    # Get physical parameters as JAX arrays
    params = config.get_physical_params_jax()

    # Define ODE term
    ode_term = dx.ODETerm(
        lambda t, y, args: jnetwork(
            t,
            y,
            args["temperature"],
            args["cr_rate"],
            args["fuv_field"],
            args["visual_extinction"],
        )
    )

    # Get solver
    solver = get_solver(config.solver)

    # Time sampling (log-spaced in years, converted to seconds)
    t_start_sec = config.t_start * SPY
    t_end_sec = config.t_end * SPY

    # Create log-spaced times with manual 0th timestep
    if config.t_start <= 0:
        # Start from very small value for log spacing (excluding t=0)
        # This captures early chemistry evolution
        t_start_log = -9  # 10^-9 years (~31.5 microseconds)
        t_log = jnp.logspace(
            t_start_log, jnp.log10(config.t_end), config.n_snapshots - 1
        )
        # Prepend t=0 as the 0th timestep
        t_snapshots = jnp.concatenate([jnp.array([0.0]), t_log])
        t_snapshots_sec = t_snapshots * SPY
    else:
        # If t_start > 0, still include it as the 0th timestep
        t_log = jnp.logspace(
            jnp.log10(config.t_start), jnp.log10(config.t_end), config.n_snapshots - 1
        )
        t_snapshots = jnp.concatenate([jnp.array([config.t_start]), t_log])
        t_snapshots_sec = t_snapshots * SPY

    # Solve
    solution = dx.diffeqsolve(
        ode_term,
        solver,
        t0=t_start_sec,
        t1=t_end_sec,
        dt0=1e-6,  # Initial timestep [s]
        y0=y0,
        stepsize_controller=dx.PIDController(
            atol=config.atol,
            rtol=config.rtol,
        ),
        saveat=dx.SaveAt(ts=t_snapshots_sec),
        args=params,
        max_steps=config.max_steps,
    )

    return solution


def compute_derivatives(
    jnetwork: JNetwork,
    solution: dx.Solution,
    config: SimulationConfig,
) -> jnp.ndarray:
    """
    Recompute dy/dt at solution snapshots.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled network
    solution : dx.Solution
        Integration solution
    config : SimulationConfig
        Configuration with physical parameters

    Returns
    -------
    derivatives : jnp.ndarray
        Time derivatives [n_snapshots, n_species]

    Notes
    -----
    Useful for analyzing formation/destruction rates.
    Evaluated at actual solution points (not interpolated).
    """
    params = config.get_physical_params_jax()

    dy = jnp.zeros_like(solution.ys)

    for i, (t, y) in enumerate(zip(solution.ts, solution.ys)):
        dy_i = jnetwork(
            t,
            y,
            params["temperature"],
            params["cr_rate"],
            params["fuv_field"],
            params["visual_extinction"],
        )
        dy = dy.at[i].set(dy_i)

    return dy


def compute_reaction_rates(
    network: eqx.Module,
    jnetwork: JNetwork,
    solution: dx.Solution,
    config: SimulationConfig,
) -> jnp.ndarray:
    """
    Compute reaction rates at solution snapshots.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled network
    solution : dx.Solution
        Integration solution
    config : SimulationConfig
        Configuration with physical parameters

    Returns
    -------
    rates : jnp.ndarray
        Reaction rates [n_snapshots, n_reactions]

    Notes
    -----
    Raw rate coefficients (not multiplied by abundances).
    Units depend on reaction type (typically cm^3/s for bimolecular).
    """
    params = config.get_physical_params_jax()

    n_snapshots = len(solution.ts)
    n_reactions = len(network.reactions)
    rates = jnp.zeros((n_snapshots, n_reactions))

    for i in range(n_snapshots):
        rates_i = jnetwork.get_rates(
            params["temperature"],
            params["cr_rate"],
            params["fuv_field"],
            params["visual_extinction"],
            solution.ys[i],  # Load abundances from solution at snapshot i
        )
        rates = rates.at[i].set(rates_i)

    return rates
