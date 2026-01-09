"""
ODE solver wrapper for chemical kinetics integration.

Wraps Diffrax solvers with appropriate settings for stiff chemistry ODEs.
"""

from typing import Tuple
import time

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp

from .config import SimulationConfig
from .network import JNetwork

import lineax as lx  # Add this import
import optimistix as optx

# Seconds per year
SPY = 3600.0 * 24 * 365.0

def get_solver(config: SimulationConfig):
    # Standardize names
    s_name = getattr(config, "solver", "kvaerno5").lower()
    ls_name = getattr(config, "linear_solver", "lu").lower()

    if s_name in ["kvaerno5", "kvaerno3"]:
        # Select linear solver
        if ls_name == "sparse":
            l_solver = lx.AutoLinearSolver(well_posed=True)
        elif ls_name == "qr":
            l_solver = lx.QR()
        else:
            l_solver = lx.LU()

        # Build the 'brain'
        root_finder = optx.Newton(
            rtol=config.rtol, 
            atol=config.atol, 
            linear_solver=l_solver
        )
        
        print(f"--- JIT Init: Implicit {s_name} with {ls_name} ---")
        return dx.Kvaerno5(root_finder=root_finder) if s_name == "kvaerno5" else dx.Kvaerno3(root_finder=root_finder)

    elif s_name in ["dopri5", "tsit5"]:
        print(f"--- JIT Init: Explicit {s_name} ---")
        return dx.Dopri5() if s_name == "dopri5" else dx.Tsit5()
    
    raise ValueError(f"Unknown solver: {s_name}")

    
# def get_solver(solver_name: str):
#     """Get Diffrax solver instance from name.

#     Parameters
#     ----------
#     solver_name : str
#         Solver identifier: 'dopri5', 'kvaerno5', 'tsit5'

#     Returns
#     -------
#     solver : diffrax.AbstractSolver
#         Configured solver instance

#     Notes
#     -----
#     - dopri5: Explicit RK method, good for non-stiff
#     - kvaerno5: SDIRK method, good for stiff chemistry (recommended)
#     - tsit5: Explicit RK method, efficient for moderate stiffness
#     """
#     if solver_name.lower() == "kvaerno5":
#         return dx.Kvaerno5()
#     solvers = {
#         "dopri5": dx.Dopri5,
#         "tsit5": dx.Tsit5,
#     }

#     if solver_name.lower() not in solvers:
#         raise ValueError(
#             f"Unknown solver: {solver_name}. Available: {list(solvers.keys())}"
#         )

#     return solvers[solver_name.lower()]()


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
    # # Get physical parameters as JAX arrays
    # params = config.get_physical_params_jax()
    # Use the physics model from config (assumed to be present for outflow models)
    physics = config.physics_model
    if physics is None:
        raise ValueError("config.physics_model must be set for the solver.")
    
    # Define ODE term
    def _ode_func(t, y, args):
        physics = args
        # Get dynamic physical conditions from the physics model
        n, T, av, r = physics.get_conditions(t)

        # Chemical source/sink term (jnetwork captured from closure)
        dy_chem = jnetwork(t, y, T, config.cr_rate, config.fuv_field, av)

        # Dilution term due to spherical expansion: d(n_i)/dt = -2 * (v/r) * n_i
        v_cgs = physics.vexp * physics.KM_CM
        dilution = -2 * (v_cgs / r) * y
        return dy_chem + dilution

    ode_term = dx.ODETerm(_ode_func)


    # Time sampling (log-spaced in years, converted to seconds)
    t_start_sec = config.t_start * SPY
    t_end_sec = config.t_end * SPY

    # # Create log-spaced times with manual 0th timestep
    # if config.t_start <= 0:
    #     # Start from very small value for log spacing (excluding t=0)
    #     # This captures early chemistry evolution
    #     t_start_log = -9  # 10^-9 years (~31.5 microseconds)
    #     t_log = jnp.logspace(
    #         t_start_log, jnp.log10(config.t_end), config.n_snapshots - 1
    #     )
    #     # Prepend t=0 as the 0th timestep
    #     t_snapshots = jnp.concatenate([jnp.array([0.0]), t_log])
    #     t_snapshots_sec = t_snapshots * SPY
    # else:
    #     # If t_start > 0, still include it as the 0th timestep
    #     t_log = jnp.logspace(
    #         jnp.log10(config.t_start), jnp.log10(config.t_end), config.n_snapshots - 1
    #     )
    #     t_snapshots = jnp.concatenate([jnp.array([config.t_start]), t_log])
    #     t_snapshots_sec = t_snapshots * SPY

    # Create log-spaced times
    if config.t_start <= 0:
        # For t_start=0, use a very small starting time for log spacing
        t_start_for_log = 1e-9  # years
        t_snapshots = jnp.logspace(
            jnp.log10(t_start_for_log), 
            jnp.log10(config.t_end), 
            config.n_snapshots
        )
        t_snapshots_sec = t_snapshots * SPY
        # NOW FIX: Overwrite first element to be exactly t_start_sec
        t_snapshots_sec = t_snapshots_sec.at[0].set(t_start_sec)
    else:
        # Normal log spacing
        t_snapshots = jnp.logspace(
            jnp.log10(config.t_start), 
            jnp.log10(config.t_end), 
            config.n_snapshots
        )
        t_snapshots_sec = t_snapshots * SPY
    
    # Ensure last point is exactly at t_end (avoid floating point error)
    t_snapshots_sec = t_snapshots_sec.at[-1].set(t_end_sec)

    # print(f"Debug time array:")
    # print(f"  config.t_start: {config.t_start}")
    # print(f"  config.t_end: {config.t_end}")
    # print(f"  t_start_sec: {t_start_sec}")
    # print(f"  t_end_sec: {t_end_sec}")
    # print(f"  t_snapshots_sec[0]: {t_snapshots_sec[0]}")
    # print(f"  t_snapshots_sec[-1]: {t_snapshots_sec[-1]}")
    # print(f"  min(t_snapshots_sec): {jnp.min(t_snapshots_sec)}")
    # print(f"  max(t_snapshots_sec): {jnp.max(t_snapshots_sec)}")
    # print(f"  First few values: {t_snapshots_sec[:3]}")
    # print(f"  Last few values: {t_snapshots_sec[-3:]}")


    # Get solver
    solver = get_solver(config)

    # root_finder = optx.Newton(rtol=config.rtol, 
    #                           atol=config.atol, 
    #                           linear_solver=lx.AutoLinearSolver())
    
    # Solve (JIT compiled for performance)
    @eqx.filter_jit
    def _solve(t0, t1, y0, args, saveat_ts):
        # my_kvaerno = dx.Kvaerno5(root_finder=root_finder)
        return dx.diffeqsolve(
            ode_term,
            solver,
            t0=t0,
            t1=t1,
            dt0=1e-6,  # Initial timestep [s]
            y0=y0,
            stepsize_controller=dx.PIDController(
                atol=config.atol,
                rtol=config.rtol,
            ),
            saveat=dx.SaveAt(ts=saveat_ts),
            args=args,
            max_steps=config.max_steps,
        )
    
    solution = _solve(t_start_sec, t_end_sec, y0, physics, t_snapshots_sec)

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
    # params = config.get_physical_params_jax()
    physics = config.physics_model

    def _compute_single(t, y):
        # Get dynamic physical conditions from the physics model
        n, T, av, r = physics.get_conditions(t)

        # Chemical source/sink term
        dy_chem = jnetwork(t, y, T, config.cr_rate, config.fuv_field, av)
        v_cgs = physics.vexp * physics.KM_CM
        dilution = -2 * (v_cgs / r) * y
        return dy_chem + dilution

    # Vectorize over time and state
    return jax.vmap(_compute_single)(solution.ts, solution.ys)


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
    physics = config.physics_model

    def _compute_single(t, y):
        n, T, av, r = physics.get_conditions(t)
        return jnetwork.get_rates(T, config.cr_rate, config.fuv_field, av, y)

    # Vectorize over time and state
    return jax.vmap(_compute_single)(solution.ts, solution.ys)
