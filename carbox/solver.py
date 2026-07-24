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


def solve_network(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    config: SimulationConfig,
    rate_modifiers: Tuple[float] = None,
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
    solution : dx.Solution
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
    # Physics model is always present (SimulationConfig builds a
    # StaticCloudPhysics from the legacy scalar fields if none is given)
    physics = config.physics_model

    # Rate modifiers (a*rate + b) are baked into the network so every
    # downstream consumer of `jnetwork` (derivatives, rates) stays consistent
    # with what was actually integrated, without recompiling the network.
    if rate_modifiers is not None:
        rate_modifier_a, rate_modifier_b = rate_modifiers
        jnetwork = jnetwork.with_rate_modifiers(rate_modifier_a, rate_modifier_b)

    # Define ODE term
    def _ode_func(t, y, args):
        physics = args
        # Get dynamic physical conditions from the physics model
        n, T, av, r = physics.get_conditions(t)

        # For thermal-balance networks (see thermo.py's ThermoRate), T is
        # integrated alongside abundances instead of prescribed by the
        # physics model. `integrates_temperature` is a static Python bool,
        # so this branch costs nothing when False (the default).
        if getattr(physics, "integrates_temperature", False):
            T = y[jnetwork.idx.TGAS]

        # Chemical source/sink term (jnetwork captured from closure)
        dy_chem = jnetwork(t, y, T, n, config.cr_rate, config.fuv_field, av)

        # Non-chemical term (e.g. expansion dilution; zero for static clouds)
        return dy_chem + physics.dilution(t, y)

    ode_term = dx.ODETerm(_ode_func)

    # Get time grid
    t_snapshots_sec = get_time_grid(config)
    t_start_sec = t_snapshots_sec[0]
    t_end_sec = t_snapshots_sec[-1]

    # Get solver
    solver = get_solver(config)

    # Solve (JIT compiled for performance)
    @eqx.filter_jit
    def _solve(t0, t1, y0, args, saveat_ts):
        # my_kvaerno = dx.Kvaerno5(root_finder=root_finder)
        return dx.diffeqsolve(
            ode_term,
            solver,
            t0=t0,
            t1=t1,
            dt0=1e-4,  # Explicit small initial step to avoid estimator overhead
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



def get_time_grid(config: SimulationConfig) -> jnp.ndarray:
    """Generate the time grid (in seconds) based on configuration."""
    t_start_sec = config.t_start * SPY
    t_end_sec = config.t_end * SPY

    # A physics model may provide its own snapshot grid (e.g. CSE spaces
    # snapshots log-uniformly in radius to avoid clustering at t=0)
    custom_grid = config.physics_model.time_grid(config)
    if custom_grid is not None:
        return custom_grid

    # Time sampling (log-spaced in years, converted to seconds)
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
    return t_snapshots_sec


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
        Compiled network. Any rate modifiers baked into it (see
        `JNetwork.with_rate_modifiers`) are applied automatically, so pass
        the same `jnetwork` that was used to produce `solution`.
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
    physics = config.physics_model

    def _compute_single(t, y):
        # Get dynamic physical conditions from the physics model
        n, T, av, r = physics.get_conditions(t)

        # See solve_network's _ode_func for why this branch is free when
        # integrates_temperature is False (the default).
        if getattr(physics, "integrates_temperature", False):
            T = y[jnetwork.idx.TGAS]

        # Chemical source/sink term (jnetwork's rate modifiers, if any, apply
        # automatically since they're baked into the network itself)
        dy_chem = jnetwork(t, y, T, n, config.cr_rate, config.fuv_field, av)
        return dy_chem + physics.dilution(t, y)

    # Vectorize over time and state
    @eqx.filter_jit
    def _compute_batch(ts, ys):
        return jax.vmap(_compute_single)(ts, ys)

    return _compute_batch(solution.ts, solution.ys)


def compute_reaction_rates(
    jnetwork: JNetwork,
    solution: dx.Solution,
    config: SimulationConfig,
    use_rate_modifiers: bool = True,
) -> jnp.ndarray:
    """
    Compute reaction rate coefficients at solution snapshots.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled network
    solution : dx.Solution
        Integration solution
    config : SimulationConfig
        Configuration with physical parameters
    use_rate_modifiers : bool, default True
        If True (default), apply `jnetwork`'s rate modifiers
        (`rate_modifier_a`/`rate_modifier_b`, see `with_rate_modifiers`) —
        the coefficients as actually used in the integration. If False,
        return the unmodified coefficients straight from the reaction
        parameterizations, ignoring any modifiers.

    Returns
    -------
    rates : jnp.ndarray
        Reaction rate coefficients [n_snapshots, n_reactions], not
        multiplied by reactant abundances.
        Units depend on reaction type (typically cm^3/s for bimolecular).
    """
    physics = config.physics_model

    def _compute_single(t, y):
        n, T, av, r = physics.get_conditions(t)
        if getattr(physics, "integrates_temperature", False):
            T = y[jnetwork.idx.TGAS]
        # Convert fractional abundances to number densities before passing to get_rates
        abundances_num_density = y * n
        rates = jnetwork.get_rates(T, config.cr_rate, config.fuv_field, av, abundances_num_density)
        return jnetwork.modify_rates(rates) if use_rate_modifiers else rates

    # Vectorize over time and state
    @eqx.filter_jit
    def _compute_batch(ts, ys):
        return jax.vmap(_compute_single)(ts, ys)

    return _compute_batch(solution.ts, solution.ys)
