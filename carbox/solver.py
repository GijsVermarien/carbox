"""ODE solver wrapper for chemical kinetics integration.

Wraps Diffrax solvers with appropriate settings for stiff chemistry ODEs.
"""

import diffrax as dx
import jax
import jax.numpy as jnp

from .config import SimulationConfig
from .network import JNetwork, Network

# Seconds per year
SPY = 3600.0 * 24 * 365.0


def get_solver(solver_name: str):
    """Get Diffrax solver instance from name.

    Parameters
    ----------
    solver_name : str
        Solver identifier: 'dopri5', 'kvaerno5', 'tsit5'

    Returns:
    -------
    solver : diffrax.AbstractSolver
        Configured solver instance

    Notes:
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
    """Solve chemical network ODE system.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial abundance vector [cm^-3]
    config : SimulationConfig
        Configuration with solver and physical parameters

    Returns:
    -------
    solution : diffrax.Solution
        Integration results with:
        - ts: time array [s]
        - ys: abundance array [n_snapshots, n_species]
        - stats: solver statistics

    Notes:
    -----
    - Uses logarithmic time sampling for astrophysical timescales
    - Physical parameters passed as args to ODE function
    - JIT compiled for performance (first call compiles)
    - Stiff solver (Kvaerno5) recommended for chemistry
    """
    # Get physical parameters as JAX arrays
    params = config.get_physical_params_jax()

    # Time sampling (log-spaced in years)
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
    else:
        # If t_start > 0, still include it as the 0th timestep
        t_log = jnp.logspace(
            jnp.log10(config.t_start), jnp.log10(config.t_end), config.n_snapshots - 1
        )
        t_snapshots = jnp.concatenate([jnp.array([config.t_start]), t_log])

    return solve_network_core(
        jnetwork=jnetwork,
        y0=y0,
        t_eval=t_snapshots,
        temperature=params["temperature"],
        cr_rate=params["cr_rate"],
        fuv_field=params["fuv_field"],
        visual_extinction=params["visual_extinction"],
        solver_name=config.solver,
        atol=config.atol,
        rtol=config.rtol,
        max_steps=config.max_steps,
    )


def solve_network_core(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    temperature: jnp.ndarray,
    cr_rate: jnp.ndarray,
    fuv_field: jnp.ndarray,
    visual_extinction: jnp.ndarray,
    solver_name: str = "kvaerno5",
    atol: float = 1e-18,
    rtol: float = 1e-12,
    max_steps: int = 4096,
) -> dx.Solution:
    """Core ODE solver with raw JAX array parameters.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial abundance vector [cm^-3]
    t_eval : jnp.ndarray
        Time points for evaluation [years]
    temperature : jnp.ndarray
        Gas temperature [K]
    cr_rate : jnp.ndarray
        Cosmic ray ionization rate [s^-1]
    fuv_field : jnp.ndarray
        FUV radiation field (Draine units)
    visual_extinction : jnp.ndarray
        Visual extinction Av [mag]
    solver_name : str
        Solver name ('dopri5', 'kvaerno5', 'tsit5')
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance
    max_steps : int
        Maximum integration steps

    Returns:
    -------
    solution : diffrax.Solution
        Integration results
    """
    # Convert time to seconds
    t_eval_sec = t_eval * SPY

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
    solver = get_solver(solver_name)

    # Physical parameters
    params = {
        "temperature": temperature,
        "cr_rate": cr_rate,
        "fuv_field": fuv_field,
        "visual_extinction": visual_extinction,
    }

    # Solve
    solution = dx.diffeqsolve(
        ode_term,
        solver,
        t0=t_eval_sec[0],
        t1=t_eval_sec[-1],
        dt0=1e-6,  # Initial timestep [s]
        y0=y0,
        stepsize_controller=dx.PIDController(atol=atol, rtol=rtol),
        saveat=dx.SaveAt(ts=t_eval_sec),
        args=params,
        max_steps=max_steps,
    )

    return solution


def solve_network_batch(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    temperatures: jnp.ndarray,
    cr_rates: jnp.ndarray,
    fuv_fields: jnp.ndarray,
    visual_extinctions: jnp.ndarray,
    solver_name: str = "kvaerno5",
    atol: float = 1e-18,
    rtol: float = 1e-12,
    max_steps: int = 4096,
) -> dx.Solution:
    """Batch solve chemical network ODE system for parameter sweeps.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial abundance vector [cm^-3] (same for all simulations)
    t_eval : jnp.ndarray
        Time points for evaluation [years] (same for all simulations)
    temperatures : jnp.ndarray
        Gas temperatures [K], shape (batch_size,)
    cr_rates : jnp.ndarray
        Cosmic ray ionization rates [s^-1], shape (batch_size,)
    fuv_fields : jnp.ndarray
        FUV radiation fields (Draine units), shape (batch_size,)
    visual_extinctions : jnp.ndarray
        Visual extinctions Av [mag], shape (batch_size,)
    solver_name : str
        Solver name ('dopri5', 'kvaerno5', 'tsit5')
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance
    max_steps : int
        Maximum integration steps

    Returns:
    -------
    solutions : diffrax.Solution
        Batch of integration results, shape (batch_size, ...)
    """
    return jax.vmap(
        lambda temp, cr, fuv, av: solve_network_core(
            jnetwork, y0, t_eval, temp, cr, fuv, av, solver_name, atol, rtol, max_steps
        ),
        in_axes=(0, 0, 0, 0),
    )(temperatures, cr_rates, fuv_fields, visual_extinctions)


def compute_derivatives(
    jnetwork: JNetwork,
    solution: dx.Solution,
    config: SimulationConfig,
) -> jnp.ndarray:
    """Recompute dy/dt at solution snapshots.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled network
    solution : dx.Solution
        Integration solution
    config : SimulationConfig
        Configuration with physical parameters

    Returns:
    -------
    derivatives : jnp.ndarray
        Time derivatives [n_snapshots, n_species]

    Notes:
    -----
    Useful for analyzing formation/destruction rates.
    Evaluated at actual solution points (not interpolated).
    """
    if not (solution.ys and solution.ts):
        raise Exception("Missing solution.ys or solution.ts.")

    params = config.get_physical_params_jax()

    dy = jnp.zeros_like(solution.ys)

    for i, (t, y) in enumerate(zip(solution.ts, solution.ys, strict=False)):
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
    network: Network,
    jnetwork: JNetwork,
    solution: dx.Solution,
    config: SimulationConfig,
) -> jnp.ndarray:
    """Compute reaction rates at solution snapshots.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled network
    solution : dx.Solution
        Integration solution
    config : SimulationConfig
        Configuration with physical parameters

    Returns:
    -------
    rates : jnp.ndarray
        Reaction rates [n_snapshots, n_reactions]

    Notes:
    -----
    Raw rate coefficients (not multiplied by abundances).
    Units depend on reaction type (typically cm^3/s for bimolecular).
    """
    if not (solution.ys and solution.ts):
        raise Exception("Missing solution.ys or solution.ts.")

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
