#!/usr/bin/env python3
"""
Run Carbox for a circumstellar-envelope (CSE) outflow.

Dedicated runner for CSEPhysics (constant-velocity spherical expansion),
as opposed to the static-cloud networks in ../run.py. Physical conditions
(density, temperature, Av, radius) are always written out alongside the
abundances -- see save_abundances() in carbox/output.py, which adds
number_density/temperature/visual_extinction/radius_cm/cr_rate/fuv_field
columns to every row of the `_abundances.csv` file. This script also
enables save_derivatives and save_rates by default, so a single run
produces:

    {run_name}_abundances.csv    species fractions + physics, per snapshot
    {run_name}_derivatives.csv   dy/dt per species + physics, per snapshot
    {run_name}_rates.csv         per-reaction rate coefficients, per snapshot
    {run_name}_metadata.json     config + network + solver stats
    {run_name}_summary.txt       human-readable summary

Example:
    python run_cse.py --network umist
    python run_cse.py --network umist_mini --mdot 5e-6 --vexp 10
    python run_cse.py --network uclchem --r-final 5e17 --n-snapshots 200

For sweeping multiple parameter combinations, use run_grid.py instead of
editing DEFAULT_PHYSICS below -- see that file's docstring.
"""

import argparse
import math
import sys
from pathlib import Path

import jax
import yaml

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.physics import CSEPhysics
from carbox.solver import SPY

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)


# Network choices for this CSE runner. Each maps to a reaction-network file,
# its format, and the O-rich parent-species initial abundances to use.
CSE_NETWORKS = {
    "umist": {
        "description": "Full UMIST22 network, O-rich CSE",
        "input_file": "../../data/umist22.csv",
        "input_format": "umist",
        "initial_conditions": "../initial_conditions/orich_cse_umist.yaml",
    },
    "umist_mini": {
        "description": "Small UMIST22 subset, O-rich CSE",
        "input_file": "../../data/umist22_mini.csv",
        "input_format": "umist",
        "initial_conditions": "../initial_conditions/orich_cse_umist_mini.yaml",
    },
    "uclchem": {
        "description": "UCLCHEM gas-phase-only network, O-rich CSE",
        "input_file": "../../data/uclchem_gas_phase_only.csv",
        "input_format": "uclchem",
        "initial_conditions": "../initial_conditions/orich_cse_uclchem.yaml",
    },
}

# Default step size in dex/step (log10 radius per snapshot), matching the
# Fortran model's RESOLUTION convention -- this is the standard way to
# control snapshot density; --n-snapshots is only used if resolution is
# explicitly disabled (--resolution none).
DEFAULT_RESOLUTION = 0.05

# Default CSE outflow parameters (O-rich AGB wind), matching benchmarks/run.py
DEFAULT_PHYSICS = {
    "mdot": 1.0e-5,   # Msun/yr
    "vexp": 15.0,     # km/s
    "t_star": 2000.0, # K, at r_star
    "r_init": 1.0e14, # cm
    "r_final": 1e18,# cm (used only to derive t_end)
    "r_star": 5.0e13, # cm
    "eps": 0.7,       # temperature power-law index
}


def _parse_resolution(value: str):
    """CLI type for --resolution: a float, or 'none'/'off' to disable it
    and fall back to --n-snapshots instead."""
    if value.strip().lower() in ("none", "off"):
        return None
    return float(value)


def resolve_t_end_years(physics: CSEPhysics, r_final: float) -> float:
    """Years to expand from r_init to r_final at constant vexp."""
    v_cgs = physics.vexp * 1.0e5
    t_end_sec = (r_final - physics.r_init) / v_cgs
    return t_end_sec / SPY


def prepare_cse_run(
    network: str = "umist",
    output: str = "results",
    run_name: str = None,
    mdot: float = DEFAULT_PHYSICS["mdot"],
    vexp: float = DEFAULT_PHYSICS["vexp"],
    t_star: float = DEFAULT_PHYSICS["t_star"],
    r_init: float = DEFAULT_PHYSICS["r_init"],
    r_final: float = DEFAULT_PHYSICS["r_final"],
    r_star: float = DEFAULT_PHYSICS["r_star"],
    eps: float = DEFAULT_PHYSICS["eps"],
    cr_rate: float = 1.0,
    fuv_field: float = 1.0,
    n_snapshots: int = 100,
    resolution: float = DEFAULT_RESOLUTION,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-20,
    solver_name: str = "kvaerno5",
    linear_solver: str = "sparse",
    max_steps: int = 65536,
    save_derivatives: bool = True,
    save_rates: bool = True,
    show_progress: bool = False,
    verbose: bool = True,
) -> tuple:
    """
    Resolve the network file and build the SimulationConfig for a CSE run,
    without actually solving anything. Shared by run_cse() (which then calls
    run_simulation()) and run_sensitivity.py (which instead differentiates
    through solve_network() directly).

    Returns (input_file, format_type, run_name, output_dir, config).
    """
    net_cfg = CSE_NETWORKS[network]
    run_name = run_name or f"cse_{network}"

    input_file = (SCRIPT_DIR / net_cfg["input_file"]).resolve()
    ic_file = (SCRIPT_DIR / net_cfg["initial_conditions"]).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Network file not found: {input_file}")
    if not ic_file.exists():
        raise FileNotFoundError(f"Initial conditions file not found: {ic_file}")

    with open(ic_file, "r") as f:
        initial_abundances = yaml.safe_load(f)["abundances"]

    physics = CSEPhysics(
        mdot=mdot,
        vexp=vexp,
        t_star=t_star,
        r_init=r_init,
        r_star=r_star,
        eps=eps,
    )
    t_end_yr = resolve_t_end_years(physics, r_final)

    n0, T0, av0, _ = physics.get_conditions(t_sec=0.0)

    # `resolution` mirrors the Fortran model's step convention (each step
    # multiplies the radius by 10**resolution, i.e. resolution is dex/step)
    # rather than a raw snapshot count -- derive n_snapshots from it so the
    # log-spaced radius grid (see CSEPhysics.time_grid) has exactly that
    # spacing. Overrides n_snapshots when given.
    if resolution is not None:
        if resolution <= 0:
            raise ValueError(f"resolution must be positive (dex/step), got {resolution}")
        log_span = math.log10(r_final) - math.log10(r_init)
        if log_span <= 0:
            raise ValueError(
                f"r_final ({r_final:.2e}) must be greater than r_init ({r_init:.2e}) to use resolution"
            )
        n_snapshots = int(round(log_span / resolution)) + 1
        if verbose:
            print(f"resolution={resolution:.3g} dex/step over {log_span:.3f} dex "
                  f"-> n_snapshots={n_snapshots}")

    if verbose:
        print("=" * 70)
        print(f"Carbox CSE run: {network} ({run_name})")
        print("=" * 70)
        print(f"Network file: {input_file}")
        print(f"Initial conditions: {ic_file}")
        print(f"CSE outflow: mdot={mdot:.2e} Msun/yr, vexp={vexp:.1f} km/s, "
              f"r_init={r_init:.2e} cm -> r_final={r_final:.2e} cm")
        print(f"Initial conditions at r_init: n={float(n0):.3e} cm^-3, "
              f"T={float(T0):.1f} K, Av={float(av0):.2f} mag")
        print(f"Integration time: 0 -> {t_end_yr:.3e} years ({n_snapshots} snapshots)")
        print()

    # Resolve relative to this script's directory (not the caller's cwd), so
    # output always lands in a predictable place regardless of where/how
    # this is invoked -- pass an absolute path to override.
    output_dir = Path(output)
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        cr_rate=cr_rate,
        fuv_field=fuv_field,
        t_start=0.0,
        t_end=t_end_yr,
        n_snapshots=n_snapshots,
        rtol=rtol,
        atol=atol,
        solver=solver_name,
        linear_solver=linear_solver,
        max_steps=max_steps,
        output_dir=str(output_dir),
        run_name=run_name,
        save_abundances=True,       # always includes physics columns (n, T, Av, r)
        save_derivatives=save_derivatives,
        save_rates=save_rates,
        show_progress=show_progress,
        initial_abundances=initial_abundances,
        physics_model=physics,
    )

    return input_file, net_cfg["input_format"], run_name, output_dir, config


def run_cse(
    network: str = "umist",
    output: str = "results",
    run_name: str = None,
    mdot: float = DEFAULT_PHYSICS["mdot"],
    vexp: float = DEFAULT_PHYSICS["vexp"],
    t_star: float = DEFAULT_PHYSICS["t_star"],
    r_init: float = DEFAULT_PHYSICS["r_init"],
    r_final: float = DEFAULT_PHYSICS["r_final"],
    r_star: float = DEFAULT_PHYSICS["r_star"],
    eps: float = DEFAULT_PHYSICS["eps"],
    cr_rate: float = 1.0,
    fuv_field: float = 1.0,
    n_snapshots: int = 100,
    resolution: float = DEFAULT_RESOLUTION,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-20,
    solver_name: str = "kvaerno5",
    linear_solver: str = "sparse",
    max_steps: int = 65536,
    no_derivatives: bool = False,
    no_rates: bool = False,
    show_progress: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run one CSE simulation. Callable directly (e.g. from run_grid.py) so a
    parameter sweep can loop over this function in-process, instead of
    editing DEFAULT_PHYSICS or shelling out to this script per grid point.
    """
    input_file, format_type, run_name, output_dir, config = prepare_cse_run(
        network=network, output=output, run_name=run_name,
        mdot=mdot, vexp=vexp, t_star=t_star, r_init=r_init, r_final=r_final,
        r_star=r_star, eps=eps, cr_rate=cr_rate, fuv_field=fuv_field,
        n_snapshots=n_snapshots, resolution=resolution, rtol=rtol, atol=atol,
        solver_name=solver_name, linear_solver=linear_solver, max_steps=max_steps,
        save_derivatives=not no_derivatives, save_rates=not no_rates,
        show_progress=show_progress, verbose=verbose,
    )

    results = run_simulation(
        network_file=str(input_file),
        config=config,
        format_type=format_type,
        verbose=verbose,
    )

    if verbose:
        print()
        print("Outputs (physics columns -- number_density, temperature,")
        print("visual_extinction, radius_cm -- are included in every row):")
        print(f"  {output_dir / f'{run_name}_abundances.csv'}")
        if not no_derivatives:
            print(f"  {output_dir / f'{run_name}_derivatives.csv'}")
        if not no_rates:
            print(f"  {output_dir / f'{run_name}_rates.csv'}")
        print(f"  {output_dir / f'{run_name}_metadata.json'}")
        print(f"  {output_dir / f'{run_name}_summary.txt'}")

    results["output_dir"] = output_dir
    results["run_name"] = run_name
    return results


def add_common_cse_args(parser: argparse.ArgumentParser) -> None:
    """Add the --network/physics/radiation/solver args shared by run_cse.py
    and run_sensitivity.py, so both scripts take the same physical setup."""
    parser.add_argument("--network", default="umist", choices=list(CSE_NETWORKS.keys()))

    phys = parser.add_argument_group("CSE outflow physics")
    phys.add_argument("--mdot", type=float, default=DEFAULT_PHYSICS["mdot"], help="Mass-loss rate [Msun/yr]")
    phys.add_argument("--vexp", type=float, default=DEFAULT_PHYSICS["vexp"], help="Expansion velocity [km/s]")
    phys.add_argument("--t-star", type=float, default=DEFAULT_PHYSICS["t_star"], help="Temperature at r_star [K]")
    phys.add_argument("--r-init", type=float, default=DEFAULT_PHYSICS["r_init"], help="Initial radius [cm]")
    phys.add_argument("--r-final", type=float, default=DEFAULT_PHYSICS["r_final"], help="Final radius [cm] (sets t_end)")
    phys.add_argument("--r-star", type=float, default=DEFAULT_PHYSICS["r_star"], help="Stellar radius, T-profile normalization [cm]")
    phys.add_argument("--eps", type=float, default=DEFAULT_PHYSICS["eps"], help="Temperature power-law exponent")

    env = parser.add_argument_group("Radiation environment")
    env.add_argument("--cr-rate", type=float, default=1.0, help="Cosmic-ray ionization rate, relative to standard")
    env.add_argument("--fuv-field", type=float, default=1.0, help="FUV field [Draine units]")

    solver = parser.add_argument_group("Solver")
    solver.add_argument("--n-snapshots", type=int, default=100,
                         help="Number of saved radii/times; only used if --resolution is "
                              "disabled (--resolution none)")
    solver.add_argument("--resolution", type=_parse_resolution, default=DEFAULT_RESOLUTION,
                         help="Fortran-model-style step size in dex/step (log10 radius per step); "
                              "derives n_snapshots from log10(r_final/r_init)/resolution + 1, "
                              f"overriding --n-snapshots (default: {DEFAULT_RESOLUTION}). "
                              "Pass 'none' to fall back to --n-snapshots instead.")
    solver.add_argument("--rtol", type=float, default=1.0e-5)
    solver.add_argument("--atol", type=float, default=1.0e-20)
    solver.add_argument("--solver-name", dest="solver_name", default="kvaerno5")
    solver.add_argument("--linear-solver", default="sparse", choices=["lu", "sparse"])
    solver.add_argument("--max-steps", type=int, default=65536)
    solver.add_argument("--show-progress", action="store_true",
                         help="Print the current radius to the terminal as the solve runs")


def main():
    parser = argparse.ArgumentParser(
        description="Run Carbox for a CSE (circumstellar envelope) outflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available networks:\n"
        + "\n".join(f"  {name:<12} - {cfg['description']}" for name, cfg in CSE_NETWORKS.items()),
    )

    parser.add_argument("--output", default="results",
                         help="Output directory (relative paths resolve against this script's "
                              "directory, not your shell's cwd; pass an absolute path to override)")
    parser.add_argument("--run-name", default=None, help="Overrides the default run name")
    add_common_cse_args(parser)

    # Output toggles -- physics is always included in the abundances CSV;
    # these control the extra derivatives/rates diagnostics.
    out = parser.add_argument_group("Output")
    out.add_argument("--no-derivatives", action="store_true", help="Skip writing dy/dt CSV")
    out.add_argument("--no-rates", action="store_true", help="Skip writing per-reaction rates CSV")

    args = parser.parse_args()
    run_cse(**vars(args))


if __name__ == "__main__":
    main()
