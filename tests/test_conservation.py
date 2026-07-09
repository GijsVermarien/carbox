"""
Elemental conservation test for CSE outflow integrations.

Guards the abundance-convention refactor: whatever the internal state
variable (number densities on dev4, fractional abundances afterwards),
the elemental composition *relative to the total gas density n(t)* must
stay constant along the outflow — chemistry conserves nuclei and the
dilution affects every species identically.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from carbox import SimulationConfig, run_simulation
from carbox.cse_physics import CSEPhysics

SPY = 3600.0 * 24 * 365.0

# Mirror the orich_cse_mini benchmark setup (fast: ~2 s integration)
CSE_PARAMS = dict(
    mdot=1.0e-5,      # M_sun/yr
    vexp=15.0,        # km/s
    t_star=2000.0,    # K
    r_init=1.0e16,    # cm
    r_star=5.0e13,    # cm
    eps=0.7,
)
R_FINAL = 1.1e17  # cm


@pytest.fixture(scope="module")
def cse_solution(tmp_path_factory):
    network_file = PROJECT_ROOT / "data" / "umist22_mini.csv"
    ic_file = PROJECT_ROOT / "benchmarks" / "initial_conditions" / "orich_cse_umist.yaml"
    with open(ic_file) as f:
        initial_abundances = yaml.safe_load(f)["abundances"]

    physics = CSEPhysics(**CSE_PARAMS)
    v_cgs = CSE_PARAMS["vexp"] * 1.0e5
    t_end_yr = (R_FINAL - CSE_PARAMS["r_init"]) / v_cgs / SPY

    initial_n, _, _, _ = physics.get_conditions(t_sec=0.0)

    config = SimulationConfig(
        number_density=float(initial_n),
        temperature=CSE_PARAMS["t_star"],
        cr_rate=1.0,
        fuv_field=1.0,
        t_start=0.0,
        t_end=t_end_yr,
        n_snapshots=32,
        rtol=1.0e-5,
        atol=1.0e-20,
        solver="kvaerno5",
        linear_solver="sparse",
        max_steps=65536,
        output_dir=str(tmp_path_factory.mktemp("cse_conservation")),
        run_name="test_conservation",
        save_abundances=False,
        save_derivatives=False,
        save_rates=False,
        initial_abundances=initial_abundances,
    )
    config.physics_model = physics

    results = run_simulation(str(network_file), config, format_type="umist", verbose=False)
    return results, physics


def test_elemental_conservation_relative_to_gas_density(cse_solution):
    results, physics = cse_solution
    solution = results["solution"]
    network = results["network"]

    elements = ["H", "O", "C"]
    elemental_content = np.asarray(
        network.get_elemental_contents(elements=elements + ["charge"])
    )

    ts = np.asarray(solution.ts)
    ys = np.asarray(solution.ys)

    n_gas = np.array([float(physics.get_conditions(t)[0]) for t in ts])

    # Elemental totals per snapshot, normalized by total gas density
    totals = ys @ elemental_content.T          # [n_snapshots, n_elements+1]
    fractional = totals / n_gas[:, None]

    for i, elem in enumerate(elements):
        x = fractional[:, i]
        assert x[0] > 0, f"initial {elem} abundance should be positive"
        drift = np.abs(x / x[0] - 1.0).max()
        assert drift < 1e-3, (
            f"Element {elem} not conserved relative to n(t): "
            f"max relative drift {drift:.2e}"
        )


def test_abundances_stay_finite_and_nonnegative(cse_solution):
    results, _ = cse_solution
    ys = np.asarray(results["solution"].ys)
    assert np.all(np.isfinite(ys)), "non-finite abundances in solution"
    # tiny negative excursions are solver noise; anything sizable is a bug
    assert ys.min() > -1e-10 * ys.max(), f"significantly negative abundance: {ys.min():.3e}"
