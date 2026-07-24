"""
Unit tests for carbox/thermo.py (ThermoRate / ThermoRateTerm), isolated from
the parser/solver -- see test_thermal_balance.py for the end-to-end path via
run_simulation, and test_cooling.py for the individual cooling channels.
"""

import jax.numpy as jnp
import pytest

from carbox.cooling import cooling_h2, cooling_lyalpha, cooling_oi
from carbox.index import Idx
from carbox.thermo import KBOLTZMANN_ERG, ThermoRate, ThermoRateTerm

IDX = Idx({"H": 0, "E": 1, "O": 2, "H2": 3, "H2O+": 4})


def test_molecularity_is_forced_to_one_not_zero():
    """ThermoRate has no reactants (len(reactants) == 0), which would
    naturally give molecularity 0; it must be 1 so JNetwork's
    density**(molecularity - 1) rescaling of this already-absolute dT/dt
    term is a no-op."""
    reaction = ThermoRate("THERMO", [], ["TGAS"])
    assert reaction.molecularity == 1


def test_factory_requires_idx():
    reaction = ThermoRate("THERMO", [], ["TGAS"])
    with pytest.raises(ValueError):
        reaction._reaction_rate_factory(idx=None)


def test_factory_returns_bound_rate_term():
    reaction = ThermoRate("THERMO", [], ["TGAS"])
    term = reaction._reaction_rate_factory(IDX)
    assert isinstance(term, ThermoRateTerm)
    assert term.idx is IDX


def test_rate_matches_sum_of_cooling_channels():
    x = jnp.array([1e2, 1e2, 1e2, 1e2, 1e-4])
    tgas = 5e3

    term = ThermoRateTerm(IDX)
    rate = term(tgas, cr_rate=1.0, uv_field=1.0, visual_extinction=1.0, abundance_vector=x)

    expected_cooling = (
        cooling_lyalpha(x, tgas, IDX) + cooling_oi(x, tgas, IDX) + cooling_h2(x, tgas, IDX)
    )
    expected_rate = -expected_cooling / (KBOLTZMANN_ERG * jnp.sum(x))
    assert rate == pytest.approx(float(expected_rate))


def test_rate_is_nonpositive_cooling_only():
    """No heating channel exists yet (see module docstring), so dT/dt <= 0."""
    x = jnp.array([1e2, 1e2, 1e2, 1e2, 1e-4])
    term = ThermoRateTerm(IDX)
    rate = term(3e3, cr_rate=1.0, uv_field=1.0, visual_extinction=1.0, abundance_vector=x)
    assert rate <= 0


def test_rate_is_zero_when_cooling_reactants_absent():
    """n_tot != 0 (H2O+ present) but H/E/O/H2 are all zero, so every cooling
    channel is zero and the rate should be exactly zero, not NaN."""
    x = jnp.array([0.0, 0.0, 0.0, 0.0, 1e2])
    term = ThermoRateTerm(IDX)
    rate = term(1e3, cr_rate=1.0, uv_field=1.0, visual_extinction=1.0, abundance_vector=x)
    assert rate == 0.0
