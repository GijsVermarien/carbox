"""
Unit tests for the individual cooling channels in carbox/cooling.py.

These test each channel in isolation (no network/parser/solver involved),
complementing the end-to-end thermal-balance tests in
test_thermal_balance.py.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from carbox.cooling import cooling_h2, cooling_lyalpha, cooling_oi
from carbox.index import Idx

# A minimal Idx covering every species the cooling channels look up.
IDX = Idx({"H": 0, "E": 1, "O": 2, "H2": 3})


def _abundances(h=0.0, e=0.0, o=0.0, h2=0.0):
    return jnp.array([h, e, o, h2])


class TestCoolingLyAlpha:
    def test_zero_when_no_hydrogen_or_electrons(self):
        assert cooling_lyalpha(_abundances(h=0.0, e=1e2), 1e4, IDX) == 0.0
        assert cooling_lyalpha(_abundances(h=1e2, e=0.0), 1e4, IDX) == 0.0

    def test_positive_for_positive_inputs(self):
        rate = cooling_lyalpha(_abundances(h=1e2, e=1e2), 1e4, IDX)
        assert rate > 0

    def test_scales_linearly_with_each_reactant(self):
        base = cooling_lyalpha(_abundances(h=1e2, e=1e2), 8e3, IDX)
        doubled_h = cooling_lyalpha(_abundances(h=2e2, e=1e2), 8e3, IDX)
        doubled_e = cooling_lyalpha(_abundances(h=1e2, e=2e2), 8e3, IDX)
        assert doubled_h == pytest.approx(2 * base)
        assert doubled_e == pytest.approx(2 * base)

    def test_increases_with_temperature(self):
        """Ly-alpha cooling switches on at high T (10.2 eV excitation gap)."""
        low = cooling_lyalpha(_abundances(h=1e2, e=1e2), 3e3, IDX)
        high = cooling_lyalpha(_abundances(h=1e2, e=1e2), 3e4, IDX)
        assert high > low

    def test_negligible_at_low_temperature(self):
        rate = cooling_lyalpha(_abundances(h=1e4, e=1e4), 10.0, IDX)
        assert rate < 1e-100


class TestCoolingOI:
    def test_zero_when_no_oxygen_or_electrons(self):
        assert cooling_oi(_abundances(o=0.0, e=1e2), 1e3, IDX) == 0.0
        assert cooling_oi(_abundances(o=1e2, e=0.0), 1e3, IDX) == 0.0

    def test_positive_for_positive_inputs(self):
        rate = cooling_oi(_abundances(o=1e2, e=1e2), 1e3, IDX)
        assert rate > 0

    def test_increases_with_temperature(self):
        low = cooling_oi(_abundances(o=1e2, e=1e2), 3e2, IDX)
        high = cooling_oi(_abundances(o=1e2, e=1e2), 3e3, IDX)
        assert high > low

    def test_switches_on_at_lower_temperature_than_lyalpha(self):
        """OI's excitation gap (2 eV) is much smaller than Ly-alpha's (10.2 eV),
        so at a fixed moderate T, OI cooling should be relatively more
        significant than Ly-alpha (both given the same reactant densities)."""
        t = 2e3
        x = _abundances(h=1e2, e=1e2, o=1e2)
        oi = cooling_oi(x, t, IDX)
        lyalpha = cooling_lyalpha(x, t, IDX)
        assert oi > lyalpha


class TestCoolingH2:
    def test_zero_when_no_h2(self):
        assert cooling_h2(_abundances(h=1e2, h2=0.0), 500.0, IDX) == 0.0

    def test_zero_when_no_atomic_hydrogen(self):
        """LDL (H-collision) branch vanishes without H, but HDL should still
        allow a nonzero rate at higher densities/temperatures; here we only
        assert the function stays finite and non-negative."""
        rate = cooling_h2(_abundances(h=0.0, h2=1e2), 500.0, IDX)
        assert np.isfinite(rate)
        assert rate >= 0

    @pytest.mark.parametrize(
        "temperature",
        [10.0, 50.0, 1e2, 5e2, 1e3, 2e3, 5e3, 6e3, 8e3, 1e4, 2e4],
    )
    def test_finite_and_nonnegative_across_all_branches(self, temperature):
        """Exercises every jax.lax.cond branch boundary in both the HDL
        (split at 2e3, 1e4) and LDL (split at 1e2, 1e3, 6e3) piecewise fits."""
        rate = cooling_h2(_abundances(h=1e2, h2=1e2), temperature, IDX)
        assert np.isfinite(rate)
        assert rate >= 0

    def test_scales_linearly_with_h2_abundance(self):
        base = cooling_h2(_abundances(h=1e2, h2=1e2), 1e3, IDX)
        doubled = cooling_h2(_abundances(h=1e2, h2=2e2), 1e3, IDX)
        assert doubled == pytest.approx(2 * base)
