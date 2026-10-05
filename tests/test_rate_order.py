"""Vectorised rate assembly must keep every rate on its own reaction.

The latent_tgas network file interleaves mechanisms (KA, CR, KA, H2Form, CR, FUV), so a rate vector stacked group by
group is not in the column order of the incidence matrix; JNetwork.get_rates must undo the grouping.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from carbox.network import Network
from carbox.parsers import parse_chemical_network

jax.config.update("jax_enable_x64", True)
NETWORK_FILE = Path(__file__).resolve().parents[1] / "data" / "simple_latent_tgas.csv"


@pytest.fixture(scope="module")
def parsed():
    return parse_chemical_network(str(NETWORK_FILE), "latent_tgas")


def _rhs_and_rates(parsed, vectorize, abundances, temperature):
    net = Network(parsed.species, parsed.reactions, use_sparse=False, vectorize_reactions=vectorize)
    ode = net.get_ode()
    n = ode.reactions_number
    rhs = ode(0.0, abundances, temperature, 3e-17, 1e-4, 2.0, jnp.ones(n), jnp.zeros(n))
    rates = ode.get_rates(temperature, 3e-17, 1e-4, 2.0, abundances)
    return np.asarray(rhs), np.asarray(rates)


@pytest.mark.parametrize("temperature", [20.0, 300.0, 3000.0])
def test_vectorised_matches_per_reaction(parsed, temperature):
    abundances = jnp.asarray(10 ** np.random.default_rng(0).uniform(-6, 2, len(parsed.species)))
    rhs_v, rates_v = _rhs_and_rates(parsed, True, abundances, temperature)
    rhs_s, rates_s = _rhs_and_rates(parsed, False, abundances, temperature)
    np.testing.assert_allclose(rates_v, rates_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(rhs_v, rhs_s, rtol=1e-10, atol=1e-30)


def test_rates_follow_file_rows(parsed):
    """Spot-check a few rows against their closed-form rate laws (T = 300 K, zeta = 3e-17, G0 = 1e-4)."""
    abundances = jnp.ones(len(parsed.species))
    _, rates = _rhs_and_rates(parsed, True, abundances, 300.0)
    by_row = {(tuple(r.reactants), tuple(r.products)): k for r, k in zip(parsed.reactions, rates)}
    assert by_row[(("O",), ("O+", "E"))] == pytest.approx(2.8 * 3e-17)          # cosmic-ray ionisation
    assert by_row[(("C+", "E"), ("C",))] == pytest.approx(4.4e-12)              # KA at T = 300 K
    assert by_row[(("H+", "E"), ("H",))] == pytest.approx(3.5e-12)
    assert by_row[(("H", "H"), ("H2",))] == pytest.approx(2.121e-17)            # grain formation, 100 * 0.01 * alpha
    assert by_row[(("H2",), ("H", "H"))] == pytest.approx(0.1 * 3e-17)          # cosmic-ray dissociation
    assert by_row[(("H2O",), ("OH", "H"))] == pytest.approx(7.72e-10 * 1e-4)     # FUV


def test_get_ode_is_idempotent(parsed):
    net = Network(parsed.species, parsed.reactions, use_sparse=False)
    first = net.get_ode()
    second = net.get_ode()
    assert len(second.reactions) == len(first.reactions)
