"""The fused per-cell Rosenbrock23 solver (carbox.fused) against diffrax Kvaerno5 on the example network."""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("warp")
pytest.importorskip("gradsolve")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_enable_x64", True)
import diffrax as dx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from carbox.fused import fused_batch_solver, network_rhs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NETWORK = os.path.join(HERE, "..", "data", "simple_latent_tgas.csv")


def _network():
    from carbox.parsers.latent_tgas_parser import LatentTGASParser

    net = LatentTGASParser().parse_network(NETWORK)
    return net.get_ode(), net


def _cells(n_species, n=64, seed=0):
    rng = np.random.default_rng(seed)
    nH = 10 ** rng.uniform(0.5, 3.0, n)  # 3 .. 1000 cm^-3
    x = np.full((n, n_species), 1e-12)  # fractional abundances
    return x, nH


@pytest.mark.skipif(not os.path.exists(NETWORK), reason="example network not found")
def test_fused_matches_diffrax():
    jnet, net = _network()
    names = [getattr(s, "name", str(s)) for s in net.species]
    n_sp = jnet.incidence.shape[0]
    y0, nH = _cells(n_sp)

    # start from a mostly molecular, neutral gas: H2 carries the hydrogen, C+ and O the heavy elements
    def put(name, frac):
        if names and name in names:
            y0[:, names.index(name)] = frac

    put("H2", 0.5)
    put("C+", 1.4e-4)
    put("E", 1.4e-4)
    put("O", 3.2e-4)
    T, crate, g0, av = 30.0, 3e-17, 1e-4, 2.0
    f = network_rhs(jnet, T, crate, g0, av)
    dt = np.full(len(y0), 1e11)
    params = nH[:, None]

    def one(y, p):  # diffrax reference at the same tolerances
        return dx.diffeqsolve(
            dx.ODETerm(lambda t, y, a: f(t, y, a)),
            dx.Kvaerno5(),
            0.0,
            float(dt[0]),
            1e3,
            y,
            args=p,
            stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-15),
            saveat=dx.SaveAt(t1=True),
            max_steps=16384,
            throw=False,
        ).ys[0]

    ref = np.asarray(jax.vmap(one)(jnp.asarray(y0), jnp.asarray(params)))
    solve = fused_batch_solver(f, n_sp, n_params=1, rtol=1e-6, atol=1e-15, max_steps=4096, cache_key="test")
    out = np.asarray(jax.jit(solve)(jnp.asarray(y0), jnp.asarray(params), jnp.asarray(dt)))
    assert np.isfinite(out).all()
    xa = np.log10(np.maximum(ref, 1e-30))
    xb = np.log10(np.maximum(out, 1e-30))
    significant = ref > 1e-12
    assert np.abs(xb - xa)[significant].max() < 1e-3  # both solvers at rtol 1e-6: agree to well below 1e-3 dex
