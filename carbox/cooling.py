"""
Cooling-function building blocks for thermal balance.

Each function here computes one cooling *channel* -- a physical process that
removes thermal energy from the gas -- and is combined with the others (and,
eventually, a heating term) into the net rate used by
:class:`carbox.thermo.ThermoRate`. See the ``Thermal balance`` page in the
docs for how these channels combine into ``dT/dt``.

Every function has the same shape: ``(x, tgas, idx) -> jnp.ndarray``, where

- ``x`` is the number-density abundance vector [cm^-3] (*not* fractional --
  the caller, :class:`carbox.network.JNetwork`, evaluates rate laws on
  number densities so self-shielding/cooling terms that expect cm^-3 don't
  need to know about the fractional-abundance ODE convention),
- ``tgas`` is the gas temperature [K],
- ``idx`` is the network's species lookup (:class:`carbox.index.Idx`), used
  to pick out the specific species (H, e-, O, H2, ...) each channel depends
  on.

They return a cooling rate in erg cm^-3 s^-1 (energy lost per unit volume
per unit time), and are pure functions of traced values -- no Python
branching on `x`/`tgas`, only ``jax.lax.cond`` -- so they compose freely
inside `jax.jit`/`vmap` and stay differentiable.

Only cooling is implemented so far; there is no heating channel yet, so
:class:`carbox.thermo.ThermoRate`'s net rate is currently pure cooling
(``dT/dt <= 0``, gas always cools or holds temperature, never heats).
"""

import jax
import jax.numpy as jnp


def cooling_lyalpha(x, tgas, idx):
    r"""
    Lyman-alpha cooling: collisional excitation of atomic hydrogen by
    electrons, followed by radiative decay (n=2 -> n=1, 1216 Angstrom).

    .. math::
        \Lambda_{Ly\alpha} = 7.3\times10^{-19}\, n_H\, n_e\,
        \exp(-118400 / T)\ \ [\mathrm{erg\ cm^{-3}\ s^{-1}}]

    This is the dominant coolant of warm (:math:`T \gtrsim 10^4` K),
    partially ionized atomic gas; the exponential cutoff reflects the large
    energy gap (10.2 eV) of the transition, so the channel switches off
    sharply below a few :math:`10^3` K.

    Parameters
    ----------
    x : jnp.ndarray
        Number-density abundance vector [cm^-3].
    tgas : jnp.ndarray
        Gas temperature [K].
    idx : carbox.index.Idx
        Species lookup; requires ``H`` and ``E`` to be present in the
        network.

    Returns
    -------
    jnp.ndarray
        Cooling rate [erg cm^-3 s^-1].
    """
    return 7.3e-19 * x[idx.H] * x[idx.E] * jnp.exp(-118400.0 / tgas)


def cooling_oi(x, tgas, idx):
    r"""
    [OI] 630 nm fine-structure line cooling: collisional excitation of
    neutral atomic oxygen by electrons, followed by radiative decay.

    .. math::
        \Lambda_{OI} = 1.8\times10^{-24}\, n_O\, n_e\,
        \exp(-22800 / T)\ \ [\mathrm{erg\ cm^{-3}\ s^{-1}}]

    A minor but non-negligible coolant wherever both free electrons and
    neutral oxygen coexist (e.g. partially ionized atomic gas); the smaller
    excitation energy (2 eV) than Lyman-alpha lets it stay active down to
    lower temperatures.

    Parameters
    ----------
    x : jnp.ndarray
        Number-density abundance vector [cm^-3].
    tgas : jnp.ndarray
        Gas temperature [K].
    idx : carbox.index.Idx
        Species lookup; requires ``O`` and ``E`` to be present in the
        network.

    Returns
    -------
    jnp.ndarray
        Cooling rate [erg cm^-3 s^-1].
    """
    return 1.8e-24 * x[idx.O] * x[idx.E] * jnp.exp(-22800.0 / tgas)


def cooling_h2(x, tgas, idx):
    r"""
    H2 ro-vibrational collisional cooling (collisions with atomic H).

    Uses the Hollenbach & McKee (1979) high-density-limit (HDL) and
    low-density-limit (LDL) fitting formulas -- piecewise polynomials in
    :math:`\log_{10}(T/1000\,\mathrm{K})` over several temperature ranges,
    combined via the standard two-limit interpolation

    .. math::
        \Lambda_{H_2} = \frac{n_{H_2}}
        {1/\mathrm{HDL}(T) + 1/(n_H\, \mathrm{LDL}(T))}
        \ \ [\mathrm{erg\ cm^{-3}\ s^{-1}}]

    HDL is the cooling rate per H2 molecule at densities high enough for
    LTE level populations; LDL (scaled by the H-atom density, the dominant
    collision partner assumed here) is the rate in the optically-thin,
    collision-limited regime. The functional form and coefficients follow
    the implementation in KROME (Grassi et al. 2014), itself based on
    Hollenbach & McKee (1979); each of the four temperature sub-ranges
    (``T < 100``, ``100-1000``, ``1000-6000``, ``> 6000`` K for LDL; two
    ranges split at 2000 K for HDL) uses `jax.lax.cond` rather than a
    Python `if`, so the branch is resolved per-element under `jit`/`vmap`
    without retracing.

    Parameters
    ----------
    x : jnp.ndarray
        Number-density abundance vector [cm^-3].
    tgas : jnp.ndarray
        Gas temperature [K].
    idx : carbox.index.Idx
        Species lookup; requires ``H`` and ``H2`` to be present in the
        network.

    Returns
    -------
    jnp.ndarray
        Cooling rate [erg cm^-3 s^-1].
    """
    t3 = tgas * 1e-3
    logt3 = jnp.log10(t3)
    logt3_powers = [logt3**n for n in range(2, 9)]
    logt32, logt33, logt34, logt35, logt36, logt37, logt38 = logt3_powers

    hdl = jax.lax.cond(
        jnp.less(tgas, 2e3),
        lambda _: (9.5e-22 * t3**3.76)
        / (1.0 + 0.12 * t3**2.1)
        * jnp.exp(-((0.13 / t3) ** 3))
        + 3.0e-24 * jnp.exp(-0.51 / t3)
        + 6.7e-19 * jnp.exp(-5.86 / t3)
        + 1.6e-18 * jnp.exp(-11.7 / t3),
        lambda _: jax.lax.cond(
            jnp.less_equal(tgas, 1e4),
            lambda _: 1e1
            ** (
                -2.0584225e1
                + 5.0194035 * logt3
                - 1.5738805 * logt32
                - 4.7155769 * logt33
                + 2.4714161 * logt34
                + 5.4710750 * logt35
                - 3.9467356 * logt36
                - 2.2148338 * logt37
                + 1.8161874 * logt38
            ),
            lambda _: 5.531333679406485e-19,
            None,
        ),
        None,
    )

    ldl = x[idx.H] * jax.lax.cond(
        jnp.less_equal(tgas, 1e2),
        lambda _: 1e1
        ** (
            -16.818342e0
            + 3.7383713e1 * logt3
            + 5.8145166e1 * logt32
            + 4.8656103e1 * logt33
            + 2.0159831e1 * logt34
            + 3.8479610e0 * logt35
        ),
        lambda _: jax.lax.cond(
            jnp.less_equal(tgas, 1e3),
            lambda _: 1e1
            ** (
                -2.4311209e1
                + 3.5692468e0 * logt3
                - 1.1332860e1 * logt32
                - 2.7850082e1 * logt33
                - 2.1328264e1 * logt34
                - 4.2519023e0 * logt35
            ),
            lambda _: jax.lax.cond(
                jnp.less_equal(tgas, 6e3),
                lambda _: 1e1
                ** (
                    -2.4311209e1
                    + 4.6450521e0 * logt3
                    - 3.7209846e0 * logt32
                    + 5.9369081e0 * logt33
                    - 5.5108049e0 * logt34
                    + 1.5538288e0 * logt35
                ),
                lambda _: 1.862314467912518e-22,
                None,
            ),
            None,
        ),
        None,
    )

    return x[idx.H2] / (1e0 / (hdl + 1e-100) + 1e0 / (ldl + 1e-100))
