import equinox as eqx
import numpy as np
from chem_commons import *

import jax
import jax.numpy as jnp


@eqx.filter_jit
def get_cooling(x, tgas):
    cool = 7.3e-19 * x[idx_H] * x[idx_E] * jnp.exp(-118400.0 / tgas)  # Ly-alpha
    cool += 1.8e-24 * x[idx_O] * x[idx_E] * jnp.exp(-22800 / tgas)  # OI 630nm
    cool += cooling_H2(x, tgas)

    return cool


@eqx.filter_jit
def cooling_H2(x, temp):
    # High density cooling function for H2
    t3 = temp * 1e-3  # (T/1000)
    logt3 = jnp.log10(t3)

    logt32 = logt3 * logt3
    logt33 = logt32 * logt3
    logt34 = logt33 * logt3
    logt35 = logt34 * logt3
    logt36 = logt35 * logt3
    logt37 = logt36 * logt3
    logt38 = logt37 * logt3

    HDL = jax.lax.cond(
        # temp < 2e3,
        jax.numpy.less(temp, 2e3),
        # IF temp < 2e3
        lambda _: (9.5e-22 * t3**3.76)
        / (1.0 + 0.12 * t3**2.1)
        * jnp.exp(-((0.13 / t3) ** 3))
        + 3.0e-24 * jnp.exp(-0.51 / t3)
        + 6.7e-19 * jnp.exp(-5.86 / t3)
        + 1.6e-18 * jnp.exp(-11.7 / t3),
        # ELSE temp >= 2e3
        lambda _: jax.lax.cond(
            jax.numpy.less_equal(temp, 1e4),
            # IF: temp <= 1e4,
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
            # ELSE: temp > 1e4
            lambda _: 5.531333679406485e-19,
            None,
        ),
        None,
    )

    # if temp < 2e3:
    #     # High Density Limit (HDL) and Low Density Limit (LDL)
    #     HDLR = (9.5e-22 * t3**3.76) / (1.0 + 0.12 * t3**2.1) * np.exp(
    #         -((0.13 / t3) ** 3)
    #     ) + 3.0e-24 * np.exp(-0.51 / t3)
    #     HDLV = 6.7e-19 * np.exp(-5.86 / t3) + 1.6e-18 * np.exp(-11.7 / t3)
    #     HDL = HDLR + HDLV
    # elif 2e3 <= temp <= 1e4:
    #     HDL = 1e1 ** (
    #         -2.0584225e1
    #         + 5.0194035 * logt3
    #         - 1.5738805 * logt32
    #         - 4.7155769 * logt33
    #         + 2.4714161 * logt34
    #         + 5.4710750 * logt35
    #         - 3.9467356 * logt36
    #         - 2.2148338 * logt37
    #         + 1.8161874 * logt38
    #     )
    # else:
    #     HDL = 5.531333679406485e-19

    LDL = x[idx_H] * jax.lax.cond(
        jnp.less_equal(temp, 1e2),
        # IF temp <= 1e2
        lambda _: 1e1
        ** (
            -16.818342e0
            + 3.7383713e1 * logt3
            + 5.8145166e1 * logt32
            + 4.8656103e1 * logt33
            + 2.0159831e1 * logt34
            + 3.8479610e0 * logt35
        ),
        # ELSE temp > 1e2
        lambda _: jax.lax.cond(
            jnp.less_equal(temp, 1e3),
            # IF: temp <= 1e3,
            lambda _: 1e1
            ** (
                -2.4311209e1
                + 3.5692468e0 * logt3
                - 1.1332860e1 * logt32
                - 2.7850082e1 * logt33
                - 2.1328264e1 * logt34
                - 4.2519023e0 * logt35
            ),
            # ELSE: temp > 1e3
            lambda _: jax.lax.cond(
                jnp.less_equal(temp, 6e3),
                # IF: temp <= 6e3,
                lambda _: 1e1
                ** (
                    -2.4311209e1
                    + 4.6450521e0 * logt3
                    - 3.7209846e0 * logt32
                    + 5.9369081e0 * logt33
                    - 5.5108049e0 * logt34
                    + 1.5538288e0 * logt35
                ),
                # ELSE: temp > 6e3
                lambda _: 1.862314467912518e-22,
                None,
            ),
            None,
        ),
        None,
    )

    # jax.debug.print("cooling {LDL}, {HDL}", LDL=LDL, HDL=HDL)
    return x[idx_H2] / (1e0 / (HDL + 1e-100) + 1e0 / (LDL + 1e-100))
    # cool =

    # return cool * (jnp.equal(LDL*HDL, 0.0) * 0.0)
