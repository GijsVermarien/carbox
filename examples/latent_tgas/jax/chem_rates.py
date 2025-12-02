import numpy as np
from chem_commons import dust2gas, nreactions

import jax
import jax.numpy as jnp

from functools import partial

reaction_strings = [
    "O+ + H2 -> OH+ + H",
    "OH+ + H2 -> H2O+ + H",
    "H2O+ + H2 -> H3O+ + H",
    "H3O+ + E -> H2O + H",
    "H2O+ + E -> OH + H",
    "H2O+ + E -> O + H2",
    "OH+ + E -> O + H",
    "O+ + E -> O",
    "O -> O+ + E",
    "C -> C+ + E",
    "CO -> C + O",
    "C+ + E -> C",
    "C + OH -> CO + H",
    "C+ + OH -> CO+ + H",
    "CO+ + H -> CO + H+",
    "CO+ + H2 -> HCO+ + H",
    "HCO+ + E -> CO + H",
    "H+ + E -> H",
    "H + H -> H2",
    "H2 -> H + H",
    "H2 -> H + H",
    "C -> C+ + E",
    "CO -> C + O",
    "H2O -> OH + H",
]


@jax.jit
def get_rates(tgas, cr_rate, gnot):
    # O+ + H2 -> OH+ + H
    k0 = 1.6e-9

    # OH+ + H2 -> H2O+ + H
    k1 = 1e-9

    # H2O+ + H2 -> H3O+ + H
    k2 = 6.1e-10

    # H3O+ + E -> H2O + H
    k3 = 1.1e-7 / jnp.sqrt(tgas / 3e2)

    # H2O+ + E -> OH + H
    k4 = 8.6e-8 / jnp.sqrt(tgas / 3e2)

    # H2O+ + E -> O + H2
    k5 = 3.9e-8 / jnp.sqrt(tgas / 3e2)

    # OH+ + E -> O + H
    k6 = 6.3e-9 * (tgas / 3e2) ** (-0.48)

    # O+ + E -> O
    k7 = 3.4e-12 * (tgas / 3e2) ** (-0.63)

    # O -> O+ + E
    k8 = 2.8 * cr_rate

    # C -> C+ + E
    k9 = 2.62 * cr_rate

    # CO -> C + O
    k10 = 5.0 * cr_rate

    # C+ + E -> C
    k11 = 4.4e-12 * (tgas / 3e2) ** (-0.61)

    # C + OH -> CO + H
    k12 = 1.15e-10 * (tgas / 3e2) ** (-0.339)

    # C+ + OH -> CO+ + H
    k13 = 9.15e-10 * (0.62 + 0.4767 * 5.5 * jnp.sqrt(300 / tgas))

    # CO+ + H -> CO + H+
    k14 = 4e-10

    # CO+ + H2 -> HCO+ + H
    k15 = 7.28e-10

    # HCO+ + E -> CO + H
    k16 = 2.8e-7 * (tgas / 3e2) ** (-0.69)

    # H+ + E -> H
    k17 = 3.5e-12 * (tgas / 3e2) ** (-0.7)

    # H + H -> H2
    k18 = 2.121e-17 * dust2gas / 1e-2

    # H2 -> H + H
    k19 = 1e-1 * cr_rate

    # H2 -> H + H
    k20 = 5.68e-11 * gnot

    # C -> C+ + E
    k21 = 3.39e-10 * gnot

    # CO -> C + O
    k22 = 2.43e-10 * gnot

    # H2O -> OH + H
    k23 = 7.72e-10 * gnot

    return jnp.hstack(
        [
            k0,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
            k8,
            k9,
            k10,
            k11,
            k12,
            k13,
            k14,
            k15,
            k16,
            k17,
            k18,
            k19,
            k20,
            k21,
            k22,
            k23,
        ]
    )
