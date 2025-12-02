import numpy as np
from chem_commons import *


def get_heating(x, tgas, cr_rate, gnot):
    rate_H2 = 5.68e-11 * gnot
    heats = [
        cr_rate * (5.5e-12 * x[idx_H] + 2.5e-11 * x[idx_H2]),
        get_photoelectric_heating(x, tgas, gnot),
        6.4e-13 * rate_H2 * x[idx_H2],
    ]

    return np.sum(heats)


def get_photoelectric_heating(x, tgas, gnot):
    number_density = np.sum(x)
    bet = 0.735e0 * tgas ** (-0.068)

    if x[idx_E] > 0e0:
        psi = gnot * np.sqrt(tgas) / x[idx_E]
    else:
        return 0e0

    # grains recombination cooling
    recomb_cool = 4.65e-30 * tgas**0.94 * psi**bet * x[idx_E] * x[idx_H]

    eps = 4.9e-2 / (1e0 + 4e-3 * psi**0.73) + 3.7e-2 * (tgas * 1e-4) ** 0.7 / (
        1e0 + 2e-4 * psi
    )

    # net photoelectric heating
    return (1.3e-24 * eps * gnot * number_density - recomb_cool) * dust2gas
