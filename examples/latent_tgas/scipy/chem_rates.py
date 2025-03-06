import numpy as np
from chem_commons import dust2gas, nreactions


def get_rates(tgas, cr_rate, gnot):
    k = np.zeros(nreactions)

    # O+ + H2 -> OH+ + H
    k[0] = 1.6e-9

    # OH+ + H2 -> H2O+ + H
    k[1] = 1e-9

    # H2O+ + H2 -> H3O+ + H
    k[2] = 6.1e-10

    # H3O+ + E -> H2O + H
    k[3] = 1.1e-7 / np.sqrt(tgas / 3e2)

    # H2O+ + E -> OH + H
    k[4] = 8.6e-8 / np.sqrt(tgas / 3e2)

    # H2O+ + E -> O + H2
    k[5] = 3.9e-8 / np.sqrt(tgas / 3e2)

    # OH+ + E -> O + H
    k[6] = 6.3e-9 * (tgas / 3e2) ** (-0.48)

    # O+ + E -> O
    k[7] = 3.4e-12 * (tgas / 3e2) ** (-0.63)

    # O -> O+ + E
    k[8] = 2.8 * cr_rate

    # C -> C+ + E
    k[9] = 2.62 * cr_rate

    # CO -> C + O
    k[10] = 5.0 * cr_rate

    # C+ + E -> C
    k[11] = 4.4e-12 * (tgas / 3e2) ** (-0.61)

    # C + OH -> CO + H
    k[12] = 1.15e-10 * (tgas / 3e2) ** (-0.339)

    # C+ + OH -> CO+ + H
    k[13] = 9.15e-10 * (0.62 + 0.4767 * 5.5 * np.sqrt(300 / tgas))

    # CO+ + H -> CO + H+
    k[14] = 4e-10

    # CO+ + H2 -> HCO+ + H
    k[15] = 7.28e-10

    # HCO+ + E -> CO + H
    k[16] = 2.8e-7 * (tgas / 3e2) ** (-0.69)

    # H+ + E -> H
    k[17] = 3.5e-12 * (tgas / 3e2) ** (-0.7)

    # H + H -> H2
    k[18] = 2.121e-17 * dust2gas / 1e-2

    # H2 -> H + H
    k[19] = 1e-1 * cr_rate

    # H2 -> H + H
    k[20] = 5.68e-11 * gnot

    # C -> C+ + E
    k[21] = 3.39e-10 * gnot

    # CO -> C + O
    k[22] = 2.43e-10 * gnot

    # H2O -> OH + H
    k[23] = 7.72e-10 * gnot

    return k
