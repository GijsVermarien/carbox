import numpy as np
from chem_commons import *
from chem_cooling import get_cooling
from chem_heating import get_heating
from chem_rates import get_rates


def fex(t, y, cr_rate, gnot):
    tgas = y[idx_tgas]

    k = get_rates(tgas, cr_rate, gnot)

    dy = np.zeros(nspecies + 1)
    flux = np.zeros(nreactions)

    gamma_ad = 1.4

    number_density = np.sum(y[:nspecies])
    cool = get_cooling(y[:nspecies], tgas)
    heat = get_heating(y[:nspecies], tgas, cr_rate, gnot)

    # dy[idx_tgas] = (gamma_ad - 1e0) * (heat - cool) / kboltzmann / number_density
    # Disable heating and cooling:
    dy[idx_tgas] = 0.0

    # enable this to figure out the heating and cooling comparison
    # print(f"scipy,{t:2.2e},{heat},{cool}")

    flux[0] = k[0] * y[idx_Oj] * y[idx_H2]
    flux[1] = k[1] * y[idx_OHj] * y[idx_H2]
    flux[2] = k[2] * y[idx_H2Oj] * y[idx_H2]
    flux[3] = k[3] * y[idx_H3Oj] * y[idx_E]
    flux[4] = k[4] * y[idx_H2Oj] * y[idx_E]
    flux[5] = k[5] * y[idx_H2Oj] * y[idx_E]
    flux[6] = k[6] * y[idx_OHj] * y[idx_E]
    flux[7] = k[7] * y[idx_Oj] * y[idx_E]
    flux[8] = k[8] * y[idx_O]
    flux[9] = k[9] * y[idx_C]
    flux[10] = k[10] * y[idx_CO]
    flux[11] = k[11] * y[idx_Cj] * y[idx_E]
    flux[12] = k[12] * y[idx_C] * y[idx_OH]
    flux[13] = k[13] * y[idx_Cj] * y[idx_OH]
    flux[14] = k[14] * y[idx_COj] * y[idx_H]
    flux[15] = k[15] * y[idx_COj] * y[idx_H2]
    flux[16] = k[16] * y[idx_HCOj] * y[idx_E]
    flux[17] = k[17] * y[idx_Hj] * y[idx_E]
    flux[18] = k[18] * y[idx_H] * y[idx_H]
    flux[19] = k[19] * y[idx_H2]
    flux[20] = k[20] * y[idx_H2]
    flux[21] = k[21] * y[idx_C]
    flux[22] = k[22] * y[idx_CO]
    flux[23] = k[23] * y[idx_H2O]

    dy[idx_C] = -flux[9] + flux[10] + flux[11] - flux[12] - flux[21] + flux[22]
    dy[idx_Cj] = +flux[9] - flux[11] - flux[13] + flux[21]
    dy[idx_CO] = -flux[10] + flux[12] + flux[14] + flux[16] - flux[22]
    dy[idx_COj] = +flux[13] - flux[14] - flux[15]
    dy[idx_E] = (
        -flux[3]
        - flux[4]
        - flux[5]
        - flux[6]
        - flux[7]
        + flux[8]
        + flux[9]
        - flux[11]
        - flux[16]
        - flux[17]
        + flux[21]
    )  # + flux[23]
    dy[idx_H] = (
        +flux[0]
        + flux[1]
        + flux[2]
        + flux[3]
        + flux[4]
        + flux[6]
        + flux[12]
        + flux[13]
        - flux[14]
        + flux[15]
        + flux[16]
        + flux[17]
        - flux[18]
        - flux[18]
        + flux[19]
        + flux[19]
        + flux[20]
        + flux[20]
        + flux[23]
    )
    dy[idx_Hj] = +flux[14] - flux[17]
    dy[idx_H2] = (
        -flux[0]
        - flux[1]
        - flux[2]
        + flux[5]
        - flux[15]
        + flux[18]
        - flux[19]
        - flux[20]
    )
    dy[idx_H2O] = +flux[3] - flux[23]
    dy[idx_H2Oj] = +flux[1] - flux[2] - flux[4] - flux[5]
    dy[idx_H3Oj] = +flux[2] - flux[3]
    dy[idx_HCOj] = +flux[15] - flux[16]
    dy[idx_O] = +flux[5] + flux[6] + flux[7] - flux[8] + flux[10] + flux[22]
    dy[idx_Oj] = -flux[0] - flux[7] + flux[8]
    dy[idx_OH] = +flux[4] - flux[12] - flux[13] + flux[23]
    dy[idx_OHj] = +flux[0] - flux[1] - flux[6]

    return dy
