import numpy as np
from chem_commons import *


def get_cooling(x, tgas):
    cool = 7.3e-19 * x[idx_H] * x[idx_E] * np.exp(-118400.0 / tgas)  # Ly-alpha
    cool += 1.8e-24 * x[idx_O] * x[idx_E] * np.exp(-22800 / tgas)  # OI 630nm
    cool += cooling_H2(x, tgas)

    return cool


def cooling_H2(x, temp):
    t3 = temp * 1e-3
    logt3 = np.log10(t3)

    logt32 = logt3 * logt3
    logt33 = logt32 * logt3
    logt34 = logt33 * logt3
    logt35 = logt34 * logt3
    logt36 = logt35 * logt3
    logt37 = logt36 * logt3
    logt38 = logt37 * logt3

    if temp < 2e3:
        HDLR = (9.5e-22 * t3**3.76) / (1.0 + 0.12 * t3**2.1) * np.exp(
            -((0.13 / t3) ** 3)
        ) + 3.0e-24 * np.exp(-0.51 / t3)
        HDLV = 6.7e-19 * np.exp(-5.86 / t3) + 1.6e-18 * np.exp(-11.7 / t3)
        HDL = HDLR + HDLV
    elif 2e3 <= temp <= 1e4:
        HDL = 1e1 ** (
            -2.0584225e1
            + 5.0194035 * logt3
            - 1.5738805 * logt32
            - 4.7155769 * logt33
            + 2.4714161 * logt34
            + 5.4710750 * logt35
            - 3.9467356 * logt36
            - 2.2148338 * logt37
            + 1.8161874 * logt38
        )
    else:
        HDL = 5.531333679406485e-19

    if temp <= 1e2:
        f = 1e1 ** (
            -16.818342e0
            + 3.7383713e1 * logt3
            + 5.8145166e1 * logt32
            + 4.8656103e1 * logt33
            + 2.0159831e1 * logt34
            + 3.8479610e0 * logt35
        )
    elif 1e2 < temp <= 1e3:
        f = 1e1 ** (
            -2.4311209e1
            + 3.5692468e0 * logt3
            - 1.1332860e1 * logt32
            - 2.7850082e1 * logt33
            - 2.1328264e1 * logt34
            - 4.2519023e0 * logt35
        )
    elif 1e3 < temp <= 6e3:
        f = 1e1 ** (
            -2.4311209e1
            + 4.6450521e0 * logt3
            - 3.7209846e0 * logt32
            + 5.9369081e0 * logt33
            - 5.5108049e0 * logt34
            + 1.5538288e0 * logt35
        )
    else:
        f = 1.862314467912518e-22

    LDL = f * x[idx_H]

    if LDL * HDL == 0e0:
        return 0e0

    cool = x[idx_H2] / (1e0 / HDL + 1e0 / LDL)

    return cool
