"""
UCLCHEM photoreactions module - JAX-compatible implementation.

Provides photodissociation and photoionization rates for H2, CO, and C
with self-shielding and dust extinction effects.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator

# Constants from UCLCHEM (may need adjustment from constants/defaultparameters)
UV_FAC = 3.02
ICE_GAS_PHOTO_CROSSSECTION_RATIO = 0.3  # Kalvans 2018

# Physical constants
PARSEC_TO_CM = 3.0857e18

# Wavelength grid for dust extinction (Savage & Mathis 1979)
LAMBDA_GRID = jnp.array(
    [
        910.0,
        950.0,
        1000.0,
        1050.0,
        1110.0,
        1180.0,
        1250.0,
        1390.0,
        1490.0,
        1600.0,
        1700.0,
        1800.0,
        1900.0,
        2000.0,
        2100.0,
        2190.0,
        2300.0,
        2400.0,
        2500.0,
        2740.0,
        3440.0,
        4000.0,
        4400.0,
        5500.0,
        7000.0,
        9000.0,
        12500.0,
        22000.0,
        34000.0,
        1.0e9,
    ]
)

XLAMBDA_GRID = jnp.array(
    [
        5.76,
        5.18,
        4.65,
        4.16,
        3.73,
        3.40,
        3.11,
        2.74,
        2.63,
        2.62,
        2.54,
        2.50,
        2.58,
        2.78,
        3.01,
        3.12,
        2.86,
        2.58,
        2.35,
        2.00,
        1.58,
        1.42,
        1.32,
        1.00,
        0.75,
        0.48,
        0.28,
        0.12,
        0.05,
        0.00,
    ]
)

# 12CO self-shielding data from van Dishoeck & Black (1988)
NCO_GRID = jnp.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
NH2_GRID = jnp.array([18.0, 19.0, 20.0, 21.0, 22.0, 23.0])

SCO_GRID = jnp.array(
    [
        [0.000, -1.408e-02, -1.099e-01, -4.400e-01, -1.154, -1.888, -2.760, -4.001],
        [
            -8.539e-02,
            -1.015e-01,
            -2.104e-01,
            -5.608e-01,
            -1.272,
            -1.973,
            -2.818,
            -4.055,
        ],
        [
            -1.451e-01,
            -1.612e-01,
            -2.708e-01,
            -6.273e-01,
            -1.355,
            -2.057,
            -2.902,
            -4.122,
        ],
        [
            -4.559e-01,
            -4.666e-01,
            -5.432e-01,
            -8.665e-01,
            -1.602,
            -2.303,
            -3.146,
            -4.421,
        ],
        [-1.303, -1.312, -1.367, -1.676, -2.305, -3.034, -3.758, -5.077],
        [-3.883, -3.888, -3.936, -4.197, -4.739, -5.165, -5.441, -6.446],
    ]
).T  # Transpose to match (NCO, NH2) indexing


@partial(jax.jit, static_argnums=())
def xlambda(wavelength: float) -> float:
    """Ratio of optical depth at wavelength to visual (Savage & Mathis 1979)."""
    lambda_clipped = jnp.clip(wavelength, LAMBDA_GRID[0], LAMBDA_GRID[-1])
    return jnp.interp(lambda_clipped, LAMBDA_GRID, XLAMBDA_GRID)


@partial(jax.jit, static_argnums=())
def scatter(wavelength: float, av: float) -> float:
    """Dust scattering attenuation (Wagenblast & Hartquist 1989, g=0.8, ω=0.3)."""
    tv = av / 1.086
    tl = tv * xlambda(wavelength)

    c = jnp.array([1.0, 2.006, -1.438, 7.364e-01, -5.076e-01, -5.920e-02])
    k = jnp.array([7.514e-01, 8.490e-01, 1.013, 1.282, 2.005, 5.832])

    scatter_low = c[0] * jnp.exp(-k[0] * tl)
    expos = k[1:] * tl
    scatter_high = jnp.sum(c[1:] * jnp.exp(-jnp.clip(expos, None, 100.0)))

    return jnp.where(tl < 1.0, scatter_low, scatter_high)


@partial(jax.jit, static_argnums=())
def h2_self_shielding(nh2: float, doppler_width: float, rad_width: float) -> float:
    """H2 self-shielding (Federman, Glassgold & Kwan 1979)."""
    fpara = 0.5
    fosc = 1.0e-2

    taud = fpara * nh2 * 1.5e-2 * fosc / (doppler_width + 1e-30)

    sj = jnp.where(
        taud == 0.0,
        1.0,
        jnp.where(
            taud < 2.0,
            jnp.exp(-0.6666667 * taud),
            jnp.where(
                taud < 10.0,
                0.638 * taud ** (-1.25),
                jnp.where(
                    taud < 100.0, 0.505 * taud ** (-1.15), 0.344 * taud ** (-1.0667)
                ),
            ),
        ),
    )

    r = rad_width / (1.7724539 * doppler_width + 1e-30)
    t = 3.02 * (r * 1.0e3) ** (-0.064)
    u = jnp.sqrt(taud * r) / t
    sr = r / (t * jnp.sqrt(0.78539816 + u**2))
    sr = jnp.where(rad_width == 0.0, 0.0, sr)

    return sj + sr


@partial(jax.jit, static_argnums=())
def h2_photo_diss_rate(
    nh2: float, rad_field: float, av: float, turb_vel: float
) -> float:
    """H2 photodissociation rate with self-shielding."""
    base_rate = 5.18e-11
    xl = 1000.0
    rad_width = 8.0e7

    doppler_width = turb_vel / (xl * 1.0e-8)

    return (
        base_rate
        * (rad_field / 1.7)
        * scatter(xl, av)
        * h2_self_shielding(nh2, doppler_width, rad_width)
    )


@partial(jax.jit, static_argnums=())
def lbar(nco: float, nh2: float) -> float:
    """Mean wavelength of CO dissociating bands (van Dishoeck & Black 1988)."""
    lu = jnp.log10(jnp.abs(nco) + 1.0)
    lw = jnp.log10(jnp.abs(nh2) + 1.0)

    lambda_bar = (
        (5675.0 - 200.6 * lw)
        - (571.6 - 24.09 * lw) * lu
        + (18.22 - 0.7664 * lw) * lu**2
    )

    return jnp.clip(lambda_bar, 913.6, 1076.1)


@partial(jax.jit, static_argnums=())
def co_self_shielding(nh2: float, nco: float) -> float:
    """CO self-shielding (van Dishoeck & Black 1988)."""
    lognco = jnp.log10(nco + 1.0)
    lognh2 = jnp.log10(nh2 + 1.0)

    lognco_clip = jnp.clip(lognco, NCO_GRID[0], NCO_GRID[-1])
    lognh2_clip = jnp.clip(lognh2, NH2_GRID[0], NH2_GRID[-1])

    interp = RegularGridInterpolator(
        (NCO_GRID, NH2_GRID),
        SCO_GRID,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    log_ssf = interp(jnp.array([lognco_clip, lognh2_clip]))
    return 10.0**log_ssf


@partial(jax.jit, static_argnums=())
def co_photo_diss_rate(nh2: float, nco: float, rad_field: float, av: float) -> float:
    """CO photodissociation rate with self-shielding."""
    ssf = co_self_shielding(nh2, nco)
    lambda_bar = lbar(nco, nh2)
    sca = scatter(lambda_bar, av)

    return 2.0e-10 * (rad_field / 1.7) * ssf * sca


@partial(jax.jit, static_argnums=())
def c_ionization_rate(
    alpha: float,
    gamma: float,
    gas_temp: float,
    nc: float,
    nh2: float,
    av: float,
    rad_field: float,
) -> float:
    """Carbon photoionization rate. [alpha, gamma need UCLCHEM values]"""
    tauc = gamma * av + 1.1e-17 * nc + 0.9 * gas_temp**0.27 * (nh2 / 1.59e21) ** 0.45
    return alpha * (rad_field / 1.7) * jnp.exp(-tauc)


@partial(jax.jit, static_argnums=())
def compute_column_density(absolute_density: float, radius_pc: float) -> float:
    """Column density: N = absolute_density × radius [cm^-2].

    Note: Carbox uses absolute densities (cm^-3) in the ODE state vector,
    unlike UCLCHEM which uses fractional abundances.
    """
    radius_cm = radius_pc * PARSEC_TO_CM
    return absolute_density * radius_cm
