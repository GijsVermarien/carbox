"""
Physics models for Carbox simulations.

A physics model prescribes the local physical conditions (density,
temperature, visual extinction, position) as a function of time, plus any
non-chemical source term for the ODE. The solver and output code only talk
to this interface, so static clouds, CSE outflows, and future models are
interchangeable.

All methods that the solver touches run inside ``jax.jit``/``jax.vmap`` —
implementations must be traceable (no Python branching on traced values).
"""

from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp

# Constants
PC_TO_CM = 3.086e18
SPY = 3600.0 * 24 * 365.0
MSUN_G = 1.989e33
YR_S = 3.15576e7
KM_CM = 1.0e5
MH = 1.67e-24


class AbstractPhysics(eqx.Module):
    """
    Interface for physical-conditions models.

    Subclasses must implement ``get_conditions``. The remaining methods have
    sensible defaults.
    """

    def get_conditions(self, t_sec) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Physical conditions at time t [s].

        Returns
        -------
        n, T, av, r : total gas density [cm^-3], temperature [K],
            visual extinction [mag], position [cm] (0.0 if not meaningful).

        Must be vmappable: every returned value has to depend on ``t_sec``
        (broadcast constants against it, see StaticCloudPhysics).
        """
        raise NotImplementedError

    def dilution(self, t_sec, y) -> jnp.ndarray:
        """
        Additive non-chemical term for the ODE (e.g. expansion dilution).

        Default: no dilution.
        """
        return jnp.zeros_like(y)

    def initial_number_density(self, t_start_sec: float = 0.0) -> float:
        """Total gas density at the start of the integration [cm^-3]."""
        n, _, _, _ = self.get_conditions(t_start_sec)
        return n

    def time_grid(self, config) -> Optional[jnp.ndarray]:
        """
        Optional custom snapshot grid [s] for this model.

        Return None to use the solver's generic log-time grid.
        """
        return None


class StaticCloudPhysics(AbstractPhysics):
    """
    Constant-conditions cloud (the original Carbox setup).

    Av is either the given constant or computed self-consistently from the
    column density: Av = base_av + n * L / 1.6e21.
    """

    number_density: float   # Total H number density [cm^-3]
    temperature: float      # Gas temperature [K]
    visual_extinction: float = 2.0   # [mag], used unless self-consistent
    use_self_consistent_av: bool = False
    base_av: float = 0.0             # Base Av before column contribution
    cloud_radius_pc: float = 1.0     # Cloud radius [pc]

    def _av(self) -> float:
        if not self.use_self_consistent_av:
            return self.visual_extinction
        column_density = self.cloud_radius_pc * PC_TO_CM * self.number_density
        return self.base_av + column_density / 1.6e21

    def get_conditions(self, t_sec):
        # Broadcast constants against t so the method stays vmappable
        zero = 0.0 * jnp.asarray(t_sec, dtype=float)
        return (
            self.number_density + zero,
            self.temperature + zero,
            self._av() + zero,
            zero,  # radius not meaningful for a static cloud
        )


class CSEPhysics(AbstractPhysics):
    """
    Circumstellar-envelope outflow: constant-velocity spherical expansion.

    Density follows mass conservation (n ~ r^-2), temperature a power law
    normalized at the stellar radius, Av from the radial column density.
    """

    mdot: float       # Mass loss rate [M_sun/yr]
    vexp: float       # Expansion velocity [km/s]
    t_star: float     # Stellar effective temperature (at r_star) [K]
    r_init: float     # Initial radius [cm]
    r_star: float     # Stellar radius, normalization of the temperature profile [cm]
    eps: float        # Temperature power law exponent

    # Constants (kept as class attributes for backward compatibility)
    MSUN_G = MSUN_G
    YR_S = YR_S
    KM_CM = KM_CM
    MH = MH
    MU = 2.3          # Mean molecular weight (H2 + He)

    def get_conditions(self, t_sec):
        # Current radius: r = r_init + v * t
        v_cgs = self.vexp * self.KM_CM
        r = self.r_init + v_cgs * jnp.asarray(t_sec)

        # Density profile (n ~ r^-2): n = Mdot / (4 pi r^2 v mu mH)
        mdot_cgs = self.mdot * self.MSUN_G / self.YR_S
        rho = mdot_cgs / (4 * jnp.pi * r**2 * v_cgs)
        n = rho / (self.MU * self.MH)

        # Temperature profile (T ~ r^-eps)
        T = self.t_star * (r / self.r_star) ** (-self.eps)

        # Extinction: for 1/r^2 density, N_H ~ n * r; Av = N_H / 1.87e21
        av = n * r / 1.87e21

        return n, T, av, r

    # No dilution override: with the fractional-abundance ODE state
    # x_i = n_i / n(t), the spherical-expansion dilution term
    # d(n_i)/dt|dilution = -2(v/r) n_i cancels exactly against the same term
    # in dn/dt, leaving dx_i/dt = chem_i / n(t) with no separate dilution
    # contribution. See AbstractPhysics.dilution (default: zero).

    def time_grid(self, config):
        """Log-spaced radius grid mapped back to time (avoids clustering at t=0)."""
        v_cgs = self.vexp * self.KM_CM
        t_start_sec = config.t_start * SPY
        t_end_sec = config.t_end * SPY

        r_start = self.r_init + v_cgs * t_start_sec
        r_end = self.r_init + v_cgs * t_end_sec

        r_snapshots = jnp.logspace(
            jnp.log10(r_start), jnp.log10(r_end), config.n_snapshots
        )
        t_snapshots_sec = (r_snapshots - self.r_init) / v_cgs

        # Ensure boundaries are exact
        t_snapshots_sec = t_snapshots_sec.at[0].set(t_start_sec)
        t_snapshots_sec = t_snapshots_sec.at[-1].set(t_end_sec)
        return t_snapshots_sec
