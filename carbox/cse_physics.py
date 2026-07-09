"""
Physics module for Circumstellar Envelope (CSE) outflow models.
"""

from typing import Tuple
import equinox as eqx
import jax.numpy as jnp

class CSEPhysics(eqx.Module):
    """
    Physical parameters for a CSE outflow model.
    """
    mdot: float       # Mass loss rate [M_sun/yr]
    vexp: float       # Expansion velocity [km/s]
    t_star: float     # Stellar effective temperature (at r_star) [K]
    r_init: float     # Initial radius [cm]
    r_star: float     # Stellar radius, normalization of the temperature profile [cm]
    eps: float        # Temperature power law exponent
    
    # Constants
    MSUN_G = 1.989e33
    YR_S = 3.15576e7
    KM_CM = 1.0e5
    MH = 1.67e-24
    MU = 2.3          # Mean molecular weight (H2 + He)

    def get_conditions(self, t_sec: float) -> Tuple[float, float, float, float]:
        """
        Calculate physical conditions at a given time t.
        
        Returns
        -------
        n, T, av, r
        """
        # Current radius: r = r_init + v * t
        v_cgs = self.vexp * self.KM_CM
        r = self.r_init + v_cgs * t_sec
        
        # Density profile (n ~ r^-2)
        # n = Mdot / (4 pi r^2 v mu mH)
        mdot_cgs = self.mdot * self.MSUN_G / self.YR_S
        rho = mdot_cgs / (4 * jnp.pi * r**2 * v_cgs)
        n = rho / (self.MU * self.MH)
        
        # Temperature profile (T ~ r^-eps)
        T = self.t_star * (r / self.r_star)**(-self.eps)
        
        # Extinction (Av)
        # For 1/r^2 density, column density N_H ~ n * r
        # Standard conversion: Av = N_H / 1.87E21 (approx)
        col_dens = n * r 
        av = col_dens / 1.87E21 
        
        
        return n, T, av, r
