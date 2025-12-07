"""Callbacks for the ODE solver."""

import diffrax as dx
import jax.numpy as jnp


class FloorCallback:
    """Callback to enforce a minimum abundance floor during integration."""

    def __init__(self, floor_value: float = 1e-40):
        """Initialize the callback with a floor value."""
        self.floor_value = floor_value

    def __call__(self, t, y, args):
        """Reset abundances to the floor value if they are below it."""
        return jnp.where(y < self.floor_value, self.floor_value, y)
