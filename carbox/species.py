import equinox as eqx
import jax.numpy as jnp
from dataclasses import dataclass


class JSpecies(eqx.Module):
    pass


@dataclass
class Species:
    name: str
    mass: float
    charge: int = 0

    def __str__(self):
        return f"{self.name} ({self.mass}, {self.charge})"

    def __repr__(self):
        return f"Species({self.name}, {self.mass}, {self.charge})"

    def __eq__(self, other):
        if isinstance(other, Species):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self):
        return hash(self.name)
