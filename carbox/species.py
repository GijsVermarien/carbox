import equinox as eqx
import jax.numpy as jnp
import dataclass as dataclass


class JSpecies(eqx.Module):
    pass


class Species(dataclass):
    name: str
    mass: float

    def __str__(self):
        return f"{self.name} ({self.mass}, {self.charge})"

    def __repr__(self):
        return f"Species({self.name}, {self.mass}, {self.charge})"

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
