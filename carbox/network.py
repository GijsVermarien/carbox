from typing import List

from reactions import Reaction
from species import Species
import dataclass as dataclass
import jax.numpy as jnp
import equinox as eqx


class JNetwork(eqx.Module):
    pass

    def __call__(
        self, x: jnp.Array, temperature: jnp.Array, density: jnp.Array
    ) -> jnp.array:
        return None  # dx


class Network(dataclass):
    species: List[Species]
    reactions: List[Reaction]

    def __init__(self, species, reactions):
        self.species = species
        self.reactions = reactions

    def __call__(self):
        self.species_by_idx = {species: idx for idx, species in enumerate(self.species)}
        self.reactions_by_idx = {
            reactions: idx for idx, reactions in enumerate(self.reactions)
        }
        # convert species to JSpecies
        # convert reactions to JReactions
        return JNetwork(self.species, self.reactions)
