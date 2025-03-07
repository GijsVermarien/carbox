from typing import List

from reactions import Reaction
from species import Species
from dataclasses import dataclass
import jax.numpy as jnp
import equinox as eqx


class JNetwork(eqx.Module):
    incidence: jnp.array
    reactions: List[Reaction]
    reactant_multiply_indices: jnp.array

    def __init__(self, incidence, reactions):
        self.incidence = incidence
        self.reactions = reactions
        self.reactant_multiply_indices = jnp.argwhere(self.incidence.T < 0)

    def __call__(
        self,
        abundances: jnp.array,
        temperature: jnp.array,
        #        density: jnp.array,
        cr_rate: jnp.array,
        fuv_rate: jnp.array,
    ) -> jnp.array:
        # Calculate the reaction rates
        rates = jnp.zeros(len(self.reactions))
        for i, reaction in enumerate(self.reactions):
            rates = rates.at[i].set(reaction(temperature, cr_rate, fuv_rate))
        # Get the matrix that encodes the reactants that need to be multiplied to get the flux
        for reac_idx, species_idx in self.reactant_multiply_indices:
            rates = rates.at[i].set(rates[reac_idx] * abundances[species_idx])
        # Calculate the change in abundances
        return jnp.matmul(self.incidence, rates)


@dataclass
class Network:
    species: List[Species]
    reactions: List[Reaction]
    incidence: jnp.array

    def __init__(self, species, reactions):
        self.species = species  # S
        self.reactions = reactions  # R
        self.incidence = self.construct_incidence(self.species, self.reactions)  # S, R
        self.jreactions = []

    def construct_incidence(self, species, reactions):
        index = {species: idx for idx, species in enumerate(species)}
        incidence = jnp.zeros((len(species), len(reactions)))
        # Fill the incidence matrix with all terms:
        for j, reaction in enumerate(reactions):
            for reactant in reaction.reactants:
                incidence = incidence.at[index[reactant], j].set(-1)
            for product in reaction.products:
                incidence = incidence.at[index[product], j].set(1)
        return incidence

    def get_ode(self):
        self.jreactions = [reaction() for reaction in self.reactions]
        return JNetwork(self.incidence, self.jreactions)

        # convert species to JSpecies
        # convert reactions to JReactions
        # return JNetwork(self.species, self.reactions)
