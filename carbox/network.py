from typing import List

from reactions import Reaction
from species import Species
from dataclasses import dataclass
import jax.numpy as jnp
from jax.experimental import sparse
import jax
import equinox as eqx


class JNetwork(eqx.Module):
    incidence: jnp.array
    reactions: List[Reaction]
    reactant_multiply_indices: jnp.array

    def __init__(self, incidence, reactions):
        self.incidence = incidence
        self.reactions = reactions
        self.reactant_multiply_indices = jnp.argwhere(self.incidence.todense().T < 0)

    def __call__(
        self,
        time: jnp.array,
        abundances: jnp.array,
        temperature: jnp.array,
        # density: jnp.array,
        cr_rate: jnp.array,
        fuv_rate: jnp.array,
    ) -> jnp.array:
        # abundances = abundances * density
        # Calculate the reaction rates
        rates = jnp.zeros(len(self.reactions))
        for i, reaction in enumerate(self.reactions):
            rates = rates.at[i].set(reaction(temperature, cr_rate, fuv_rate))
        # jax.debug.print("rates: {rates}", rates=rates)
        # Get the matrix that encodes the reactants that need to be multiplied to get the flux
        for reac_idx, species_idx in self.reactant_multiply_indices:
            rates = rates.at[reac_idx].set(rates[reac_idx] * abundances[species_idx])
        # Calculate the change in abundances
        # TODO: check that we are not loosing too much precision with the matmul?
        # Use BCCOO to avoid conversion to dense
        return self.incidence @ rates
        # Regular implmentation with dense matrix and highest precision
        # return jnp.matmul(self.incidence, rates, precision=jax.lax.Precision.HIGHEST)


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
        incidence = jnp.zeros((len(species), len(reactions)), dtype=jnp.int8)  # S, R
        # Fill the incidence matrix with all terms:
        for j, reaction in enumerate(reactions):
            for reactant in reaction.reactants:
                incidence = incidence.at[index[reactant], j].add(-1)
            for product in reaction.products:
                incidence = incidence.at[index[product], j].add(1)
        if True:
            incidence = sparse.BCOO.fromdense(incidence)
        return incidence

    def get_index(self, species: str) -> int:
        """
        Get the index of a species in the network.
        """
        return self.species.index(species)

    def get_elemental_contents(self, elements=["C", "H", "O", "charge"]):
        """
        Get the elemental contents of the species in the network.
        """
        # Create a dictionary to map species to their elemental content
        element_map = {species: idx for idx, species in enumerate(elements)}
        # Create an empty array to store the elemental content
        elemental_content = jnp.zeros((len(elements), len(self.species)))
        # Fill the elemental content array with the elemental content of each species
        for i, species in enumerate(self.species):
            for element in elements:
                if element in species:
                    # acount for number of atoms in the species
                    species_string_index = species.index(element)
                    # Get the number of atoms of the element in the species
                    if (
                        species_string_index + 1 < len(species)
                        and species[species_string_index + 1].isdigit()
                    ):
                        number_of_atoms = int(species[species_string_index + 1])
                    else:
                        number_of_atoms = 1
                    elemental_content = elemental_content.at[
                        element_map[element], i
                    ].set(number_of_atoms)

        return elemental_content

    def get_ode(self):
        self.jreactions = [reaction() for reaction in self.reactions]
        return JNetwork(self.incidence, self.jreactions)

        # convert species to JSpecies
        # convert reactions to JReactions
        # return JNetwork(self.species, self.reactions)
