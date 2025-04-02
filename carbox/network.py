from typing import List

from reactions import JReactionRateTerm, Reaction
from species import Species
from dataclasses import dataclass
import jax.numpy as jnp
from jax.experimental import sparse
import jax
import equinox as eqx

from functools import partial


class JNetwork(eqx.Module):
    incidence: jnp.array
    reactions: List[JReactionRateTerm]
    reactant_multipliers: jnp.array

    def __init__(self, incidence, reactions, dense=True):
        self.incidence = incidence  # S, R
        self.reactions = reactions  # R
        self.reactant_multipliers = self.get_reactant_multipliers(incidence)

    def get_reactant_multipliers(self, incidence):
        # In order to correctly get the flux, we need to multiply the rates per reaction
        # by the abundances of the reactants. This is done by getting the indices of the
        # reactants that need to be multiplied by the abundances and ensure they are repeated
        # the correct number of times. Use double entries to avoid power in the computation.
        if isinstance(incidence, sparse.BCOO):
            reactants_for_multiply = jnp.argwhere(incidence.todense().T < 0)
            times_for_multiply = -incidence[
                reactants_for_multiply[:, 1], reactants_for_multiply[:, 0]
            ].todense()
        else:
            reactants_for_multiply = jnp.argwhere(incidence.T < 0)
            times_for_multiply = -incidence[
                reactants_for_multiply[:, 1], reactants_for_multiply[:, 0]
            ]
        # We cannot do multiplies with duplicate entries, so create an array
        # with two columns, one for each of the reactants. The second row
        # is filled with an unreachable index, which we ignore by using "drop" in the
        # multiply operation. See: https://github.com/jax-ml/jax/issues/9296
        filler_value = incidence.shape[0] + 1
        reactant_multiplier = jnp.full((incidence.shape[1], 2), filler_value)
        for (reactant_idx, spec_idx), multiplier in zip(
            reactants_for_multiply, times_for_multiply
        ):
            # multiplier allows us to reactions with identical reactants: H + H -> H2
            for i in range(multiplier):
                # Write the first column if there is still a filler value:
                if reactant_multiplier[reactant_idx, 0] == filler_value:
                    reactant_multiplier = reactant_multiplier.at[reactant_idx, 0].set(
                        spec_idx
                    )
                # Else, write the second column:
                else:
                    reactant_multiplier = reactant_multiplier.at[reactant_idx, 1].set(
                        spec_idx
                    )
        return reactant_multiplier

    @jax.jit
    def get_rates(self, temperature, cr_rate, fuv_rate):
        """
        Get the reaction rates for the given temperature, cosmic ray ionisation rate,
        and FUV radiation field.
        """
        # TODO: optimization: The most Jax way to do optimize would be to create one class with all the reactions of one type and all their constants.
        # rates = jnp.empty(len(self.reactions))
        # for i, reaction in enumerate(self.reactions):
        #     rates = rates.at[i].set(reaction(temperature, cr_rate, fuv_rate))
        # return rates
        return jnp.hstack(
            [reaction(temperature, cr_rate, fuv_rate) for reaction in self.reactions]
        )

    @jax.jit
    def multiply_rates_by_abundance(self, rates, abundances):
        """
        Multiply the rates by the abundances of the reactants.
        """
        # We scatter the abunndances in two columns, with unity if it is monomolecular
        # This is achieved by "dropping" values we cannnot reach. Then take the product of each row, and mulitply it with the rates.
        rates_multiplier = jnp.ones_like(self.reactant_multipliers)
        rates_multiplier = jnp.prod(
            abundances.at[self.reactant_multipliers].get(mode="drop", fill_value=1.0),
            axis=1,
        )
        return rates * rates_multiplier

    @partial(jax.profiler.annotate_function, name="JNetwork._call__")
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
        rates = self.get_rates(temperature, cr_rate, fuv_rate)
        # jax.debug.print("rates: {rates}", rates=rates)
        # Get the matrix that encodes the reactants that need to be multiplied to get the flux
        rates = self.multiply_rates_by_abundance(rates, abundances)
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
    use_sparse: bool
    vectorize_reactions: bool

    def __init__(self, species, reactions, use_sparse=True, vectorize_reactions=True):
        self.species = species  # S
        self.reactions = reactions  # R
        self.use_sparse = use_sparse
        self.vectorize_reactions = vectorize_reactions
        self.jreactions = []
        # Sort the reactions by type if we need to vectorize them
        if vectorize_reactions:
            self.reactions = sorted(self.reactions, key=lambda x: x.reaction_type)
        # Create the incidence matrix
        self.incidence = self.construct_incidence(self.species, self.reactions)  # S, R

    def species_count(self):
        """
        Get the number of species in the network.
        """
        return self.incidence.shape[0]

    def reaction_count(self):
        """
        Get the number of reactions in the network.
        """
        return self.incidence.shape[1]

    def construct_incidence(self, species, reactions):
        index = {species: idx for idx, species in enumerate(species)}
        incidence = jnp.zeros((len(species), len(reactions)), dtype=jnp.int16)  # S, R
        # Fill the incidence matrix with all terms:
        for j, reaction in enumerate(reactions):
            for reactant in reaction.reactants:
                incidence = incidence.at[index[reactant], j].add(-1)
            for product in reaction.products:
                incidence = incidence.at[index[product], j].add(1)
        if self.use_sparse:
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
        elemental_content = jnp.zeros(
            (len(elements), self.species_count())
        )  # ELEMENTS, SPECIES
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
        # Always reset the jreactions
        self.jcreations = []
        if self.vectorize_reactions:
            reaction_groups = {}
            for reaction in self.reactions:
                if reaction.reaction_type not in reaction_groups:
                    reaction_groups[reaction.reaction_type] = []
                reaction_groups[reaction.reaction_type].append(reaction)
            reaction_classes = {
                reaction.reaction_type: type(reaction) for reaction in self.reactions
            }
            for reaction_type, grouped_reactions in reaction_groups.items():
                # Gather parameters for vectorization
                params = {
                    key: [getattr(reaction, key) for reaction in grouped_reactions]
                    for key in vars(grouped_reactions[0])
                }
                # The molecularity is infered from the number of reactants
                del params["molecularity"]
                vectorized_reaction = reaction_classes[reaction_type](**params)
                self.jreactions.append(vectorized_reaction())
        else:
            self.jreactions = [reaction() for reaction in self.reactions]
        return JNetwork(self.incidence, self.jreactions)
