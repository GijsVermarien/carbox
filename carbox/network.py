"""Defines the chemical network composed of species and the reactions between them."""

from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from reactions import JReactionRateTerm, Reaction
from species import Species


class JNetwork(eqx.Module):
    """Jax Jit compiled Network."""

    incidence: jnp.ndarray
    reactions: list[JReactionRateTerm]
    reactant_multipliers: jnp.ndarray

    def __init__(self, incidence, reactions, reactant_multipliers):
        self.incidence = incidence  # S, R
        self.reactions = reactions  # R
        self.reactant_multipliers = reactant_multipliers

    @jax.jit
    def get_rates(self, temperature, cr_rate, fuv_rate, visual_extinction, abundances):
        """Get the reaction rates for the given temperature, cosmic ray ionisation rate, FUV radiation field, and abundance vector."""
        # TODO: optimization: The most Jax way to do optimize would be to create one class with all the reactions of one type and all their constants.
        # rates = jnp.empty(len(self.reactions))
        # for i, reaction in enumerate(self.reactions):
        #     rates = rates.at[i].set(reaction(temperature, cr_rate, fuv_rate))
        # return rates
        return jnp.hstack(
            [
                reaction(temperature, cr_rate, fuv_rate, visual_extinction, abundances)
                for reaction in self.reactions
            ]
        )

    @jax.jit
    def multiply_rates_by_abundance(self, rates, abundances):
        """Multiply the rates by the abundances of the reactants."""
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
        visual_extinction: jnp.array,
    ) -> jnp.array:
        # abundances = abundances * density
        # Calculate the reaction rates (pass abundances for self-shielding reactions)
        rates = self.get_rates(
            temperature, cr_rate, fuv_rate, visual_extinction, abundances
        )
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
    """A full chemical network defined by the reactions between species."""

    species: list[Species]
    reactions: list[Reaction]
    incidence: jnp.ndarray
    reactant_multipliers: jnp.ndarray
    use_sparse: bool
    vectorize_reactions: bool

    def __init__(  # noqa
        self,
        species: list[Species],
        reactions: list[Reaction],
        use_sparse: bool = True,
        vectorize_reactions: bool = True,
    ) -> None:
        self.species = species  # S
        self.reactions = reactions  # R
        self.use_sparse = use_sparse
        self.vectorize_reactions = vectorize_reactions
        self.jreactions = []

        # Create the incidence matrix (S species, R reactions) and reactant multipliers
        self.incidence, self.reactant_multipliers = self.construct_incidence(
            self.species, self.reactions
        )

    def species_count(self):
        """Get the number of species in the network."""
        return self.incidence.shape[0]

    def reaction_count(self):
        """Get the number of reactions in the network."""
        return self.incidence.shape[1]

    def construct_incidence(self, species: list[Species], reactions):
        index = {sp.name: idx for idx, sp in enumerate(species)}
        incidence = jnp.zeros((len(species), len(reactions)), dtype=jnp.int16)  # S, R
        # Fill the incidence matrix with all terms:
        for j, reaction in enumerate(reactions):
            for reactant in reaction.reactants:
                incidence = incidence.at[index[reactant], j].add(-1)
            for product in reaction.products:
                incidence = incidence.at[index[product], j].add(1)

        # Compute reactant multipliers from dense incidence before sparsifying
        reactant_multipliers = self.compute_reactant_multipliers(incidence)

        if self.use_sparse:
            incidence = sparse.BCOO.fromdense(incidence)
        return incidence, reactant_multipliers

    def compute_reactant_multipliers(self, incidence):
        """Compute reactant multipliers from dense incidence matrix."""
        # In order to correctly get the flux, we need to multiply the rates per reaction
        # by the abundances of the reactants. This is done by getting the indices of the
        # reactants that need to be multiplied by the abundances and ensure they are repeated
        # the correct number of times. Use double entries to avoid power in the computation.
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

    def get_index(self, species: str) -> int:
        """Get the index of a species in the network."""
        return [sp.name for sp in self.species].index(species)

    def get_elemental_contents(self, elements=["C", "H", "O", "charge"]):
        """Get the elemental contents of the species in the network."""
        # Create a dictionary to map species to their elemental content
        element_map = {element: idx for idx, element in enumerate(elements)}
        # Create an empty array to store the elemental content
        elemental_content = jnp.zeros(
            (len(elements), self.species_count())
        )  # ELEMENTS, SPECIES
        # Fill the elemental content array with the elemental content of each species
        for i, species_obj in enumerate(self.species):
            species_name = species_obj.name
            for element in elements:
                if element in species_name:
                    # acount for number of atoms in the species
                    species_string_index = species_name.index(element)
                    # Get the number of atoms of the element in the species
                    if (
                        species_string_index + 1 < len(species_name)
                        and species_name[species_string_index + 1].isdigit()
                    ):
                        number_of_atoms = int(species_name[species_string_index + 1])
                    else:
                        number_of_atoms = 1
                    elemental_content = elemental_content.at[
                        element_map[element], i
                    ].set(number_of_atoms)
        return elemental_content

    def to_networkx(self):
        """Convert the reaction network to a NetworkX directed graph.

        Returns:
            networkx.DiGraph: A directed graph where:
                - Nodes represent chemical species
                - Edges represent reactions (reactant -> product)
                - Edge attributes include reaction index and reaction object
        """
        import networkx as nx

        # Create a directed graph
        G = nx.DiGraph()

        # Add all species as nodes
        for species in self.species:
            G.add_node(species.name, species=species)

        # Process each reaction (column in incidence matrix)
        for j, reaction in enumerate(self.reactions):
            # Get reactants and products for this reaction
            reactants = reaction.reactants
            products = reaction.products

            # Create reaction label
            reactants_str = " + ".join(reactants)
            products_str = " + ".join(products)
            reaction_label = f"{reactants_str} -> {products_str}"

            # Add edges from each reactant to each product
            for reactant in reactants:
                for product in products:
                    # Check if edge already exists
                    if G.has_edge(reactant, product):
                        # Append to existing reactions list
                        G[reactant][product]["reactions"].append(
                            {"index": j, "reaction": reaction, "label": reaction_label}
                        )
                    else:
                        # Create new edge with reactions list
                        G.add_edge(
                            reactant,
                            product,
                            reactions=[
                                {
                                    "index": j,
                                    "reaction": reaction,
                                    "label": reaction_label,
                                }
                            ],
                        )

        return G

    def get_ode(self):
        # Always reset the jreactions
        self.jcreations = []

        # Import special reaction types that should not be vectorized
        from .reactions.reactions import (
            CIonizationReaction,
            COPhotoDissReaction,
            H2PhotoDissReaction,
        )

        # Types that should not be vectorized due to unique parameters
        non_vectorizable_types = (
            H2PhotoDissReaction,
            COPhotoDissReaction,
            CIonizationReaction,
        )

        if self.vectorize_reactions:
            reaction_groups = {}
            non_vectorizable_reactions = []

            for reaction in self.reactions:
                # Skip vectorization for special photoreactions
                if isinstance(reaction, non_vectorizable_types):
                    non_vectorizable_reactions.append(reaction)
                else:
                    if reaction.reaction_type not in reaction_groups:
                        reaction_groups[reaction.reaction_type] = []
                    reaction_groups[reaction.reaction_type].append(reaction)

            reaction_classes = {
                reaction.reaction_type: type(reaction)
                for reaction in self.reactions
                if not isinstance(reaction, non_vectorizable_types)
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

            # Add non-vectorizable reactions individually
            for reaction in non_vectorizable_reactions:
                self.jreactions.append(reaction())
        else:
            self.jreactions = [reaction() for reaction in self.reactions]
        return JNetwork(self.incidence, self.jreactions, self.reactant_multipliers)
