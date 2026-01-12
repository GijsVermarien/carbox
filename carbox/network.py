from dataclasses import dataclass
from functools import partial
from typing import List, Union
from typing import List, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np

from .reactions import JReactionRateTerm, Reaction
from .species import Species


class _ScalarRateTermWrapper(eqx.Module):
    term: JReactionRateTerm

    def __call__(self, *args, **kwargs):
        return jnp.reshape(self.term(*args, **kwargs), (1,))


class JNetwork(eqx.Module):
    incidence: Union[jnp.ndarray, sparse.BCOO]
    reactions: List[JReactionRateTerm]
    reactant_multipliers: jnp.array

    def __init__(self, incidence, reactions, reactant_multipliers):
        # Ensure incidence is treated as a static structure if possible, 
        # but as a Module field it's fine as a jnp.array.
        self.incidence = incidence 
        self.reactions = reactions  
        self.reactant_multipliers = reactant_multipliers
        
        

    # @jax.jit
    # def get_rates(self, temperature, cr_rate, fuv_rate, visual_extinction, abundances):
    #     """
    #     Get the reaction rates for the given temperature, cosmic ray ionisation rate,
    #     FUV radiation field, and abundance vector.
    #     """
    #     # TODO: optimization: The most Jax way to do optimize would be to create one class with all the reactions of one type and all their constants.
    #     # rates = jnp.empty(len(self.reactions))
    #     # for i, reaction in enumerate(self.reactions):
    #     #     rates = rates.at[i].set(reaction(temperature, cr_rate, fuv_rate))
    #     # return rates
    #     return jnp.hstack(
    #         [
    #             reaction(temperature, cr_rate, fuv_rate, visual_extinction, abundances)
    #             for reaction in self.reactions
    #         ]
    #     )

    def get_rates(self, temperature, cr_rate, fuv_rate, visual_extinction, abundances):
        # List comprehension is okay for JIT, but it unrolls the loop.
        # For 'large' networks, this can make the graph massive.
        
        # for i, reaction in enumerate(self.reactions):
        #     rates = rates.at[i].set(reaction(temperature, cr_rate, fuv_rate))
        #     if i == 8259:
        #         jax.debug.print("Reaction 8259 rate: {rate}", rate=rates[i])
        
        return jnp.hstack([
            r(temperature, cr_rate, fuv_rate, visual_extinction, abundances)
            for r in self.reactions
        ])

    def multiply_rates_by_abundance(self, rates, abundances):
        """
        Multiply the rates by the abundances of the reactants.
        """
        # Optimization: Pad abundances with 1.0 to handle filler indices (set to N_species)
        # This avoids the slower .at[...].get(mode="fill") operation.
        padded_abundances = jnp.concatenate([abundances, jnp.array([1.0])])
        
        # Gather abundances: shape (n_reactions, 2)
        reactant_abunds = padded_abundances[self.reactant_multipliers]
        
        # Multiply the two columns: shape (n_reactions,)
        rates_multiplier = reactant_abunds[:, 0] * reactant_abunds[:, 1]
        
        return rates * rates_multiplier

    @partial(jax.profiler.annotate_function, name="JNetwork._call__")
    def __call__(
        self,
        time: jnp.array,
        abundances: jnp.array,
        temperature: jnp.array,
        cr_rate: jnp.array,
        fuv_rate: jnp.array,
        visual_extinction: jnp.array,
    ) -> jnp.array:
        # abundances = abundances * density
        # Calculate the reaction rates (pass abundances for self-shielding reactions)
        rates = self.get_rates(
            temperature, cr_rate, fuv_rate, visual_extinction, abundances
        )
        # jax.debug.print("Reaction 8258 rate: {rate}", rate=rates[8258])
        # jax.debug.print("Reaction 8259 rate: {rate}", rate=rates[8259])
        # jax.debug.print("Reaction 8260 rate: {rate}", rate=rates[8260])


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

        # Create the incidence matrix (S species, R reactions)
        self.incidence = self.construct_incidence(self.species, self.reactions)

    def get_reactant_multipliers(self, incidence):
        # In order to correctly get the flux, we need to multiply the rates per reaction
        # by the abundances of the reactants. This is done by getting the indices of the
        # reactants that need to be multiplied by the abundances and ensure they are repeated
        # the correct number of times. Use double entries to avoid power in the computation.
        # Extract data to CPU numpy arrays for fast processing
        if isinstance(incidence, sparse.BCOO):
            # Extract indices and data where stoichiometry is negative (reactants)
            # BCOO indices are (nse, 2) -> (species_idx, reaction_idx)
            indices = np.array(incidence.indices)
            data = np.array(incidence.data)

            mask = data < 0
            s_indices = indices[mask, 0]
            r_indices = indices[mask, 1]
            multipliers = -data[mask]

            # Stack as (reaction_idx, species_idx) for clarity
            reactants_for_multiply = np.stack((r_indices, s_indices), axis=1)
            times_for_multiply = multipliers
        else:
            # Dense case: convert to numpy to avoid JAX dispatch overhead in loop
            inc_np = np.array(incidence)
            # argwhere is slow, let's use np.where
            r_indices, s_indices = np.where(inc_np.T < 0)
            reactants_for_multiply = np.stack((r_indices, s_indices), axis=1)
            times_for_multiply = -inc_np[s_indices, r_indices]

        # --- Vectorized replacement for the loop ---
        
        # We cannot do multiplies with duplicate entries, so create an array
        # with two columns, one for each of the reactants.
        filler_value = incidence.shape[0]
        reactant_multiplier = np.full((incidence.shape[1], 2), filler_value, dtype=np.int32)
        
        # Repeat reactant entries based on their stoichiometry (multiplier)
        # e.g., for H+H, the entry for H is repeated twice
        expanded_reactants = np.repeat(reactants_for_multiply, times_for_multiply.astype(np.int32), axis=0)
        
        # Get unique reaction indices and the index where each reaction's reactants first appear
        unique_r_indices, first_occurrence_indices = np.unique(expanded_reactants[:, 0], return_index=True)
        
        # Assign the first reactant for each reaction
        reactant_multiplier[unique_r_indices, 0] = expanded_reactants[first_occurrence_indices, 1]
        
        # To find the second reactant, we can remove the first occurrences and repeat the process.
        # This is an efficient way to find the "second" item in each group.
        # Create a mask to select all but the first occurrences
        mask = np.ones(len(expanded_reactants), dtype=bool)
        mask[first_occurrence_indices] = False
        
        # Get the remaining reactants (which are the second reactants for each group)
        second_reactant_entries = expanded_reactants[mask]
        
        # Get unique reaction indices and their first appearance in this *reduced* set
        unique_r_indices_second, first_occurrence_indices_second = np.unique(second_reactant_entries[:, 0], return_index=True)
        
        # Assign the second reactant
        reactant_multiplier[unique_r_indices_second, 1] = second_reactant_entries[first_occurrence_indices_second, 1]

        return jnp.array(reactant_multiplier)


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
        import numpy as np
        from scipy import sparse as sp_sparse

        index = {sp.name: idx for idx, sp in enumerate(species)}
        
        # Build lists for COO construction on CPU
        rows = []
        cols = []
        data = []

        for j, reaction in enumerate(reactions):
            for reactant in reaction.reactants:
                rows.append(index[reactant])
                cols.append(j)
                data.append(-1)
            for product in reaction.products:
                rows.append(index[product])
                cols.append(j)
                data.append(1)

        shape = (len(species), len(reactions))
        
        if self.use_sparse:
            # Create JAX BCOO directly from coordinate lists
            indices = jnp.array(np.column_stack((rows, cols)))
            data_arr = jnp.array(data, dtype=jnp.int16)
            return sparse.BCOO((data_arr, indices), shape=shape)
        else:
            # Create dense matrix via scipy to avoid OOM on large zeros()
            coo = sp_sparse.coo_matrix((data, (rows, cols)), shape=shape, dtype=np.int16)
            return jnp.array(coo.todense())

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
        """
        Convert the reaction network to a NetworkX directed graph.

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
        self.jreactions = []

        # Import special reaction types that should not be vectorized
        from .reactions import (
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
            reordered_reactions = []

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
                
                # Track reactions in the new order for incidence matrix reconstruction
                reordered_reactions.extend(grouped_reactions)

            # Add non-vectorizable reactions individually
            for reaction in non_vectorizable_reactions:
                self.jreactions.append(_ScalarRateTermWrapper(reaction()))
                reordered_reactions.append(reaction)
            
            # Rebuild incidence matrix to match the vectorized order
            # This is crucial: column j in incidence must match the j-th rate in the rate vector
            incidence = self.construct_incidence(self.species, reordered_reactions)

            # Update the network's reactions to match the reordered list
            # This ensures that outputs (which iterate self.reactions) match the JNetwork rates
            self.reactions = reordered_reactions
            self.incidence = incidence
        else:
            self.jreactions = [reaction() for reaction in self.reactions]
            incidence = self.incidence

        reactant_multipliers = self.get_reactant_multipliers(incidence)

        return JNetwork(incidence, self.jreactions, reactant_multipliers=reactant_multipliers)
