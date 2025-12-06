"""Using the carbox parser for parsing umist network."""

import numpy as np
import pandas as pd

from ..network import Network
from ..reactions import CRPReaction, FUVReaction, KAReaction, Reaction
from ..species import Species
from .base_parser import BaseParser


class UMISTParser(BaseParser):
    """Parser for UMIST reaction format - adapted from existing parser_umist.py.

    This is a legacy adapter to integrate the existing UMIST parser
    with the unified parser architecture.
    """

    def __init__(self):  # noqa
        format_type = "umist"
        super().__init__(format_type)

        # UMIST reaction type mapping
        self.reaction_type_mapping = {
            "AD": "associative_detachment",
            "CD": "collisional_dissociation",
            "CE": "charge_exchange",
            "CP": "cosmic_ray_proton",
            "CR": "cosmic_ray",
            "DR": "dissociative_recombination",
            "IA": "ion_association",
            "IN": "ion_neutral",
            "MN": "mutual_neutralization",
            "NN": "neutral_neutral",
            "PH": "photoionization",
            "PD": "photodissociation",
            "RA": "radiative_association",
            "REA": "radiative_electron_attachment",
            "RR": "radiative_recombination",
        }

    def parse_network(self, filepath: str) -> Network:
        """Parse UMIST reactions file and return Network."""
        # Read colon-separated file
        reactions_data = []

        # Read colon-separated file using pandas
        df = pd.read_csv(
            filepath, sep=":", comment="#", names=range(46), header=None
        ).iloc[:, :18]

        # Convert to list format for compatibility with existing code
        reactions_data = df.values.tolist()
        # Convert to DataFrame for easier processing
        columns = [
            "reaction_number",
            "reaction_type",
            "reactant_1",
            "reactant_2",
            "product_1",
            "product_2",
            "product_3",
            "product_4",
            "stoich_reactant_1",
            "alpha",
            "beta",
            "gamma",
            "tlow",
            "thigh",
            "uncertainty_flag",
            "source",
            "reference_1",
            "reference_2",
            "notes",
        ]
        df = pd.DataFrame(reactions_data, columns=columns[: len(reactions_data[0])])

        # Parse reactions
        reactions = []
        species_set = set()

        for _, row in df.iterrows():
            reaction = self.parse_reaction(row)
            if reaction is not None:
                reactions.append(reaction)
                species_set.update(reaction.reactants)
                species_set.update(reaction.products)

        # Create species list
        species = [Species(name, 0.0) for name in sorted(species_set)]

        # Create network
        return Network(species, reactions, use_sparse=True, vectorize_reactions=True)

    def parse_reaction(self, row) -> Reaction | None:
        """Parse a single UMIST reaction row."""
        try:
            # Parse reactants and products
            reactants = self._parse_species_list(
                row["reactant_1"]
            ) + self._parse_species_list(row["reactant_2"])
            products = (
                self._parse_species_list(row["product_1"])
                + self._parse_species_list(row["product_2"])
                + self._parse_species_list(row["product_3"])
                + self._parse_species_list(row["product_4"])
            )

            # Get reaction type
            reaction_type = row["reaction_type"]

            # Normalize parameters to standard Arrhenius form
            alpha, beta, gamma = self.normalize_arrhenius_params(row, "umist")

            # Map to appropriate reaction class
            if reaction_type in ["CP", "CR"]:
                return CRPReaction(reaction_type, reactants, products, alpha)
            elif reaction_type in ["PH", "PD"]:
                return FUVReaction(reaction_type, reactants, products, alpha)
            else:
                return KAReaction(
                    reaction_type, reactants, products, alpha, beta, gamma
                )

        except Exception as e:
            print(f"Warning: Failed to parse UMIST reaction: {e}")
            return None

    def _parse_species_list(self, species_str: str) -> list[str]:
        """Parse UMIST species list (space or + separated)."""
        if (
            isinstance(species_str, str)
            and (not species_str or species_str.strip() == "")
            or isinstance(species_str, float)
            and np.isnan(species_str)
        ):
            return []

        # Handle both space and + separators
        species = species_str.replace("+", " ").split()

        # Clean species names
        cleaned_species = []
        for sp in species:
            sp = sp.strip()
            if sp and sp != "hv":  # Remove photon notation
                cleaned_species.append(sp)

        return cleaned_species
