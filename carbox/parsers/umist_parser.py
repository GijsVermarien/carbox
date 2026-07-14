from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..network import Network
from ..reactions import CRPReaction, KAReaction, UMISTPhotoReaction
from ..species import Species
from .base_parser import BaseParser


# UMIST reaction-rate accuracy classes (McElroy et al. 2013, Table 3):
# A = uncertain to 25%, B = 50%, C = a factor of 2, D = an order of
# magnitude, E = worse than an order of magnitude / unknown. Expressed here
# as a multiplicative factor F such that the tabulated rate is believed to
# lie within roughly [k/F, k*F] (A/B use 1+fraction rather than a true
# ratio, matching the convention the fractions were given in).
DEFAULT_UNCERTAINTY_FACTORS = {"A": 1.25, "B": 1.50, "C": 2.0, "D": 10.0, "E": 10.0}


class UMISTParser(BaseParser):
    """
    Parser for UMIST reaction format - adapted from existing parser_umist.py

    This is a legacy adapter to integrate the existing UMIST parser
    with the unified parser architecture.
    """

    def __init__(self, uncertainty_factors: Optional[Dict[str, float]] = None):
        super().__init__()
        self.format_type = "umist"
        # Customizable mapping from the UMIST accuracy letter (A-E) to a
        # numeric multiplicative uncertainty factor. Override to use a
        # different convention without touching this file.
        self.uncertainty_factors = uncertainty_factors or DEFAULT_UNCERTAINTY_FACTORS

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
        """Parse UMIST reactions file and return Network"""
        # Read colon-separated file
        reactions_data = []

        # Read colon-separated file using pandas
        df = pd.read_csv(
            filepath, sep=":", comment="#", names=range(46), header=None
        ).iloc[:, :18]

        # Convert to list format for compatibility with existing code
        reactions_data = df.values.tolist()
        # Convert to DataFrame for easier processing.
        # Note: columns 15/16 are named for what they actually contain, not
        # the generic names an earlier version of this parser used --
        # verified against data/umist22.csv: column 15 ("measurement_type")
        # holds letters {L, M, C, E} (literature/measured/calculated/
        # estimated); column 16 ("accuracy_flag", directly before the DOI)
        # holds the real {A, B, C, D, E} accuracy classes from McElroy et
        # al. (2013), Table 3 -- see DEFAULT_UNCERTAINTY_FACTORS above.
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
            "measurement_type",
            "accuracy_flag",
            "reference_1",
            "reference_2",
            "notes",
        ]
        df = pd.DataFrame(reactions_data, columns=columns[: len(reactions_data[0])])


        # # Parse reactions
        # parsed_reactions = df.apply(self.parse_reaction, axis=1)
        # reactions = parsed_reactions.dropna().tolist()

        # # Get species set
        # species_set = set()
        # for reaction in reactions:
        #     species_set.update(reaction.reactants)
        #     species_set.update(reaction.products)

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
        return Network(species, reactions, use_sparse=False, vectorize_reactions=True)

    def parse_reaction(self, row) -> Optional[KAReaction]:
        """Parse a single UMIST reaction row"""
        try:
            # Parse reactants and products
            reactants = (
                self._parse_species_list(row["reactant_1"]) 
                + self._parse_species_list(row["reactant_2"])
                )
            products = (
                self._parse_species_list(row["product_1"])
                + self._parse_species_list(row["product_2"])
                + self._parse_species_list(row["product_3"])
                + self._parse_species_list(row["product_4"])
            )

            # Get reaction type
            reaction_type = row["reaction_type"]
            reaction_id = row["reaction_number"] # Get reaction_id

            # Normalize parameters to standard Arrhenius form
            alpha, beta, gamma = self.normalize_arrhenius_params(row, "umist")

            # Map to appropriate reaction class
            if reaction_type in ["CP", "CR"]:
                reaction = CRPReaction(reaction_type, reactants, products, alpha, reaction_id=reaction_id)
            elif reaction_type in ["PH", "PD"]:
                reaction = UMISTPhotoReaction(reaction_type, reactants, products, alpha, beta, gamma, reaction_id=reaction_id)
            else:
                reaction = KAReaction(
                    reaction_type, reactants, products, alpha, beta, gamma, reaction_id=reaction_id
                )

            # Attach the accuracy classification as metadata (not a
            # constructor arg -- see network.get_ode(), which strips these
            # before rebuilding vectorized reaction groups). Defaults to a
            # multiplicative factor of 1.0 (no info) for missing/unknown flags.
            accuracy_flag = row.get("accuracy_flag")
            reaction.uncertainty_flag = accuracy_flag
            reaction.uncertainty_factor = self.uncertainty_factors.get(accuracy_flag, 1.0)

            return reaction

        except Exception as e:
            print(f"Warning: Failed to parse UMIST reaction: {e}")
            return None

    def _parse_species_list(self, species_str: str) -> List[str]:
        """Parse a single UMIST reactant/product field.

        Each reactant/product already occupies its own column in the UMIST
        rate file, so a field holds exactly one species name. It must NOT be
        split on "+", since "+" is part of many species' names (e.g. "CO+",
        "H2+", "HCO+") rather than a separator between multiple species --
        stripping it silently turns every cation into its neutral
        counterpart and merges the two species' abundances.
        """
        if isinstance(species_str, str) and (
            not species_str or species_str.strip() == ""
        ):
            return []
        elif isinstance(species_str, float) and np.isnan(species_str):
            return []

        sp = species_str.strip()
        if not sp or sp == "hv":  # Remove photon notation
            return []

        return [sp]
