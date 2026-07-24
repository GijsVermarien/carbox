from typing import List, Optional

import pandas as pd

from ..network import Network
from ..reactions import CRPReaction, FUVReaction, KAReaction
from ..species import Species
from ..thermo import ThermoRate
from .base_parser import BaseParser


class LatentTGASParser(BaseParser):
    """
    Parser for latent_tgas reaction format - adapted from existing parser_latent_tgas.py

    This is a legacy adapter to integrate the existing latent_tgas parser
    with the unified parser architecture.
    """

    def __init__(self):
        super().__init__()
        self.format_type = "latent_tgas"

        # latent_tgas uses simplified 2-reactant → 2-product format
        self.expected_columns = [
            "reactant1",
            "reactant2",
            "product1",
            "product2",
            "alpha",
            "beta",
            "gamma",
            "reaction_type",
        ]

    def parse_network(self, filepath: str) -> Network:
        """Parse latent_tgas reactions file and return Network"""
        # Read CSV file ('#'-prefixed lines are comments, not data)
        df = pd.read_csv(filepath, comment="#")

        # Parse reactions
        reactions = []
        species_set = set()
        has_thermo = False

        for _, row in df.iterrows():
            reaction = self.parse_reaction(row)
            if reaction is not None:
                reactions.append(reaction)
                species_set.update(reaction.reactants)
                species_set.update(reaction.products)
                if isinstance(reaction, ThermoRate):
                    has_thermo = True

        species_set.discard("TGAS")

        # Create species list. TGAS (gas temperature, folded into the
        # abundance vector for thermal-balance networks) must be last so
        # ThermoRate/AbstractPhysics.integrates_temperature can rely on a
        # fixed position for it.
        species = [Species(name, 0.0) for name in sorted(species_set)]
        if has_thermo:
            species.append(Species("TGAS", 0.0))

        # Create network
        return Network(species, reactions, use_sparse=True, vectorize_reactions=True)

    def parse_reaction(self, row) -> Optional[KAReaction]:
        """Parse a single latent_tgas reaction row"""
        try:
            # Parse reactants and products (simplified 2→2 format)
            reactants = row.loc[["r1", "r2"]].dropna().tolist()
            products = row.loc[["p1", "p2"]].dropna().tolist()

            # Get reaction type if available
            reaction_type = row.get("mechanism", None)

            # Normalize parameters to standard Arrhenius form
            alpha, beta, gamma = self.normalize_arrhenius_params(row, "latent_tgas")

            # Create standard Arrhenius reaction
            if reaction_type == "CR":
                return CRPReaction(reaction_type, reactants, products, alpha)
            elif reaction_type == "FUV":
                return FUVReaction(reaction_type, reactants, products, alpha)
            elif reaction_type == "H2Form":
                return CRPReaction(reaction_type, reactants, products, alpha)
            elif reaction_type == "KA":
                return KAReaction(
                    reaction_type, reactants, products, alpha, beta, gamma
                )
            elif reaction_type == "THERMO":
                return ThermoRate(reaction_type, reactants, products)
            else:
                raise ValueError(f"Unknown reaction type: {reaction_type}")

        except Exception as e:
            print(f"Warning: Failed to parse latent_tgas reaction: {e}")
            return None
