from typing import Optional

import pandas as pd

from ..network import Network
from ..reactions import (
    CIonizationReaction,
    COPhotoDissReaction,
    CRPhotoReaction,
    CRPReaction,
    GARReaction,
    H2PhotoDissReaction,
    IonPol1Reaction,
    IonPol2Reaction,
    KAReaction,
    UCLCHEMH2FormReaction,
    UCLCHEMPhotonReaction,
)
from ..species import Species
from .base_parser import BaseParser

# Mapping different format reaction types to existing Carbox classes
UNIFIED_REACTION_MAPPING = {
    # UCLCHEM reaction types (gas-phase only)
    "PHOTON": UCLCHEMPhotonReaction,  # Direct photodissociation → FUV
    "CRP": CRPReaction,  # Cosmic ray proton → CR
    "CRPHOT": CRPhotoReaction,  # Cosmic ray photodissociation → CRPhoto
    "TWOBODY": KAReaction,  # Standard bimolecular → Arrhenius
    "IONOPOL1": IonPol1Reaction,  # Ion-polar molecule (KIDA formula 1)
    "IONOPOL2": IonPol2Reaction,  # Ion-polar molecule (KIDA formula 2)
    "H2FORM": UCLCHEMH2FormReaction,  # H2 formation on grains (Cazaux & Tielens)
}


class UCLCHEMParser(BaseParser):
    """Parser for UCLCHEM reaction format - gas-phase reactions only"""

    def __init__(self, cloud_radius_pc: float = 1.0, number_density: float = 1e4):
        super().__init__()
        self.format_type = "uclchem"

        # Cloud geometry parameters for photoreaction shielding
        self.cloud_radius_pc = cloud_radius_pc
        self.number_density = number_density

        # Gas-phase reaction types (including H2 formation on grains)
        self.gas_phase_reaction_types = {
            "PHOTON",
            "CRP",
            "CRPHOT",
            "TWOBODY",
            "IONOPOL1",
            "IONOPOL2",
            "CRS",
            "GAR",
            "H2FORM",
        }

        # Surface/grain reaction types to exclude
        self.surface_reaction_types = {
            "FREEZE",
            "DESORB",
            "ER",
            "ERDES",
            "LH",
            "LHDES",
            "BULKSWAP",
            "SURFSWAP",
            "THERM",
            "DESOH2",
            "DESCR",
            "DEUVCR",
            "EXSOLID",
            "EXRELAX",
        }

    def parse_network(self, filepath: str) -> Network:
        """Parse UCLCHEM reactions file and return Network"""
        # Read CSV file
        df = pd.read_csv(filepath)

        # Filter to gas-phase reactions only
        df_gas = self._filter_gas_phase_reactions(df)

        # Parse reactions
        reactions = []
        species_set = set()

        for _, row in df_gas.iterrows():
            reaction = self.parse_reaction(row)

            if reaction is not None:
                reactions.append(reaction)
                species_set.update(reaction.reactants)
                species_set.update(reaction.products)

        # Create species list (simplified - masses would need to be calculated)
        species = [Species(name, 0.0) for name in sorted(species_set)]

        # Create species name -> index mapping
        species_name_to_index = {s.name: i for i, s in enumerate(species)}

        # Separate special photoreactions and move them to the end
        # This ensures they don't interfere with vectorization of regular reactions
        non_vectorizable_types = (
            H2PhotoDissReaction,
            COPhotoDissReaction,
            CIonizationReaction,
        )

        regular_reactions = []
        special_reactions = []
        special_reaction_count = 0

        for reaction in reactions:
            if isinstance(reaction, non_vectorizable_types):
                special_reactions.append(reaction)
                special_reaction_count += 1
                # Write the indices of key species into the special reactions
                if isinstance(reaction, H2PhotoDissReaction):
                    reaction.h2_species_index = species_name_to_index["H2"]
                elif isinstance(reaction, COPhotoDissReaction):
                    reaction.h2_species_index = species_name_to_index["H2"]
                    reaction.co_species_index = species_name_to_index["CO"]
                elif isinstance(reaction, CIonizationReaction):
                    reaction.c_species_index = species_name_to_index["C"]
                    reaction.h2_species_index = species_name_to_index["H2"]
            else:
                regular_reactions.append(reaction)

        # Reorder: regular reactions first, then special reactions at the end
        reactions = regular_reactions + special_reactions

        print(f"✓ Detected {special_reaction_count} special photoreactions")
        # Create network with vectorization enabled
        return Network(species, reactions, use_sparse=True, vectorize_reactions=True)

    def parse_reaction(self, row) -> Optional[KAReaction]:
        """Parse a single UCLCHEM reaction row"""
        # Skip surface reactions
        if not self._is_gas_phase_reaction(row):
            return None

        # Extract and normalize species names
        reactants = [
            self._clean_species_name(row["Reactant 1"]),
            self._clean_species_name(row["Reactant 2"]),
            self._clean_species_name(row["Reactant 3"]),
        ]
        products = [
            self._clean_species_name(row["Product 1"]),
            self._clean_species_name(row["Product 2"]),
            self._clean_species_name(row["Product 3"]),
            self._clean_species_name(row["Product 4"]),
        ]

        # Filter out None values and special reactants
        reactants = [
            r
            for r in reactants
            if r is not None and r not in self.gas_phase_reaction_types
        ]
        products = [p for p in products if p is not None]

        # Get reaction type and map to Carbox reaction class
        reaction_type = self._identify_reaction_type(row)
        reaction_class = UNIFIED_REACTION_MAPPING.get(reaction_type, None)

        # Normalize parameters
        alpha, beta, gamma = self.normalize_arrhenius_params(row, "uclchem")

        # Check for special photoreaction patterns requiring self-shielding
        if reaction_class == UCLCHEMPhotonReaction:
            # Pattern: H2 + PHOTON -> H + H
            if reactants == ["H2"] and products == ["H", "H"]:
                # Use unique type so it doesn't get vectorized with other PHOTON reactions
                return H2PhotoDissReaction(
                    "H2PHOTODISS",  # Unique type for special H2 photodissociation
                    reactants,
                    products,
                    cloud_radius_pc=self.cloud_radius_pc,
                    number_density=self.number_density,
                    h2_species_index=None,
                )
            # Pattern: CO + PHOTON -> O + C (note: order in file is O, C not C, O)
            elif reactants == ["CO"] and set(products) == {"C", "O"}:
                # Use unique type so it doesn't get vectorized with other PHOTON reactions
                return COPhotoDissReaction(
                    "COPHOTODISS",  # Unique type for special CO photodissociation
                    reactants,
                    products,
                    cloud_radius_pc=self.cloud_radius_pc,
                    number_density=self.number_density,
                    h2_species_index=None,
                    co_species_index=None,
                )
            # Pattern: C + PHOTON -> C+ + E-
            elif reactants == ["C"] and set(products) == {"C+", "E-"}:
                # Use unique type so it doesn't get vectorized with other PHOTON reactions
                return CIonizationReaction(
                    "CPHOTOION",  # Unique type for special C photoionization
                    reactants,
                    products,
                    alpha=alpha,
                    gamma=gamma,
                    cloud_radius_pc=self.cloud_radius_pc,
                    number_density=self.number_density,
                    c_species_index=None,
                    h2_species_index=None,
                )
            else:
                # Default photon reaction
                return UCLCHEMPhotonReaction(
                    reaction_type, reactants, products, alpha, beta, gamma
                )
        # Create appropriate reaction based on type
        elif reaction_class == CRPReaction:
            return CRPReaction(reaction_type, reactants, products, alpha)
        elif reaction_class == CRPhotoReaction:
            return CRPhotoReaction(
                reaction_type, reactants, products, alpha, beta, gamma
            )
        elif reaction_class == UCLCHEMH2FormReaction:
            # H2 formation uses physics-based rate, ignore alpha/beta/gamma
            return UCLCHEMH2FormReaction(reaction_type, reactants, products)
        elif reaction_class in [IonPol1Reaction, IonPol2Reaction, GARReaction]:
            return reaction_class(
                reaction_type, reactants, products, alpha, beta, gamma
            )
        elif reaction_class == KAReaction:  # Default to KAReaction (Arrhenius)
            return KAReaction(reaction_type, reactants, products, alpha, beta, gamma)
        else:
            raise ValueError(f"Unhandled reaction class: {reaction_class}")

    def _identify_reaction_type(self, row) -> str:
        """Identify reaction type from UCLCHEM format"""
        reactants = [row["Reactant 1"], row["Reactant 2"], row["Reactant 3"]]

        # Check third reactant first, then second (UCLCHEM convention)
        if pd.notna(reactants[2]) and reactants[2] in self.gas_phase_reaction_types:
            return reactants[2]
        elif pd.notna(reactants[1]) and reactants[1] in self.gas_phase_reaction_types:
            return reactants[1]
        else:
            return "TWOBODY"  # Standard bimolecular reaction

    def _is_gas_phase_reaction(self, row) -> bool:
        """Filter out surface/grain reactions for gas-phase only analysis"""
        reaction_type = self._identify_reaction_type(row)

        # Exclude surface reaction types
        if reaction_type in self.surface_reaction_types:
            return False

        # Exclude reactions involving surface species (@, #)
        all_species = [
            row["Reactant 1"],
            row["Reactant 2"],
            row["Reactant 3"],
            row["Product 1"],
            row["Product 2"],
            row["Product 3"],
            row["Product 4"],
        ]

        for species in all_species:
            if (
                pd.notna(species)
                and isinstance(species, str)
                and ("@" in species or "#" in species)
            ):
                return False

        return True

    def _filter_gas_phase_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter UCLCHEM reactions to include only gas-phase chemistry"""
        # Filter by reaction type
        gas_phase_mask = pd.Series(True, index=df.index)

        for col in ["Reactant 2", "Reactant 3"]:
            if col in df.columns:
                gas_phase_mask &= ~df[col].isin(self.surface_reaction_types)

        # Filter by surface species notation
        species_cols = [
            "Reactant 1",
            "Reactant 2",
            "Reactant 3",
            "Product 1",
            "Product 2",
            "Product 3",
            "Product 4",
        ]

        for col in species_cols:
            if col in df.columns:
                # Convert to string first to handle mixed types
                gas_phase_mask &= ~df[col].astype(str).str.contains("@|#", na=False)

        return df[gas_phase_mask]

    def _clean_species_name(self, name: str) -> Optional[str]:
        """Normalize UCLCHEM species names to Carbox format"""
        if pd.isna(name) or name == "NAN":
            return None

        # Normalize electron notation
        if name == "E-":
            return "E-"

        return str(name).strip()
