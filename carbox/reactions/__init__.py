"""Reactions imports.

Concept:
The reaction network consists of abstract Species and Reactions, which are subclassed to reflect
different reactions. They are objects that interact nicely at the user level.
Each of these reactions needs to produce some ReactionRateTerm that takes: (T, density, ...) with its
own secret bits implemented as pytree variables + a function transform. These reactions can then be combined
by into a ChemicalNetwork that has a immutable copy of both the reaction network (objects) and the terms (pytree).
"""

from .latent_gas_reactions import FUVReaction, H2FormReaction
from .reactions import JReactionRateTerm, Reaction
from .uclchem_photoreactions import (
    c_ionization_rate,
    co_photo_diss_rate,
    co_self_shielding,
    compute_column_density,
    h2_photo_diss_rate,
    h2_self_shielding,
    lbar,
    scatter,
    xlambda,
)
from .uclchem_reactions import (
    CIonizationReaction,
    COPhotoDissReaction,
    GARReaction,
    H2PhotoDissReaction,
    IonPol1Reaction,
    IonPol2Reaction,
    UCLCHEMH2FormReaction,
    UCLCHEMPhotonReaction,
)
from .umist_reactions import UMISTPhotoReaction
from .universal_reactions import (
    CRPhotoReaction,
    CRPReaction,
    KAFixedReaction,
    KAReaction,
)

__all__ = [
    "FUVReaction",
    "H2FormReaction",
    "JReactionRateTerm",
    "Reaction",
    "c_ionization_rate",
    "co_photo_diss_rate",
    "co_self_shielding",
    "compute_column_density",
    "h2_photo_diss_rate",
    "h2_self_shielding",
    "lbar",
    "scatter",
    "xlambda",
    "UCLCHEMH2FormReaction",
    "UCLCHEMPhotonReaction",
    "IonPol1Reaction",
    "IonPol2Reaction",
    "GARReaction",
    "H2PhotoDissReaction",
    "COPhotoDissReaction",
    "CIonizationReaction",
    "UMISTPhotoReaction",
    "KAReaction",
    "KAFixedReaction",
    "CRPReaction",
    "CRPhotoReaction",
]
