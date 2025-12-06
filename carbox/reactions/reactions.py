"""Defines schemas for reactions."""

from dataclasses import dataclass

import equinox as eqx
import numpy as np

REACTION_SKIP_LIST = ["CRPHOT", "CRP", "PHOTON"]


class JReactionRateTerm(eqx.Module):
    """Base class for JAX-compatible reaction rate terms.

    All subclasses must implement __call__ with signature:
        __call__(self, temperature, cr_rate, uv_field, visual_extinction, abundance_vector)

    This ensures consistent signatures for JIT compilation.
    Reactions that don't need abundance_vector can simply ignore it.
    """


def valid_species_check(species):
    """Check if the species are valid, i.e., not in the skip list."""
    valid = False
    if isinstance(species, float):
        valid = ~np.isnan(species)
    elif isinstance(species, str):
        valid = species not in REACTION_SKIP_LIST
    return valid


@dataclass
class Reaction:
    """Dataclass for invidiual reactions."""

    reaction_type: str
    reactants: list[str]
    products: list[str]
    molecularity: int

    def __init__(self, reaction_type, reactants, products):  # noqa
        self.reactants = [r for r in reactants if valid_species_check(r)]
        self.products = [p for p in products if valid_species_check(p)]
        self.reaction_type = reaction_type
        self.molecularity = np.array(self.reactants).shape[-1]

    def __str__(self):
        return f"{self.reactants} -> {self.products}"

    def __repr__(self):
        return f"Reaction({self.reaction_type}, {self.reactants}, {self.products})\n"

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        # Abstract function to implement in subclasses
        raise NotImplementedError

    def __call__(self):
        return self._reaction_rate_factory()
