import equinox as eqx
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

REACTION_SKIP_LIST = ["CRPHOT", "CRP", "PHOTON"]


class JReactionRateTerm(eqx.Module):
    pass


# Concept:
# The reaction network consists of abstract Species and Reactions, which are subclassed to reflect
# different reactions. They are objects that interact nicely at the user level.
# Each of these reactions needs to produce some ReactionRateTerm that takes: (T, density, ...) with its
# own secret bits implemented as pytree variables + a function transform. These reactions can then be combined
# by into a ChemicalNetwork that has a immutable copy of both the reaction network (objects) and the terms (pytree).


def valid_species_check(species):
    """
    Check if the species are valid, i.e., not in the skip list.
    """
    valid = False
    if isinstance(species, float):
        valid = ~np.isnan(species)
    elif isinstance(species, str):
        valid = species not in REACTION_SKIP_LIST
    return valid


@dataclass
class Reaction:
    reaction_type: str
    reactants: list[str]
    products: list[str]
    molecularity: int

    def __init__(self, reaction_type, reactants, products):
        self.reactants = [r for r in reactants if valid_species_check(r)]
        self.products = [p for p in products if valid_species_check(p)]
        self.reaction_type = reaction_type
        self.molecularity = np.array(self.reactants).shape[-1]

    def __str__(self):
        return f"{self.reactants} -> {self.products}"

    def __repr__(self):
        return f"Reaction({self.reaction_type}, {self.reactants}, {self.products})\n"

    def _reaction_rate_factory() -> JReactionRateTerm:
        # Abstract function to implement in subclasses
        raise NotImplementedError

    def __call__(self):
        return self._reaction_rate_factory()


class KAReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class KAReactionRateTerm(JReactionRateTerm):
            alpha: float
            beta: float
            gamma: float

            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                # α(T/300K​)^βexp(−γ/T)
                return (
                    self.alpha
                    * jnp.power(0.0033333333333333335 * temperature, self.beta)
                    * jnp.exp(-self.gamma / temperature)
                )

        # Ensure that the parameters are JAX arrays!
        return KAReactionRateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )


class KAFixedReaction(Reaction):
    def __init__(
        self, reaction_type, reactants, products, alpha, beta, gamma, temperature
    ):
        super().__init__(reaction_type, reactants, products)
        self.reaction_coeff = (
            alpha
            * jnp.power(0.0033333333333333335 * temperature, beta)
            * jnp.exp(-gamma / temperature)
        )

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class KAFixedReactionRateTerm(JReactionRateTerm):
            reaction_coeff: float

            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                # α(T/300K​)βexp(−γ/T)
                return self.reaction_coeff

        return KAFixedReactionRateTerm(jnp.array(self.reaction_coeff))


class CRReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class CRReactionRateTerm(JReactionRateTerm):
            alpha: float

            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                return self.alpha * cr_rate

        return CRReactionRateTerm(jnp.array(self.alpha))


class FUVReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class FUVReactionRateTerm(JReactionRateTerm):
            alpha: float

            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                return self.alpha * uv_field

        return FUVReactionRateTerm(jnp.array(self.alpha))


class H2FormReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, gas2dust):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.gas2dust = gas2dust

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class H2ReactionRateTerm(JReactionRateTerm):
            alpha: float
            gas2dust: float

            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                return 100.0 * self.gas2dust * self.alpha

        return H2ReactionRateTerm(jnp.array(self.alpha), jnp.array(self.gas2dust))


# Reactions for UMIST:
class CPReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class CPReactionRateTerm(JReactionRateTerm):
            alpha: float

            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                return self.alpha

        return CPReactionRateTerm(jnp.array(self.alpha))


class PHReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class PHReactionRateTerm(JReactionRateTerm):
            alpha: float
            beta: float
            gamma: float

            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                return self.alpha * jnp.exp(-self.gamma * visual_extinction * 4.65)

        return PHReactionRateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )


class CRPhotoReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class CRReactionRateTerm(JReactionRateTerm):
            alpha: float
            beta: float
            gamma: float

            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                # α(T/300K​)^βexp(−γ/T)
                return (
                    cr_rate
                    * jnp.power(0.0033333333333333335 * temperature, self.beta)
                    * self.gamma
                    / (1 - 0.5)  # hardcoded omega value
                )

        return CRReactionRateTerm(
            jnp.array(self.alpha),
            jnp.array(self.beta),
            jnp.array(self.gamma),
        )


# UCLCHEM-specific reaction types
class IonPol1Reaction(Reaction):
    """UCLCHEM IONOPOL1: Ion-polar molecule reaction (KIDA formula 1)
    k = α * β * (0.62 + 0.4767 * γ * sqrt(300/T))
    """
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
    
    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class IonPol1RateTerm(JReactionRateTerm):
            alpha: float
            beta: float
            gamma: float
            
            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                return self.alpha * self.beta * (0.62 + 0.4767 * self.gamma * jnp.sqrt(300.0 / temperature))
                
        return IonPol1RateTerm(jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma))


class IonPol2Reaction(Reaction):
    """UCLCHEM IONOPOL2: Ion-polar molecule reaction (KIDA formula 2)
    k = α * β * (1.0 + 0.0967 * γ * sqrt(300/T) + γ² * 300/(10.526 * T))
    """
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class IonPol2RateTerm(JReactionRateTerm):
            alpha: float
            beta: float 
            gamma: float
            
            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                sqrt_term = 0.0967 * self.gamma * jnp.sqrt(300.0 / temperature)
                quadratic_term = self.gamma**2 * 300.0 / (10.526 * temperature)
                return self.alpha * self.beta * (1.0 + sqrt_term + quadratic_term)
                
        return IonPol2RateTerm(jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma))


class GARReaction(Reaction):
    """UCLCHEM GAR: Grain-assisted recombination (Weingartner & Draine 2001)
    Simplified implementation for gas-phase comparison
    """
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class GARRateTerm(JReactionRateTerm):
            alpha: float
            beta: float
            gamma: float
            
            def __call__(self, temperature, cr_rate, uv_field, visual_extinction):
                # Simplified implementation - treat as standard Arrhenius for gas-phase comparison
                # Full GAR formula requires grain parameters not available in gas-phase mode
                return self.alpha * jnp.power(temperature / 300.0, self.beta) * jnp.exp(-self.gamma / temperature)
                
        return GARRateTerm(jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma))
