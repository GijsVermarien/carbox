import equinox as eqx
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass


class JReactionRateTerm(eqx.Module):
    pass


# Concept:
# The reaction network consists of abstract Species and Reactions, which are subclassed to reflect
# different reactions. They are objects that interact nicely at the user level.
# Each of these reactions needs to produce some ReactionRateTerm that takes: (T, density, ...) with its
# own secret bits implemented as pytree variables + a function transform. These reactions can then be combined
# by into a ChemicalNetwork that has a immutable copy of both the reaction network (objects) and the terms (pytree).


@dataclass
class Reaction:
    reaction_type: str
    reactants: list[str]
    products: list[str]
    molecularity: int

    def __init__(self, reaction_type, reactants, products):
        self.reactants = [
            r for r in reactants if (~np.isnan(r) if isinstance(r, float) else True)
        ]
        self.products = [
            p for p in products if (~np.isnan(p) if isinstance(p, float) else True)
        ]
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

            def __call__(self, temperature, cr_rate, uv_field):
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

            def __call__(self, temperature, cr_rate, uv_field):
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

            def __call__(self, temperature, cr_rate, uv_field):
                return self.alpha * cr_rate

        return CRReactionRateTerm(jnp.array(self.alpha))


class FUVReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class FUVReactionRateTerm(JReactionRateTerm):
            alpha: float

            def __call__(self, temperature, cr_rate, uv_field):
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

            def __call__(self, temperature, cr_rate, uv_field):
                return 100.0 * self.gas2dust * self.alpha

        return H2ReactionRateTerm(jnp.array(self.alpha), jnp.array(self.gas2dust))
