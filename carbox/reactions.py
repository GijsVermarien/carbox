import equinox as eqx
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

    # def __init__(self, reactants, products, reaction_type):
    #     self.reactants = reactants
    #     self.products = products
    #     self.reaction_type = reaction_type

    def __str__(self):
        return f"{self.reactants} -> {self.products}"

    def __repr__(self):
        return f"Reaction({self.reaction_type}, {self.reactants}, {self.products})"

    def _reaction_rate_factory() -> JReactionRateTerm:
        # Abstract function to implement in subclasses
        raise NotImplementedError

    def __call__(self):
        return self._reaction_rate_factory()


class KAReaction(Reaction):
    def __init__(self, name, reactants, products, alpha, beta, gamma):
        super().__init__(name, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class KAReactionRateTerm(JReactionRateTerm):
            alpha: float
            beta: float
            gamma: float

            def __call__(self, temperature, cr_rate, uv_field):
                # α(T/300K​)βexp(−γ/T)
                return (
                    self.alpha
                    * jnp.power(0.0033333333333333335 * temperature, self.beta)
                    * jnp.exp(-self.gamma / temperature)
                )

        return KAReactionRateTerm(self.alpha, self.beta, self.gamma)


class CRReaction(Reaction):
    def __init__(self, name, reactants, products, alpha):
        super().__init__(name, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class CRReactionRateTerm(JReactionRateTerm):
            alpha: float

            def __call__(self, temperature, cr_rate, uv_field):
                return self.alpha * cr_rate

        return CRReactionRateTerm(self.alpha)


class FUVReaction(Reaction):
    def __init__(self, name, reactants, products, alpha):
        super().__init__(name, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class FUVReactionRateTerm(JReactionRateTerm):
            alpha: float

            def __call__(self, temperature, cr_rate, uv_field):
                return self.alpha * uv_field

        return FUVReactionRateTerm(self.alpha)


class H2FormReaction(Reaction):
    def __init__(self, name, reactants, products, alpha, gas2dust):
        super().__init__(name, reactants, products)
        self.alpha = alpha
        self.gas2dust = gas2dust

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class UVReactionRateTerm(JReactionRateTerm):
            alpha: float
            gas2dust: float

            def __call__(self, temperature, cr_rate, uv_field):
                return 0.01 * self.alpha * self.gas2dust

        return UVReactionRateTerm(self.alpha, self.gas2dust)
