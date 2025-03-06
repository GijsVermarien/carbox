import equinox as eqx
import jax.numpy as jnp
import dataclass as dataclass


class JReactionRateTerm(eqx.Module):
    pass


# Concept:
# The reaction network consists of abstract Species and Reactions, which are subclassed to reflect
# different reactions. They are objects that interact nicely at the user level.
# Each of these reactions needs to produce some ReactionRateTerm that takes: (T, density, ...) with its
# own secret bits implemented as pytree variables + a function transform. These reactions can then be combined
# by into a ChemicalNetwork that has a immutable copy of both the reaction network (objects) and the terms (pytree).


class Reaction(dataclass):
    reactants: list[str]
    products: list[str]

    def __init__(self, name, reactants, products, rate):
        self.reactants = reactants
        self.products = products
        self.rate = rate

    def __str__(self):
        return f"{self.reactants} -> {self.products}"

    def __repr__(self):
        return f"Reaction({self.name}, {self.reactants}, {self.products}, {self.rate})"

    def _reaction_rate_factory() -> JReactionRateTerm:
        def reaction_rate(temperature, density):
            return 1e-20 * temperature

        return None

    def __call__(self):
        return self._reaction_rate_factory()


class KAReaction(Reaction):
    def __init__(self, name, reactants, products, rate):
        super().__init__(name, reactants, products, rate)

    def _reaction_rate_factory(*, alpha, beta, gamma) -> JReactionRateTerm:
        class KAReactionRateTerm(JReactionRateTerm):
            alpha: float
            beta: float
            gamma: float

            def __call__(self, temperature):
                # α(T/300K​)βexp(−γ/T)
                return (
                    alpha
                    * jnp.power(0.0033333333333333335 * temperature, beta)
                    * jnp.exp(-gamma / temperature)
                )

        return KAReactionRateTerm


class CRReaction(Reaction):
    def __init__(self, name, reactants, products, rate):
        super().__init__(name, reactants, products, rate)

    def _reaction_rate_factory(*, alpha, beta, gamma) -> JReactionRateTerm:
        class CRReactionRateTerm(JReactionRateTerm):
            alpha: float
            beta: float
            gamma: float

            def __call__(self, temperature):
                # α(T/300K​)βexp(−γ/T)
                return (
                    alpha
                    * jnp.power(0.0033333333333333335 * temperature, beta)
                    * jnp.exp(-gamma / temperature)
                )

        return CRReactionRateTerm
