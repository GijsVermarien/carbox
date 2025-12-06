"""Latent_tgas / prizmo specific reactions."""

import jax.numpy as jnp
from jax import Array

from . import JReactionRateTerm, Reaction


class FUVReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class FUVReactionRateTerm(JReactionRateTerm):
            alpha: Array

            def __call__(
                self,
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
            ):
                return self.alpha * uv_field

        return FUVReactionRateTerm(jnp.array(self.alpha))


class H2FormReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, gas2dust):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.gas2dust = gas2dust

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class H2ReactionRateTerm(JReactionRateTerm):
            alpha: Array
            gas2dust: Array

            def __call__(
                self,
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
            ):
                return 100.0 * self.gas2dust * self.alpha

        return H2ReactionRateTerm(jnp.array(self.alpha), jnp.array(self.gas2dust))
