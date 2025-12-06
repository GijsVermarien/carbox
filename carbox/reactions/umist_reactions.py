"""Umist specific reactions."""

import jax.numpy as jnp
from jax import Array

from . import JReactionRateTerm, Reaction


class UMISTPhotoReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class PHReactionRateTerm(JReactionRateTerm):
            alpha: Array
            beta: Array
            gamma: Array

            def __call__(
                self,
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
            ):
                return self.alpha * jnp.exp(-self.gamma * visual_extinction * 4.65)

        return PHReactionRateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )
