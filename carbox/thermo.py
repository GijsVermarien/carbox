from .reactions import JReactionRateTerm, Reaction
from .cooling import cooling_Lyalpha, cooling_OI, cooling_H2
import jax.numpy as jnp


class ThermoRate(Reaction):
    def __init__(self, reaction_type, reactants, products):
        super().__init__(reaction_type, reactants, products)

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class ThermoRateTerm(JReactionRateTerm):
            #alpha: float

            def __call__(
                self,
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
                idx
            ):
                cool = cooling_Lyalpha(abundance_vector, temperature, idx) \
                    + cooling_OI(abundance_vector, temperature, idx) \
                    + cooling_H2(abundance_vector, temperature, idx)

                kboltzmann_erg = 1.380649e-16  # Boltzmann constant in erg/K
                ntot = jnp.sum(abundance_vector)
                print(abundance_vector.shape)
                return - cool / kboltzmann_erg / ntot

        return ThermoRateTerm()