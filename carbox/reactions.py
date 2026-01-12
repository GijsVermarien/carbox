from dataclasses import dataclass
from typing import List, Union, Optional

import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from . import uclchem_photoreactions

REACTION_SKIP_LIST = ["CRPHOT", "CRP", "PHOTON"]


class JReactionRateTerm(eqx.Module):
    """
    Base class for JAX-compatible reaction rate terms.

    All subclasses must implement __call__ with signature:
        __call__(self, temperature, cr_rate, uv_field, visual_extinction, abundance_vector)

    This ensures consistent signatures for JIT compilation.
    Reactions that don't need abundance_vector can simply ignore it.
    """

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
    reaction_id: Optional[int] = None

    def __init__(self, reaction_type, reactants, products, reaction_id: Optional[int] = None):
        self.reactants = [r for r in reactants if valid_species_check(r)]
        self.products = [p for p in products if valid_species_check(p)]
        self.reaction_type = reaction_type
        self.molecularity = np.array(self.reactants).shape[-1]
        self.reaction_id = reaction_id

    def __str__(self):
        return f"{self.reactants} -> {self.products}"

    def __repr__(self):
        return f"Reaction(id={self.reaction_id}, type={self.reaction_type}, {self.reactants}, {self.products})\n"

    def _reaction_rate_factory() -> JReactionRateTerm:
        # Abstract function to implement in subclasses
        raise NotImplementedError

    def __call__(self):
        return self._reaction_rate_factory()

# --- Rate Term Classes (Defined at module level for efficiency) ---

class KAReactionRateTerm(JReactionRateTerm):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        # α(T/300K​)^βexp(−γ/T)
        return (
            self.alpha
            * jnp.power(0.0033333333333333335 * temperature, self.beta)
            * jnp.exp(-self.gamma / temperature)
        )

class KAFixedReactionRateTerm(JReactionRateTerm):
    reaction_coeff: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        return self.reaction_coeff

class CRPReactionRateTerm(JReactionRateTerm):
    alpha: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        return cr_rate * self.alpha

class CRPhotoReactionRateTerm(JReactionRateTerm):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        return (
            # 1.31e-17
            self.alpha
            * cr_rate
            * jnp.power(0.0033333333333333335 * temperature, self.beta)
            * self.gamma
            / (1 - 0.5)  # hardcoded omega value
        )

class FUVReactionRateTerm(JReactionRateTerm):
    alpha: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        return self.alpha * uv_field

class H2ReactionRateTerm(JReactionRateTerm):
    alpha: jnp.ndarray
    gas2dust: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        return 100.0 * self.gas2dust * self.alpha

class UCLCHEMH2FormRateTerm(JReactionRateTerm):
    # Silicate parameters
    silicate_mu: jnp.ndarray
    silicate_e_s: jnp.ndarray
    silicate_e_h2: jnp.ndarray
    silicate_e_hp: jnp.ndarray
    silicate_e_hc: jnp.ndarray
    silicate_nu_h2: jnp.ndarray
    silicate_nu_hc: jnp.ndarray
    silicate_cross_section: jnp.ndarray

    # Graphite parameters
    graphite_mu: jnp.ndarray
    graphite_e_s: jnp.ndarray
    graphite_e_h2: jnp.ndarray
    graphite_e_hp: jnp.ndarray
    graphite_e_hc: jnp.ndarray
    graphite_nu_h2: jnp.ndarray
    graphite_nu_hc: jnp.ndarray
    graphite_cross_section: jnp.ndarray

    hflux: jnp.ndarray
    alpha: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        gas_temp = temperature
        dust_temp = temperature

        # Mean thermal velocity of H atoms (cm s^-1)
        thermal_velocity = 1.45e5 * jnp.sqrt(gas_temp / 100.0)

        # Sticking coefficient (Hollenbach & McKee 1979, eqn 3.7)
        sticking_coeff = 1.0 / (
            1.0
            + 0.04 * jnp.sqrt(gas_temp + dust_temp)
            + 0.2 * (gas_temp / 100.0)
            + 0.08 * (gas_temp / 100.0) ** 2
        )

        # SILICATE formation efficiency
        factor1_sil = (
            self.silicate_mu
            * self.hflux
            / (
                2.0
                * self.silicate_nu_h2
                * jnp.exp(-self.silicate_e_h2 / dust_temp)
            )
        )

        sqrt_term_sil = jnp.sqrt(
            (self.silicate_e_hc - self.silicate_e_s)
            / (self.silicate_e_hp - self.silicate_e_s)
        )

        factor2_sil = (
            (1.0 + sqrt_term_sil) ** 2
            / 4.0
            * jnp.exp(-self.silicate_e_s / dust_temp)
        )

        epsilon_sil = 1.0 / (
            1.0
            + self.silicate_nu_hc
            / (2.0 * self.hflux)
            * jnp.exp(-1.5 * self.silicate_e_hc / dust_temp)
            * (1.0 + sqrt_term_sil) ** 2
        )

        silicate_efficiency = (
            1.0 / (1.0 + factor1_sil + factor2_sil) * epsilon_sil
        )

        # GRAPHITE formation efficiency
        factor1_gra = (
            self.graphite_mu
            * self.hflux
            / (
                2.0
                * self.graphite_nu_h2
                * jnp.exp(-self.graphite_e_h2 / dust_temp)
            )
        )

        sqrt_term_gra = jnp.sqrt(
            (self.graphite_e_hc - self.graphite_e_s)
            / (self.graphite_e_hp - self.graphite_e_s)
        )

        factor2_gra = (
            (1.0 + sqrt_term_gra) ** 2
            / 4.0
            * jnp.exp(-self.graphite_e_s / dust_temp)
        )

        epsilon_gra = 1.0 / (
            1.0
            + self.graphite_nu_hc
            / (2.0 * self.hflux)
            * jnp.exp(-1.5 * self.graphite_e_hc / dust_temp)
            * (1.0 + sqrt_term_gra) ** 2
        )

        graphite_efficiency = (
            1.0 / (1.0 + factor1_gra + factor2_gra) * epsilon_gra
        )

        # Combined rate (Cazaux & Tielens 2002, 2004)
        rate = (
            0.5
            * thermal_velocity
            * (
                self.silicate_cross_section * silicate_efficiency
                + self.graphite_cross_section * graphite_efficiency
            )
            * sticking_coeff
        )

        return rate * self.alpha

class UCLCHEMPhotonRateTerm(JReactionRateTerm):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        return (
            self.alpha
            * jnp.exp(-self.gamma * visual_extinction)
            * uv_field
            / 1.7
        )

class PHReactionRateTerm(JReactionRateTerm):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        rate = self.alpha * uv_field * jnp.exp(-self.gamma * visual_extinction)
        # jax.debug.print("PHReactionRateTerm: alpha={alpha}, uv_field={uv_field}, visual_extinction={visual_extinction}, gamma={gamma}, rate={rate}", alpha=self.alpha, uv_field=uv_field, visual_extinction=visual_extinction, gamma=self.gamma, rate=rate)
        return rate

class IonPol1RateTerm(JReactionRateTerm):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        return (
            self.alpha
            * self.beta
            * (0.62 + 0.4767 * self.gamma * jnp.sqrt(300.0 / temperature))
        )

class IonPol2RateTerm(JReactionRateTerm):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        sqrt_term = 0.0967 * self.gamma * jnp.sqrt(300.0 / temperature)
        quadratic_term = self.gamma**2 * 300.0 / (10.526 * temperature)
        return self.alpha * self.beta * (1.0 + sqrt_term + quadratic_term)

class H2PhotoDissRateTerm(JReactionRateTerm):
    cloud_radius_pc: jnp.ndarray
    turb_vel: jnp.ndarray
    h2_species_index: int

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        n_h2 = abundance_vector[self.h2_species_index]
        n_h2_column = uclchem_photoreactions.compute_column_density(n_h2, self.cloud_radius_pc)
        rate = uclchem_photoreactions.h2_photo_diss_rate(
            n_h2_column, uv_field, visual_extinction, self.turb_vel
        )
        return rate

class COPhotoDissRateTerm(JReactionRateTerm):
    cloud_radius_pc: jnp.ndarray
    h2_species_index: int
    co_species_index: int

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        n_h2 = abundance_vector[self.h2_species_index]
        n_co = abundance_vector[self.co_species_index]

        n_h2_column = uclchem_photoreactions.compute_column_density(n_h2, self.cloud_radius_pc)
        n_co_column = uclchem_photoreactions.compute_column_density(n_co, self.cloud_radius_pc)

        return uclchem_photoreactions.co_photo_diss_rate(
            n_h2_column, n_co_column, uv_field, visual_extinction
        )

class CIonizationRateTerm(JReactionRateTerm):
    alpha: jnp.ndarray
    gamma: jnp.ndarray
    cloud_radius_pc: jnp.ndarray
    c_species_index: int
    h2_species_index: int

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        n_c = abundance_vector[self.c_species_index]
        n_h2 = abundance_vector[self.h2_species_index]

        n_c_column = uclchem_photoreactions.compute_column_density(n_c, self.cloud_radius_pc)
        n_h2_column = uclchem_photoreactions.compute_column_density(n_h2, self.cloud_radius_pc)

        return uclchem_photoreactions.c_ionization_rate(
            self.alpha,
            self.gamma,
            temperature,
            n_c_column,
            n_h2_column,
            visual_extinction,
            uv_field,
        )


# Universal reactions:


class KAReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma, reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products, reaction_id=reaction_id)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return KAReactionRateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )


class KAFixedReaction(Reaction):
    def __init__(
        self, reaction_type, reactants, products, alpha, beta, gamma, temperature, reaction_id: Optional[int] = None
    ):
        super().__init__(reaction_type, reactants, products, reaction_id=reaction_id)
        self.reaction_coeff = (
            alpha
            * jnp.power(0.0033333333333333335 * temperature, beta)
            * jnp.exp(-gamma / temperature)
        )

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return KAFixedReactionRateTerm(jnp.array(self.reaction_coeff))


class CRPReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products, reaction_id=reaction_id)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return CRPReactionRateTerm(jnp.array(self.alpha))


class CRPhotoReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma, reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return CRPhotoReactionRateTerm(
            jnp.array(self.alpha),
            jnp.array(self.beta),
            jnp.array(self.gamma),
        )


# Latent_tgas / prizmo specific reactions:


class FUVReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha,reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return FUVReactionRateTerm(jnp.array(self.alpha))


class H2FormReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, gas2dust,reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.gas2dust = gas2dust

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return H2ReactionRateTerm(jnp.array(self.alpha), jnp.array(self.gas2dust))


# UCLCHEM reactions:
class UCLCHEMH2FormReaction(Reaction):
    """
    H2 formation on grains using UCLCHEM's Cazaux & Tielens (2002, 2004) treatment.

    Implements the h2FormEfficiency function from UCLCHEM's surfacereactions.f90.
    This accounts for:
    - Sticking coefficient (Hollenbach & McKee 1979)
    - Formation efficiency on silicate and graphite grains
    - Physisorption vs chemisorption sites
    - Temperature-dependent desorption

    Note: Assumes dust temperature = gas temperature for simplicity.
    For full treatment, pass dust_temp separately.
    """

    def __init__(
        self,
        reaction_type,
        reactants,
        products,
        alpha=1.0,
        **kwargs,
    ):
        """
        Initialize H2 formation reaction.

        Args:
            reaction_type: Type identifier for the reaction
            reactants: List of reactant species
            products: List of product species
            alpha: Multiplicative scaling factor for rate (default=1.0)
            **kwargs: Ignored. Allows vectorization to pass extra params.
        """
        super().__init__(reaction_type, reactants, products)

        self.alpha = alpha

        # Silicate grain properties
        self.silicate_mu = 0.005  # Fraction of H2 staying on surface
        self.silicate_e_s = 110.0  # Saddle point energy (K)
        self.silicate_e_h2 = 320.0  # H2 desorption energy (K)
        self.silicate_e_hp = 450.0  # Physisorbed H desorption (K)
        self.silicate_e_hc = 3.0e4  # Chemisorbed H desorption (K)
        self.silicate_nu_h2 = 3.0e12  # H2 vibrational frequency (s^-1)
        self.silicate_nu_hc = 1.3e13  # H vibrational frequency (s^-1)
        self.silicate_cross_section = 8.473e-22  # cm^-2/nucleus

        # Graphite grain properties
        self.graphite_mu = 0.005
        self.graphite_e_s = 260.0
        self.graphite_e_h2 = 520.0
        self.graphite_e_hp = 800.0
        self.graphite_e_hc = 3.0e4
        self.graphite_nu_h2 = 3.0e12
        self.graphite_nu_hc = 1.3e13
        self.graphite_cross_section = 7.908e-22  # cm^-2/nucleus

        self.hflux = 1.0e-10  # Monolayers per second
        
        # Allow kwargs to override defaults (crucial for vectorization)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return UCLCHEMH2FormRateTerm(
            silicate_mu=jnp.array(self.silicate_mu),
            silicate_e_s=jnp.array(self.silicate_e_s),
            silicate_e_h2=jnp.array(self.silicate_e_h2),
            silicate_e_hp=jnp.array(self.silicate_e_hp),
            silicate_e_hc=jnp.array(self.silicate_e_hc),
            silicate_nu_h2=jnp.array(self.silicate_nu_h2),
            silicate_nu_hc=jnp.array(self.silicate_nu_hc),
            silicate_cross_section=jnp.array(self.silicate_cross_section),
            graphite_mu=jnp.array(self.graphite_mu),
            graphite_e_s=jnp.array(self.graphite_e_s),
            graphite_e_h2=jnp.array(self.graphite_e_h2),
            graphite_e_hp=jnp.array(self.graphite_e_hp),
            graphite_e_hc=jnp.array(self.graphite_e_hc),
            graphite_nu_h2=jnp.array(self.graphite_nu_h2),
            graphite_nu_hc=jnp.array(self.graphite_nu_hc),
            graphite_cross_section=jnp.array(self.graphite_cross_section),
            hflux=jnp.array(self.hflux),
            alpha=jnp.array(self.alpha),
        )


class UCLCHEMPhotonReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma,reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return UCLCHEMPhotonRateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )


# UMIST reactions:


class UMISTPhotoReaction(Reaction):
    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma, reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products, reaction_id=reaction_id)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return PHReactionRateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )


# UCLCHEM-specific reaction types
class IonPol1Reaction(Reaction):
    """UCLCHEM IONOPOL1: Ion-polar molecule reaction (KIDA formula 1)
    k = α * β * (0.62 + 0.4767 * γ * sqrt(300/T))
    """

    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma,reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return IonPol1RateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )


class IonPol2Reaction(Reaction):
    """UCLCHEM IONOPOL2: Ion-polar molecule reaction (KIDA formula 2)
    k = α * β * (1.0 + 0.0967 * γ * sqrt(300/T) + γ² * 300/(10.526 * T))
    """

    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma,reaction_id: Optional[int] = None):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return IonPol2RateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )


class GARReaction(Reaction):
    """UCLCHEM GAR: Grain-assisted recombination (Weingartner & Draine 2001)
    Simplified implementation for gas-phase comparison
    """

    def __init__(self, reaction_type, reactants, products):
        super().__init__(reaction_type, reactants, products)

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class GARRateTerm(JReactionRateTerm):
            def __call__(
                self,
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
            ):
                return NotImplementedError

        return NotImplementedError


class H2PhotoDissReaction(Reaction):
    """
    H2 photodissociation with self-shielding and dust extinction.
    Uses UCLCHEM's treatment from photoreactions module.
    Requires cloud geometry and H2 abundance from state vector.
    """

    def __init__(
        self,
        reaction_type,
        reactants,
        products,
        cloud_radius_pc=1.0,
        number_density=1e4,
        turb_vel=1e5,
        h2_species_index=None,
    ):
        super().__init__(reaction_type, reactants, products)
        self.cloud_radius_pc = cloud_radius_pc
        self.number_density = number_density
        self.turb_vel = turb_vel
        self.h2_species_index = h2_species_index

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return H2PhotoDissRateTerm(
            jnp.array(self.cloud_radius_pc),
            jnp.array(self.turb_vel),
            self.h2_species_index,
        )


class COPhotoDissReaction(Reaction):
    """
    CO photodissociation with self-shielding from H2 and CO.
    Requires both H2 and CO abundances from state vector.
    """

    def __init__(
        self,
        reaction_type,
        reactants,
        products,
        cloud_radius_pc=1.0,
        number_density=1e4,
        h2_species_index=None,
        co_species_index=None,
    ):
        super().__init__(reaction_type, reactants, products)
        self.cloud_radius_pc = cloud_radius_pc
        self.number_density = number_density
        self.h2_species_index = h2_species_index
        self.co_species_index = co_species_index

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return COPhotoDissRateTerm(
            jnp.array(self.cloud_radius_pc),
            self.h2_species_index,
            self.co_species_index,
        )


class CIonizationReaction(Reaction):
    """
    Carbon photoionization with dust extinction and gas-phase shielding.
    Requires C, H2 abundances from state vector.
    Uses UCLCHEM's treatment with dust and gas-phase shielding.
    """

    def __init__(
        self,
        reaction_type,
        reactants,
        products,
        alpha=3.5e-10,
        gamma=3.0,
        cloud_radius_pc=1.0,
        number_density=1e4,
        c_species_index=None,
        h2_species_index=None,
    ):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.gamma = gamma
        self.cloud_radius_pc = cloud_radius_pc
        self.number_density = number_density
        self.c_species_index = c_species_index
        self.h2_species_index = h2_species_index

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        return CIonizationRateTerm(
            jnp.array(self.alpha),
            jnp.array(self.gamma),
            jnp.array(self.cloud_radius_pc),
            self.c_species_index,
            self.h2_species_index,
        )
