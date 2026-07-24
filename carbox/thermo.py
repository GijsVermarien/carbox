r"""
Thermal balance (cooling/heating) as a pseudo-reaction.

Cooling and heating are modeled as the rate coefficient of a reaction with no
reactants and a single product, the pseudo-species ``TGAS`` (gas
temperature, in Kelvin, folded into the abundance vector as its last entry):

.. math::
    \text{NOTHING} \rightarrow \text{TGAS}, \quad
    k = -\Lambda_{\rm cool} / (k_B\, n_{\rm tot})

so that ``dTGAS/dt = k`` (the reactants are missing, so they're treated as
unitary -- rate law times 1.0 -- matching every other zero-reactant rate law
in this codebase, e.g. ``CRPReaction``). See :mod:`carbox.cooling` for the
individual cooling channels (:math:`\Lambda_{\rm cool}`) that are summed
here; there is no heating channel yet, so the net rate is currently pure
cooling.

Two things make this pseudo-reaction different from a normal chemical
reaction, both handled explicitly in this module and in
:class:`carbox.network.JNetwork`:

- **Units.** Every other reaction rate in this codebase ends up as a
  fractional-abundance rate (:math:`dx_i/dt`), rescaled by
  ``density ** (molecularity - 1)`` in ``JNetwork.__call__`` (see
  ``carbox/network.py``). ``ThermoRateTerm``'s rate is already an *absolute*
  ``dT/dt`` [K/s], so :class:`ThermoRate` forces ``molecularity = 1``
  (rather than the ``0`` that ``len(reactants) == 0`` would naturally give)
  to make that rescaling a no-op (``density**0 == 1``).
- **Feedback.** Whether the *rest* of the network's rate laws see this
  integrated ``TGAS`` value, or a temperature prescribed by the physics
  model, is a solver-level choice -- see
  ``AbstractPhysics.integrates_temperature`` in ``carbox/physics.py``.
"""

import jax.numpy as jnp

from .cooling import cooling_h2, cooling_lyalpha, cooling_oi
from .index import Idx
from .reactions import JReactionRateTerm, Reaction

KBOLTZMANN_ERG = 1.380649e-16  # erg / K


class ThermoRateTerm(JReactionRateTerm):
    """
    JAX rate-law implementation for :class:`ThermoRate`.

    Sums the cooling channels in :mod:`carbox.cooling` and converts the
    result from an energy-loss rate [erg cm^-3 s^-1] to a temperature-loss
    rate [K/s] via :math:`\\dot{T} = -\\Lambda_{\\rm cool} / (k_B\\, n_{\\rm
    tot})` -- the standard ideal-monatomic-gas thermal-energy relation
    :math:`u = \\tfrac{3}{2} n k_B T` up to the constant prefactor (kept as
    in the original implementation; a factor of 2/3 is not currently
    applied, so this is an approximation rather than a rigorously derived
    energy-conservation equation).

    Attributes
    ----------
    idx : carbox.index.Idx
        The network's species lookup, baked in at construction time (via
        `ThermoRate._reaction_rate_factory`) so the cooling channels can
        find ``H``, ``E``, ``O``, ``H2`` regardless of species ordering.
        Has zero pytree leaves (see `Idx`), so carrying it here is free
        under `jit`/`vmap`.
    """

    idx: Idx

    def __call__(
        self,
        temperature,
        cr_rate,
        uv_field,
        visual_extinction,
        abundance_vector,
    ):
        """
        Parameters
        ----------
        temperature : jnp.ndarray
            Gas temperature [K] -- the *current* value being integrated
            over when `AbstractPhysics.integrates_temperature` is True.
        cr_rate, uv_field, visual_extinction : jnp.ndarray
            Unused by this reaction type; present for a uniform
            `JReactionRateTerm.__call__` signature across all reaction
            classes.
        abundance_vector : jnp.ndarray
            Number-density abundance vector [cm^-3] (all species except
            TGAS itself, see `carbox.network.JNetwork.__call__`).

        Returns
        -------
        jnp.ndarray
            ``dT/dt`` [K/s]. Negative (cooling) given the current cooling-
            only implementation; zero if every cooling channel's inputs are
            zero.
        """
        cooling = (
            cooling_lyalpha(abundance_vector, temperature, self.idx)
            + cooling_oi(abundance_vector, temperature, self.idx)
            + cooling_h2(abundance_vector, temperature, self.idx)
        )
        n_tot = jnp.sum(abundance_vector)
        return -cooling / (KBOLTZMANN_ERG * n_tot)


class ThermoRate(Reaction):
    """
    Thermal-balance pseudo-reaction: ``NOTHING -> TGAS``.

    Parsed from a network row with no reactants and product ``TGAS`` (e.g.
    the ``latent_tgas`` format's ``mechanism == "THERMO"``, see
    `carbox.parsers.latent_tgas_parser.LatentTGASParser`). See the module
    docstring for why `molecularity` is forced to ``1``.

    Parameters
    ----------
    reaction_type : str
        The parser's reaction-type label (e.g. ``"THERMO"``).
    reactants : list[str]
        Expected empty for this reaction type.
    products : list[str]
        Expected to be exactly ``["TGAS"]``.
    reaction_id : int, optional
        Original network-file identifier, see `carbox.reactions.Reaction`.
    """

    def __init__(self, reaction_type, reactants, products, reaction_id=None):
        super().__init__(reaction_type, reactants, products, reaction_id=reaction_id)
        # Absolute dT/dt, not fractional -- see module docstring.
        self.molecularity = 1

    def _reaction_rate_factory(self, idx=None) -> JReactionRateTerm:
        """
        Build the `ThermoRateTerm` for this reaction.

        Parameters
        ----------
        idx : carbox.index.Idx, optional
            The network's species lookup; required (unlike every other
            reaction type, which ignores `idx`) because the cooling
            channels need to find specific species by name.

        Returns
        -------
        ThermoRateTerm
        """
        if idx is None:
            raise ValueError("ThermoRate requires the network's species Idx")
        return ThermoRateTerm(idx)
