# Thermal balance

By default, Carbox treats temperature the same way UCLCHEM's static-cloud
mode does: it's a fixed input you set on the physics model, and chemistry
has no effect on it. Thermal balance is the opt-in alternative -- radiative
cooling feeds back into an integrated gas temperature, so `T` evolves
alongside the chemical abundances instead of staying constant.

Turn it on when the gas you're modeling cools enough over the run that
holding temperature fixed would give you the wrong chemistry -- e.g. warm,
partially-ionized atomic gas relaxing toward a molecular cloud, or any setup
where you want to check whether cooling reaches equilibrium rather than
assume it.

!!! note "Cooling only, for now"
    Only cooling is implemented; there's no heating term yet, so with
    thermal balance on, the gas can cool or hold steady, but never heat up.

## Enabling it

Thermal balance is a property of the *physics model*, not a separate
simulation mode -- set `integrates_temperature=True` on whichever physics
model you're using:

```python
from carbox import SimulationConfig, run_simulation
from carbox.physics import StaticCloudPhysics

physics = StaticCloudPhysics(
    number_density=1e4,
    temperature=300.0,      # now just the *initial* temperature
    integrates_temperature=True,
)
config = SimulationConfig(
    number_density=1e4,
    temperature=300.0,
    t_end=1e3,
    initial_abundances={"H": 0.1, "H2": 0.1, "O": 1e-4, "E": 1e-4},
)
config.physics_model = physics

results = run_simulation("network_with_thermo_row.csv", config, format_type="latent_tgas")
network, solution = results["network"], results["solution"]
tgas = solution.ys[:, network.get_index("TGAS")]  # temperature trajectory [K]
```

Two things have to be true for this to do anything:

1. **The network file needs a thermal-balance row.** Only `latent_tgas`
   format networks support this today: add a row with an empty reactant
   list, product `TGAS`, and mechanism `THERMO`. Without it, there's no
   `TGAS` species for the integrated temperature to live in, and
   `integrates_temperature` has no effect.
2. **`integrates_temperature=True` on the physics model.** If the network
   has a `TGAS` row but this stays `False` (the default), `TGAS` still gets
   tracked in the output, but the rest of the network's rate laws keep using
   the physics model's prescribed temperature instead of the integrated
   one -- i.e. cooling is computed and recorded, but doesn't feed back into
   the chemistry. This is also exactly the behavior of every network from
   before thermal balance existed, and of any network without a `TGAS`
   species at all -- turning this feature on never changes results for runs
   that don't use it.

See `tests/test_thermal_balance.py` for a complete runnable example with a
minimal test network, and `tests/test_cooling.py` for tests of the
individual cooling channels.

## What's cooling the gas

The net cooling rate is the sum of three channels, implemented in
[`carbox.cooling`](api/cooling.md):

| Channel | Physical process | Dominant regime |
|---|---|---|
| `cooling_lyalpha` | H excited by e-, decays via Ly-alpha (1216 A) | warm, partially ionized atomic gas |
| `cooling_oi` | O excited by e-, decays via [OI] 630 nm | cooler atomic/ionized gas |
| `cooling_h2` | H2 ro-vibrational cooling via collisions with H | molecular gas |

`cooling_h2` is a piecewise fit (Hollenbach & McKee 1979, as implemented in
KROME) across several temperature ranges. See the
[API reference](api/cooling.md) for the exact functional forms, units, and
which species each channel needs present in the network.

## Reading the output

The integrated temperature shows up as a `TGAS` column in the abundances
CSV, right alongside the species columns -- see
[Output files](outputs.md#abundances-_abundancescsv) for the full column
layout. It's in Kelvin, not a fractional abundance, so don't divide it by
`number_density` the way you would a real species column.

## How it's implemented

The short version: temperature is folded into the ODE state as a
pseudo-species (`NOTHING -> TGAS`, rate = net cooling), reusing all of the
existing reaction machinery rather than adding a special-cased integration
path. If you're extending this (e.g. adding a heating channel) or just
curious about the JAX-level details -- the molecularity trick that keeps
`TGAS`'s rate from being rescaled like a real species, why it's a
compile-time branch with zero overhead when unused, the elemental-
conservation edge case -- see the "Thermal balance" section in
[CONTRIBUTING.md](https://github.com/gijsvermarien/carbox/blob/main/CONTRIBUTING.md).
