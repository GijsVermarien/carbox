# Physics models

The physical conditions a network evolves under -- density, temperature,
visual extinction, position, and any non-chemical ODE source term -- come
from a `physics_model` on [`SimulationConfig`](configuration.md), not from
scalar fields directly. If you don't set one, `SimulationConfig` builds a
`StaticCloudPhysics` from the legacy scalar fields (`number_density`,
`temperature`, `visual_extinction`, ...) automatically, so simple runs never
need to touch this.

## Static cloud (the default)

Constant density and temperature -- the classic single-zone astrochemistry
setup. This is what you get implicitly if you just set `number_density` /
`temperature` on the config; setting it up explicitly looks like:

```python
from carbox import SimulationConfig
from carbox.physics import StaticCloudPhysics

physics = StaticCloudPhysics(
    number_density=1e4,      # cm^-3
    temperature=50.0,        # K
    visual_extinction=2.0,   # mag
)
config = SimulationConfig(t_end=1e6, physics_model=physics)
```

Set `use_self_consistent_av=True` (with `base_av`/`cloud_radius_pc`) to
compute `Av` from the column density instead of fixing it; see
[`StaticCloudPhysics`](api/physics.md#carbox.physics.StaticCloudPhysics)
for the exact formula. It also accepts `integrates_temperature=True` to let
[thermal balance](thermal_balance.md) evolve `temperature` instead of
holding it fixed.

## Circumstellar envelope (CSE) outflow

A wind expanding at constant velocity away from a star: density falls off
as `n ~ r^-2` (mass conservation) and temperature follows a power law
`T ~ r^-eps`, normalized at the stellar radius.

```python
from carbox import SimulationConfig, run_simulation
from carbox.physics import CSEPhysics

physics = CSEPhysics(
    mdot=1e-5,      # mass-loss rate, M_sun/yr
    vexp=15.0,      # expansion velocity, km/s
    t_star=2000.0,  # temperature at r_star, K
    r_init=1e16,    # starting radius, cm
    r_star=5e13,    # stellar radius, cm
    eps=0.7,        # temperature power-law index
)
config = SimulationConfig(t_end=1e5, physics_model=physics)
results = run_simulation("data/network.csv", config, format_type="latent_tgas")
```

Use this instead of `StaticCloudPhysics` whenever the chemistry you care
about depends on how far material has traveled from the star -- density can
drop by orders of magnitude over a run, which is exactly what the
fractional-abundance ODE state (below) is built to handle cleanly.

## Both are `AbstractPhysics`

`StaticCloudPhysics` and `CSEPhysics` are both
[`AbstractPhysics`](api/physics.md#carbox.physics.AbstractPhysics)
subclasses; the solver and output code only ever talk to that interface, so
swapping between them -- or writing your own -- doesn't touch any other part
of a config. To add a new model (e.g. a collapsing core), subclass
`AbstractPhysics` and implement `get_conditions(t_sec) -> (n, T, av, r)`;
only override `dilution`/`time_grid` if the model needs a non-chemical ODE
source term or a custom snapshot grid. Full field/method reference:
[carbox.physics](api/physics.md).

## Why this matters for the ODE state

Regardless of which model you use, Carbox always integrates *fractional*
abundance, `x_i = n_i / n(t)`, relative to whatever total density the
physics model prescribes at each moment -- not raw number density. That's
what keeps solver tolerances meaningful across a CSE outflow where `n` can
drop by orders of magnitude, and it's why static and dynamic-density models
can share the same solver code without special-casing.

See [Thermal balance](thermal_balance.md) for `integrates_temperature`, the
field both concrete models above use to opt into self-consistent
temperature.
