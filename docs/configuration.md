# Configuration

Every simulation is driven by a single [`SimulationConfig`](api/config.md)
object: physical conditions, initial abundances, solver settings, and what
gets written to disk. This page is a task-oriented tour of the options you're
most likely to touch; see the source docstring for the exhaustive field list.

## Setting it up in Python

```python
from carbox import SimulationConfig

config = SimulationConfig(
    number_density=1e4,       # cm^-3
    temperature=50.0,         # K
    cr_rate=1.3e-17,          # s^-1
    visual_extinction=2.0,    # mag
    initial_abundances={"H2": 1.0, "O": 2e-4, "C": 1e-4},
    t_end=1e6,                # years
    run_name="my_run",
)
```

Every field has a sensible default (a cold, moderately dense molecular
cloud), so you only need to set the ones that differ from that.

## From a YAML or JSON file

The same fields can live in a config file instead, which is usually more
convenient once you're running more than a handful of simulations:

```yaml
# my_config.yaml
number_density: 1.0e4
temperature: 50.0
t_end: 1.0e6
initial_abundances:
  H2: 1.0
  O: 2.0e-4
  C: 1.0e-4
run_name: my_run
```

```python
config = SimulationConfig.from_yaml("my_config.yaml")
# or: SimulationConfig.from_json("my_config.json")
```

`config.to_yaml(path)` / `config.to_json(path)` write it back out (handy for
recording exactly what a run used alongside its output).

From the command line, pass the file with `--config`:

```bash
carbox-sim --input data/network.csv --config my_config.yaml
```

`--output`, `--name`, and `--solver` on the command line override whatever
the config file says for that run; run `carbox-sim --help` for the full list.

## The fields, grouped by what they control

**Physical conditions** -- `number_density`, `temperature`,
`visual_extinction`, `cr_rate`, `fuv_field`, `gas_to_dust_ratio`. These feed
either directly into the solver, or into the default
[`StaticCloudPhysics`](physics.md) model built for you if you don't set
`physics_model` explicitly. If you need density/temperature that change over
time (an outflow, a self-consistently integrated temperature, ...), set
`physics_model` yourself -- see [Physics models](physics.md) and
[Thermal balance](thermal_balance.md).

**Initial abundances** -- `initial_abundances` is a `{species: fractional
abundance}` dict; anything you don't list starts at `abundance_floor`
(default `1e-30`, not zero, for numerical stability). Species named in
`initial_abundances` that aren't in the network print a warning and are
otherwise ignored.

**Integration** -- `t_start`/`t_end` (years), `n_snapshots` (log-spaced
output times), `solver` (`kvaerno5` default; `kvaerno3`, `dopri5`, `tsit5`
also available), `atol`/`rtol`, `max_steps`. The defaults are tuned for
stiff astrochemical networks; if a run fails to converge, `kvaerno5` with
tighter tolerances is the first thing to try before switching solvers.

**Output** -- see [Output files](outputs.md) for what each `save_*` flag
produces and where it ends up.

## Reusing a compiled network across configs

Parsing a network file and JIT-compiling it (`Network.get_ode()`) is the
expensive part of a run. If you're sweeping over `SimulationConfig`s for the
*same* network -- e.g. a parameter scan -- `carbox.main.parse_network()` and
`carbox.main.solve()` split that out so you only pay the compilation cost
once:

```python
from carbox.main import parse_network, solve

bundle = parse_network("data/network.csv", format_type="latent_tgas")

for density in [1e3, 1e4, 1e5]:
    config = SimulationConfig(number_density=density, t_end=1e6)
    solution = solve(bundle, config)
```

`solve()` also accepts `rate_modifiers=(a, b)` to scale/override specific
reaction rates without recompiling -- see
[`JNetwork.with_rate_modifiers`](api/network.md) for the sensitivity-analysis
use case this is built for.
