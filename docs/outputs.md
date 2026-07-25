# Output files

A run writes into `config.output_dir` (default `output/`), with every
filename prefixed by `config.run_name`. Which files get written is
controlled by the `save_*` flags on [`SimulationConfig`](configuration.md):

| Flag | Default | File | Contents |
|---|---|---|---|
| `save_abundances` | on | `<run_name>_abundances.csv` | Species abundances at every snapshot |
| `save_derivatives` | off | `<run_name>_derivatives.csv` | `dy/dt` at every snapshot |
| `save_rates` | off | `<run_name>_rates.csv` | Per-reaction rate coefficients at every snapshot |
| `save_metadata` | on | `<run_name>_metadata.json` | Config, network stats, solver stats |
| `save_summary` | on | `<run_name>_summary.txt` | Human-readable run report |

Set `save_all = True` on the config to turn everything on regardless of the
individual flags (or `False` to turn everything off); leave it as `None`
(the default) to respect the flags above.

## Abundances (`*_abundances.csv`)

One row per snapshot, log-spaced in time between `t_start` and `t_end`
(`n_snapshots` rows). Columns:

- `time_seconds`, `time_years`
- `radius_cm`, `number_density`, `temperature`, `cr_rate`, `fuv_field`,
  `visual_extinction` -- the physical conditions at that snapshot (from the
  run's `physics_model`; constant across rows for a static cloud, varying
  for e.g. a [CSE outflow](physics.md))
- one column per species, named by species -- **fractional abundance**
  relative to the total gas density at that time/position, `x_i =
  n_i / n_gas`, which is also exactly the ODE state Carbox integrates. To
  get a number density, multiply by that row's `number_density`.

If the network uses [thermal balance](thermal_balance.md), the integrated
gas temperature is its own column, named `TGAS` (in Kelvin, *not*
fractional -- see that page for why it's still bundled into the same state
vector as the real species).

## Derivatives and rates

`*_derivatives.csv` mirrors the abundances file but with `d<species>_dt`
columns instead -- useful for spotting which species are still evolving
fastest at a given time, or for debugging a run that isn't converging.

`*_rates.csv` has one numbered column per reaction (`0`, `1`, `2`, ...,
matching the order reactions were parsed in -- cross-reference against the
network file, or `network.reactions[i]`, to see which is which) with the
rate *coefficient* at each snapshot -- not multiplied by reactant
abundances, so these are directly comparable to the `alpha`/`beta`/`gamma`
values in the input network file.

Both are computed by re-evaluating the compiled network at the solution's
own snapshots (`carbox.solver.compute_derivatives` /
`compute_reaction_rates`), so they reflect whatever `rate_modifiers` were
baked into the network for that run.

## Metadata (`*_metadata.json`)

A structured record of what produced the run: the physics model and its
initial conditions, integration settings (solver, tolerances, time range),
initial abundances, network size (species/reaction counts and names), and
solver statistics (accepted/rejected step counts). Meant to be checked in
or archived alongside the CSVs so a plot can always be traced back to the
config that made it, without re-running anything.

## Summary (`*_summary.txt`)

A short human-readable printout -- final abundances of the most abundant
species, run duration, step counts -- meant for a quick sanity check right
after a run finishes, not for programmatic use (parse the CSV/JSON outputs
for that).
