# Contributing to Carbox

Thanks for looking at Carbox. This document covers setting up a development
environment, running the tests, and the codebase's architecture at the
level of detail you need to make a change with confidence. For how to
*use* Carbox rather than change it, see the
[docs site](https://gijsvermarien.github.io/carbox/) and the
[README](README.md).

## Development setup

```bash
pip install -e .[dev]
pre-commit install
```

This installs `pytest` and `ruff` alongside the package itself in editable
mode. Run the linter from the repository root:

```bash
ruff check .
```

`[tool.ruff]` in `pyproject.toml` scopes this to `carbox/` and `tests/`
(correctness and import-hygiene rules only -- see the comment there for
why) and skips `benchmarks/`, `examples/`, `notebooks/`, and
`sensitivity_analysis/`, which haven't had a lint pass yet.

Run the test suite from the repository root:

```bash
python -m pytest                              # full suite
python -m pytest tests/test_conservation.py    # one file
python -m pytest tests/test_reaction_id.py -k umist   # one test
```

Both `ruff check .` and `pytest` run in CI on every push and pull request
(`.github/workflows/ci.yml`); a PR won't merge cleanly until both pass.

To build and preview the documentation site locally:

```bash
pip install -e .[docs]
mkdocs serve   # live preview at http://127.0.0.1:8000
mkdocs build   # static site in site/
```

Docs are deployed automatically to
[gijsvermarien.github.io/carbox](https://gijsvermarien.github.io/carbox/)
on every push to `main` that touches `docs/`, `mkdocs.yml`, or `carbox/`
(`.github/workflows/docs.yml`).

## Known issues

**JAX version sensitivity.** Newer JAX (0.4.30+) has triggered a
`TracerBoolConversionError` in this codebase. If you hit that, the
previously-working pin is:

```
jax[cpu]==0.4.26
jaxlib==0.4.26
equinox==0.11.4
lineax==0.0.4
diffrax==0.5.0
```

## Architecture

### The pipeline

A run is four stages, chained by `carbox.main.run_simulation()`:

1. **Parse** -- `parse_chemical_network(filepath, format_type)` reads a
   network file (UCLCHEM, UMIST, or LATENT-TGAS format) into a `Network`:
   plain Python `Species` and `Reaction` objects, easy to inspect and
   modify.
2. **Compile** -- `Network.get_ode()` turns that into a `JNetwork`: an
   `equinox.Module` (a JAX pytree) that's the actual, JIT-traceable ODE
   right-hand side. This step groups same-type reactions together so their
   rate laws can be evaluated as a batch instead of one Python object at a
   time (see "Reactions" below).
3. **Solve** -- `solve_network()` builds the ODE term from a physics model
   and integrates it with Diffrax, wrapped in `eqx.filter_jit` so the first
   call compiles and subsequent calls (e.g. across a parameter sweep) are
   fast.
4. **Save** -- `carbox/output.py` writes whichever of abundances /
   derivatives / rates / metadata / summary the `SimulationConfig` asked
   for.

If you want to reuse a compiled network across multiple solves (a
parameter scan, a sensitivity sweep) without re-parsing and re-compiling
each time, `carbox.main.parse_network()` and `carbox.main.solve()` expose
steps 1-2 and step 3 separately -- see
[Configuration](https://gijsvermarien.github.io/carbox/configuration/#reusing-a-compiled-network-across-configs)
for the user-facing version of this.

### Physics models (`carbox/physics.py`)

The physical conditions a network evolves under: density, temperature,
visual extinction, position, and any non-chemical ODE source term,  come
from a `physics_model: AbstractPhysics` on `SimulationConfig`, not from
scalar fields directly. If none is given, `SimulationConfig.__post_init__`
builds a `StaticCloudPhysics` from the scalar fields
(`number_density`, `temperature`, `visual_extinction`,
`use_self_consistent_av`, `base_av`, `cloud_radius_pc`) for backward
compatibility.

- `AbstractPhysics.get_conditions(t_sec) -> (n, T, av, r)` is the one
  method every subclass must implement. It runs inside `jax.jit`/`vmap`,
  so it has to be traceable; no branching on traced values, and
  every returned value must actually depend on `t_sec` (broadcast
  constants against it rather than returning a bare Python float).
- `dilution(t_sec, y)` is an additive non-chemical ODE term (e.g.
  expansion dilution of an outflow); it defaults to zero.
- `time_grid(config)` is an optional custom snapshot grid -- `CSEPhysics`
  uses one log-spaced in radius rather than time, to avoid clustering
  snapshots near `t=0`; the default `None` falls back to the solver's
  generic log-time grid.
- `StaticCloudPhysics` is constant density/temperature; `CSEPhysics` is a
  circumstellar-envelope outflow (`n ~ r^-2` from mass conservation,
  `T ~ r^-eps`). `CSEPhysics`'s dilution term cancels analytically against
  the fractional-abundance ODE -- see the comment above its `time_grid`
  method for why it deliberately does *not* override `dilution`.
- `carbox/cse_physics.py` is just a backward-compat shim re-exporting
  `CSEPhysics` from `physics.py`.


### Adding a new physics model
To add a new model (e.g. a collapsing core), subclass `AbstractPhysics`
and implement `get_conditions`; only override `dilution`/`time_grid` if
the model needs a non-chemical source term or a custom grid. See
[Physics models](https://gijsvermarien.github.io/carbox/physics/) for the
user-facing side.

### Why the ODE state is fractional abundance

The integrated state vector is `x_i = n_i / n(t)` (fractional abundance
relative to total gas density at time `t`), not number density. This is
what keeps solver tolerances meaningful across a CSE outflow where `n` can
drop by orders of magnitude, and it's what lets the same `JNetwork.
__call__` serve both static and dynamic-density physics models without
special cases. Rates are evaluated on number densities internally
(`abundances * density`), since self-shielding/column-density terms expect
cm^-3, then rescaled by `density ** (molecularities - 1)` to convert back
to the fractional-abundance ODE: a molecularity-1 reaction (photo/cosmic-ray)
needs no density factor, a molecularity-2 (bimolecular) reaction needs one.

### `Network` vs `JNetwork` (`carbox/network.py`)

- `Network` (a plain Python dataclass) holds the parsed `Species`/
  `Reaction` objects and the incidence matrix. `Network.get_ode()` groups
  same-type reactions together to batch their rate-law evaluation --
  except a few `non_vectorizable_types` (like `H2PhotoDissReaction`) that
  have unique per-reaction parameters and can't be batched -- and returns
  a compiled `JNetwork`.
- `JNetwork` (an `equinox.Module`, i.e. a JAX pytree) is the actual ODE
  right-hand side, called as `jnetwork(t, y, T, n, cr_rate, fuv_rate, av)`
  inside the solver.

**Rate modifiers.** `JNetwork` carries `rate_modifier_a`/`rate_modifier_b`
fields (identity by default: `a=1, b=0`) that let you scale or override
specific reaction rates (`rate = rate*a + b`) *without recompiling the
network* -- this is central to how Carbox is used for sensitivity
analysis. Never pass modifiers as call-time arguments; call
`jnetwork.with_rate_modifiers(a, b)` to get an updated immutable copy
(built with `eqx.tree_at`), and pass that copy to everything downstream
(`solve_network`, `compute_derivatives`, `compute_reaction_rates`) so they
stay consistent with what was actually integrated.
`compute_reaction_rates(..., use_rate_modifiers=True)` (the default)
returns the effective/modified coefficients; pass `False` for the raw
parameterization instead. `JNetwork.modify_rates(rates)` is the one place
the `a*rate + b` transform happens -- reuse it rather than re-deriving the
formula elsewhere.

### Reactions (`carbox/reactions.py`)

Two parallel class hierarchies: `Reaction` subclasses (plain Python
objects describing a reaction -- `KAReaction`, `CRPReaction`,
`H2PhotoDissReaction`, and so on) and `JReactionRateTerm` subclasses (the
matching `equinox.Module` rate-law implementations, e.g.
`KAReactionRateTerm`), defined at module level rather than nested inside a
method. Every `Reaction` carries an optional `reaction_id` that survives
parsing, reordering, and vectorization, and is what output CSVs use to
label rate columns (not a reconstructed reaction string) -- if you add a
new reaction type, thread `reaction_id` through its constructor the same
way the existing ones do.

### Parsers (`carbox/parsers/`)

`UnifiedChemicalParser` dispatches to a per-format parser
(`uclchem_parser.py`, `umist_parser.py`, `latent_tgas_parser.py`) based on
`format_type` or auto-detection from the file.
`parse_chemical_network(filepath, format_type=None)` is the public entry
point. See `carbox/parsers/README.md` for format-specific details --
column layouts, special reaction handling like IONOPOL/GAR.

### Config (`carbox/config.py`)

`SimulationConfig` is a plain dataclass (`from_yaml`/`from_json`/
`to_yaml`/`to_json` for serialization; `physics_model` is excluded from
serialization since it isn't trivially round-trippable). `save_all`
overrides the individual `save_abundances`/`save_derivatives`/
`save_rates`/`save_metadata`/`save_summary` flags when set. See
[Configuration](https://gijsvermarien.github.io/carbox/configuration/) for
the full field-by-field guide.

### Solver (`carbox/solver.py`)

`solve_network()` builds the ODE term from the physics model and
JIT-compiles the integration. `get_solver()` maps `config.solver`
(`kvaerno5`/`kvaerno3`/`dopri5`/`tsit5`) and `config.linear_solver`
(`sparse`/`qr`/`lu`) to the corresponding Diffrax/Lineax solver objects.
It's highly recommend/needed to use Stiff solvers! 
`compute_derivatives()` and `compute_reaction_rates()` recompute `dy/dt`
and per-reaction rates at the solution's existing snapshots (vmapped over
time/state) -- both take the same `jnetwork` that produced the solution,
so rate modifiers stay consistent.

### Thermal balance internals

Thermal balance ([user docs](https://gijsvermarien.github.io/carbox/thermal_balance/))
folds gas temperature into the ODE state as a pseudo-species `TGAS`, with
cooling expressed as a pseudo-reaction (`NOTHING -> TGAS`,
`carbox.thermo.ThermoRate`). This reuses all of the network's existing
machinery -- vectorization, the incidence-matrix multiply, rate modifiers
-- instead of adding a special-cased integration path. Four consequences
of that design are handled explicitly rather than left as an accident of
how the numbers happen to work out:

1. **`TGAS`'s rate must not be rescaled like a real species'.** Every
   other reaction's rate is rescaled in `JNetwork.__call__` by
   `density ** (molecularity - 1)` (see "Why the ODE state is fractional
   abundance" above). `ThermoRateTerm`'s rate is already an absolute
   `dT/dt` [K/s]. `ThermoRate.__init__` forces `molecularity = 1` (rather
   than the `0` its empty reactant list would naturally produce
   otherwise) specifically so that rescaling becomes a no-op
   (`density**0 == 1`).
2. **`TGAS` must never be picked up by element-conservation bookkeeping.**
   `Network.get_elemental_contents` explicitly skips a species named
   `TGAS` -- it has no elemental content, and "TGAS" also happens to
   contain the substring `"S"`, which the substring-matching element
   parser would otherwise misread as one sulfur atom.
3. **Feeding the integrated temperature back into the rest of the network
   is a compile-time branch, not a runtime one.**
   `AbstractPhysics.integrates_temperature` is a *static* Equinox field
   (`eqx.field(static=True)`), so the solver's branch on it
   (`T = y[jnetwork.idx.TGAS] if integrates_temperature else ...`) is
   resolved at trace time. Networks that leave it at the default `False`
   get exactly the same compiled graph as before thermal balance existed
   -- zero JAX overhead for not using the feature. Keep it that way if you
   extend this: no new field here should be a traced array unless it
   genuinely needs to vary per-call.
4. **The species lookup (`carbox.index.Idx`) is a zero-leaf pytree.**
   `ThermoRateTerm` needs to find `H`, `E`, `O`, `H2` by name regardless of
   species ordering, so it carries an `Idx` (passed in at construction via
   `Reaction._reaction_rate_factory(self, idx=None)`). `Idx.tree_flatten`
   returns no leaves -- the name->index mapping lives entirely in
   aux_data -- so carrying it around costs nothing under `jit`/`vmap`;
   `idx.H` is a plain Python `int` (a static index), not a traced value.
   If you add another reaction type that needs species-level indexing,
   follow this pattern rather than passing `idx` through every
   `JReactionRateTerm.__call__` -- only reaction types that actually need
   it should take it.

Only cooling is implemented (`carbox.cooling`: Lyman-alpha, [OI] 630nm, H2
ro-vibrational); there is no heating channel yet, so `ThermoRate`'s net
rate is currently pure cooling. Adding heating means adding a channel
function with the same `(x, tgas, idx) -> jnp.ndarray` signature and
summing it (with the opposite sign) into `ThermoRateTerm.__call__` --
everything downstream (molecularity handling, the static branch, the
`Idx` plumbing) should need no changes.

## Adding a reaction type

1. Add a `Reaction` subclass in `carbox/reactions.py` describing the
   parameters (mirrors the network file's columns).
2. Add the matching `JReactionRateTerm` subclass (module-level, not
   nested inside a method) implementing `__call__(self, temperature,
   cr_rate, uv_field, visual_extinction, abundance_vector)`.
3. Thread `reaction_id` through the constructor like the existing
   reaction types.
4. If the reaction type has unique per-instance parameters that don't
   vectorize cleanly against others of the same type, add it to
   `non_vectorizable_types` in `Network.get_ode`.
5. Wire parsing for it in the relevant format parser under
   `carbox/parsers/` (see `carbox/parsers/README.md` for the per-format
   column layouts).

## Contributing something completely new
Since this code was written with the goal of being a batteries included
reimplementation of existing astrochemistry codes, you might to just
be able to use this code directly, if not, please do feel free to adapt
this code for your personal needs. If you do so, please do consider creating
a pull request to contribute back to the project, two/three-phase chemistry, better photochemistry, radiative transfer, better
solvers and a lot else would still be welcome. Also don't shy way from
opening an issue if you need something, but aren't sure how to obtain it
from this code.