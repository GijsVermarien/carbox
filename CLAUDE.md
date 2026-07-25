# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Carbox is a JAX-accelerated astrochemical kinetic reaction network simulator (JAX + Diffrax + Equinox). It parses a chemical reaction network from a CSV/text file (UCLCHEM, UMIST, or LATENT-TGAS format), compiles it into a JAX-traceable ODE right-hand side, and integrates species abundances over time under a chosen physical-conditions model (static cloud or CSE outflow).

## Commands

Use `.conda/bin/python` for all scripts/tests in this repo (project-local conda env).

```bash
# Install (editable, with dev extras: pytest, pre-commit, ruff)
.conda/bin/pip install -e .[dev]
pre-commit install

# Run the full test suite (config lives in pytest.ini: testpaths=tests, verbose, strict-markers)
.conda/bin/python -m pytest

# Run a single test file / test
.conda/bin/python -m pytest tests/test_conservation.py
.conda/bin/python -m pytest tests/test_reaction_id.py -k umist

# CLI entry point (installed as `carbox-sim`, or run as a module)
.conda/bin/python -m carbox.main --input data/network.csv --format umist
carbox-sim --input data/network.csv --config my_config.yaml

# Reproduce paper benchmark figures
cd benchmarks && ./run_benchmarks.sh && python plot_publication_comparison.py
cd sensitivity_analysis && ./run_cr_sensitivity.sh
```

There is no configured lint/format command beyond `ruff` being a dev dependency (no `ruff.toml`/`[tool.ruff]` section exists yet — don't assume `ruff check` is wired into CI).

**Known JAX version sensitivity** (see CONTRIBUTING.md's "Known issues" section for detail): newer JAX (0.4.30+) has triggered `TracerBoolConversionError` in this codebase. If you hit that, the previously-working pin is `jax[cpu]==0.4.26`, `jaxlib==0.4.26`, `equinox==0.11.4`, `lineax==0.0.4`, `diffrax==0.5.0`.

## Architecture

### Pipeline

`parse_chemical_network()` (file → `Network` of `Species`/`Reaction` objects) → `Network.get_ode()` (compiles to a `JNetwork`, a JAX pytree) → `solve_network()` (integrates via Diffrax) → `output.py` save functions. `carbox/main.py:run_simulation()` wires the whole pipeline together and is the main high-level entry point; `carbox/main.py:parse_network()`/`solve()` expose the two halves separately for callers that want to reuse a compiled network across multiple solves (e.g. sweeping `rate_modifiers`).

### Physics models (`carbox/physics.py`)

The physical conditions a network evolves under (density, temperature, visual extinction, position, and any non-chemical ODE source term) come from a `physics_model: AbstractPhysics` on `SimulationConfig`, not from scalar fields directly. If none is given, `SimulationConfig.__post_init__` builds a `StaticCloudPhysics` from the legacy scalar fields (`number_density`, `temperature`, `visual_extinction`, `use_self_consistent_av`, `base_av`, `cloud_radius_pc`) for backward compatibility.

- `AbstractPhysics.get_conditions(t_sec) -> (n, T, av, r)` is the one method every subclass must implement. It runs inside `jax.jit`/`vmap`, so it must be traceable — no Python branching on traced values, and every returned value must depend on `t_sec` (broadcast constants against it).
- `dilution(t_sec, y)`: additive non-chemical ODE term (e.g. expansion dilution); default zero.
- `time_grid(config)`: optional custom snapshot grid (e.g. `CSEPhysics` uses a grid log-spaced in radius, not time, to avoid clustering near t=0); default `None` falls back to the solver's generic log-time grid.
- `StaticCloudPhysics`: constant density/temperature.
- `CSEPhysics`: circumstellar-envelope outflow, `n ~ r^-2` from mass conservation, `T ~ r^-eps` power law. Its dilution term cancels analytically against the fractional-abundance ODE (see the comment above `CSEPhysics.time_grid` for why it deliberately does *not* override `dilution`).
- `carbox/cse_physics.py` is just a backward-compat shim re-exporting `CSEPhysics` from `physics.py` — don't add new code there.

To add a new model (e.g. a collapsing core), subclass `AbstractPhysics` and implement `get_conditions`; only override `dilution`/`time_grid` if the model needs a non-chemical source term or a custom grid.

### ODE state is fractional abundance

The integrated state vector is `x_i = n_i / n(t)` (fractional abundance relative to total gas density at time t), not number density. This is what keeps solver tolerances meaningful across a CSE outflow where `n` can drop by orders of magnitude, and lets the same `JNetwork.__call__` serve both static and dynamic-density physics models without special-casing. Rates are evaluated on number densities internally (`abundances * density`) since self-shielding/column-density terms expect cm^-3, then rescaled by `density ** (molecularities - 1)` to convert back to the fractional-abundance ODE (molecularity-1 power: m=1 photo/CR reactions need no density factor, m=2 bimolecular reactions need one).

### `Network` vs `JNetwork` (`carbox/network.py`)

- `Network` (plain Python dataclass): holds parsed `Species`/`Reaction` objects and the incidence matrix. `Network.get_ode()` does reaction vectorization (grouping same-type reactions to batch their rate-law evaluation, except a few `non_vectorizable_types` like `H2PhotoDissReaction` with unique per-reaction params) and returns a compiled `JNetwork`.
- `JNetwork` (an `equinox.Module`, i.e. a JAX pytree): the actual ODE right-hand side, called as `jnetwork(t, y, T, n, cr_rate, fuv_rate, av)` inside the solver's `_ode_func`.

**Rate modifiers**: `JNetwork` carries `rate_modifier_a`/`rate_modifier_b` fields (default identity: `a=1, b=0`) that let you scale/override specific reaction rates (`rate = rate*a + b`) *without recompiling the network* — central to how this framework is used for sensitivity work. Never pass modifiers as call-time args; instead call `jnetwork.with_rate_modifiers(a, b)` to get an updated immutable copy (uses `eqx.tree_at`), and that copy is what everything downstream (`solve_network`, `compute_derivatives`, `compute_reaction_rates`) should be given so they stay consistent with what was actually integrated. `compute_reaction_rates(..., use_rate_modifiers=True)` (default) returns the effective/modified coefficients; pass `False` to get the raw parameterization output instead. `JNetwork.modify_rates(rates)` is the single place the `a*rate+b` transform is applied — reuse it rather than re-deriving the formula.

### Reactions (`carbox/reactions.py`)

Two parallel class hierarchies: `Reaction` subclasses (plain Python objects describing a reaction, e.g. `KAReaction`, `CRPReaction`, `H2PhotoDissReaction`) and `JReactionRateTerm` subclasses (the corresponding `equinox.Module` rate-law implementations, e.g. `KAReactionRateTerm`). Every `Reaction` carries an optional `reaction_id` that survives parsing/reordering/vectorization and is used to label output columns (rates CSVs are keyed by `reaction_id`, not by a reconstructed reaction string) — if you add a new reaction type, thread `reaction_id` through its constructor like the existing ones.

### Parsers (`carbox/parsers/`)

`UnifiedChemicalParser` (`unified_parser.py`) dispatches to per-format parsers (`uclchem_parser.py`, `umist_parser.py`, `latent_tgas_parser.py`) based on `format_type` or auto-detection from the file. `parse_chemical_network(filepath, format_type=None)` is the public entry point. See `carbox/parsers/README.md` for format-specific details (column layouts, special reaction handling like IONOPOL/GAR).

### Config (`carbox/config.py`)

`SimulationConfig` is a plain dataclass (`from_yaml`/`from_json`/`to_yaml`/`to_json` for serialization; `physics_model` is excluded from serialization since it isn't trivially round-trippable). `save_all` overrides the individual `save_abundances`/`save_derivatives`/`save_rates`/`save_metadata`/`save_summary` flags when set (checked in `main.run_simulation`).

### Solver (`carbox/solver.py`)

`solve_network()` builds the ODE term from the physics model and JIT-compiles the integration (`_solve` wrapped in `eqx.filter_jit`). `get_solver()` maps `config.solver` (`kvaerno5`/`kvaerno3`/`dopri5`/`tsit5`) and `config.linear_solver` (`sparse`/`qr`/`lu`) to Diffrax/Lineax solver objects. `compute_derivatives()` and `compute_reaction_rates()` recompute dy/dt and per-reaction rates at existing solution snapshots (vmapped over time/state) — both take the same `jnetwork` that produced the solution so rate modifiers stay consistent.
