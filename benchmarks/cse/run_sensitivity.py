#!/usr/bin/env python3
"""
Sensitivity of a CSE run's abundances to specific reactions' rate coefficients.

Computes d(n_i)/d(scale_j) by differentiating *through the whole ODE solve*
with jax.jacrev (reverse-mode; diffrax's default adjoint is a custom_vjp, so
forward-mode/jacfwd is not available here). `scale_j` is the per-reaction
multiplicative knob added in carbox/reactions.py: every rate law evaluates
as `k_j(T) * scale_j`, with scale_j = 1.0 giving the nominal rate -- so this
is exactly d(n_i)/d(ln k_j) at the nominal rate, independent of how k_j
happens to be parameterized (Arrhenius, ion-polar, photo, ...).

For UMIST reactions this is combined with the parsed accuracy class
(uncertainty_factor, from the A-E column -- see umist_parser.py) to report
an "uncertainty-propagated shift": raw_sensitivity * ln(uncertainty_factor),
a first-order estimate of how much n_i could plausibly move given how
uncertain that reaction's rate actually is (reactions with no accuracy info
get uncertainty_factor=1.0, so this shift is reported as 0 -- that means
"no assumed uncertainty", not "this reaction doesn't matter"). Since
d_species_d_scale = d(abundance)/d(ln k), uncertainty_shift is already an
absolute shift in the abundance itself, not a relative one.

Combining every reaction's uncertainty_shift for one species (in quadrature,
assuming independent rate uncertainties) gives an actual error bar on that
species' predicted abundance -- see summarize_uncertainty() /
`{run_name}_sensitivity_summary.csv`.

Cost note: jax.jacrev's cost scales with the number of OUTPUT components
requested (species x snapshots), not with how many reactions you
differentiate against -- reverse-mode AD gives the full gradient w.r.t.
*all* inputs in one pass, at a cost independent of input size. So
--reaction-id defaults to *every* reaction (essentially free); --species and
--snapshot-index are the real cost levers, and stay explicit/opt-in for
"all". The whole trajectory is solved once regardless -- "all radii" reuses
that single solve rather than re-solving per snapshot.

Examples:
    # Sensitivity of H2, HCO+, CO+ to reaction 3247, at the final radius
    python run_sensitivity.py --network umist_mini --reaction-id 3247 \\
        --species H2 HCO+ CO+

    # Same species, every reaction, every saved radius (one solve, one jacrev)
    python run_sensitivity.py --network umist_mini --species H2 HCO+ CO+ \\
        --snapshot-index all

    # Full analysis: every reaction, every species, every radius (expensive --
    # cost ~ n_species x n_snapshots reverse passes; fine on umist_mini, slow
    # on the full UMIST network)
    python run_sensitivity.py --network umist_mini --snapshot-index all
"""

import argparse
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from carbox.initial_conditions import initialize_abundances
from carbox.parsers import parse_chemical_network
from carbox.solver import get_time_grid, solve_network, SPY

from run_cse import CSE_NETWORKS, add_common_cse_args, prepare_cse_run  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)


def _scale_term(rate_term):
    """Unwrap _ScalarRateTermWrapper (used for non-vectorized special
    photoreactions) to reach the underlying rate term's `scale` field."""
    return getattr(rate_term, "term", rate_term).scale


def build_reaction_index(network, jnetwork) -> dict:
    """Map reaction_id -> (group_idx, position_in_group, group_ndim).

    network.reactions is reordered by Network.get_ode() to align exactly,
    group by group, with jnetwork.reactions -- see network.py's
    vectorization loop, which extends/appends both lists in lockstep.
    """
    index = {}
    flat_pos = 0
    for group_idx, rate_term in enumerate(jnetwork.reactions):
        scale = _scale_term(rate_term)
        ndim = scale.ndim
        size = int(jnp.atleast_1d(scale).shape[0])
        for pos in range(size):
            reaction = network.reactions[flat_pos]
            index[reaction.reaction_id] = (group_idx, pos, ndim)
            flat_pos += 1
    return index


def compute_sensitivity(
    network, jnetwork, y0, config, reaction_ids=None, species_names=None, snapshot_index=-1,
):
    """
    Returns a DataFrame with one row per (reaction_id, species, snapshot).

    reaction_ids=None means every reaction in the network -- cheap, since
    jacrev's cost doesn't depend on how many inputs are differentiated.
    snapshot_index=-1 (etc.) evaluates at one snapshot; "all" evaluates at
    every saved radius/time from a single solve.
    """
    reaction_index = build_reaction_index(network, jnetwork)

    if reaction_ids is None:
        reaction_ids = sorted(reaction_index)
    else:
        missing = [rid for rid in reaction_ids if rid not in reaction_index]
        if missing:
            raise ValueError(f"reaction_id(s) not found in this network: {missing}")

    reactions_by_id = {r.reaction_id: r for r in network.reactions}
    group_indices = sorted({reaction_index[rid][0] for rid in reaction_ids})

    species_names = species_names or [s.name for s in network.species]
    try:
        species_idx = np.array([network.get_index(s) for s in species_names])
    except ValueError as e:
        raise ValueError(
            f"Unknown species requested: {e}. "
            f"Available species: {[s.name for s in network.species]}"
        )

    all_snapshots = snapshot_index == "all"
    t_snapshots_sec = get_time_grid(config)  # deterministic, doesn't need solving
    selected_times = t_snapshots_sec if all_snapshots else jnp.atleast_1d(t_snapshots_sec[snapshot_index])
    n_dyn, T_dyn, av_dyn, r_dyn = jax.vmap(config.physics_model.get_conditions)(selected_times)

    # Nominal (scale=1 everywhere) abundances at the same snapshots, for the
    # relative-uncertainty column -- one plain forward solve, no differentiation.
    nominal_sol = solve_network(jnetwork, y0, config)
    nominal_ys = nominal_sol.ys[:, species_idx] if all_snapshots else nominal_sol.ys[snapshot_index][species_idx][None, :]
    nominal_ys = np.asarray(nominal_ys)  # (n_out_snapshots, n_species)

    n_out_snapshots = len(selected_times)
    print(
        f"Output: {len(species_names)} species x {len(reaction_ids)} reaction(s) x "
        f"{n_out_snapshots} snapshot(s) = {len(species_names) * n_out_snapshots} reverse-mode "
        f"passes (reaction count doesn't affect cost)"
    )

    # Mark only the touched groups' `scale` leaves as differentiable;
    # everything else (all other reactions' parameters, incidence, etc.)
    # stays fixed/static.
    false_tree = jax.tree_util.tree_map(lambda _: False, jnetwork)
    where = lambda jn: tuple(_scale_term(jn.reactions[g]) for g in group_indices)  # noqa: E731
    filter_spec = eqx.tree_at(where, false_tree, replace=tuple(True for _ in group_indices))
    diff_params, static_params = eqx.partition(jnetwork, filter_spec)

    def run(params):
        net = eqx.combine(params, static_params)
        sol = solve_network(net, y0, config)
        if all_snapshots:
            return sol.ys[:, species_idx]  # (n_snapshots, n_species)
        return sol.ys[snapshot_index][species_idx][None, :]  # (1, n_species)

    jac = jax.jacrev(run)(diff_params)  # leaves gain a (n_out_snapshots, n_species, ...) prefix

    rows = []
    for rid in reaction_ids:
        group_idx, pos, ndim = reaction_index[rid]
        block = _scale_term(jac.reactions[group_idx])  # (n_snap, n_species) or (n_snap, n_species, group_size)
        column = block if ndim == 0 else block[..., pos]  # (n_snap, n_species)

        reaction = reactions_by_id[rid]
        uncertainty_flag = getattr(reaction, "uncertainty_flag", None)
        uncertainty_factor = getattr(reaction, "uncertainty_factor", 1.0)
        reaction_str = f"{' + '.join(reaction.reactants)} -> {' + '.join(reaction.products)}"

        column = np.asarray(column)
        for snap_i in range(n_out_snapshots):
            for sp_i, (sp_name, sensitivity) in enumerate(zip(species_names, column[snap_i])):
                rows.append({
                    "reaction_id": rid,
                    "reaction_type": reaction.reaction_type,
                    "reaction": reaction_str,
                    "uncertainty_flag": uncertainty_flag,
                    "uncertainty_factor": uncertainty_factor,
                    "species": sp_name,
                    "time_years": float(selected_times[snap_i]) / SPY,
                    "radius_cm": float(r_dyn[snap_i]),
                    "nominal_abundance": nominal_ys[snap_i, sp_i],
                    "d_species_d_scale": sensitivity,
                    "uncertainty_shift": sensitivity * np.log(uncertainty_factor),
                })

    return pd.DataFrame(rows)


def summarize_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine every reaction's uncertainty_shift into one error bar per
    (species, snapshot), assuming independent rate uncertainties: variances
    add, so the combined 1-sigma-like shift is the quadrature sum (root of
    the sum of squares) rather than a plain sum (which would assume every
    reaction is simultaneously wrong in the same direction -- a worst-case
    bound, not a typical error bar).

    d_species_d_scale = d(abundance)/d(ln k), so uncertainty_shift is
    already an absolute shift in the abundance itself (same units as
    nominal_abundance) -- dividing by nominal_abundance gives the relative
    (fractional) uncertainty instead.
    """
    group_cols = ["species", "time_years", "radius_cm"]
    summary = (
        df.groupby(group_cols)
        .agg(
            nominal_abundance=("nominal_abundance", "first"),
            sigma_abundance=("uncertainty_shift", lambda s: float(np.sqrt((s**2).sum()))),
            worst_case_abundance=("uncertainty_shift", lambda s: float(s.abs().sum())),
            n_reactions=("reaction_id", "count"),
        )
        .reset_index()
    )
    summary["relative_uncertainty"] = summary["sigma_abundance"] / summary["nominal_abundance"]
    return summary


def _parse_snapshot_index(value: str):
    if value == "all":
        return "all"
    return int(value)


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity of CSE abundances to specific reactions' rate coefficients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default="results",
                         help="Output directory (relative to this script's directory unless absolute)")
    parser.add_argument("--run-name", default=None, help="Overrides the default run name")
    add_common_cse_args(parser)

    parser.add_argument(
        "--reaction-id", type=int, nargs="+", default=None,
        help="Reaction(s) to compute sensitivity for (default: every reaction in the network -- "
             "this is cheap, jacrev's cost doesn't depend on how many reactions you differentiate "
             "against). Use the stable reaction_id, e.g. the UMIST reaction number.",
    )
    parser.add_argument(
        "--species", nargs="*", default=None,
        help="Species to report sensitivity for (default: all -- expensive for large networks, "
             "since cost scales with the number of species requested)",
    )
    parser.add_argument(
        "--snapshot-index", type=_parse_snapshot_index, default=-1,
        help="Which saved snapshot to evaluate at (default: -1, the final one), or 'all' to "
             "get sensitivity vs. radius from a single solve",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Rows to print, ranked by |uncertainty_shift|")

    args = parser.parse_args()
    kwargs = vars(args)
    reaction_ids = kwargs.pop("reaction_id")
    species_names = kwargs.pop("species")
    snapshot_index = kwargs.pop("snapshot_index")
    top_n = kwargs.pop("top_n")

    if species_names is None:
        print(
            "No --species given: computing sensitivity for every species in the network. "
            "This is O(n_species) reverse-mode passes -- pass --species explicitly to speed this up."
        )

    input_file, format_type, run_name, output_dir, config = prepare_cse_run(**kwargs)

    network = parse_chemical_network(str(input_file), format_type)
    y0 = initialize_abundances(network, config)
    jnetwork = network.get_ode()

    print(f"\nComputing sensitivity of {species_names or 'all species'} "
          f"to {'all reactions' if reaction_ids is None else reaction_ids} "
          f"at snapshot(s) {snapshot_index}...")

    df = compute_sensitivity(
        network, jnetwork, y0, config,
        reaction_ids=reaction_ids, species_names=species_names, snapshot_index=snapshot_index,
    )

    out_path = output_dir / f"{run_name}_sensitivity.csv"
    df.to_csv(out_path, index=False)

    ranked = df.reindex(df["uncertainty_shift"].abs().sort_values(ascending=False).index)
    print(f"\nTop {min(top_n, len(ranked))} by |uncertainty_shift| (= d(species)/d(scale) * ln(uncertainty_factor)):")
    with pd.option_context("display.max_rows", top_n, "display.width", 160):
        print(ranked.head(top_n).to_string(index=False))

    print(f"\nSaved full table ({len(df)} rows): {out_path}")

    summary = summarize_uncertainty(df)
    summary_path = output_dir / f"{run_name}_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(
        "\nPer-species error bar (quadrature sum of every reaction's uncertainty_shift; "
        "assumes independent rate uncertainties):"
    )
    with pd.option_context("display.max_rows", len(summary), "display.width", 160):
        print(summary.to_string(index=False))
    print(f"\nSaved summary ({len(summary)} rows): {summary_path}")


if __name__ == "__main__":
    main()
