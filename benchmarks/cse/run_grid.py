#!/usr/bin/env python3
"""
Run a grid of CSE simulations, sweeping one or more parameters of run_cse.py.

Don't edit DEFAULT_PHYSICS in run_cse.py to scan parameters -- that just
changes the fallback for every future single run and leaves no record of
what was actually run. Instead, sweep parameters either from the command
line (--param/--fixed) or from an input file (--input-file), matching the
":"/";" convention from the Fortran model this is replacing. Each grid
point calls run_cse.run_cse() directly (in-process, not via subprocess),
and results land in one output directory with a manifest CSV tying each
run's files back to its parameter values.

--- Input-file grid syntax (applies to every parameter, not just physics) ---

One `name = value` per line (blank lines and lines starting with # are
skipped). A plain value is fixed across every run. A value containing ":"
is a *zip* sweep: every ":"-parameter's i-th value is used together in the
i-th run, so all ":" parameters in the file must list the same number of
values. A value containing ";" is a *cartesian* sweep: every combination
across all ";" parameters (and the single zip group, if any) gets its own
run.

    r_init = 1e14:1e15
    r_final = 1e16:1e17
    # -> 2 runs (zipped): (r_init=1e14, r_final=1e16), (r_init=1e15, r_final=1e17)

    r_init = 1e14;1e15
    r_final = 1e16;1e17
    # -> 4 runs (cartesian): every r_init combined with every r_final

":" and ";" can both appear in the same file: the ":" group collapses into
one combined axis, which is then cartesian-multiplied against each ";"
parameter. Any parameter of run_cse() can be swept this way, including
`network` itself (e.g. `network = umist_mini;uclchem`).

Examples:
    # File-based grid, Fortran-model style
    python run_grid.py --input-file my_grid.in --output results/grid1

    # 3x3 grid over mdot and vexp, everything else at run_cse defaults
    python run_grid.py --network umist_mini \\
        --param mdot=5e-6,1e-5,2e-5 --param vexp=10,15,20 \\
        --output results/grid_mdot_vexp

    # See the planned grid without running anything
    python run_grid.py --input-file my_grid.in --dry-run
"""

import argparse
import inspect
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_cse import run_cse, CSE_NETWORKS, SCRIPT_DIR  # noqa: E402

# Parameters of run_cse() that make sense to sweep or fix -- excludes
# output/run_name/verbose, which this script manages itself. `network` IS
# included: an input file can legitimately sweep across networks too.
_SIGNATURE = inspect.signature(run_cse)
SWEEPABLE_PARAMS = [
    name for name in _SIGNATURE.parameters
    if name not in ("output", "run_name", "verbose")
]


def _infer_caster(name: str):
    """Pick a type caster for a run_cse() parameter from its type annotation.

    Prefer the annotation over the default value: several parameters (e.g.
    `resolution: float = None`) default to None but still expect a numeric
    value when actually given, which a default-based guess would get wrong.
    """
    annotation = _SIGNATURE.parameters[name].annotation
    if annotation is bool:
        return lambda v: v.strip().lower() in ("1", "true", "yes", "on")
    if annotation is int:
        return int
    if annotation is float:
        return float
    return str


def _format_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _validate_name(name: str):
    if name not in SWEEPABLE_PARAMS:
        raise ValueError(f"Unknown run_cse parameter {name!r}. Available: {SWEEPABLE_PARAMS}")


# --- CLI-style grid: --param name=v1,v2,... (cartesian) + --fixed name=value ---

def _parse_param_arg(spec: str) -> tuple:
    """Parse 'name=v1,v2,v3' into (name, [typed values])."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected name=v1,v2,... got: {spec!r}")
    name, raw_values = spec.split("=", 1)
    name = name.strip()
    _validate_name(name)
    caster = _infer_caster(name)
    return name, [caster(v) for v in raw_values.split(",")]


def build_combinations_from_params(param_specs, fixed_specs) -> tuple:
    """--param/--fixed path: plain cartesian product over --param dimensions.

    Returns (combinations, swept_names) -- swept_names excludes --fixed
    overrides, so run names reflect only what's actually varying.
    """
    swept = [_parse_param_arg(p) for p in param_specs]
    fixed = {}
    for f in fixed_specs:
        name, values = _parse_param_arg(f)
        if len(values) != 1:
            raise ValueError(f"--fixed {f} must be a single value, not a list")
        fixed[name] = values[0]

    names = [name for name, _ in swept]
    value_lists = [values for _, values in swept]
    combinations = [
        {**fixed, **dict(zip(names, combo))}
        for combo in itertools.product(*value_lists)
    ]
    return combinations, set(names)


# --- Input-file grid: ":" = zip sweep, ";" = cartesian sweep ---

def parse_input_file(path: Path) -> dict:
    """Read `name = value` lines into {name: (mode, [typed values])}.

    mode is "fixed" (no separator), "zip" (":"-separated), or
    "cartesian" (";"-separated).
    """
    parsed = {}
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: expected 'name = value', got: {raw_line!r}")
        name, raw_value = (part.strip() for part in line.split("=", 1))
        _validate_name(name)
        caster = _infer_caster(name)

        if ";" in raw_value:
            mode, raw_values = "cartesian", raw_value.split(";")
        elif ":" in raw_value:
            mode, raw_values = "zip", raw_value.split(":")
        else:
            mode, raw_values = "fixed", [raw_value]

        parsed[name] = (mode, [caster(v.strip()) for v in raw_values])
    return parsed


def build_combinations_from_file(path: Path) -> tuple:
    """Returns (combinations, swept_names) -- swept_names excludes fixed
    (unseparated) values, so run names reflect only what's actually varying."""
    parsed = parse_input_file(path)

    fixed = {name: values[0] for name, (mode, values) in parsed.items() if mode == "fixed"}
    zipped = {name: values for name, (mode, values) in parsed.items() if mode == "zip"}
    cartesian = {name: values for name, (mode, values) in parsed.items() if mode == "cartesian"}
    swept_names = set(zipped) | set(cartesian)

    axes = []  # each axis is a list of {name: value, ...} dicts to cartesian-multiply
    if zipped:
        lengths = {name: len(values) for name, values in zipped.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"':'-swept (zip) parameters must all list the same number of values: {lengths}"
            )
        n = next(iter(lengths.values()))
        axes.append([
            {name: values[i] for name, values in zipped.items()}
            for i in range(n)
        ])
    for name, values in cartesian.items():
        axes.append([{name: v} for v in values])

    if not axes:
        return [dict(fixed)], swept_names

    combinations = []
    for parts in itertools.product(*axes):
        combo = dict(fixed)
        for part in parts:
            combo.update(part)
        combinations.append(combo)
    return combinations, swept_names


def _run_name_for(overrides: dict, all_swept_names) -> str:
    swept_only = {k: v for k, v in overrides.items() if k in all_swept_names}
    if not swept_only:
        return "cse_grid"
    return "cse_" + "_".join(f"{n}{_format_value(v)}" for n, v in swept_only.items())


def run_grid(combinations: list, output: str, swept_names=None, dry_run: bool = False) -> list:
    """Run every combination (a list of run_cse() kwarg dicts), returning manifest rows."""
    all_swept_names = set(swept_names) if swept_names is not None else {
        k for combo in combinations for k in combo
    }

    print(f"Grid: {len(combinations)} run(s)")
    if dry_run:
        for combo in combinations:
            print("  ", combo)
        return []

    manifest_rows = []
    for i, overrides in enumerate(combinations):
        run_name = _run_name_for(overrides, all_swept_names)
        print(f"\n[{i + 1}/{len(combinations)}] {run_name}: {overrides}")

        row = {"run_name": run_name, **overrides}
        t0 = time.perf_counter()
        try:
            results = run_cse(output=output, run_name=run_name, verbose=False, **overrides)
            row["success"] = True
            row["error"] = ""
            row["n_species"] = len(results["network"].species)
            row["n_reactions"] = len(results["network"].reactions)
            row["output_dir"] = str(results["output_dir"])
        except Exception as e:  # noqa: BLE001 -- keep sweeping past a bad grid point
            row["success"] = False
            row["error"] = str(e)
            print(f"  FAILED: {e}")
        row["elapsed_seconds"] = time.perf_counter() - t0
        print(f"  {'done' if row['success'] else 'failed'} in {row['elapsed_seconds']:.1f}s")
        manifest_rows.append(row)

    output_dir = Path(output)
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "grid_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    n_ok = sum(r["success"] for r in manifest_rows)
    print(f"\n{'=' * 70}")
    print(f"Grid complete: {n_ok}/{len(manifest_rows)} succeeded")
    print(f"Manifest: {manifest_path}")
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(
        description="Sweep a grid of CSE runs over one or more run_cse.py parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default="results/grid",
                         help="Directory for all grid outputs + manifest.csv "
                              "(relative to this script's directory unless absolute)")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned grid, run nothing")

    file_group = parser.add_argument_group("Input-file grid (Fortran-model style)")
    file_group.add_argument(
        "--input-file", type=Path, default=None,
        help="Parameter file using ':' (zip) / ';' (cartesian) sweep syntax -- see this script's docstring",
    )

    cli_group = parser.add_argument_group("CLI grid (alternative to --input-file)")
    cli_group.add_argument("--network", default="umist", choices=list(CSE_NETWORKS.keys()))
    cli_group.add_argument(
        "--param", action="append", default=[], metavar="NAME=V1,V2,...",
        help=f"Parameter to sweep, cartesian (repeatable). One of: {SWEEPABLE_PARAMS}",
    )
    cli_group.add_argument(
        "--fixed", action="append", default=[], metavar="NAME=VALUE",
        help="Fixed override applied to every grid run (repeatable, not swept)",
    )

    args = parser.parse_args()

    if args.input_file:
        if args.param or args.fixed:
            parser.error("--input-file cannot be combined with --param/--fixed")
        combinations, swept_names = build_combinations_from_file(args.input_file)
    else:
        if not args.param:
            parser.error("Give --input-file, or at least one --param name=v1,v2,... to sweep")
        combinations, swept_names = build_combinations_from_params(args.param, args.fixed)
        for combo in combinations:
            combo.setdefault("network", args.network)

    run_grid(combinations, output=args.output, swept_names=swept_names, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
