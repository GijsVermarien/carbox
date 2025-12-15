# Carbox

Carbox is a simulation framework for simulating astrochemical kinetic reaction systems, built on top of **JAX**, **Diffrax**, and **Equinox**.

The main code lives in the `carbox` directory, benchmarks between uclchem and carbox can be found in `benchmarks`, `examples` highlights an usage of carbox with thermochemistry.

Here is the [release paper](https://arxiv.org/abs/2511.10558), presented at [Diffsys 2025](https://differentiable-systems.github.io/workshop-eurips-2025/).


## Installation

Install the core Carbox library from the repository root with:

```bash
pip install .
```

This installs the Python package `carbox` together with a small command-line entry
point `carbox-sim`.

To install the additional tools needed to reproduce the UCLCHEM / Fortran
benchmarks, use the `benchmarks` extra:

```bash
pip install .[benchmarks]
```

## Python API

A minimal end-to-end example that runs a network from a CSV file:

```python
from carbox import SimulationConfig, run_simulation

config = SimulationConfig(
    number_density=1e4,
    temperature=50.0,
    t_end=1e6,
    run_name="example_run",
)

results = run_simulation("data/network.csv", config, format_type="latent_tgas")
solution = results["solution"]
network = results["network"]
```

The same high-level API is used internally by the benchmarking suite under
`benchmarks/`.

## Command line usage

The package also provides a small CLI wrapper around the same functionality:

```bash
carbox-sim --input data/network.csv --format latent_tgas
```

You can optionally provide a configuration file:

```bash
carbox-sim --input data/network.csv --config config.yaml
```

Run `carbox-sim --help` to see all available options.

## Reproducing paper figures

In order to reproduce figure 1 from the whitepaper, ensure the UCLCHEM submodule
is present and installed with `pip`, install the Carbox requirements, and run:

```bash
cd benchmarks
./run_benchmarks.sh
python plot_publication_comparison.py
```

Figure 2 can be reproduced by following the same steps but running:

```bash
cd sensitivity_analysis
./run_cr_sensitivity.sh
```
