# Carbox

Carbox is a simulation framework for simulating astrochemical kinetic reaction systems. 

The main code lives in the `carbox` directory, benchmarks between uclchem and carbox can be found in `benchmarks`, `examples` highlights an usage of carbox with thermochemistry.

Here is the [release paper](https://arxiv.org/abs/2511.10558), presented at [Diffsys 2025](https://differentiable-systems.github.io/workshop-eurips-2025/).


## Usage
In order to reproduce figure 1. from the whitepaper, ensure the uclchem submodule is present and install it with `pip`, install the `requirements.txt` for carbox and run `run_benchmark.sh` and `plot_publication_comparison.py`. 
Figure 2. can be reproduced by following the same steps but running `run_cr_sensitivity.sh` instead.
