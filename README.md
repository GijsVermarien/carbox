
# Note
This is a fork of the original repository primarily for testing features. 

Full credit goes to GijsVermarien and the original repository listed [here](https://github.com/GijsVermarien/carbox.git).
The whitepaper they published is also linked here: [Carbox: an end-to-end differentiable astrochemical simulation framework](https://arxiv.org/abs/2511.10558).

# Carbox

Carbox is a simulation framework for simulating astrochemical kinetic reaction systems.

The main code lives in the `carbox` directory, benchmarks between uclchem and carbox can be found in `benchmarks`, `examples` highlights an usage of carbox with thermochemistry.


## Usage
In order to reproduce figure 1. from the whitepaper, ensure the uclchem submodule is present and install it with `pip`, install the `requirements.txt` for carbox and run `run_benchmark.sh` and `plot_publication_comparison.py`.
Figure 2. can be reproduced by following the same steps but running `run_cr_sensitivity.sh` instead.
