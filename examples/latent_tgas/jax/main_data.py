from datetime import datetime
import logging
import sys

import diffrax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chem_commons import idx_C, idx_H2, idx_O, names, nspecies
from chem_ode import fex
from tqdm import tqdm

import equinox as eqx

import jax
import jax.numpy as jnp
from jax import random
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

from chem_rates import get_rates, reaction_strings


# x = random.uniform(random.key(0), (1000,), dtype=jnp.float64)
# print(x.dtype)  # --> dtype('float64')

simulation_parameters = {
    # Hydrogen number density
    "ntot": 1e4,  # [1e2 - 1e6] cm^-3
    # Fractional abunadnce of oxygen
    "O_fraction": 2e-4,  # [1e-5, 1e-3] -
    # Fractional abundance of carbon
    "C_fraction": 1e-4,  # [1e-5, 1e-3] -
    # Cosmic ray ionisation rate
    "cr_rate": jnp.array(1e-17),  # enchance up to 1e-14 s^-1
    # Radiation field
    "gnot": jnp.array(1e0),  # enchance up to 1e5
    # Initial gas temperature
    "t_gas_init": 5e1,  # [1e1, 1e5]
}

# logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)

# seconds per year
spy = 3600.0 * 24 * 365.0

# Initialize all species at the numerical minimum of 10^-20
y0 = np.zeros(nspecies + 1) + 1e-20 * simulation_parameters["ntot"]
# Gas temperature
y0[-1] = simulation_parameters["t_gas_init"]
# The initial molecular hydrogen abundance
y0[idx_H2] = simulation_parameters["ntot"]
# The intial carbon and oxygen abundances
y0[idx_O] = simulation_parameters["ntot"] * simulation_parameters["O_fraction"]
y0[idx_C] = simulation_parameters["ntot"] * simulation_parameters["C_fraction"]

y0 = jnp.array(y0)

# Cosmic ray ionisation rate and radiation field

# Integrate the system for 1 Myr
# tend = 1e6 * spy

# Define the initial state and time span
initial_state = y0
# t0 = 0.0
# t1 = tend

from diffrax import Dopri5, Kvaerno5, ODETerm, PIDController, SaveAt, diffeqsolve

# solver = Dopri5()
solver = Kvaerno5()

# Define the differential equation problem'


cr_rate = jnp.array(simulation_parameters["cr_rate"])
gnot = jnp.array(simulation_parameters["gnot"])


problem = ODETerm(lambda t, y, args: fex(t, y, args[0], args[1]))


tend = 1e6


# Solve the problem
@eqx.filter_jit
def solver_wrap(y0):
    print("compiling")
    return diffeqsolve(
        problem,
        solver,
        t0=0.0,
        t1=spy * tend,
        dt0=1e-6,
        y0=y0,
        stepsize_controller=PIDController(
            atol=1e-18,
            rtol=1e-12,
        ),
        saveat=SaveAt(ts=spy * jnp.logspace(-5, np.log10(tend), 1000)),
        args=(cr_rate, gnot),
        max_steps=16**3,
    )


solver_wrap(y0)

samples = 10

# with jax.profiler.trace("/tmp/latent_tgas", create_perfetto_trace=True):
start = datetime.now()
for i in range(samples):
    solution = solver_wrap(y0)
print(
    f"Average time taken for {samples} samples: ",
    (datetime.now() - start) / samples,
)
print(f"Solver report: {solution.stats}")

# Extract the solution
sol_t = solution.ts
sol_y = solution.ys.T

df = pd.DataFrame(solution.ys)
df.columns = names
df = pd.DataFrame(solution.ys, index=solution.ts, columns=names)
df.to_csv("jax_no_heating.csv")


# Reevaluate the evaluations
dy = jnp.zeros_like(sol_y)
for i, (t, y) in enumerate(zip(sol_t, sol_y.T)):
    dy = dy.at[:, i].set(fex(t, y, cr_rate, gnot))
print(sol_t.shape, sol_y.shape)
df = pd.DataFrame(dy).T
df.columns = names
df.index = sol_t
df.to_csv("jax_dy_no_heating.csv")

rates = np.zeros((len(sol_t), len(reaction_strings)))
for i, (t, y) in enumerate(zip(sol_t, sol_y.T)):
    rates[i] = get_rates(
        simulation_parameters["t_gas_init"],
        simulation_parameters["cr_rate"],
        simulation_parameters["gnot"],
    )
df = pd.DataFrame(rates)
df.columns = reaction_strings
df.index = sol_t
df.to_csv("jax_rates.csv")

from jax import make_jaxpr

with open("jax_jaxpr.txt", "w") as f:
    f.write(str(make_jaxpr(fex)(0.0, y0, 1e-17, 1e0)))
