import logging
import sys

import diffrax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chem_commons import idx_C, idx_H2, idx_O, names, nspecies
from chem_ode import fex
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

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

logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)

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
tend = 1e6 * spy

# Define the initial state and time span
initial_state = y0
t0 = 0.0
t1 = tend

from diffrax import Dopri5, Kvaerno5, ODETerm, PIDController, SaveAt, diffeqsolve

# solver = Dopri5()
solver = Kvaerno5()

# Define the differential equation problem'
problem = ODETerm(lambda t, y, args: fex(t, y, args[0], args[1]))
# Solve the problem
solution = diffeqsolve(
    problem,
    solver,
    t0=0.0,
    dt0=0.001 * tend,
    t1=tend,
    y0=y0,
    stepsize_controller=PIDController(
        atol=1e-18,
        rtol=1e-6,
    ),
    saveat=SaveAt(ts=spy * jnp.logspace(-14, 6, 1000)),
    args=[simulation_parameters["cr_rate"], simulation_parameters["gnot"]],
    max_steps=16**3,
)

# print(solution)
# solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)

# Extract the solution
sol_t = solution.ts
sol_y = solution.ys.T


# Print the minimum abundance to check convergence
# print("Minimum solver abundance: ", sol_y.min())

df = pd.DataFrame(solution.ys)
df.columns = names
# print(solution)
ions = [n for n in names if n[-1] == "+"]
# print(ions)
# print("Ion density: ", df[ions].sum(axis=1))
# print("Electron density: ", df["E"])

# Plot the abundances
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
lss = ["-", "--", ":"]
# print(solution.ts.shape)
# print(solution.ys.shape)

df = pd.DataFrame(solution.ys, index=solution.ts, columns=names)
df.to_csv("jax.csv")
# print(df.head())
for i, lab in enumerate(names[:-1]):
    plt.loglog(
        solution.ts / spy,
        solution.ys[:, i],
        label=lab,
        color=colors[i % len(colors)],
        ls=lss[i // len(colors)],
    )

plt.loglog(solution.ts / spy, solution.ys[:, -1], label="Tgas", color="k")
plt.legend(loc="best", ncol=2, fontsize=6)
