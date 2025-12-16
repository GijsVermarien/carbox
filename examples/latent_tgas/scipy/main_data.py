from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from chem_commons import idx_C, idx_H2, idx_O, names, nspecies
from chem_ode import fex
from chem_rates import get_rates, reaction_strings
from scipy.integrate import solve_ivp
from tqdm import tqdm

simulation_parameters = {
    # Hydrogen number density
    "number_density": 1e4,  # [1e2 - 1e6]
    # Fractional abunadnce of oxygen
    "O_fraction": 2e-4,  # [1e-5, 1e-3]
    # Fractional abundance of carbon
    "C_fraction": 1e-4,  # [1e-5, 1e-3]
    # Cosmic ray ionisation rate
    "cr_rate": 1e-17,  # enchance up to 1e-14
    # Radiation field
    "gnot": 1e0,  # enchance up to 1e5
    # t_gas_init
    "t_gas_init": 5e1,  # [1e1, 1e6]
}

# seconds per year
spy = 3600.0 * 24 * 365.0

# Initialize all species at the numerical minimum of 10^-20
y0 = np.zeros(nspecies + 1) + 1e-20 * simulation_parameters["number_density"]
# Gas temperature
y0[-1] = simulation_parameters["t_gas_init"]
# The initial molecular hydrogen abundance
y0[idx_H2] = simulation_parameters["number_density"]
# The intial carbon and oxygen abundances
y0[idx_O] = (
    simulation_parameters["number_density"] * simulation_parameters["O_fraction"]
)
y0[idx_C] = (
    simulation_parameters["number_density"] * simulation_parameters["C_fraction"]
)

# Cosmic ray ionisation rate and radiation field

# Integrate the system for 1 Myr
tend = 1e6 * spy
# Solve the system using the BDF method

# Increase this number for benchmarking
samples = 10
start = datetime.now()
for i in range(samples):
    sol = solve_ivp(
        fex,
        (0, tend),
        y0,
        "BDF",
        atol=1e-18,
        rtol=1e-12,
        args=(simulation_parameters["cr_rate"], simulation_parameters["gnot"]),
    )
print(f"Average time taken for {samples} samples: ", (datetime.now() - start) / samples)


# Print the error message if the integration failed
if not sol.success:
    print(sol.message)

import pandas as pd

sol_y = sol.y
sol_t = sol.t

df = pd.DataFrame(sol_y.T, index=sol_t, columns=names)
df.to_csv("scipy_no_heating.csv")

# Reevaluate the evaluations
dy = np.zeros_like(sol_y)
for i, (t, y) in enumerate(zip(sol_t, sol_y.T)):
    dy[:, i] = fex(
        t, y, simulation_parameters["cr_rate"], simulation_parameters["gnot"]
    )
df = pd.DataFrame(dy).T
df.columns = names
df.index = sol_t
df.to_csv("scipy_dy_no_heating.csv")


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
df.to_csv("scipy_rates.csv")
