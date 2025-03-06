import matplotlib.pyplot as plt
import numpy as np
from chem_commons import idx_C, idx_H2, idx_O, names, nspecies
from chem_ode import fex
from tqdm import tqdm

from scipy.integrate import solve_ivp

simulation_parameters = {
    # Hydrogen number density
    "ntot": 1e4,  # [1e2 - 1e6]
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
y0 = np.zeros(nspecies + 1) + 1e-20 * simulation_parameters["ntot"]
# Gas temperature
y0[-1] = simulation_parameters["t_gas_init"]
# The initial molecular hydrogen abundance
y0[idx_H2] = simulation_parameters["ntot"]
# The intial carbon and oxygen abundances
y0[idx_O] = simulation_parameters["ntot"] * simulation_parameters["O_fraction"]
y0[idx_C] = simulation_parameters["ntot"] * simulation_parameters["C_fraction"]

# Cosmic ray ionisation rate and radiation field

# Integrate the system for 1 Myr
tend = 1e6 * spy
# Solve the system using the BDF method
sol = solve_ivp(
    fex,
    (0, tend),
    y0,
    "BDF",
    atol=1e-40,
    rtol=1e-12,
    args=(simulation_parameters["cr_rate"], simulation_parameters["gnot"]),
)

# Print the error message if the integration failed
if not sol.success:
    print(sol.message)

# Print the minimum abundance to check convergence
# print("Minimum solver abundance: ", sol.y.min())


# Plot the abundances
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
lss = ["-", "--", ":"]
for i, lab in enumerate(names[:-1]):
    plt.loglog(
        sol.t / spy,
        sol.y[i],
        label=lab,
        color=colors[i % len(colors)],
        ls=lss[i // len(colors)],
    )
plt.loglog(sol.t / spy, sol.y[-1], label="Tgas", color="k")
plt.legend(loc="best", ncol=2, fontsize=6)
# plt.show()

import pandas as pd

df = pd.DataFrame(sol.y.T, index=sol.t, columns=names)
df.to_csv("scipy.csv")
