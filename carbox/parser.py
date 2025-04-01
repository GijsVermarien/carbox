from datetime import datetime
import pandas as pd
import numpy as np

from network import Network
from constants import elemental_dict
from reactions import KAReaction, CRReaction, FUVReaction, H2FormReaction
import matplotlib.pyplot as plt


import diffrax as dx
import equinox as eqx

import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

GAS2DUST = 0.01


# FUTURE: 2 times reactant  4 times product
reaction_by_shorthand_name = {
    "KA": lambda r1, r2, p1, p2, a, b, c, rtype: KAReaction(
        rtype, (r1, r2), (p1, p2), a, b, c
    ),
    "CR": lambda r1, r2, p1, p2, a, b, c, rtype: CRReaction(
        rtype, (r1, r2), (p1, p2), a
    ),
    "FUV": lambda r1, r2, p1, p2, a, b, c, rtype: FUVReaction(
        rtype, (r1, r2), (p1, p2), a
    ),
    "H2Form": lambda r1, r2, p1, p2, a, b, c, rtype: H2FormReaction(
        rtype, (r1, r2), (p1, p2), a, GAS2DUST
    ),
}


def parse_atoms(name, mode="mass"):
    import itertools

    atoms = sorted(elemental_dict.keys(), key=lambda x: len(x), reverse=True)
    ps = ["".join(x) for x in itertools.product("qzxj", repeat=4)][: len(atoms)]
    proxy = {a: p for a, p in zip(atoms, ps)}
    proxy_rev = {p: a for a, p in proxy.items()}

    pname = name.strip()
    for a in atoms:
        pname = pname.replace(a, "$" + proxy[a] + "$")

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    alist = [x for x in pname.split("$") if x != ""]
    expl = []
    pold = None
    for p in alist:
        if not is_number(p):
            expl += [p]
        else:
            expl += [pold] * max(int(p) - 1, 1)
        pold = p
    exploded = sorted([proxy_rev[x] for x in expl])
    if mode == "mass":
        mass = sum([elemental_dict[x]["mass"] for x in exploded])
    elif mode == "atomic_weight":
        mass = sum([elemental_dict[x]["atomic_weight"] for x in exploded])
    else:
        raise ValueError("Mode not recognized")
    return exploded, mass


def parse_weights(molecules):
    return [parse_atoms(x, mode="atomic_weight") for x in molecules]


# md = load_mass_dict("atom_mass.dat")

# print(parse("HCO3HCO+", md))

simulation_parameters = {
    # Hydrogen number density
    "ntot": 1e4,  # [1e2 - 1e6]
    # Fractional abunadnce of oxygen
    "O_fraction": 2e-4,  # [1e-5, 1e-3]
    # Fractional abundance of carbon
    "C_fraction": jnp.array(1e-4),  # [1e-5, 1e-3]
    # Cosmic ray ionisation rate
    "cr_rate": jnp.array(1e-17),  # enchance up to 1e-14
    # Radiation field
    "gnot": 1e0,  # enchance up to 1e5
    # t_gas_init
    "t_gas_init": jnp.array(5e1),  # [1e1, 1e6]
}

spy = 3600.0 * 24 * 365.0


if __name__ == "__main__":
    reactions_file = pd.read_csv("../data/simple_latent_tgas.csv")
    # Get the unique species from the reaction file
    species = list(
        set(reactions_file["r1"])
        | set(reactions_file["r2"])
        | set(reactions_file["p1"])
        | set(reactions_file["p2"])
    )
    species.remove(np.nan)

    # Sort the species by atomic weight
    species = sorted(species, key=lambda x: parse_atoms(x, mode="atomic_weight")[1])

    # Parse the reactions:
    reactions = [
        reaction_by_shorthand_name[reac.iloc[-1]](*reac)
        for idx, reac in reactions_file.iterrows()
    ]

    # Create the reaction network and get the ODE system
    reaction_network = Network(species, reactions)
    system = reaction_network.get_ode()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Plot the reaction network
    ax.imshow(reaction_network.incidence.T)
    ax.set_xticks(np.arange(len(reaction_network.species)))
    ax.set_yticks(np.arange(len(reaction_network.reactions)))
    ax.set_xticklabels(reaction_network.species, rotation=90)
    ax.set_yticklabels(reaction_network.reactions)
    # plt.show()

    y0 = jnp.ones(len(reaction_network.species)) * 1e-20 * simulation_parameters["ntot"]
    # The initial molecular hydrogen abundance
    y0 = y0.at[reaction_network.get_index("H2")].set(simulation_parameters["ntot"])
    # The intial carbon and oxygen abundances
    y0 = y0.at[reaction_network.get_index("O")].set(
        simulation_parameters["ntot"] * simulation_parameters["O_fraction"]
    )
    y0 = y0.at[reaction_network.get_index("C")].set(
        simulation_parameters["ntot"] * simulation_parameters["C_fraction"]
    )

    tend = 1e1

    @eqx.filter_jit
    def get_solution(system, y0, tend, simulation_parameters):
        print("compiling")
        return dx.diffeqsolve(
            dx.ODETerm(lambda t, y, args: system(t, y, args[0], args[1], args[2])),
            dx.Kvaerno5(),
            y0=y0,
            t0=0.0,
            t1=spy * tend,
            dt0=1e-6,
            saveat=dx.SaveAt(ts=spy * jnp.logspace(-5, np.log10(tend), 1000)),
            stepsize_controller=dx.PIDController(
                atol=1e-18,
                rtol=1e-12,
            ),
            args=[
                simulation_parameters["t_gas_init"],
                simulation_parameters["cr_rate"],
                simulation_parameters["gnot"],
            ],
            max_steps=16**3,
        )

    get_solution(system, y0, tend, simulation_parameters)

    samples = 1
    with jax.profiler.trace("/tmp/carbox", create_perfetto_trace=True):
        start = datetime.now()
        for i in range(samples):
            solution = get_solution(system, y0, tend, simulation_parameters)
    print(
        f"Average time taken for {samples} samples: ",
        (datetime.now() - start) / samples,
    )

    # Extract the solution
    print(f"Solver report: {solution.stats}")
    sol_t = solution.ts
    sol_y = solution.ys.T

    # Plot the solution
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    df = pd.DataFrame(solution.ys)
    df.columns = reaction_network.species
    ions = [n for n in reaction_network.species if n[-1] == "+"]

    # Plot the abundances
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lss = ["-", "--", ":"]

    df = pd.DataFrame(solution.ys, index=solution.ts, columns=reaction_network.species)
    df.to_csv("carbox.csv")
    for i, lab in enumerate(reaction_network.species[:-1]):
        ax[0].loglog(
            solution.ts / spy,
            solution.ys[:, i],
            label=lab,
            color=colors[i % len(colors)],
            ls=lss[i // len(colors)],
        )

    # plt.loglog(solution.ts / spy, solution.ys[:, -1], label="Tgas", color="k")
    ax[0].legend(loc="best", ncol=2, fontsize=6)
    ax[0].set_xlim(1e-20, 1e6)

    conservation = (reaction_network.get_elemental_contents() @ df.T).T
    conservation.columns = ["C", "H", "O", "charge"]
    conservation.plot(loglog=True, ax=ax[1])

    # Reevaluate the function evaluations
    dy = jnp.zeros_like(sol_y)
    for i, (t, y) in enumerate(zip(sol_t, sol_y.T)):
        dy = dy.at[:, i].set(
            system(
                t,
                y,
                simulation_parameters["t_gas_init"],
                simulation_parameters["cr_rate"],
                simulation_parameters["gnot"],
            )
        )

    df = pd.DataFrame(dy).T
    df.columns = reaction_network.species
    df.index = sol_t
    df.to_csv("carbox_dy_no_heating.csv")

    rates = jnp.zeros((len(sol_t), len(reaction_network.reactions)))
    for i, (t, y) in enumerate(zip(sol_t, sol_y.T)):
        for j, reaction in enumerate(system.reactions):
            rates = rates.at[i, j].set(
                reaction(
                    simulation_parameters["t_gas_init"],
                    simulation_parameters["cr_rate"],
                    simulation_parameters["gnot"],
                )
            )
    df = pd.DataFrame(rates)
    df.columns = [r.reaction_type for r in reaction_network.reactions]
    df.index = sol_t
    df.to_csv("carbox_rates.csv")
