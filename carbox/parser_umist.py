from datetime import datetime

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import elemental_dict
from network import Network
from reactions import (
    CPReaction,
    CRPhotoReaction,
    KAReaction,
    PHReaction,
)

jax.config.update("jax_enable_x64", True)

#

reaction_by_shorthand_name = {
    r: lambda rtype,
    r1,
    r2,
    p1,
    p2,
    p3,
    p4,
    _,
    a,
    b,
    c,
    tmin,
    tmax,
    _1,
    _2,
    _3,
    _4,
    _5: KAReaction(rtype, (r1, r2), (p1, p2, p3, p4), float(a), float(b), float(c))
    for r in ["AD", "CD", "CE", "DR", "IN", "MN", "NN", "RA", "REA", "RR"]
} | {
    # CP: k = cr_rate
    "CP": lambda rtype,
    r1,
    r2,
    p1,
    p2,
    p3,
    p4,
    _,
    a,
    b,
    c,
    tmin,
    tmax,
    _1,
    _2,
    _3,
    _4,
    _5: CPReaction(rtype, (r1, r2), (p1, p2, p3, p4), float(a)),
    # PH: F_UV * exp(-gamma * visual_extinction)
    "PH": lambda rtype,
    r1,
    r2,
    p1,
    p2,
    p3,
    p4,
    _,
    a,
    b,
    c,
    tmin,
    tmax,
    _1,
    _2,
    _3,
    _4,
    _5: PHReaction(rtype, (r1, r2), (p1, p2, p3, p4), float(a), float(b), float(c)),
    # CR: cr_rate * (T/300)^beta gamma / (1-omega)
    "CR": lambda rtype,
    r1,
    r2,
    p1,
    p2,
    p3,
    p4,
    _,
    a,
    b,
    c,
    tmin,
    tmax,
    _1,
    _2,
    _3,
    _4,
    _5: CRPhotoReaction(
        rtype, (r1, r2), (p1, p2, p3, p4), float(a), float(b), float(c)
    ),
}


def parse_atoms(name, mode="mass"):
    if name == "e-":
        name = "E"
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


simulation_parameters = {
    # Hydrogen number density
    "number_density": 1e4,  # [1e2 - 1e6]
    # Fractional abunadnce of oxygen
    "O_fraction": 2e-4,  # [1e-5, 1e-3]
    # Fractional abundance of carbon
    "C_fraction": jnp.array(1e-4),  # [1e-5, 1e-3]
    # Cosmic ray ionisation rate
    "cr_rate": jnp.array(1e-17),  # enchance up to 1e-14
    # Radiation field
    "gnot": 0e0,  # enchance up to 1e5
    # t_gas_init
    "t_gas_init": jnp.array(800.0),  # [1e1, 1e6],
    # base visual extinction
    "base_av": jnp.array(0.5),  # Base Av to account for the edge visual extinction
}
simulation_parameters["visual_extinction"] = (
    simulation_parameters["base_av"]
    + 3.086e18 * simulation_parameters["number_density"]
) / 1.6e21

spy = 3600.0 * 24 * 365.0


if __name__ == "__main__":
    # reactions_file = pd.read_csv("../data/umist22.csv")
    reactions_file = pd.read_csv(
        "../data/umist22.csv",
        sep=":",
        skiprows=1,
        header=None,
        on_bad_lines=lambda x: x[:19],
        engine="python",
    )
    reactions_file = reactions_file.set_index(0)
    reactions_file.columns = [
        "reaction_type",
        "r1",
        "r2",
        "p1",
        "p2",
        "p3",
        "p4",
        "charge",
        "alpha",
        "beta",
        "gamma",
        "tmin",
        "tmax",
        "source",
        "accuracy",
        "link",
        "citation",
        "empty",
    ]

    # Get the unique species from the reaction file
    species = list(
        set(reactions_file["r1"])
        | set(reactions_file["r2"])
        | set(reactions_file["p1"])
        | set(reactions_file["p2"])
        | set(reactions_file["p3"])
        | set(reactions_file["p4"])
    )
    for to_remove in [np.nan, "CRPHOT", "CRP", "PHOTON", "nan"]:
        if to_remove in species:
            species.remove(to_remove)

    # Sort the species by atomic weight
    species = sorted(species, key=lambda x: parse_atoms(x, mode="atomic_weight")[1])

    # Parse the reactions:
    reactions = [
        reaction_by_shorthand_name[reac.iloc[0]](*reac)
        for idx, reac in reactions_file.iterrows()
    ]
    # Create the reaction network and get the ODE system
    print(datetime.now(), "Creating reaction network...")
    reaction_network = Network(species, reactions)
    print(datetime.now(), "Converting network to ODE.")
    system = reaction_network.get_ode()
    print(datetime.now(), "Network created.")

    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # # Plot the reaction network
    # ax.imshow(reaction_network.incidence.T)
    # ax.set_xticks(np.arange(len(reaction_network.species)))
    # ax.set_yticks(np.arange(len(reaction_network.reactions)))
    # ax.set_xticklabels(reaction_network.species, rotation=90)
    # ax.set_yticklabels(reaction_network.reactions)
    # # plt.show()
    initial_abundances = {
        "H2": 5.00e-01,
        "He": 0.85e-1,
        "CO": 4.00e-04,
        "N2": 2.00e-05,
        "CH4": 1.75e-06,
        "NH3": 3.00e-08,
        "H2S": 2.00e-09,
        "HCP": 1.25e-08,
        "H2O": 1.275e-06,
        "C2H2": 2.19e-05,
        "HCN": 2.045e-05,
        "CS": 0.53e-05,
        "SiC2": 0.935e-05,
        "HCl": 1.625e-07,
        "HF": 0.85e-08,
        "C2H4": 3.425e-08,
        "SiO": 2.51e-06,
        "SiS": 2.99e-06,
        "Mg": 0.5e-08,
        "Na": 0.5e-08,
        "Fe": 0.5e-08,
    }

    y0 = (
        jnp.ones(len(reaction_network.species))
        * 1e-25
        * simulation_parameters["number_density"]
    )

    for k, v in initial_abundances.items():
        y0 = y0.at[reaction_network.get_index(k)].set(
            v * simulation_parameters["number_density"]
        )

    # Index all charged molecules, then sum them to obtain the number of electrons:
    y0 = y0.at[reaction_network.get_index("e-")].set(
        sum(
            [
                y0[reaction_network.get_index(s)]
                for s in reaction_network.species
                if s[-1] == "+"
            ]
        )
    )

    tend = 6e4

    start = datetime.now()

    @eqx.filter_jit
    def get_solution(system, y0, tend, simulation_parameters):
        return dx.diffeqsolve(
            dx.ODETerm(
                lambda t, y, args: system(t, y, args[0], args[1], args[2], args[3])
            ),
            dx.Kvaerno5(),
            y0=y0,
            t0=0.0,
            t1=spy * tend + 1e11,
            dt0=1e-6,
            saveat=dx.SaveAt(ts=spy * jnp.logspace(-1, np.log10(tend), 100)),
            stepsize_controller=dx.PIDController(
                atol=1e-18,
                rtol=1e-12,
            ),
            args=[
                simulation_parameters["t_gas_init"],
                simulation_parameters["cr_rate"],
                simulation_parameters["gnot"],
                simulation_parameters["visual_extinction"],
            ],
            max_steps=16**4,
            throw=False,
        )

    solution = get_solution(system, y0, tend, simulation_parameters)
    print("Compilation + run time: ", datetime.now() - start)

    samples = 1
    # # with jax.profiler.trace("/tmp/carbox", create_perfetto_trace=True):
    start = datetime.now()
    for i in range(samples):
        solution = get_solution(system, y0, tend, simulation_parameters)
    print(
        f"Average time taken for only running {samples} samples: ",
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
            # ls=lss[i // len(colors)],
        )

    # plt.loglog(solution.ts / spy, solution.ys[:, -1], label="Tgas", color="k")
    ax[0].legend(loc="best", ncol=2, fontsize=6)
    ax[0].set_xlim(1e-20, 1e6)

    conservation = (reaction_network.get_elemental_contents() @ df.T).T
    conservation.columns = ["C", "H", "O", "charge"]
    conservation.plot(loglog=True, ax=ax[1])

    fig.savefig("test_umist.png", dpi=300)

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
                simulation_parameters["visual_extinction"],
            )
        )

    df = pd.DataFrame(dy).T
    df.columns = reaction_network.species
    df.index = sol_t
    df.to_csv("carbox_dy_no_heating.csv")

    rates = jnp.zeros((len(sol_t), reaction_network.reaction_count()))
    for i, (t, y) in enumerate(zip(sol_t, sol_y.T)):
        rates = rates.at[i].set(
            system.get_rates(
                simulation_parameters["t_gas_init"],
                simulation_parameters["cr_rate"],
                simulation_parameters["gnot"],
                simulation_parameters["visual_extinction"],
            )
        )
    df = pd.DataFrame(rates)
    df.columns = [r.reaction_type for r in reaction_network.reactions]
    df.index = sol_t
    df.to_csv("carbox_rates.csv")

    from jax import make_jaxpr

    with open("carbox_jaxpr.txt", "w") as f:
        f.write(str(make_jaxpr(system)(0.0, y0, 1e1, 1e-17, 1e0)))
