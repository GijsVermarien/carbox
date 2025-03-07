import pandas as pd
import numpy as np

from network import Network
from constants import elemental_dict
from reactions import KAReaction, CRReaction, FUVReaction, H2FormReaction

import diffrax as dx

GAS2DUST = 0.01

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


if __name__ == "__main__":
    reactions_file = pd.read_csv("data/simple_latent_tgas.csv")
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
        reaction_by_shorthand_name[reac[-1]](*reac)
        for idx, reac in reactions_file.iterrows()
    ]
    # Jreaction = [reaction() for reaction in reactions]

    reaction_network = Network(species, reactions)
    # print(reaction_network.incidence)
    system = reaction_network.get_ode()
    system(np.ones(16) * 1e-5, 10.0, 1e-17, 1e0)
