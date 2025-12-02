import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# HCO+
elements = ["C", "H", "O", "charge"]
elemental_content = np.array(
    [
        [0, 1, 0, 0],  # C
        [0, 1, 0, 1],  # C+
        [0, 1, 1, 0],  # CO
        [0, 1, 1, 1],  # CO+
        [0, 0, 0, -1],  # E
        [1, 0, 0, 0],  # H
        [1, 0, 0, 1],  # H+
        [2, 0, 0, 0],  # H2
        [2, 0, 1, 0],  # H2O
        [2, 0, 1, 1],  # H2O+
        [3, 0, 1, 1],  # H3O+
        [1, 1, 1, 1],  # HCO+
        [0, 0, 1, 0],  # O
        [0, 0, 1, 1],  # O+
        [1, 0, 1, 0],  # OH
        [1, 0, 1, 1],  # OH+
    ]
).T


def get_elemental_conservation(array):
    conservation = elemental_content @ array.T
    return conservation


if __name__ == "__main__":
    jax = pd.read_csv("jax/jax.csv", index_col=0)
    scipy = pd.read_csv("scipy/scipy.csv", index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(19, 7), sharex=True, sharey=True)
    ax = axes[0]
    ax.set_prop_cycle(plt.cycler("color", plt.cm.tab20.colors))
    scipy.plot(ax=ax, loglog=True)
    ax.set_prop_cycle(plt.cycler("color", plt.cm.tab20.colors))
    jax.plot(ax=ax, loglog=True, linestyle="--", legend=False)
    # handles, labels = ax.get_legend_handles_labels()
    # handles.extend([
    #     Line2D([0], [0], color='black', lw=2, label='scipy'),
    #     Line2D([0], [0], color='black', lw=2, linestyle='--', label='jax')
    # ])
    # ax.legend(handles=handles)
    ax.axvline(1e10)

    ax = axes[1]
    ax.axvline(1e10)
    scipy_conservation = get_elemental_conservation(scipy.values[:, :-1])
    jax_conservation = get_elemental_conservation(jax.values[:, :-1])

    ax.set_prop_cycle(plt.cycler("color", plt.cm.tab20.colors))
    ax.plot(scipy.index, scipy_conservation.T, label=["H", "C", "O", "charge"])
    ax.set_prop_cycle(plt.cycler("color", plt.cm.tab20.colors))
    ax.plot(jax.index, jax_conservation.T, linestyle="--", label="jax")
    ax.legend()

    # BITS to compare the heating and cooling rate by plotting them:
    # Enable plotting heatcool in both chem_ode files

    # scipy_hc = pd.read_csv("scipy/scipy_heatcool.txt")
    # scipy_hc.columns = ["name", "time", "heat", "cool"]
    # jax_hc = pd.read_csv("jax/jax_heatcool.txt")
    # jax_hc.columns = ["name", "time", "heat", "cool"]

    # ax = axes[2]

    # scipy_hc.plot(ax=ax, x="time", y=["heat", "cool"], loglog=True)
    # jax_hc.plot(ax=ax, x="time", y=["heat", "cool"], loglog=True, lw=0, marker=".")

    fig.savefig("compare.png")
